"""Single-device supervised finetuning from a base checkpoint with exact resume."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scratch_llm.data.hub import load_hub_parquet_cache
from scratch_llm.data.sft_sources import SFTConversationDataset, get_sft_dataset_spec
from scratch_llm.chat.loader import (
    FiniteSFTSource,
    SFT_LOADER_STATE_FORMAT,
    SFTConversationLoader,
    SFTMixtureEntry,
    build_fresh_sft_validation_loader,
    load_jsonl_conversation_source,
)
from scratch_llm.chat.rendering import CHAT_RENDERER_ID
from scratch_llm.config import ProjectConfig, SFTConfig, SFTSourceConfig
from scratch_llm.evaluation.sft_bpb import (
    SFTAssistantBPBCallback,
    SFTAssistantBPBResult,
    SFTValidationCheckpointState,
    SFTValidationError,
    advance_sft_validation_state,
    sft_validation_identity,
)
from scratch_llm.identity import canonical_json_identity, file_identity
from scratch_llm.diagnostics.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
)
from scratch_llm.diagnostics.oom import (
    OOMDiagnostic,
    PretrainingOOMError,
    diagnose_out_of_memory,
)
from scratch_llm.model import GPT
from scratch_llm.run import RunPaths
from scratch_llm.tokenization.artifacts import build_token_byte_lengths
from scratch_llm.tokenization.bpe import RegexBPETokenizer
from scratch_llm.tokenization.tokenizer import (
    NANOCHAT_SPECIAL_TOKENS,
    ByteTokenizer,
    Tokenizer,
)
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.tracking_state import TrackingState
from scratch_llm.training.checkpoint import (
    ExactTrainingState,
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from scratch_llm.training.loop import (
    OptimizerStepResult,
    derive_grad_accum_steps,
    run_training_steps,
)
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.rng_state import (
    capture_training_rng_state,
    restore_training_rng_state,
)
from scratch_llm.training.telemetry import peak_flops_basis_from_config
from scratch_llm.utils import get_device, set_seed


class SFTTrainingError(RuntimeError):
    """The requested SFT initialization, data, or continuation is unsafe."""


class SFTTrainingOOMError(PretrainingOOMError):
    """SFT command failure carrying shared, SFT-addressed OOM advice."""


@dataclass(frozen=True)
class SFTTrainingResult:
    """Artifacts and bounded step history produced by one SFT command."""

    paths: RunPaths
    metrics_path: Path
    checkpoint_path: Path
    base_checkpoint_identity: str
    initial_step: int
    final_step: int
    steps: tuple[OptimizerStepResult, ...]
    validation_state: SFTValidationCheckpointState | None
    validation_results: tuple[SFTAssistantBPBResult, ...]


@dataclass(frozen=True)
class _SFTRuntime:
    model: GPT
    tokenizer: Tokenizer
    optimizer: Optimizer
    scheduler: LRScheduler
    base_checkpoint_identity: str
    initial_step: int
    initial_total_training_time_seconds: float
    initial_total_training_flops: float
    validation_state: SFTValidationCheckpointState | None
    checkpoint_tracking_state: TrackingState | None
    continuation: ExactTrainingState | None


def build_sft_conversation_sources(
    source_configs: Sequence[SFTSourceConfig],
    *,
    config: SFTConfig,
) -> tuple[FiniteSFTSource, ...]:
    """Load only configured local JSONL or already-verified cache sources."""

    if not isinstance(config, SFTConfig):
        raise TypeError(f"config must be an SFTConfig, got {type(config).__name__}")
    config.validate()
    normalized = tuple(source_configs)
    if not normalized:
        raise SFTTrainingError("SFT conversation sources must not be empty")
    sources: list[FiniteSFTSource] = []
    for index, source_config in enumerate(normalized):
        if not isinstance(source_config, SFTSourceConfig):
            raise TypeError(
                f"source {index} must be an SFTSourceConfig, "
                f"got {type(source_config).__name__}"
            )
        source_config.validate(f"sft.sources.{index}")
        if source_config.kind == "jsonl":
            source: FiniteSFTSource = load_jsonl_conversation_source(
                source_config.path,
                shuffle=source_config.shuffle,
            )
        else:
            if source_config.dataset is None or source_config.split is None:
                raise AssertionError("validated Hub cache source lost its coordinates")
            spec = get_sft_dataset_spec(
                source_config.dataset,
                source_config.split,
            )
            cache = load_hub_parquet_cache(spec, source_config.path)
            source = SFTConversationDataset(
                cache,
                shuffle_buffer_size=config.shuffle_buffer_size,
                row_batch_size=config.row_batch_size,
                shuffle=source_config.shuffle,
            )
        sources.append(source)
    identities = [source.source_identity for source in sources]
    if len(set(identities)) != len(identities):
        raise SFTTrainingError("configured SFT sources require unique identities")
    return tuple(sources)


def sft_mixture_identity(
    sources: Sequence[FiniteSFTSource],
    *,
    seed: int,
    batch_size: int,
    max_seq_len: int,
    packing_buffer_size: int,
) -> str:
    """Return one canonical identity for a finite validation mixture view."""

    payload = {
        "batch_size": batch_size,
        "format": "scratch_llm_sft_validation_mixture_v1",
        "max_seq_len": max_seq_len,
        "packing_buffer_size": packing_buffer_size,
        "seed": seed,
        "source_identities": [source.source_identity for source in sources],
    }
    return canonical_json_identity(payload)


def run_sft_training(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    tracker: Tracker,
    base_checkpoint: str | Path | None = None,
    resume_from: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    allow_tracking_fork: bool = False,
    validation_runner: Callable[[int], SFTAssistantBPBResult] | None = None,
) -> SFTTrainingResult:
    """Initialize from base weights or exactly continue one SFT checkpoint."""

    _validate_request(
        config,
        paths=paths,
        tracker=tracker,
        base_checkpoint=base_checkpoint,
        resume_from=resume_from,
        progress=progress,
        allow_tracking_fork=allow_tracking_fork,
        validation_runner=validation_runner,
    )
    set_seed(config.run.seed)
    device = get_device(config.run.device)
    metrics_path = paths.run_dir / config.tracking.jsonl.path
    active_base_checkpoint = _resolve_base_checkpoint(
        config,
        base_checkpoint=base_checkpoint,
        resume_from=resume_from,
    )
    runtime = _initialize_runtime(
        config,
        paths=paths,
        metrics_path=metrics_path,
        tracker=tracker,
        device=device,
        base_checkpoint=active_base_checkpoint,
        resume_from=resume_from,
        allow_tracking_fork=allow_tracking_fork,
    )
    train_sources = build_sft_conversation_sources(
        config.sft.train_sources,
        config=config.sft,
    )
    validation_sources = build_sft_conversation_sources(
        config.sft.validation_sources,
        config=config.sft,
    )
    train_loader = SFTConversationLoader(
        tuple(
            SFTMixtureEntry(source, repeats=source_config.repeat_weight)
            for source, source_config in zip(
                train_sources,
                config.sft.train_sources,
                strict=True,
            )
        ),
        tokenizer=runtime.tokenizer,
        batch_size=config.sft.device_batch_size,
        max_seq_len=config.model.seq_len,
        packing_buffer_size=config.sft.packing_buffer_size,
        seed=config.run.seed,
        repeat=True,
    )
    validation_seed = config.run.seed + 1
    validation_mixture_identity = sft_mixture_identity(
        validation_sources,
        seed=validation_seed,
        batch_size=config.sft.device_batch_size,
        max_seq_len=config.model.seq_len,
        packing_buffer_size=config.sft.packing_buffer_size,
    )
    expected_validation_identity = sft_validation_identity(
        tokenizer_identity=runtime.tokenizer.get_identity(),
        renderer_identity=CHAT_RENDERER_ID,
        validation_mixture_identity=validation_mixture_identity,
        batch_budget=config.sft.eval_batches,
    )
    if (
        runtime.validation_state is not None
        and runtime.validation_state.validation_identity != expected_validation_identity
    ):
        raise SFTTrainingError(
            "resume checkpoint SFT validation identity changed: "
            f"{runtime.validation_state.validation_identity!r} != "
            f"{expected_validation_identity!r}"
        )
    if resume_from is not None:
        if runtime.continuation is None:  # pragma: no cover - stage requires exact.
            raise SFTTrainingError("SFT resume checkpoint has no exact continuation")
        _restore_exact_continuation(
            train_loader,
            runtime.continuation,
            device=device,
        )

    active_validation_runner = validation_runner
    if active_validation_runner is None:
        active_validation_runner = SFTAssistantBPBCallback(
            model=runtime.model,
            validation_loader_factory=lambda: build_fresh_sft_validation_loader(
                validation_sources,
                tokenizer=runtime.tokenizer,
                batch_size=config.sft.device_batch_size,
                max_seq_len=config.model.seq_len,
                packing_buffer_size=config.sft.packing_buffer_size,
                seed=validation_seed,
            ),
            token_bytes=build_token_byte_lengths(runtime.tokenizer),
            checkpoint_identity_prefix=f"sft:{paths.run_dir.resolve()}",
            tokenizer_identity=runtime.tokenizer.get_identity(),
            validation_mixture_identity=validation_mixture_identity,
            device=device,
            max_batches=config.sft.eval_batches,
        )
    lifecycle = _SFTCheckpointLifecycle(
        config=config,
        paths=paths,
        runtime=runtime,
        loader=train_loader,
        device=device,
        tracker=tracker,
        validation_runner=active_validation_runner,
        progress=progress,
    )
    train_config = config.sft.to_train_config(config.model.seq_len)
    grad_accum_steps = derive_grad_accum_steps(
        device_batch_size=train_config.device_batch_size,
        seq_len=config.model.seq_len,
        total_batch_size_tokens=train_config.total_batch_size_tokens,
    )
    try:
        step_results = run_training_steps(
            runtime.model,
            train_loader,
            runtime.optimizer,
            runtime.scheduler,
            max_steps=train_config.max_steps,
            grad_accum_steps=grad_accum_steps,
            grad_clip=train_config.grad_clip,
            device=device,
            tracker=tracker,
            log_every=train_config.log_every,
            on_step=lifecycle.on_step,
            initial_total_training_time_seconds=(
                runtime.initial_total_training_time_seconds
            ),
            initial_total_training_flops=runtime.initial_total_training_flops,
            initial_processed_model_tokens=(
                runtime.initial_step * train_config.total_batch_size_tokens
            ),
            peak_flops_basis=peak_flops_basis_from_config(train_config),
        )
    except torch.OutOfMemoryError as error:
        try:
            runtime.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        memory = _collect_memory_after_oom(config.run.device)
        diagnostic_config = replace(config, train=train_config)
        diagnostic = diagnose_out_of_memory(
            error,
            config=diagnostic_config,
            memory=memory,
        )
        if diagnostic is None:  # pragma: no cover - explicit torch OOM.
            raise
        _clear_accelerator_cache_after_oom(config.run.device)
        raise SFTTrainingOOMError(_sft_oom_diagnostic(diagnostic)) from error
    final_step = runtime.scheduler.last_epoch
    final_result = step_results[-1]
    checkpoint_path = lifecycle.finalize(final_step, final_result)
    return SFTTrainingResult(
        paths=paths,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        base_checkpoint_identity=runtime.base_checkpoint_identity,
        initial_step=runtime.initial_step,
        final_step=final_step,
        steps=tuple(step_results),
        validation_state=lifecycle.validation_state,
        validation_results=lifecycle.validation_results,
    )


def _validate_request(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    tracker: Tracker,
    base_checkpoint: str | Path | None,
    resume_from: str | Path | None,
    progress: Callable[[str], None] | None,
    allow_tracking_fork: bool,
    validation_runner: Callable[[int], SFTAssistantBPBResult] | None,
) -> None:
    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    config.validate()
    if not isinstance(paths, RunPaths):
        raise TypeError(f"paths must be RunPaths, got {type(paths).__name__}")
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if not isinstance(allow_tracking_fork, bool):
        raise TypeError("allow_tracking_fork must be a boolean")
    if validation_runner is not None and not callable(validation_runner):
        raise TypeError("validation_runner must be callable or None")
    if base_checkpoint is not None and resume_from is not None:
        raise SFTTrainingError(
            "base checkpoint initialization and SFT resume are mutually exclusive"
        )
    if config.sft.dtype != "float32":
        raise SFTTrainingError("SFT currently supports sft.dtype='float32' only")
    if config.sft.compile:
        raise SFTTrainingError("SFT does not support sft.compile yet")
    if config.sft.activation_checkpointing:
        raise SFTTrainingError("SFT does not support sft.activation_checkpointing yet")


def _resolve_base_checkpoint(
    config: ProjectConfig,
    *,
    base_checkpoint: str | Path | None,
    resume_from: str | Path | None,
) -> Path | None:
    if resume_from is not None:
        return None
    configured = (
        None if config.sft.base_checkpoint is None else Path(config.sft.base_checkpoint)
    )
    explicit = None if base_checkpoint is None else Path(base_checkpoint)
    if explicit is not None and configured is not None:
        if explicit.resolve() != configured.resolve():
            raise SFTTrainingError(
                "--base-checkpoint conflicts with sft.base_checkpoint"
            )
    selected = explicit or configured
    if selected is None:
        raise SFTTrainingError(
            "fresh SFT requires a base checkpoint via sft.base_checkpoint "
            "or --base-checkpoint"
        )
    return selected


def _initialize_runtime(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    metrics_path: Path,
    tracker: Tracker,
    device: torch.device,
    base_checkpoint: Path | None,
    resume_from: str | Path | None,
    allow_tracking_fork: bool,
) -> _SFTRuntime:
    tracking_state = tracker.checkpoint_state()
    train_config = config.sft.to_train_config(config.model.seq_len)
    if resume_from is None:
        _validate_existing_outputs(
            paths,
            metrics_path,
            initial_step=0,
            is_resume=False,
        )
        if base_checkpoint is None:  # pragma: no cover - resolved by caller.
            raise AssertionError("fresh SFT lost its base checkpoint")
        base_identity = file_identity(base_checkpoint)
        base = load_model_checkpoint(base_checkpoint, device=device)
        if base.training_stage != "pretrain":
            raise SFTTrainingError(
                "base initialization requires a pretraining checkpoint"
            )
        if base.config.model != config.model:
            raise SFTTrainingError(
                "base checkpoint architecture or sequence length conflicts "
                "with the SFT config"
            )
        _validate_tokenizer_contract(config, base.tokenizer)
        base.model.train()
        optimizer = build_optimizer(base.model, train_config)
        return _SFTRuntime(
            model=base.model,
            tokenizer=base.tokenizer,
            optimizer=optimizer,
            scheduler=build_lr_scheduler(optimizer, train_config),
            base_checkpoint_identity=base_identity,
            initial_step=0,
            initial_total_training_time_seconds=0.0,
            initial_total_training_flops=0.0,
            validation_state=None,
            checkpoint_tracking_state=tracking_state,
            continuation=None,
        )

    checkpoint = load_training_checkpoint(
        resume_from,
        device=device,
        expected_stage="sft",
    )
    _validate_resume_config(config, checkpoint.config)
    _validate_tokenizer_contract(config, checkpoint.tokenizer)
    if checkpoint.base_checkpoint_identity is None:  # pragma: no cover - stage check.
        raise SFTTrainingError("SFT checkpoint lost base-checkpoint provenance")
    if checkpoint.step >= config.sft.max_steps:
        raise SFTTrainingError(
            f"checkpoint step {checkpoint.step} has already reached "
            f"sft.max_steps={config.sft.max_steps}"
        )
    validation_state = checkpoint.validation
    if validation_state is not None and not isinstance(
        validation_state,
        SFTValidationCheckpointState,
    ):
        raise SFTTrainingError("SFT resume requires SFT validation state")
    checkpoint_tracking_state = tracking_state or checkpoint.tracking
    if (
        tracking_state is not None
        and tracking_state != checkpoint.tracking
        and not allow_tracking_fork
    ):
        raise SFTTrainingError(
            "enabled remote tracker identity does not match the SFT checkpoint; "
            "explicitly select a tracking fork"
        )
    _validate_existing_outputs(
        paths,
        metrics_path,
        initial_step=checkpoint.step,
        is_resume=True,
    )
    if checkpoint.continuation is None:  # pragma: no cover - SFT stage is exact.
        raise SFTTrainingError("SFT resume checkpoint has no exact continuation")
    return _SFTRuntime(
        model=checkpoint.model,
        tokenizer=checkpoint.tokenizer,
        optimizer=checkpoint.optimizer,
        scheduler=checkpoint.scheduler,
        base_checkpoint_identity=checkpoint.base_checkpoint_identity,
        initial_step=checkpoint.step,
        initial_total_training_time_seconds=(
            checkpoint.continuation.total_training_time_seconds
        ),
        initial_total_training_flops=checkpoint.continuation.total_training_flops,
        validation_state=validation_state,
        checkpoint_tracking_state=checkpoint_tracking_state,
        continuation=checkpoint.continuation,
    )


def _validate_tokenizer_contract(config: ProjectConfig, tokenizer: Tokenizer) -> None:
    if tokenizer.get_vocab_size() != config.tokenizer.vocab_size:
        raise SFTTrainingError(
            "base/SFT checkpoint tokenizer vocabulary conflicts with the config"
        )
    if tokenizer.get_special_tokens() != set(NANOCHAT_SPECIAL_TOKENS):
        raise SFTTrainingError(
            "base/SFT checkpoint tokenizer special tokens conflict with the config"
        )
    if config.tokenizer.type == "byte":
        expected_identity = ByteTokenizer().get_identity()
    else:
        if config.tokenizer.artifact_dir is None:
            raise SFTTrainingError("regex-BPE SFT requires tokenizer.artifact_dir")
        try:
            configured_tokenizer = RegexBPETokenizer.load(config.tokenizer.artifact_dir)
        except (OSError, TypeError, ValueError) as error:
            raise SFTTrainingError(
                f"could not load the configured SFT tokenizer artifacts: {error}"
            ) from error
        expected_identity = configured_tokenizer.get_identity()
    if tokenizer.get_identity() != expected_identity:
        raise SFTTrainingError(
            "base/SFT checkpoint tokenizer identity conflicts with the config"
        )


def _resume_comparison(config: ProjectConfig) -> dict[str, object]:
    values = config.to_dict()
    run = values["run"]
    if not isinstance(run, dict):  # pragma: no cover - config guarantees this.
        raise TypeError("resolved run configuration must be a dictionary")
    run.pop("name")
    run.pop("output_dir")
    return values


def _validate_resume_config(
    config: ProjectConfig,
    checkpoint_config: ProjectConfig,
) -> None:
    if _resume_comparison(config) != _resume_comparison(checkpoint_config):
        raise SFTTrainingError(
            "resume config must match the SFT checkpoint config except for "
            "run.name and run.output_dir"
        )


def _last_metric_step(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last_step: int | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError("record must be an object")
            if record.get("record_type") != "metrics":
                continue
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError("metric step must be an integer")
            if last_step is not None and step <= last_step:
                raise ValueError("metric steps must be strictly increasing")
            last_step = step
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SFTTrainingError(f"invalid metrics file {path}: {error}") from error
    return last_step


def _validate_existing_outputs(
    paths: RunPaths,
    metrics_path: Path,
    *,
    initial_step: int,
    is_resume: bool,
) -> None:
    checkpoints = sorted(paths.checkpoints_dir.glob("*.pt"))
    last_metric_step = _last_metric_step(metrics_path)
    if not is_resume:
        if checkpoints or last_metric_step is not None:
            raise SFTTrainingError(
                f"{paths.run_dir} already contains training outputs; "
                "use --resume or choose a new run.name"
            )
        return
    if last_metric_step is not None and last_metric_step > initial_step:
        raise SFTTrainingError(
            f"{metrics_path} already records step {last_metric_step}, "
            f"which is newer than resume step {initial_step}"
        )
    last_path = paths.checkpoints_dir / "last.pt"
    if last_path.exists():
        last = load_model_checkpoint(last_path, device="cpu")
        if last.step > initial_step:
            raise SFTTrainingError(
                f"{last_path} is at step {last.step}, which is newer than "
                f"resume step {initial_step}"
            )


def _capture_exact_continuation(
    loader: SFTConversationLoader,
    *,
    device: torch.device,
    step: int,
    result: OptimizerStepResult,
) -> ExactTrainingState:
    return ExactTrainingState(
        loader_format=SFT_LOADER_STATE_FORMAT,
        loader_state=loader.state_dict(),
        rng_state=capture_training_rng_state(device),
        tracker_step=step,
        total_training_time_seconds=result.total_training_time_seconds,
        total_training_flops=result.total_training_flops,
    )


def _restore_exact_continuation(
    loader: SFTConversationLoader,
    continuation: ExactTrainingState,
    *,
    device: torch.device,
) -> None:
    if continuation.loader_format != SFT_LOADER_STATE_FORMAT:
        raise SFTTrainingError(
            "checkpoint loader format does not match the SFT loader: "
            f"{continuation.loader_format!r}"
        )
    original_loader_state = loader.state_dict()
    original_rng_state = capture_training_rng_state(device)
    try:
        loader.load_state_dict(continuation.loader_state)
        restore_training_rng_state(continuation.rng_state, device=device)
    except Exception as error:
        rollback_errors: list[str] = []
        try:
            loader.load_state_dict(original_loader_state)
        except Exception as rollback_error:
            rollback_errors.append(f"loader rollback failed: {rollback_error}")
        try:
            restore_training_rng_state(original_rng_state, device=device)
        except Exception as rollback_error:
            rollback_errors.append(f"RNG rollback failed: {rollback_error}")
        suffix = f"; {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise SFTTrainingError(
            f"could not restore exact SFT continuation: {error}{suffix}"
        ) from error


class _SFTCheckpointLifecycle:
    """Coordinate strict assistant-BPB ranking with periodic and latest saves."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        paths: RunPaths,
        runtime: _SFTRuntime,
        loader: SFTConversationLoader,
        device: torch.device,
        tracker: Tracker,
        validation_runner: Callable[[int], SFTAssistantBPBResult],
        progress: Callable[[str], None] | None,
    ) -> None:
        self._config = config
        self._paths = paths
        self._runtime = runtime
        self._loader = loader
        self._device = device
        self._tracker = tracker
        self._validation_runner = validation_runner
        self._progress = progress
        self._validation_state = runtime.validation_state
        self._validation_results: list[SFTAssistantBPBResult] = []
        self._registered_events: set[str] = set()
        self._checkpoint_path = paths.checkpoints_dir / "last.pt"

    @property
    def validation_state(self) -> SFTValidationCheckpointState | None:
        return self._validation_state

    @property
    def validation_results(self) -> tuple[SFTAssistantBPBResult, ...]:
        return tuple(self._validation_results)

    def _save(
        self,
        path: Path,
        *,
        step: int,
        continuation: ExactTrainingState,
        validation: SFTValidationCheckpointState | None,
        role: Literal["best", "latest", "periodic"],
    ) -> Path:
        saved = save_checkpoint(
            path,
            model=self._runtime.model,
            optimizer=self._runtime.optimizer,
            scheduler=self._runtime.scheduler,
            config=self._config,
            step=step,
            tokenizer=self._runtime.tokenizer,
            continuation=continuation,
            validation=validation,
            tracking=self._runtime.checkpoint_tracking_state,
            training_stage="sft",
            base_checkpoint_identity=self._runtime.base_checkpoint_identity,
        )
        self._register(saved, role=role, step=step)
        return saved

    def _register(
        self,
        path: Path,
        *,
        role: Literal["best", "latest", "periodic"],
        step: int,
    ) -> None:
        event_id = f"sft-checkpoint:{role}:step:{step}"
        if event_id in self._registered_events:
            return
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"completed SFT checkpoint is not a file: {path}")
        relative = path.relative_to(self._paths.run_dir).as_posix()
        name = (
            f"sft_checkpoint_step_{step:06d}"
            if role == "periodic"
            else f"sft_checkpoint_{role}"
        )
        if isinstance(self._tracker, RunTracker):
            self._tracker.log_artifact_once(
                relative,
                name,
                "model",
                event_id=event_id,
            )
        else:
            self._tracker.log_artifact(relative, name, "model")
        self._registered_events.add(event_id)

    def _validate(
        self,
        step: int,
        result: OptimizerStepResult,
    ) -> ExactTrainingState | None:
        try:
            validation = self._validation_runner(step)
            decision = advance_sft_validation_state(
                self._validation_state,
                validation,
                validation_step=step,
            )
        except torch.OutOfMemoryError:
            raise
        except (SFTValidationError, TypeError, ValueError) as error:
            raise SFTTrainingError(
                f"SFT validation at step {step} failed: {error}"
            ) from error
        continuation = None
        if decision.improved:
            continuation = _capture_exact_continuation(
                self._loader,
                device=self._device,
                step=step,
                result=result,
            )
            self._save(
                self._paths.checkpoints_dir / "best.pt",
                step=step,
                continuation=continuation,
                validation=decision.state,
                role="best",
            )
        self._validation_state = decision.state
        self._validation_results.append(validation)
        if self._progress is not None:
            self._progress(
                f"SFT validation step {step}: assistant BPB={validation.bpb:.6f}"
            )
        return continuation

    def on_step(self, step: int, result: OptimizerStepResult) -> None:
        continuation = None
        if step % self._config.sft.eval_every == 0:
            continuation = self._validate(step, result)
        if step % self._config.sft.save_every != 0:
            return
        if continuation is None:
            continuation = _capture_exact_continuation(
                self._loader,
                device=self._device,
                step=step,
                result=result,
            )
        self._save(
            self._paths.checkpoints_dir / f"step_{step:06d}.pt",
            step=step,
            continuation=continuation,
            validation=self._validation_state,
            role="periodic",
        )
        self._save(
            self._checkpoint_path,
            step=step,
            continuation=continuation,
            validation=self._validation_state,
            role="latest",
        )

    def finalize(self, step: int, result: OptimizerStepResult) -> Path:
        continuation = None
        if (
            self._validation_state is None
            or self._validation_state.validation_step != step
        ):
            continuation = self._validate(step, result)
        if continuation is None:
            continuation = _capture_exact_continuation(
                self._loader,
                device=self._device,
                step=step,
                result=result,
            )
        return self._save(
            self._checkpoint_path,
            step=step,
            continuation=continuation,
            validation=self._validation_state,
            role="latest",
        )


def _collect_memory_after_oom(device: str) -> AcceleratorMemorySnapshot:
    try:
        return collect_accelerator_memory(device)
    except Exception as error:
        return AcceleratorMemorySnapshot(
            device=torch.device(device),
            available=False,
            unavailable_reason=f"memory collection failed after OOM: {error}",
        )


def _clear_accelerator_cache_after_oom(device: str) -> None:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError):
        return
    if resolved.type == "cuda" and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _sft_oom_diagnostic(diagnostic: OOMDiagnostic) -> OOMDiagnostic:
    recommendations = tuple(
        replace(
            recommendation,
            field=(
                recommendation.field.replace("train.", "sft.", 1)
                if recommendation.field.startswith("train.")
                else recommendation.field
            ),
            cli_overrides=tuple(
                override.replace("train.", "sft.", 1)
                if override.startswith("train.")
                else override
                for override in recommendation.cli_overrides
            ),
        )
        for recommendation in diagnostic.recommendations
    )
    return replace(diagnostic, recommendations=recommendations)


__all__ = [
    "SFTTrainingError",
    "SFTTrainingOOMError",
    "SFTTrainingResult",
    "build_sft_conversation_sources",
    "run_sft_training",
    "sft_mixture_identity",
]
