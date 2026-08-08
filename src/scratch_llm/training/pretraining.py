"""Command-level composition for tiny-text and production pretraining."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scratch_llm._validation import require_integer, require_positive_integer
from scratch_llm.attention_backends import (
    format_attention_selection,
    preflight_attention_backend,
)
from scratch_llm.diagnostics.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
)
from scratch_llm.evaluation.base_tracking import track_periodic_base_validation
from scratch_llm.training.best_checkpoint import (
    BEST_CHECKPOINT_RANKING_PROTOCOL_ID,
    BestCheckpointError,
    PeriodicValidationResult,
    ValidationCheckpointState,
    advance_validation_state,
    base_validation_identity,
)
from scratch_llm.training.activation_checkpointing import (
    configure_activation_checkpointing,
    format_activation_checkpoint_selection,
)
from scratch_llm.tokenization.bpe import RegexBPETokenizer
from scratch_llm.training.checkpoint import (
    ExactTrainingState,
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from scratch_llm.training.compilation import (
    ModelCompiler,
    build_compile_runtime,
    format_compile_selection,
    warmup_compiled_training,
)
from scratch_llm.config import ProjectConfig
from scratch_llm.data.loaders import NextTokenDataset, create_token_loader
from scratch_llm.evaluation.full_document_bpb import (
    FullDocumentProtocolConfig,
    evaluate_full_document_bpb,
)
from scratch_llm.model import GPT
from scratch_llm.evaluation.nanochat_bpb import (
    NanochatCompatibilityConfig,
    evaluate_nanochat_compatible_bpb,
)
from scratch_llm.diagnostics.oom import (
    PretrainingOOMError,
    diagnose_out_of_memory,
)
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.precision import (
    PrecisionError,
    PrecisionPolicy,
    build_precision_policy,
)
from scratch_llm.run import RunPaths
from scratch_llm.training.rng_state import (
    capture_training_rng_state,
    restore_training_rng_state,
)
from scratch_llm.tokenization.tokenizer import (
    NANOCHAT_SPECIAL_TOKENS,
    ByteTokenizer,
    Tokenizer,
)
from scratch_llm.data.tokenized import (
    TokenizedShardReader,
    tokenized_manifest_identity,
)
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.tracking_state import TrackingState
from scratch_llm.training.loop import (
    OptimizerStepResult,
    derive_grad_accum_steps,
    run_training_steps,
)
from scratch_llm.training.telemetry import peak_flops_basis_from_config
from scratch_llm.utils import get_device, set_seed


class PretrainingError(RuntimeError):
    """The requested pretraining composition is unsafe or unsupported."""


@dataclass(frozen=True)
class PretrainingResult:
    """Artifacts and bounded step history produced by one pretraining command."""

    paths: RunPaths
    metrics_path: Path
    checkpoint_path: Path
    initial_step: int
    final_step: int
    steps: tuple[OptimizerStepResult, ...]
    validation_state: ValidationCheckpointState | None
    validation_results: tuple[PeriodicValidationResult, ...]


_ValidationRunner = Callable[[int], PeriodicValidationResult]


@dataclass(frozen=True)
class _PreparedTrainingData:
    """Open data resources shared by initialization, training, and validation."""

    tokenizer: Tokenizer
    batches: Iterator[tuple[Tensor, Tensor]]
    loader: object
    production_reader: TokenizedShardReader | None
    training_tokens_per_epoch: int | None


@dataclass(frozen=True)
class _TrainingRuntime:
    """Model state and resume cursors needed by the training loop."""

    model: GPT
    tokenizer: Tokenizer
    optimizer: Optimizer
    scheduler: LRScheduler
    precision: PrecisionPolicy
    initial_step: int
    initial_total_training_time_seconds: float
    initial_total_training_flops: float
    validation_state: ValidationCheckpointState | None
    checkpoint_tracking_state: TrackingState | None


@dataclass(frozen=True)
class _PeriodicBaseValidator:
    """Run both pinned base-validation protocols against one model snapshot."""

    model: GPT
    tokenizer: Tokenizer
    reader: TokenizedShardReader
    token_bytes: Tensor
    parquet_dir: str
    run_dir: Path
    compatibility_config: NanochatCompatibilityConfig
    full_document_config: FullDocumentProtocolConfig
    device: torch.device

    def __call__(self, step: int) -> PeriodicValidationResult:
        checkpoint_identity = f"pretrain:{self.run_dir.resolve()}#step:{step}"
        compatibility = evaluate_nanochat_compatible_bpb(
            self.model,
            self.tokenizer,
            self.reader,
            self.token_bytes,
            parquet_dir=self.parquet_dir,
            checkpoint_identity=checkpoint_identity,
            config=self.compatibility_config,
            device=self.device,
        )
        full_document = evaluate_full_document_bpb(
            self.model,
            self.tokenizer,
            self.reader,
            self.token_bytes,
            checkpoint_identity=checkpoint_identity,
            config=self.full_document_config,
            device=self.device,
        )
        return PeriodicValidationResult(
            compatibility=compatibility,
            full_document=full_document,
        )


class TinyTextLoaderStateError(ValueError):
    """A tiny-text loader state is malformed or belongs to another corpus."""


_TINY_TEXT_LOADER_FORMAT = "scratch_llm_tiny_text_loader_state"
_TINY_TEXT_LOADER_FORMAT_VERSION = 1
_TINY_TEXT_LOADER_STATE_KEYS = frozenset(
    {
        "batch_size",
        "epoch",
        "epoch_seed",
        "format",
        "format_version",
        "position",
        "rng_state",
        "row_position",
        "seq_len",
        "source_identity",
    }
)
_MAX_TORCH_SEED = 2**63 - 1


class _TinyTextBatchLoader(Iterator[tuple[Tensor, Tensor]]):
    """Yield seeded shuffled tiny-text batches with exact next-batch state."""

    def __init__(
        self,
        dataset: NextTokenDataset,
        *,
        batch_size: int,
        seed: int,
        source_identity: str,
    ) -> None:
        if not isinstance(dataset, NextTokenDataset):
            raise TypeError("dataset must be a NextTokenDataset")
        batch_size = require_positive_integer(batch_size, name="batch_size")
        seed = require_integer(seed, name="seed")
        if not 0 <= seed <= _MAX_TORCH_SEED:
            raise ValueError("seed is outside the supported torch range")
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError("source_identity must be a non-empty string")
        usable_example_count = len(dataset) // batch_size * batch_size
        if usable_example_count == 0:
            raise ValueError(
                "tiny text must produce at least one complete device batch"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = dataset.seq_len
        self.source_identity = source_identity
        self.usable_example_count = usable_example_count
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._order: tuple[int, ...] = ()
        self.epoch = -1
        self.epoch_seed = 0
        self.position = 0
        self.row_position = 0
        self._start_next_epoch()

    def __iter__(self) -> _TinyTextBatchLoader:
        return self

    def __len__(self) -> int:
        return self.usable_example_count // self.batch_size

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self.row_position == len(self._order):
            self._start_next_epoch()
        indices = self._order[self.row_position : self.row_position + self.batch_size]
        if len(indices) != self.batch_size:
            raise RuntimeError("tiny-text epoch ended with an incomplete batch")
        examples = [self.dataset[index] for index in indices]
        inputs = torch.stack([example[0] for example in examples])
        targets = torch.stack([example[1] for example in examples])
        self.row_position += self.batch_size
        self.position += self.batch_size
        return inputs, targets

    def state_dict(self) -> dict[str, object]:
        """Return JSON-compatible exact next-batch state."""

        return {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "epoch_seed": self.epoch_seed,
            "format": _TINY_TEXT_LOADER_FORMAT,
            "format_version": _TINY_TEXT_LOADER_FORMAT_VERSION,
            "position": self.position,
            "rng_state": self._generator.get_state().tolist(),
            "row_position": self.row_position,
            "seq_len": self.seq_len,
            "source_identity": self.source_identity,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate a candidate completely before changing loader state."""

        if not isinstance(state, Mapping):
            raise TinyTextLoaderStateError("tiny-text loader state must be a mapping")
        if set(state) != _TINY_TEXT_LOADER_STATE_KEYS:
            missing = sorted(_TINY_TEXT_LOADER_STATE_KEYS - set(state))
            unexpected = sorted(set(state) - _TINY_TEXT_LOADER_STATE_KEYS)
            raise TinyTextLoaderStateError(
                "tiny-text loader state fields do not match format version 1; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if state["format"] != _TINY_TEXT_LOADER_FORMAT:
            raise TinyTextLoaderStateError("unknown tiny-text loader state format")
        if state["format_version"] != _TINY_TEXT_LOADER_FORMAT_VERSION:
            raise TinyTextLoaderStateError(
                "unknown tiny-text loader state format version"
            )
        for name, expected in (
            ("batch_size", self.batch_size),
            ("seq_len", self.seq_len),
            ("source_identity", self.source_identity),
        ):
            if state[name] != expected:
                raise TinyTextLoaderStateError(
                    f"tiny-text loader state {name} does not match this loader"
                )
        epoch = _tiny_state_integer(state["epoch"], name="epoch")
        epoch_seed = _tiny_state_integer(state["epoch_seed"], name="epoch_seed")
        position = _tiny_state_integer(state["position"], name="position")
        row_position = _tiny_state_integer(
            state["row_position"],
            name="row_position",
        )
        if epoch < 0:
            raise TinyTextLoaderStateError(
                "tiny-text loader epoch must be non-negative"
            )
        if not 0 <= epoch_seed <= _MAX_TORCH_SEED:
            raise TinyTextLoaderStateError("tiny-text loader epoch_seed is invalid")
        if position < 0 or position % self.batch_size:
            raise TinyTextLoaderStateError(
                "tiny-text loader position must be batch-aligned"
            )
        if (
            row_position < 0
            or row_position > self.usable_example_count
            or row_position % self.batch_size
        ):
            raise TinyTextLoaderStateError(
                "tiny-text loader row_position must be an in-epoch batch offset"
            )
        rng_state = _tiny_rng_state(state["rng_state"])
        candidate_generator = torch.Generator(device="cpu")
        try:
            candidate_generator.set_state(rng_state)
        except RuntimeError as error:
            raise TinyTextLoaderStateError(
                f"tiny-text loader rng_state is invalid: {error}"
            ) from error
        order = self._build_order(epoch_seed)

        self._generator.set_state(rng_state)
        self._order = order
        self.epoch = epoch
        self.epoch_seed = epoch_seed
        self.position = position
        self.row_position = row_position

    def _start_next_epoch(self) -> None:
        epoch_seed = int(
            torch.randint(
                0,
                _MAX_TORCH_SEED,
                (1,),
                generator=self._generator,
                dtype=torch.int64,
                device="cpu",
            ).item()
        )
        self._order = self._build_order(epoch_seed)
        self.epoch += 1
        self.epoch_seed = epoch_seed
        self.row_position = 0

    def _build_order(self, epoch_seed: int) -> tuple[int, ...]:
        order_generator = torch.Generator(device="cpu").manual_seed(epoch_seed)
        return tuple(
            torch.randperm(
                len(self.dataset),
                generator=order_generator,
                dtype=torch.int64,
                device="cpu",
            )[: self.usable_example_count].tolist()
        )


def _tiny_state_integer(value: object, *, name: str) -> int:
    try:
        return require_integer(value, name=name)
    except TypeError as error:
        raise TinyTextLoaderStateError(
            f"tiny-text loader state {name} must be an integer"
        ) from error


def _tiny_rng_state(value: object) -> Tensor:
    if not isinstance(value, list) or not value:
        raise TinyTextLoaderStateError(
            "tiny-text loader rng_state must be a non-empty byte list"
        )
    if any(
        not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 255
        for item in value
    ):
        raise TinyTextLoaderStateError(
            "tiny-text loader rng_state must contain integer bytes"
        )
    return torch.tensor(value, dtype=torch.uint8, device="cpu")


def _read_tiny_text(config: ProjectConfig) -> str:
    if config.data.profile != "tiny_text":
        raise PretrainingError(
            "the first-sprint pretrain command requires data.profile='tiny_text'"
        )
    source = Path(config.data.base_dir) / "fixtures" / "tiny.txt"
    try:
        return source.read_text(encoding="utf-8")
    except OSError as error:
        raise PretrainingError(
            f"could not read tiny-text corpus {source}: {error}"
        ) from error


def _build_batches(
    text: str,
    config: ProjectConfig,
    tokenizer: Tokenizer,
) -> _TinyTextBatchLoader:
    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(
        token_ids,
        config.model.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
    )
    if len(dataset) < config.train.device_batch_size:
        raise PretrainingError(
            "tiny text must produce at least one complete device batch; "
            f"found {len(dataset)} examples for batch size "
            f"{config.train.device_batch_size}"
        )
    source_identity = (
        "sha256:"
        + hashlib.sha256(
            text.encode("utf-8") + b"\0" + tokenizer.get_identity().encode("utf-8")
        ).hexdigest()
    )
    return _TinyTextBatchLoader(
        dataset,
        batch_size=config.train.device_batch_size,
        seed=config.run.seed,
        source_identity=source_identity,
    )


def prepare_pretraining_batch(
    batch: tuple[Tensor, ...] | list[Tensor],
    *,
    strategy: Literal["flat", "packed"],
) -> tuple[Tensor, Tensor]:
    """Normalize one flat or packed loader batch for the shared model loop.

    Packed padding positions are converted to the model's ``ignore_index=-1``
    target without mutating the loader-owned target tensor.
    """

    if strategy not in ("flat", "packed"):
        raise ValueError(f"strategy must be 'flat' or 'packed', got {strategy!r}")
    expected_values = 2 if strategy == "flat" else 3
    if not isinstance(batch, (tuple, list)) or len(batch) != expected_values:
        raise TypeError(
            f"{strategy} batches must contain exactly {expected_values} tensors"
        )
    if any(not isinstance(value, Tensor) for value in batch):
        raise TypeError(f"{strategy} batch values must all be Tensors")

    inputs, targets = batch[:2]
    if inputs.shape != targets.shape:
        raise ValueError(
            "pretraining input and target shapes must match; "
            f"got {tuple(inputs.shape)} and {tuple(targets.shape)}"
        )
    if strategy == "flat":
        return inputs, targets

    loss_mask = batch[2]
    if loss_mask.dtype != torch.bool:
        raise TypeError(
            f"packed loss mask must have dtype torch.bool, got {loss_mask.dtype}"
        )
    if loss_mask.shape != targets.shape:
        raise ValueError(
            "packed loss mask shape must match targets; "
            f"got {tuple(loss_mask.shape)} and {tuple(targets.shape)}"
        )
    masked_targets = targets.clone()
    masked_targets.masked_fill_(~loss_mask, -1)
    return inputs, masked_targets


class PreparedPretrainingBatchIterator(Iterator[tuple[Tensor, Tensor]]):
    """Adapt one infinite token loader to the shared two-tensor loop."""

    def __init__(
        self,
        batches: Iterator[tuple[Tensor, ...]],
        *,
        strategy: Literal["flat", "packed"],
    ) -> None:
        self._batches = batches
        self._strategy = strategy

    def __iter__(self) -> PreparedPretrainingBatchIterator:
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        return prepare_pretraining_batch(
            next(self._batches),
            strategy=self._strategy,
        )


def _validate_tiny_text_config(config: ProjectConfig) -> None:
    config.validate()
    if config.tokenizer.type != "byte":
        raise PretrainingError("tiny-text pretraining requires tokenizer.type='byte'")
    tokenizer = ByteTokenizer()
    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise PretrainingError(
            "tiny-text pretraining requires tokenizer.vocab_size="
            f"{tokenizer.get_vocab_size()}"
        )
    if tuple(config.tokenizer.special_tokens) != NANOCHAT_SPECIAL_TOKENS:
        raise PretrainingError(
            "tiny-text pretraining requires the ByteTokenizer special-token order"
        )


def validate_production_pretraining_config(config: ProjectConfig) -> None:
    config.validate()
    if config.data.profile != "nanochat_climbmix":
        raise PretrainingError(
            "production pretraining requires data.profile='nanochat_climbmix'"
        )
    if config.tokenizer.type != "regex_byte_bpe":
        raise PretrainingError(
            "production pretraining requires tokenizer.type='regex_byte_bpe'"
        )
    if config.tokenizer.artifact_dir is None:
        raise PretrainingError("production pretraining requires tokenizer.artifact_dir")


def load_production_tokenizer(config: ProjectConfig) -> RegexBPETokenizer:
    artifact_dir = config.tokenizer.artifact_dir
    if artifact_dir is None:  # pragma: no cover - validated by the caller.
        raise PretrainingError("production pretraining requires tokenizer.artifact_dir")
    tokenizer = RegexBPETokenizer.load(artifact_dir)
    if tokenizer.get_vocab_size() != config.tokenizer.vocab_size:
        raise PretrainingError(
            f"tokenizer artifact {artifact_dir} has vocabulary size "
            f"{tokenizer.get_vocab_size()}, but config requires "
            f"{config.tokenizer.vocab_size}"
        )
    if tokenizer.get_special_tokens() != set(config.tokenizer.special_tokens):
        raise PretrainingError(
            f"tokenizer artifact {artifact_dir} special tokens do not match "
            "the resolved config"
        )
    actual_special_ids = [
        tokenizer.encode_special(token) for token in config.tokenizer.special_tokens
    ]
    expected_special_ids = list(
        range(
            config.tokenizer.vocab_size - len(config.tokenizer.special_tokens),
            config.tokenizer.vocab_size,
        )
    )
    if actual_special_ids != expected_special_ids:
        raise PretrainingError(
            f"tokenizer artifact {artifact_dir} special-token IDs do not match "
            "the canonical vocabulary-final range"
        )
    return tokenizer


def _load_periodic_validation_token_bytes(config: ProjectConfig) -> Tensor:
    artifact_dir = config.tokenizer.artifact_dir
    if artifact_dir is None:
        raise PretrainingError(
            "periodic BPB validation requires tokenizer.artifact_dir"
        )
    path = Path(artifact_dir) / "token_bytes.pt"
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PretrainingError(
            f"could not load periodic validation token bytes {path}: {error}"
        ) from error
    if not isinstance(value, Tensor):
        raise PretrainingError(
            f"periodic validation token bytes {path} must contain a Tensor"
        )
    return value


def _resume_comparison(config: ProjectConfig) -> dict[str, Any]:
    values = config.to_dict()
    run = values["run"]
    if not isinstance(run, dict):  # pragma: no cover - ProjectConfig guarantees this.
        raise TypeError("resolved run configuration must be a dictionary")
    run.pop("name")
    run.pop("output_dir")
    return values


def _validate_resume_config(
    config: ProjectConfig,
    checkpoint_config: ProjectConfig,
) -> None:
    if _resume_comparison(config) != _resume_comparison(checkpoint_config):
        raise PretrainingError(
            "resume config must match the checkpoint config except for "
            "run.name and run.output_dir"
        )


def _last_metric_step(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last_step: int | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            if record.get("record_type") != "metrics":
                continue
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError("metric step must be an integer")
            if last_step is not None and step < last_step:
                raise ValueError("metric steps must be monotonically non-decreasing")
            last_step = step
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PretrainingError(f"invalid metrics file {path}: {error}") from error
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
            raise PretrainingError(
                f"{paths.run_dir} already contains training outputs; "
                "use --resume or choose a new run.name"
            )
        return

    if last_metric_step is not None and last_metric_step > initial_step:
        raise PretrainingError(
            f"{metrics_path} already records step {last_metric_step}, "
            f"which is newer than resume step {initial_step}"
        )
    last_checkpoint_path = paths.checkpoints_dir / "last.pt"
    if last_checkpoint_path.exists():
        last_checkpoint = load_model_checkpoint(
            last_checkpoint_path,
            device="cpu",
        )
        if last_checkpoint.step > initial_step:
            raise PretrainingError(
                f"{last_checkpoint_path} is at step {last_checkpoint.step}, "
                f"which is newer than resume step {initial_step}"
            )


def run_pretraining(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    tracker: Tracker,
    resume_from: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    allow_non_exact_resume: bool = False,
    allow_tracking_fork: bool = False,
    validation_runner: Callable[[int], PeriodicValidationResult] | None = None,
    compiler: ModelCompiler | None = None,
) -> PretrainingResult:
    """Run pretraining and transform only supported PyTorch OOM failures."""

    try:
        return _run_pretraining_impl(
            config,
            paths=paths,
            tracker=tracker,
            resume_from=resume_from,
            progress=progress,
            allow_non_exact_resume=allow_non_exact_resume,
            allow_tracking_fork=allow_tracking_fork,
            validation_runner=validation_runner,
            compiler=compiler,
        )
    except torch.OutOfMemoryError as error:
        memory = _collect_memory_after_oom(config.run.device)
        diagnostic = diagnose_out_of_memory(
            error,
            config=config,
            memory=memory,
        )
        assert diagnostic is not None
        _clear_accelerator_cache_after_oom(config.run.device)
        raise PretrainingOOMError(diagnostic) from error


def _run_pretraining_impl(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    tracker: Tracker,
    resume_from: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    allow_non_exact_resume: bool = False,
    allow_tracking_fork: bool = False,
    validation_runner: Callable[[int], PeriodicValidationResult] | None = None,
    compiler: ModelCompiler | None = None,
) -> PretrainingResult:
    """Train or resume either supported data profile through one shared loop."""

    if compiler is not None and not callable(compiler):
        raise TypeError("compiler must be callable or None")
    _validate_pretraining_request(
        config,
        paths=paths,
        tracker=tracker,
        resume_from=resume_from,
        allow_non_exact_resume=allow_non_exact_resume,
        allow_tracking_fork=allow_tracking_fork,
        validation_runner=validation_runner,
    )

    tracking_state = tracker.checkpoint_state()
    device = get_device(config.run.device)
    try:
        precision = build_precision_policy(dtype=config.train.dtype, device=device)
    except PrecisionError as error:
        raise PretrainingError(
            f"invalid pretraining precision policy: {error}"
        ) from error
    attention_preflight = preflight_attention_backend(
        config.model,
        device=device,
        dtype=config.train.dtype,
        training=True,
    )
    if progress is not None:
        progress(format_attention_selection(attention_preflight.selection))
    set_seed(config.run.seed)
    metrics_path = paths.run_dir / config.tracking.jsonl.path

    with ExitStack() as resources:
        data = _prepare_training_data(
            config,
            resources=resources,
            progress=progress,
        )
        runtime = _initialize_training_runtime(
            config,
            paths=paths,
            metrics_path=metrics_path,
            data=data,
            device=device,
            precision=precision,
            resume_from=resume_from,
            allow_non_exact_resume=allow_non_exact_resume,
            tracking_state=tracking_state,
            allow_tracking_fork=allow_tracking_fork,
        )
        activation_checkpoint_selection = configure_activation_checkpointing(
            runtime.model,
            enabled=config.train.activation_checkpointing,
        )
        if progress is not None:
            progress(
                format_activation_checkpoint_selection(activation_checkpoint_selection)
            )
        compile_runtime = build_compile_runtime(
            runtime.model,
            config.train,
            compiler=compiler,
        )
        warmup_tokens = torch.zeros(
            (config.train.device_batch_size, config.model.seq_len),
            dtype=torch.long,
            device=device,
        )
        warmup_compiled_training(
            compile_runtime,
            runtime.optimizer,
            inputs=warmup_tokens,
            targets=warmup_tokens,
            precision=runtime.precision,
            device=device,
        )
        if progress is not None:
            progress(format_compile_selection(compile_runtime.selection))
        active_validation_runner = _resolve_periodic_validation_runner(
            config,
            paths=paths,
            data=data,
            runtime=runtime,
            device=device,
            validation_runner=validation_runner,
        )
        checkpoints = _CheckpointLifecycle(
            config=config,
            paths=paths,
            runtime=runtime,
            loader=data.loader,
            device=device,
            tracker=tracker,
            validation_runner=active_validation_runner,
            progress=progress,
        )
        step_results = _execute_training(
            config,
            data=data,
            runtime=runtime,
            device=device,
            tracker=tracker,
            on_step=checkpoints.on_step,
            execution_model=compile_runtime.execution_model,
        )
        if progress is not None:
            progress(format_compile_selection(compile_runtime.selection))
        observed_attention = runtime.model.attention_backend_selection()
        if progress is not None and observed_attention != attention_preflight.selection:
            progress(format_attention_selection(observed_attention))
        final_step = runtime.scheduler.last_epoch
        final_result = step_results[-1]
        checkpoint_path = checkpoints.finalize(final_step, final_result)
        return PretrainingResult(
            paths=paths,
            metrics_path=metrics_path,
            checkpoint_path=checkpoint_path,
            initial_step=runtime.initial_step,
            final_step=final_step,
            steps=tuple(step_results),
            validation_state=checkpoints.validation_state,
            validation_results=checkpoints.validation_results,
        )


def _validate_pretraining_request(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    tracker: Tracker,
    resume_from: str | Path | None,
    allow_non_exact_resume: bool,
    allow_tracking_fork: bool,
    validation_runner: _ValidationRunner | None,
) -> None:
    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    if not isinstance(paths, RunPaths):
        raise TypeError(f"paths must be RunPaths, got {type(paths).__name__}")
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")
    if not isinstance(allow_non_exact_resume, bool):
        raise TypeError("allow_non_exact_resume must be a boolean")
    if not isinstance(allow_tracking_fork, bool):
        raise TypeError("allow_tracking_fork must be a boolean")
    if validation_runner is not None and not callable(validation_runner):
        raise TypeError("validation_runner must be callable or None")
    if allow_non_exact_resume and resume_from is None:
        raise PretrainingError(
            "allow_non_exact_resume requires an explicit resume checkpoint"
        )
    if config.data.profile == "tiny_text":
        _validate_tiny_text_config(config)
    elif config.data.profile == "nanochat_climbmix":
        validate_production_pretraining_config(config)
    else:
        raise PretrainingError(
            "pretraining data.profile must be 'tiny_text' or "
            f"'nanochat_climbmix', got {config.data.profile!r}"
        )


def _prepare_training_data(
    config: ProjectConfig,
    *,
    resources: ExitStack,
    progress: Callable[[str], None] | None,
) -> _PreparedTrainingData:
    if config.data.profile == "tiny_text":
        prepared_tokenizer = ByteTokenizer()
        tiny_loader = _build_batches(
            _read_tiny_text(config),
            config,
            prepared_tokenizer,
        )
        return _PreparedTrainingData(
            tokenizer=prepared_tokenizer,
            batches=tiny_loader,
            loader=tiny_loader,
            production_reader=None,
            training_tokens_per_epoch=None,
        )

    production_tokenizer = load_production_tokenizer(config)
    reader = resources.enter_context(
        TokenizedShardReader(
            config.data.tokenized_dir,
            tokenizer=production_tokenizer,
        )
    )
    production_loader = create_token_loader(
        reader,
        strategy=config.data.loader_strategy,
        split="train",
        batch_size=config.train.device_batch_size,
        seq_len=config.model.seq_len,
        seed=config.run.seed,
        planning_progress=progress,
    )
    return _PreparedTrainingData(
        tokenizer=production_tokenizer,
        batches=PreparedPretrainingBatchIterator(
            iter(production_loader),  # type: ignore[arg-type]
            strategy=config.data.loader_strategy,
        ),
        loader=production_loader,
        production_reader=reader,
        training_tokens_per_epoch=reader.manifest.splits["train"].token_count,
    )


def _initialize_training_runtime(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    metrics_path: Path,
    data: _PreparedTrainingData,
    device: torch.device,
    precision: PrecisionPolicy,
    resume_from: str | Path | None,
    allow_non_exact_resume: bool,
    tracking_state: TrackingState | None,
    allow_tracking_fork: bool,
) -> _TrainingRuntime:
    if resume_from is None:
        _validate_existing_outputs(
            paths,
            metrics_path,
            initial_step=0,
            is_resume=False,
        )
        model = GPT(config.model).to(device)
        optimizer = build_optimizer(model, config.train)
        return _TrainingRuntime(
            model=model,
            tokenizer=data.tokenizer,
            optimizer=optimizer,
            scheduler=build_lr_scheduler(optimizer, config.train),
            precision=precision,
            initial_step=0,
            initial_total_training_time_seconds=0.0,
            initial_total_training_flops=0.0,
            validation_state=None,
            checkpoint_tracking_state=tracking_state,
        )

    checkpoint = load_training_checkpoint(
        resume_from,
        device=device,
        allow_non_exact_resume=allow_non_exact_resume,
        expected_stage="pretrain",
        expected_precision=precision.checkpoint_state(),
    )
    checkpoint_tracking_state = tracking_state or checkpoint.tracking
    if (
        tracking_state is not None
        and tracking_state != checkpoint.tracking
        and not allow_tracking_fork
    ):
        raise PretrainingError(
            "enabled remote tracker identity does not match the resume "
            "checkpoint; explicitly select a tracking fork"
        )
    _validate_resume_config(config, checkpoint.config)
    if checkpoint.precision is None:  # pragma: no cover - exact resume requires it.
        raise PretrainingError("exact resume checkpoint has no precision state")
    try:
        precision.load_checkpoint_state(checkpoint.precision)
    except PrecisionError as error:
        raise PretrainingError(f"could not restore precision state: {error}") from error
    if checkpoint.tokenizer.get_identity() != data.tokenizer.get_identity():
        raise PretrainingError(
            "resume checkpoint tokenizer identity does not match "
            "the configured training artifacts"
        )
    validation_state = checkpoint.validation
    if validation_state is not None and not isinstance(
        validation_state,
        ValidationCheckpointState,
    ):
        raise PretrainingError(
            "pretraining resume requires base-validation checkpoint state"
        )
    if (
        validation_state is not None
        and validation_state.ranking_protocol_id != BEST_CHECKPOINT_RANKING_PROTOCOL_ID
    ):
        raise PretrainingError(
            "resume checkpoint ranking protocol changed: "
            f"{validation_state.ranking_protocol_id!r} != "
            f"{BEST_CHECKPOINT_RANKING_PROTOCOL_ID!r}"
        )
    if checkpoint.step >= config.train.max_steps:
        raise PretrainingError(
            f"checkpoint step {checkpoint.step} has already reached "
            f"train.max_steps={config.train.max_steps}"
        )
    _validate_existing_outputs(
        paths,
        metrics_path,
        initial_step=checkpoint.step,
        is_resume=True,
    )
    if checkpoint.continuation is None:
        total_training_time_seconds = 0.0
        total_training_flops = 0.0
    else:
        _restore_exact_continuation(
            data.loader,
            checkpoint.continuation,
            device=device,
        )
        total_training_time_seconds = (
            checkpoint.continuation.total_training_time_seconds
        )
        total_training_flops = checkpoint.continuation.total_training_flops
    return _TrainingRuntime(
        model=checkpoint.model,
        tokenizer=checkpoint.tokenizer,
        optimizer=checkpoint.optimizer,
        scheduler=checkpoint.scheduler,
        precision=precision,
        initial_step=checkpoint.step,
        initial_total_training_time_seconds=total_training_time_seconds,
        initial_total_training_flops=total_training_flops,
        validation_state=validation_state,
        checkpoint_tracking_state=checkpoint_tracking_state,
    )


def _resolve_periodic_validation_runner(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    data: _PreparedTrainingData,
    runtime: _TrainingRuntime,
    device: torch.device,
    validation_runner: _ValidationRunner | None,
) -> _ValidationRunner | None:
    reader = data.production_reader
    if reader is None:
        return validation_runner

    compatibility_config = NanochatCompatibilityConfig(
        device_batch_size=config.train.device_batch_size,
        context_length=config.model.seq_len,
        eval_tokens=config.train.eval_tokens,
    )
    full_document_config = FullDocumentProtocolConfig(
        device_batch_size=config.train.device_batch_size,
        context_length=config.model.seq_len,
    )
    expected_validation_identity = base_validation_identity(
        tokenizer_identity=runtime.tokenizer.get_identity(),
        validation_manifest_identity=tokenized_manifest_identity(reader.manifest),
        compatibility_reference_config=compatibility_config.to_dict(),
        full_document_reference_config=full_document_config.to_dict(),
    )
    if (
        runtime.validation_state is not None
        and validation_runner is None
        and runtime.validation_state.validation_identity != expected_validation_identity
    ):
        raise PretrainingError(
            "resume checkpoint validation identity changed: "
            f"{runtime.validation_state.validation_identity!r} != "
            f"{expected_validation_identity!r}"
        )
    if validation_runner is not None:
        return validation_runner
    return _PeriodicBaseValidator(
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        reader=reader,
        token_bytes=_load_periodic_validation_token_bytes(config),
        parquet_dir=config.data.parquet_dir,
        run_dir=paths.run_dir,
        compatibility_config=compatibility_config,
        full_document_config=full_document_config,
        device=device,
    )


class _CheckpointLifecycle:
    """Coordinate validation state with best, periodic, and final checkpoints."""

    def __init__(
        self,
        *,
        config: ProjectConfig,
        paths: RunPaths,
        runtime: _TrainingRuntime,
        loader: object,
        device: torch.device,
        tracker: Tracker,
        validation_runner: _ValidationRunner | None,
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
        self._validation_results: list[PeriodicValidationResult] = []
        self._registered_checkpoint_events: set[str] = set()
        self._checkpoint_path = paths.checkpoints_dir / "last.pt"

    @property
    def validation_state(self) -> ValidationCheckpointState | None:
        return self._validation_state

    @property
    def validation_results(self) -> tuple[PeriodicValidationResult, ...]:
        return tuple(self._validation_results)

    def _report_validation_failure(self, step: int, reason: object) -> None:
        if self._progress is not None:
            self._progress(f"Validation at step {step} was not accepted: {reason}")

    def _save(
        self,
        path: Path,
        *,
        step: int,
        continuation: ExactTrainingState,
        validation: ValidationCheckpointState | None,
        role: Literal["best", "latest", "periodic"],
    ) -> Path:
        checkpoint_path = save_checkpoint(
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
            precision=self._runtime.precision.checkpoint_state(),
        )
        self._register_checkpoint(checkpoint_path, role=role, step=step)
        return checkpoint_path

    def _register_checkpoint(
        self,
        path: Path,
        *,
        role: Literal["best", "latest", "periodic"],
        step: int,
    ) -> None:
        event_id = f"checkpoint:{role}:step:{step}"
        if event_id in self._registered_checkpoint_events:
            return
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(
                f"completed checkpoint is not a regular file: {path}"
            )
        relative_path = path.relative_to(self._paths.run_dir).as_posix()
        name = (
            f"checkpoint_step_{step:06d}"
            if role == "periodic"
            else f"checkpoint_{role}"
        )
        if isinstance(self._tracker, RunTracker):
            self._tracker.log_artifact_once(
                relative_path,
                name,
                "model",
                event_id=event_id,
            )
        else:
            self._tracker.log_artifact(relative_path, name, "model")
        self._registered_checkpoint_events.add(event_id)

    def _validate_if_due(
        self,
        step: int,
        result: OptimizerStepResult,
    ) -> ExactTrainingState | None:
        if self._validation_runner is None or step % self._config.train.eval_every != 0:
            return None
        try:
            validation = self._validation_runner(step)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            self._report_validation_failure(step, error)
            return None

        if isinstance(validation, PeriodicValidationResult):
            self._validation_results.append(validation)
        try:
            decision = advance_validation_state(
                self._validation_state,
                validation,
                validation_step=step,
            )
        except BestCheckpointError as error:
            raise PretrainingError(str(error)) from error
        except (TypeError, ValueError) as error:
            self._report_validation_failure(step, error)
            return None
        if not decision.accepted:
            self._report_validation_failure(step, decision.reason)
            return None

        candidate = decision.state
        if candidate is None:  # pragma: no cover - accepted decision invariant.
            raise RuntimeError("accepted validation produced no checkpoint state")
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
                validation=candidate,
                role="best",
            )
        self._validation_state = candidate
        track_periodic_base_validation(
            validation,
            candidate,
            tracker=self._tracker,
            step=step,
        )
        return continuation

    def on_step(self, step: int, result: OptimizerStepResult) -> None:
        continuation = self._validate_if_due(step, result)
        if step % self._config.train.save_every != 0:
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


def _execute_training(
    config: ProjectConfig,
    *,
    data: _PreparedTrainingData,
    runtime: _TrainingRuntime,
    device: torch.device,
    tracker: Tracker,
    on_step: Callable[[int, OptimizerStepResult], None],
    execution_model: nn.Module,
) -> list[OptimizerStepResult]:
    grad_accum_steps = derive_grad_accum_steps(
        device_batch_size=config.train.device_batch_size,
        seq_len=config.model.seq_len,
        total_batch_size_tokens=config.train.total_batch_size_tokens,
    )
    try:
        return run_training_steps(
            execution_model,
            data.batches,
            runtime.optimizer,
            runtime.scheduler,
            max_steps=config.train.max_steps,
            grad_accum_steps=grad_accum_steps,
            grad_clip=config.train.grad_clip,
            device=device,
            tracker=tracker,
            log_every=config.train.log_every,
            on_step=on_step,
            initial_total_training_time_seconds=(
                runtime.initial_total_training_time_seconds
            ),
            initial_total_training_flops=runtime.initial_total_training_flops,
            initial_processed_model_tokens=(
                runtime.initial_step * config.train.total_batch_size_tokens
            ),
            tokens_per_epoch=data.training_tokens_per_epoch,
            peak_flops_basis=peak_flops_basis_from_config(config.train),
            precision=runtime.precision,
        )
    except torch.OutOfMemoryError:
        try:
            runtime.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        raise


def _capture_exact_continuation(
    loader: object,
    *,
    device: torch.device,
    step: int,
    result: OptimizerStepResult,
) -> ExactTrainingState:
    state_dict = getattr(loader, "state_dict", None)
    if not callable(state_dict):
        raise PretrainingError(
            f"loader {type(loader).__name__} does not expose exact state"
        )
    loader_state = state_dict()
    if not isinstance(loader_state, dict):
        raise PretrainingError("loader state_dict must return a dictionary")
    loader_format = loader_state.get("format")
    if not isinstance(loader_format, str) or not loader_format:
        raise PretrainingError("loader state must contain a concrete format")
    return ExactTrainingState(
        loader_format=loader_format,
        loader_state=loader_state,
        rng_state=capture_training_rng_state(device),
        tracker_step=step,
        total_training_time_seconds=result.total_training_time_seconds,
        total_training_flops=result.total_training_flops,
    )


def _restore_exact_continuation(
    loader: object,
    continuation: ExactTrainingState,
    *,
    device: torch.device,
) -> None:
    state_dict = getattr(loader, "state_dict", None)
    load_state_dict = getattr(loader, "load_state_dict", None)
    if not callable(state_dict) or not callable(load_state_dict):
        raise PretrainingError(
            f"loader {type(loader).__name__} does not support exact resume"
        )
    original_loader_state = state_dict()
    if not isinstance(original_loader_state, dict):
        raise PretrainingError("loader state_dict must return a dictionary")
    if original_loader_state.get("format") != continuation.loader_format:
        raise PretrainingError(
            "checkpoint loader format does not match the configured loader: "
            f"checkpoint={continuation.loader_format!r}, "
            f"configured={original_loader_state.get('format')!r}"
        )
    original_rng_state = capture_training_rng_state(device)
    try:
        load_state_dict(continuation.loader_state)
        restore_training_rng_state(continuation.rng_state, device=device)
    except Exception as error:
        rollback_errors: list[str] = []
        try:
            load_state_dict(original_loader_state)
        except Exception as rollback_error:
            rollback_errors.append(f"loader rollback failed: {rollback_error}")
        try:
            restore_training_rng_state(original_rng_state, device=device)
        except Exception as rollback_error:
            rollback_errors.append(f"RNG rollback failed: {rollback_error}")
        suffix = f"; {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise PretrainingError(
            f"could not restore exact training continuation: {error}{suffix}"
        ) from error


def _collect_memory_after_oom(device: str) -> AcceleratorMemorySnapshot:
    try:
        return collect_accelerator_memory(device)
    except Exception as error:
        return AcceleratorMemorySnapshot(
            device=torch.device(device),
            available=False,
            unavailable_reason=(
                "memory snapshot collection failed after OOM: "
                f"{type(error).__name__}: {error}"
            ),
        )


def _clear_accelerator_cache_after_oom(device: str) -> None:
    requested = torch.device(device)
    if requested.type != "cuda":
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_tiny_pretraining(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    tracker: Tracker,
    resume_from: str | Path | None = None,
    allow_non_exact_resume: bool = False,
) -> PretrainingResult:
    """Backward-compatible entry point for the deterministic tiny-text profile."""

    if isinstance(config, ProjectConfig) and config.data.profile != "tiny_text":
        raise PretrainingError("run_tiny_pretraining requires data.profile='tiny_text'")
    return run_pretraining(
        config,
        paths=paths,
        tracker=tracker,
        resume_from=resume_from,
        allow_non_exact_resume=allow_non_exact_resume,
    )


__all__ = [
    "PretrainingError",
    "PretrainingOOMError",
    "PretrainingResult",
    "PreparedPretrainingBatchIterator",
    "load_production_tokenizer",
    "prepare_pretraining_batch",
    "run_pretraining",
    "run_tiny_pretraining",
    "validate_production_pretraining_config",
]
