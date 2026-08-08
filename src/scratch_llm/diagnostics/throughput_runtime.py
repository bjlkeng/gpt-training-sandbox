"""Production data/model composition for the bounded throughput protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
import platform
from pathlib import Path
import subprocess
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scratch_llm._validation import require_positive_integer
from scratch_llm.attention_backends import (
    format_attention_selection,
    preflight_attention_backend,
)
from scratch_llm.diagnostics.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
    reset_accelerator_memory_peak,
)
from scratch_llm.config import ProjectConfig
from scratch_llm.data.loaders import create_token_loader
from scratch_llm.model import GPT
from scratch_llm.diagnostics.oom import PretrainingOOMError, diagnose_out_of_memory
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.compilation import (
    ModelCompiler,
    build_compile_runtime,
    format_compile_selection,
    warmup_compiled_training,
)
from scratch_llm.training.activation_checkpointing import (
    configure_activation_checkpointing,
    format_activation_checkpoint_selection,
)
from scratch_llm.training.precision import PrecisionPolicy, build_precision_policy
from scratch_llm.training.pretraining import (
    PreparedPretrainingBatchIterator,
    load_production_tokenizer,
    validate_production_pretraining_config,
)
from scratch_llm.diagnostics.throughput import BenchmarkExecution
from scratch_llm.data.tokenized import (
    TokenizedShardReader,
    tokenized_manifest_identity,
)
from scratch_llm.tracking import NullTracker
from scratch_llm.training.loop import (
    OptimizerStepResult,
    derive_grad_accum_steps,
    run_training_steps,
)
from scratch_llm.training.telemetry import peak_flops_basis_from_config
from scratch_llm.utils import get_device, set_seed


@dataclass(frozen=True)
class BenchmarkStepExecution:
    """Shared training results with an aligned memory snapshot per step."""

    steps: tuple[OptimizerStepResult, ...]
    memory_snapshots: tuple[AcceleratorMemorySnapshot, ...]


def run_benchmark_training_steps(
    config: ProjectConfig,
    *,
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    warmup_steps: int,
    timed_steps: int,
    clock: Callable[[], float] = perf_counter,
    reset_memory_peak: Callable[[str | torch.device], bool] = (
        reset_accelerator_memory_peak
    ),
    collect_memory: Callable[[str | torch.device], AcceleratorMemorySnapshot] = (
        collect_accelerator_memory
    ),
    precision: PrecisionPolicy | None = None,
) -> BenchmarkStepExecution:
    """Run warmup plus timed work through the shared optimizer-step boundary."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    warmup_steps = require_positive_integer(warmup_steps, name="warmup_steps")
    timed_steps = require_positive_integer(timed_steps, name="timed_steps")
    total_steps = warmup_steps + timed_steps
    if total_steps > config.train.max_steps:
        raise ValueError(
            "warmup_steps + timed_steps cannot exceed train.max_steps; "
            f"got {total_steps} > {config.train.max_steps}"
        )
    active_precision = (
        build_precision_policy(dtype=config.train.dtype, device=config.run.device)
        if precision is None
        else precision
    )
    reset_results: list[bool] = []
    collected_snapshots: list[AcceleratorMemorySnapshot] = []

    def record_reset(device: str | torch.device) -> bool:
        reset = reset_memory_peak(device)
        if not isinstance(reset, bool):
            raise TypeError("reset_memory_peak must return a boolean")
        reset_results.append(reset)
        return reset

    def record_collection(
        device: str | torch.device,
    ) -> AcceleratorMemorySnapshot:
        snapshot = collect_memory(device)
        if not isinstance(snapshot, AcceleratorMemorySnapshot):
            raise TypeError("collect_memory must return an AcceleratorMemorySnapshot")
        collected_snapshots.append(snapshot)
        return snapshot

    results = run_training_steps(
        model,
        batches,
        optimizer,
        scheduler,
        max_steps=total_steps,
        grad_accum_steps=derive_grad_accum_steps(
            device_batch_size=config.train.device_batch_size,
            seq_len=config.model.seq_len,
            total_batch_size_tokens=config.train.total_batch_size_tokens,
        ),
        grad_clip=config.train.grad_clip,
        device=config.run.device,
        tracker=NullTracker(),
        log_every=1,
        peak_flops_basis=peak_flops_basis_from_config(config.train),
        clock=clock,
        reset_memory_peak=record_reset,
        collect_memory=record_collection,
        precision=active_precision,
    )
    if len(reset_results) != total_steps:
        raise RuntimeError(
            "shared training did not expose one memory boundary per step"
        )
    unavailable_snapshot: AcceleratorMemorySnapshot | None = None
    captured = iter(collected_snapshots)
    aligned_snapshots: list[AcceleratorMemorySnapshot] = []
    for reset in reset_results:
        if reset:
            aligned_snapshots.append(next(captured))
            continue
        if unavailable_snapshot is None:
            unavailable_snapshot = collect_memory(config.run.device)
            if not isinstance(unavailable_snapshot, AcceleratorMemorySnapshot):
                raise TypeError(
                    "collect_memory must return an AcceleratorMemorySnapshot"
                )
            if unavailable_snapshot.available:
                raise RuntimeError(
                    "memory peak reset was unavailable but collection reported "
                    "available counters"
                )
        aligned_snapshots.append(unavailable_snapshot)
    try:
        next(captured)
    except StopIteration:
        pass
    else:  # pragma: no cover - shared training owns one collection per reset.
        raise RuntimeError("shared training collected extra memory snapshots")
    return BenchmarkStepExecution(
        steps=tuple(results),
        memory_snapshots=tuple(aligned_snapshots),
    )


def execute_production_throughput_benchmark(
    config: ProjectConfig,
    *,
    warmup_steps: int,
    timed_steps: int,
    repository_root: str | Path,
    progress: Callable[[str], None] | None = None,
    compiler: ModelCompiler | None = None,
) -> BenchmarkExecution:
    """Load immutable production artifacts and execute the bounded protocol."""

    validate_production_pretraining_config(config)
    device = get_device(config.run.device)
    precision = build_precision_policy(dtype=config.train.dtype, device=device)
    preflight = preflight_attention_backend(
        config.model,
        device=device,
        dtype=config.train.dtype,
        training=True,
    )
    if progress is not None:
        progress(format_attention_selection(preflight.selection))
    set_seed(config.run.seed)
    tokenizer = load_production_tokenizer(config)
    with ExitStack() as resources:
        reader = resources.enter_context(
            TokenizedShardReader(
                config.data.tokenized_dir,
                tokenizer=tokenizer,
            )
        )
        loader = create_token_loader(
            reader,
            strategy=config.data.loader_strategy,
            split="train",
            batch_size=config.train.device_batch_size,
            seq_len=config.model.seq_len,
            seed=config.run.seed,
            planning_progress=progress,
        )
        batches = PreparedPretrainingBatchIterator(
            iter(loader),  # type: ignore[arg-type]
            strategy=config.data.loader_strategy,
        )
        model = GPT(config.model).to(device)
        optimizer = build_optimizer(model, config.train)
        scheduler = build_lr_scheduler(optimizer, config.train)
        activation_checkpoint_selection = configure_activation_checkpointing(
            model,
            enabled=config.train.activation_checkpointing,
        )
        if progress is not None:
            progress(
                format_activation_checkpoint_selection(activation_checkpoint_selection)
            )
        compile_runtime = build_compile_runtime(
            model,
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
            optimizer,
            inputs=warmup_tokens,
            targets=warmup_tokens,
            precision=precision,
            device=device,
        )
        if progress is not None:
            progress(format_compile_selection(compile_runtime.selection))
        try:
            step_execution = run_benchmark_training_steps(
                config,
                model=compile_runtime.execution_model,
                batches=batches,
                optimizer=optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                timed_steps=timed_steps,
                precision=precision,
            )
        except torch.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            diagnostic = diagnose_out_of_memory(
                error,
                config=config,
                memory=collect_accelerator_memory(device),
            )
            assert diagnostic is not None
            raise PretrainingOOMError(diagnostic) from error
        if progress is not None:
            progress(format_compile_selection(compile_runtime.selection))
        hardware, cuda, pytorch = collect_runtime_identities(device)
        return BenchmarkExecution(
            steps=step_execution.steps,
            memory_snapshots=step_execution.memory_snapshots,
            tokenizer_identity=tokenizer.get_identity(),
            manifest_identity=tokenized_manifest_identity(reader.manifest),
            hardware_identity=hardware,
            cuda_identity=cuda,
            pytorch_identity=pytorch,
            code_identity=collect_code_identity(repository_root),
            attention_selection=model.attention_backend_selection(),
            compile_selection=compile_runtime.selection,
            activation_checkpoint_selection=activation_checkpoint_selection,
        )


def collect_runtime_identities(
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Return explicit hardware, CUDA, and PyTorch identities without timing."""

    hardware: dict[str, object] = {
        "device": str(device),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware.update(
            {
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "device_name": torch.cuda.get_device_name(device),
                "total_memory_bytes": properties.total_memory,
            }
        )
    cuda = {
        "available": bool(torch.cuda.is_available()),
        "compiled_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    pytorch = {
        "debug_build": bool(torch.version.debug),
        "git_version": torch.version.git_version,
        "version": torch.__version__,
    }
    return hardware, cuda, pytorch


def collect_code_identity(repository_root: str | Path) -> dict[str, object]:
    """Return the local Git commit and tracked-dirty state, or an explicit reason."""

    root = Path(repository_root).resolve()
    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    if commit is None or status is None:
        return {
            "available": False,
            "commit": None,
            "tracked_dirty": None,
            "unavailable_reason": "Git identity could not be resolved",
        }
    return {
        "available": True,
        "commit": commit,
        "tracked_dirty": bool(status),
        "unavailable_reason": None,
    }


def _run_git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


__all__ = [
    "BenchmarkStepExecution",
    "collect_code_identity",
    "collect_runtime_identities",
    "execute_production_throughput_benchmark",
    "run_benchmark_training_steps",
]
