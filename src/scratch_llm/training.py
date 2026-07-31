"""Educational single-device training loops and optimizer-step mechanics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sized
from dataclasses import dataclass, replace
import math
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from scratch_llm._validation import require_positive_integer, require_positive_real
from scratch_llm.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
    reset_accelerator_memory_peak,
)
from scratch_llm.config import ProjectConfig
from scratch_llm.data import NextTokenDataset
from scratch_llm.model import GPT
from scratch_llm.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tokenizer import NANOCHAT_SPECIAL_TOKENS, ByteTokenizer, Tokenizer
from scratch_llm.tracking import NullTracker, Tracker
from scratch_llm.training_telemetry import (
    PeakFlopsBasis,
    TrainingStepTelemetry,
    estimate_gpt_training_flops,
    peak_flops_basis_from_config,
)
from scratch_llm.utils import get_device, set_seed


def derive_grad_accum_steps(
    *,
    device_batch_size: int,
    seq_len: int,
    total_batch_size_tokens: int,
) -> int:
    """Return the exact number of microbatches in one optimizer step."""

    device_batch_size = require_positive_integer(
        device_batch_size,
        name="device_batch_size",
    )
    seq_len = require_positive_integer(seq_len, name="seq_len")
    total_batch_size_tokens = require_positive_integer(
        total_batch_size_tokens,
        name="total_batch_size_tokens",
    )

    tokens_per_microbatch = device_batch_size * seq_len
    grad_accum_steps, remainder = divmod(
        total_batch_size_tokens,
        tokens_per_microbatch,
    )
    if remainder:
        raise ValueError(
            "total_batch_size_tokens must be divisible by "
            "device_batch_size * seq_len "
            f"({tokens_per_microbatch}); got {total_batch_size_tokens}"
        )
    return grad_accum_steps


@dataclass(frozen=True)
class OptimizerStepResult:
    """Metrics produced by one completed gradient-accumulation window."""

    loss: float
    grad_norm: float
    telemetry: TrainingStepTelemetry | None = None

    @property
    def step_duration_seconds(self) -> float:
        return 0.0 if self.telemetry is None else self.telemetry.duration_seconds

    @property
    def total_training_time_seconds(self) -> float:
        if self.telemetry is None:
            return 0.0
        return self.telemetry.total_training_time_seconds

    @property
    def total_training_flops(self) -> float:
        if self.telemetry is None:
            return 0.0
        return self.telemetry.total_training_flops


@dataclass(frozen=True)
class TinyTextTrainingResult:
    """Reusable state and step history from the phase-one text training path."""

    model: GPT
    tokenizer: Tokenizer
    optimizer: Optimizer
    scheduler: LRScheduler
    steps: tuple[OptimizerStepResult, ...]


def run_optimizer_step(
    optimizer: Optimizer,
    micro_losses: Iterable[Tensor],
    *,
    grad_accum_steps: int,
    grad_clip: float,
) -> OptimizerStepResult:
    """Accumulate scaled losses, clip gradients, update once, and clear gradients.

    ``micro_losses`` may be a lazy iterable so each forward pass can release its
    graph after backward. Exactly ``grad_accum_steps`` losses are consumed.
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            f"optimizer must be an Optimizer, got {type(optimizer).__name__}"
        )
    grad_accum_steps = require_positive_integer(
        grad_accum_steps,
        name="grad_accum_steps",
    )
    grad_clip = require_positive_real(grad_clip, name="grad_clip")

    loss_iterator = iter(micro_losses)
    loss_sum: Tensor | None = None
    try:
        for microstep in range(grad_accum_steps):
            try:
                micro_loss = next(loss_iterator)
            except StopIteration as error:
                raise ValueError(
                    "micro_losses ended after "
                    f"{microstep} of {grad_accum_steps} required losses"
                ) from error
            if not isinstance(micro_loss, Tensor):
                raise TypeError(
                    f"micro loss {microstep} must be a Tensor, "
                    f"got {type(micro_loss).__name__}"
                )
            if micro_loss.ndim != 0:
                raise ValueError(
                    f"micro loss {microstep} must be scalar, "
                    f"got shape {tuple(micro_loss.shape)}"
                )

            detached_loss = micro_loss.detach()
            loss_sum = detached_loss if loss_sum is None else loss_sum + detached_loss
            (micro_loss / grad_accum_steps).backward()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise

    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    grad_norm = clip_grad_norm_(parameters, grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if loss_sum is None:  # pragma: no cover - positive steps make this unreachable.
        raise RuntimeError("optimizer step completed without a loss")
    return OptimizerStepResult(
        loss=float((loss_sum / grad_accum_steps).item()),
        grad_norm=float(grad_norm.item()),
    )


def _repeat_batches(
    batches: Iterable[tuple[Tensor, Tensor]],
) -> Iterator[tuple[Tensor, Tensor]]:
    """Repeat a re-iterable batch source and reject an empty training set."""

    while True:
        yielded_batch = False
        for batch in batches:
            yielded_batch = True
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise TypeError(
                    "each training batch must contain exactly inputs and targets"
                )
            inputs, targets = batch
            if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
                raise TypeError("training batch inputs and targets must be Tensors")
            yield inputs, targets
        if not yielded_batch:
            raise ValueError("batches must yield at least one training batch")


def run_validation(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: str | torch.device,
) -> float:
    """Return target-weighted mean loss without changing trainable state."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    resolved_device = get_device(device)
    module_modes = [(module, module.training) for module in model.modules()]
    weighted_loss = 0.0
    target_count = 0

    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise TypeError(
                        "each validation batch must contain exactly inputs and targets"
                    )
                inputs, targets = batch
                if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
                    raise TypeError(
                        "validation batch inputs and targets must be Tensors"
                    )

                targets = targets.to(resolved_device)
                batch_target_count = int(targets.ne(-1).sum().item())
                if batch_target_count == 0:
                    continue
                loss = model(inputs.to(resolved_device), targets)
                if not isinstance(loss, Tensor):
                    raise TypeError(
                        "validation model must return a Tensor loss when given targets"
                    )
                if loss.ndim != 0:
                    raise ValueError(
                        "validation model loss must be scalar, "
                        f"got shape {tuple(loss.shape)}"
                    )
                weighted_loss += float(loss.item()) * batch_target_count
                target_count += batch_target_count
    finally:
        for module, training_mode in module_modes:
            module.training = training_mode

    if target_count == 0:
        raise ValueError("validation batches must contain at least one target")
    return weighted_loss / target_count


def run_training_steps(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    *,
    max_steps: int,
    grad_accum_steps: int,
    grad_clip: float,
    device: str | torch.device,
    tracker: Tracker | None = None,
    log_every: int = 1,
    on_step: Callable[[int, OptimizerStepResult], None] | None = None,
    initial_total_training_time_seconds: float = 0.0,
    initial_total_training_flops: float = 0.0,
    peak_flops_basis: PeakFlopsBasis | None = None,
    clock: Callable[[], float] | None = None,
    reset_memory_peak: Callable[[str | torch.device], bool] | None = None,
    collect_memory: (
        Callable[[str | torch.device], AcceleratorMemorySnapshot] | None
    ) = None,
) -> list[OptimizerStepResult]:
    """Train to ``max_steps`` and call ``on_step`` after each completed step."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            f"optimizer must be an Optimizer, got {type(optimizer).__name__}"
        )
    if not isinstance(scheduler, LRScheduler):
        raise TypeError(
            f"scheduler must be an LRScheduler, got {type(scheduler).__name__}"
        )
    max_steps = require_positive_integer(max_steps, name="max_steps")
    grad_accum_steps = require_positive_integer(
        grad_accum_steps,
        name="grad_accum_steps",
    )
    grad_clip = require_positive_real(grad_clip, name="grad_clip")
    log_every = require_positive_integer(log_every, name="log_every")
    if tracker is None:
        tracker = NullTracker()
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")
    if on_step is not None and not callable(on_step):
        raise TypeError(
            f"on_step must be callable or None, got {type(on_step).__name__}"
        )
    if peak_flops_basis is not None and not isinstance(
        peak_flops_basis,
        PeakFlopsBasis,
    ):
        raise TypeError(
            "peak_flops_basis must be a PeakFlopsBasis or None, got "
            f"{type(peak_flops_basis).__name__}"
        )
    active_clock = perf_counter if clock is None else clock
    active_reset_memory_peak = (
        reset_accelerator_memory_peak
        if reset_memory_peak is None
        else reset_memory_peak
    )
    active_collect_memory = (
        collect_accelerator_memory if collect_memory is None else collect_memory
    )
    for name, function in (
        ("clock", active_clock),
        ("reset_memory_peak", active_reset_memory_peak),
        ("collect_memory", active_collect_memory),
    ):
        if not callable(function):
            raise TypeError(f"{name} must be callable")
    total_training_time_seconds = _non_negative_finite_counter(
        initial_total_training_time_seconds,
        name="initial_total_training_time_seconds",
    )
    total_training_flops = _non_negative_finite_counter(
        initial_total_training_flops,
        name="initial_total_training_flops",
    )
    resolved_device = get_device(device)
    batches_per_epoch = len(batches) if isinstance(batches, Sized) else None
    if batches_per_epoch is not None and batches_per_epoch <= 0:
        batches_per_epoch = None

    model.to(resolved_device)
    model.train()
    flops_estimate = (
        estimate_gpt_training_flops(model.config) if isinstance(model, GPT) else None
    )
    batch_iterator = iter(_repeat_batches(batches))
    results: list[OptimizerStepResult] = []
    initial_step = scheduler.last_epoch
    if initial_step > max_steps:
        raise ValueError(
            f"scheduler step {initial_step} exceeds max_steps target {max_steps}"
        )

    for step in range(initial_step + 1, max_steps + 1):
        base_learning_rate = float(scheduler.base_lrs[0])
        learning_rate_multiplier = (
            float(scheduler.get_last_lr()[0]) / base_learning_rate
        )
        should_log = step % log_every == 0
        memory_window_started = (
            bool(active_reset_memory_peak(resolved_device)) if should_log else False
        )
        step_started_at = _clock_value(active_clock(), name="clock start")
        processed_model_tokens = 0
        supervised_target_tokens = 0

        def micro_losses() -> Iterator[Tensor]:
            nonlocal processed_model_tokens, supervised_target_tokens
            for _ in range(grad_accum_steps):
                inputs, targets = next(batch_iterator)
                if inputs.ndim != 2 or targets.shape != inputs.shape:
                    raise ValueError(
                        "training inputs and targets must have matching "
                        "two-dimensional shapes"
                    )
                if (
                    flops_estimate is not None
                    and inputs.shape[1] != flops_estimate.sequence_length
                ):
                    raise ValueError(
                        "training input sequence length must match the FLOPs "
                        f"estimate ({flops_estimate.sequence_length})"
                    )
                loss = model(
                    inputs.to(resolved_device),
                    targets.to(resolved_device),
                )
                processed_model_tokens += inputs.numel()
                supervised_target_tokens += int(targets.ne(-1).sum().item())
                yield loss

        result = run_optimizer_step(
            optimizer,
            micro_losses(),
            grad_accum_steps=grad_accum_steps,
            grad_clip=grad_clip,
        )
        scheduler.step()
        step_finished_at = _clock_value(active_clock(), name="clock finish")
        step_duration = step_finished_at - step_started_at
        if step_duration <= 0:
            raise ValueError("measured optimizer-step duration must be positive")
        total_training_time_seconds += step_duration
        if flops_estimate is not None:
            step_flops = flops_estimate.flops_for_tokens(processed_model_tokens)
            total_training_flops += step_flops
            peak_memory_mib: float | None = None
            if memory_window_started:
                memory = active_collect_memory(resolved_device)
                if not isinstance(memory, AcceleratorMemorySnapshot):
                    raise TypeError(
                        "collect_memory must return an AcceleratorMemorySnapshot"
                    )
                if not memory.available or memory.peak_allocated_mib is None:
                    raise RuntimeError(
                        "peak memory reset succeeded but collection was unavailable"
                    )
                peak_memory_mib = memory.peak_allocated_mib
            mfu = (
                None
                if peak_flops_basis is None
                else step_flops / step_duration / peak_flops_basis.flops_per_second
            )
            result = replace(
                result,
                telemetry=TrainingStepTelemetry(
                    processed_model_tokens=processed_model_tokens,
                    supervised_target_tokens=supervised_target_tokens,
                    duration_seconds=step_duration,
                    tokens_per_second=processed_model_tokens / step_duration,
                    step_flops=step_flops,
                    total_training_flops=total_training_flops,
                    total_training_time_seconds=total_training_time_seconds,
                    mfu=mfu,
                    peak_flops_basis=peak_flops_basis,
                    peak_memory_mib=peak_memory_mib,
                    flops_estimate=flops_estimate,
                ),
            )
        results.append(result)
        if should_log:
            metrics = {
                "train/loss": result.loss,
                "train/lrm": learning_rate_multiplier,
                "train/dt": step_duration,
                "train/grad_norm": result.grad_norm,
                "train/total_training_flops": result.total_training_flops,
                "train/total_training_time": result.total_training_time_seconds,
            }
            if batches_per_epoch is not None:
                metrics["train/epoch"] = step * grad_accum_steps / batches_per_epoch
            tracker.log(metrics, step=step)
        if on_step is not None:
            on_step(step, result)

    return results


def _non_negative_finite_counter(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _clock_value(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def train_tiny_text(
    text: str,
    config: ProjectConfig,
    *,
    tracker: Tracker | None = None,
) -> TinyTextTrainingResult:
    """Compose the byte tokenizer, tiny dataset, GPT, AdamW, and train loop."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    config.validate()
    if config.tokenizer.type != "byte":
        raise ValueError(
            "tiny-text training requires tokenizer.type='byte', "
            f"got {config.tokenizer.type!r}"
        )

    set_seed(config.run.seed)
    device = get_device(config.run.device)
    tokenizer = ByteTokenizer()
    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise ValueError(
            "tiny-text training requires tokenizer.vocab_size="
            f"{tokenizer.get_vocab_size()}, got {config.tokenizer.vocab_size}"
        )
    if tuple(config.tokenizer.special_tokens) != NANOCHAT_SPECIAL_TOKENS:
        raise ValueError(
            "tiny-text training requires the ByteTokenizer special-token order"
        )

    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(
        token_ids,
        config.model.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
    )
    if len(dataset) < config.train.device_batch_size:
        raise ValueError(
            "tiny text must produce at least one complete device batch; "
            f"found {len(dataset)} examples for batch size "
            f"{config.train.device_batch_size}"
        )

    data_generator = torch.Generator().manual_seed(config.run.seed)
    batches = DataLoader(
        dataset,
        batch_size=config.train.device_batch_size,
        shuffle=True,
        drop_last=True,
        generator=data_generator,
    )
    model = GPT(config.model).to(device)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    grad_accum_steps = derive_grad_accum_steps(
        device_batch_size=config.train.device_batch_size,
        seq_len=config.model.seq_len,
        total_batch_size_tokens=config.train.total_batch_size_tokens,
    )
    steps = run_training_steps(
        model,
        batches,
        optimizer,
        scheduler,
        max_steps=config.train.max_steps,
        grad_accum_steps=grad_accum_steps,
        grad_clip=config.train.grad_clip,
        device=device,
        tracker=tracker,
        log_every=config.train.log_every,
        peak_flops_basis=peak_flops_basis_from_config(config.train),
    )
    return TinyTextTrainingResult(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        steps=tuple(steps),
    )


__all__ = [
    "OptimizerStepResult",
    "TinyTextTrainingResult",
    "derive_grad_accum_steps",
    "run_optimizer_step",
    "run_training_steps",
    "run_validation",
    "train_tiny_text",
]
