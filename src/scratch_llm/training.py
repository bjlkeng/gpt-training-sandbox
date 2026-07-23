"""Model-independent mechanics for single-device training steps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


def _require_positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_positive_real(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return numeric


def derive_grad_accum_steps(
    *,
    device_batch_size: int,
    seq_len: int,
    total_batch_size_tokens: int,
) -> int:
    """Return the exact number of microbatches in one optimizer step."""

    device_batch_size = _require_positive_integer(
        device_batch_size,
        name="device_batch_size",
    )
    seq_len = _require_positive_integer(seq_len, name="seq_len")
    total_batch_size_tokens = _require_positive_integer(
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
    grad_accum_steps = _require_positive_integer(
        grad_accum_steps,
        name="grad_accum_steps",
    )
    grad_clip = _require_positive_real(grad_clip, name="grad_clip")

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


__all__ = [
    "OptimizerStepResult",
    "derive_grad_accum_steps",
    "run_optimizer_step",
]
