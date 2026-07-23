"""Optimizer and learning-rate helpers for single-device training."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scratch_llm.config import TrainConfig


def get_lr_multiplier(step: int, config: TrainConfig) -> float:
    """Return the configured LR multiplier for a zero-based optimizer step."""

    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError(f"step must be an integer, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if not isinstance(config, TrainConfig):
        raise TypeError(f"config must be a TrainConfig, got {type(config).__name__}")
    config.validate()

    return _get_lr_multiplier(
        step,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        warmdown_ratio=config.warmdown_ratio,
        final_lr_frac=config.final_lr_frac,
    )


def _get_lr_multiplier(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> float:
    bounded_step = min(step, max_steps)
    if bounded_step < warmup_steps:
        return (bounded_step + 1) / warmup_steps

    warmdown_steps = round(warmdown_ratio * max_steps)
    warmdown_start = max_steps - warmdown_steps
    if warmdown_steps == 0 or bounded_step <= warmdown_start:
        return 1.0

    progress = (max_steps - bounded_step) / warmdown_steps
    return progress + (1.0 - progress) * final_lr_frac


class WarmupConstantWarmdownLR(LRScheduler):
    """Linearly warm up, hold steady, then linearly warm down."""

    def __init__(
        self,
        optimizer: Optimizer,
        config: TrainConfig,
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(config, TrainConfig):
            raise TypeError(
                f"config must be a TrainConfig, got {type(config).__name__}"
            )
        config.validate()
        self.max_steps = config.max_steps
        self.warmup_steps = config.warmup_steps
        self.warmdown_ratio = config.warmdown_ratio
        self.final_lr_frac = config.final_lr_frac
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Return scheduled rates for the current zero-based step."""

        multiplier = _get_lr_multiplier(
            self.last_epoch,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            warmdown_ratio=self.warmdown_ratio,
            final_lr_frac=self.final_lr_frac,
        )
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler progress and its current optimizer rates."""

        super().load_state_dict(state_dict)
        restored_lrs = self.get_lr()
        if len(restored_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "scheduler state has a different number of parameter groups "
                "than the optimizer"
            )
        for group, learning_rate in zip(
            self.optimizer.param_groups,
            restored_lrs,
            strict=True,
        ):
            group["lr"] = learning_rate
        self._last_lr = restored_lrs


def build_lr_scheduler(
    optimizer: Optimizer,
    config: TrainConfig,
) -> WarmupConstantWarmdownLR:
    """Build the reusable three-stage scheduler from training config."""

    return WarmupConstantWarmdownLR(optimizer, config)


def build_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    """Build AdamW from ``config`` using only trainable model parameters."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(config, TrainConfig):
        raise TypeError(f"config must be a TrainConfig, got {type(config).__name__}")
    config.validate()

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("model has no trainable parameters")

    return AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


__all__ = [
    "WarmupConstantWarmdownLR",
    "build_lr_scheduler",
    "build_optimizer",
    "get_lr_multiplier",
]
