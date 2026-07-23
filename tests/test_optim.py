"""Tests for optimizer construction and learning-rate scheduling."""

from __future__ import annotations

import pytest
from torch import nn
from torch.optim import AdamW

from scratch_llm.config import TrainConfig
from scratch_llm.optim import (
    WarmupConstantWarmdownLR,
    build_lr_scheduler,
    build_optimizer,
    get_lr_multiplier,
)


def test_build_optimizer_uses_config_and_only_trainable_parameters() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)
    config = TrainConfig(
        max_steps=10,
        learning_rate=0.003,
        weight_decay=0.2,
        beta1=0.8,
        beta2=0.88,
        warmup_steps=2,
        warmdown_ratio=0.4,
    )

    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, AdamW)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(config.learning_rate)
    assert group["weight_decay"] == pytest.approx(config.weight_decay)
    assert group["betas"] == pytest.approx((config.beta1, config.beta2))
    assert {id(parameter) for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 0.5),
        (1, 1.0),
        (2, 1.0),
        (6, 1.0),
        (7, 0.8),
        (9, 0.4),
        (10, 0.2),
        (11, 0.2),
    ],
)
def test_lr_multiplier_covers_schedule_boundaries(step: int, expected: float) -> None:
    config = TrainConfig(
        max_steps=10,
        warmup_steps=2,
        warmdown_ratio=0.4,
        final_lr_frac=0.2,
    )

    assert get_lr_multiplier(step, config) == pytest.approx(expected)


def test_lr_scheduler_applies_the_multiplier_to_the_optimizer() -> None:
    model = nn.Linear(2, 1)
    config = TrainConfig(
        max_steps=10,
        learning_rate=0.003,
        warmup_steps=2,
        warmdown_ratio=0.4,
        final_lr_frac=0.2,
    )
    optimizer = build_optimizer(model, config)

    scheduler = build_lr_scheduler(optimizer, config)

    assert isinstance(scheduler, WarmupConstantWarmdownLR)
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0015)

    optimizer.step()
    scheduler.step()

    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.003)


def test_lr_scheduler_state_round_trip_restores_progress_and_rate() -> None:
    config = TrainConfig(
        max_steps=10,
        learning_rate=0.003,
        warmup_steps=2,
        warmdown_ratio=0.4,
        final_lr_frac=0.2,
    )
    optimizer = build_optimizer(nn.Linear(2, 1), config)
    scheduler = build_lr_scheduler(optimizer, config)
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    saved_state = scheduler.state_dict()
    saved_lr = optimizer.param_groups[0]["lr"]

    restored_optimizer = build_optimizer(nn.Linear(2, 1), config)
    restored_scheduler = build_lr_scheduler(restored_optimizer, config)
    restored_scheduler.load_state_dict(saved_state)

    assert restored_scheduler.state_dict() == saved_state
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)

    optimizer.step()
    scheduler.step()
    restored_optimizer.step()
    restored_scheduler.step()

    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
