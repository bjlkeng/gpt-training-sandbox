"""Tests for model-independent training-step mechanics."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD

import scratch_llm.training as training
from scratch_llm.training import derive_grad_accum_steps, run_optimizer_step


def test_derive_grad_accum_steps_uses_the_exact_token_budget() -> None:
    assert (
        derive_grad_accum_steps(
            device_batch_size=4,
            seq_len=128,
            total_batch_size_tokens=65_536,
        )
        == 128
    )


@pytest.mark.parametrize("total_batch_size_tokens", [0, -1])
def test_derive_grad_accum_steps_rejects_non_positive_token_budgets(
    total_batch_size_tokens: int,
) -> None:
    with pytest.raises(ValueError, match="total_batch_size_tokens must be positive"):
        derive_grad_accum_steps(
            device_batch_size=2,
            seq_len=8,
            total_batch_size_tokens=total_batch_size_tokens,
        )


def test_derive_grad_accum_steps_rejects_non_divisible_token_budgets() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        derive_grad_accum_steps(
            device_batch_size=2,
            seq_len=8,
            total_batch_size_tokens=33,
        )


def test_optimizer_step_scales_micro_losses_and_orders_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = SGD([parameter], lr=0.1)
    real_clip_grad_norm = training.clip_grad_norm_
    real_step = optimizer.step
    real_zero_grad = optimizer.zero_grad

    def record_clip_grad_norm(
        parameters: Iterable[Tensor],
        max_norm: float,
    ) -> Tensor:
        events.append("clip")
        return real_clip_grad_norm(parameters, max_norm)

    def record_step(*args: Any, **kwargs: Any) -> Any:
        events.append("step")
        return real_step(*args, **kwargs)

    def record_zero_grad(*args: Any, **kwargs: Any) -> Any:
        events.append("zero")
        return real_zero_grad(*args, **kwargs)

    monkeypatch.setattr(training, "clip_grad_norm_", record_clip_grad_norm)
    monkeypatch.setattr(optimizer, "step", record_step)
    monkeypatch.setattr(optimizer, "zero_grad", record_zero_grad)

    result = run_optimizer_step(
        optimizer,
        [parameter.square(), parameter.square()],
        grad_accum_steps=2,
        grad_clip=0.5,
    )

    assert events == ["clip", "step", "zero"]
    assert result.loss == pytest.approx(4.0)
    assert result.grad_norm == pytest.approx(4.0)
    assert parameter.item() == pytest.approx(1.95)
    assert parameter.grad is None


def test_accumulated_cpu_step_matches_the_equivalent_larger_batch() -> None:
    torch.manual_seed(7)
    full_inputs = torch.tensor(
        [
            [0.2, -0.4, 0.5],
            [1.0, 0.3, -0.2],
            [-0.7, 0.1, 0.8],
            [0.6, -0.5, -0.1],
        ]
    )
    full_targets = torch.tensor([[0.3], [-0.2], [0.7], [0.1]])
    accumulated_model = nn.Linear(3, 1)
    large_batch_model = deepcopy(accumulated_model)
    accumulated_optimizer = SGD(accumulated_model.parameters(), lr=0.05)
    large_batch_optimizer = SGD(large_batch_model.parameters(), lr=0.05)
    microbatches = zip(
        full_inputs.chunk(2),
        full_targets.chunk(2),
        strict=True,
    )

    accumulated_result = run_optimizer_step(
        accumulated_optimizer,
        (
            F.mse_loss(accumulated_model(inputs), targets)
            for inputs, targets in microbatches
        ),
        grad_accum_steps=2,
        grad_clip=1_000.0,
    )
    large_batch_result = run_optimizer_step(
        large_batch_optimizer,
        [F.mse_loss(large_batch_model(full_inputs), full_targets)],
        grad_accum_steps=1,
        grad_clip=1_000.0,
    )

    torch.testing.assert_close(
        accumulated_model.weight,
        large_batch_model.weight,
    )
    torch.testing.assert_close(
        accumulated_model.bias,
        large_batch_model.bias,
    )
    assert accumulated_result.loss == pytest.approx(large_batch_result.loss)
    assert accumulated_result.grad_norm == pytest.approx(large_batch_result.grad_norm)
