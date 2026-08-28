"""Tests for measured base-training telemetry and FLOPs assumptions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch
from torch import Tensor

from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.config import GPTConfig, TrainConfig
from scratch_llm.model import GPT
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tracking import Tracker
from scratch_llm.training.loop import run_training_steps
from scratch_llm.training.telemetry import (
    TRAINING_FLOPS_FORMULA_ID,
    PeakFlopsBasis,
    base_training_metrics,
    estimate_gpt_training_flops,
)


@pytest.mark.parametrize("tie_weights", [True, False])
def test_gpt_training_flops_matches_hand_calculation_without_tie_aliasing(
    tie_weights: bool,
) -> None:
    config = GPTConfig(
        vocab_size=32,
        seq_len=8,
        n_layer=2,
        n_head=2,
        n_embd=4,
        mlp_ratio=3,
        tie_weights=tie_weights,
    )

    estimate = estimate_gpt_training_flops(config)

    # Per layer, QKV + attention output + two MLP matrices execute
    # (4 + 2 * mlp_ratio) * width**2 matrix weights. The output projection
    # executes width * vocab weights even when it aliases the input embedding.
    expected_weight_elements = 2 * (4 + 2 * 3) * 4**2 + 4 * 32
    expected_linear_flops_per_token = 6 * expected_weight_elements
    expected_attention_flops_per_token = 12 * 2 * 4 * 8
    assert estimate.formula_id == TRAINING_FLOPS_FORMULA_ID
    assert estimate.executed_weight_elements == expected_weight_elements
    assert estimate.linear_flops_per_token == expected_linear_flops_per_token
    assert estimate.attention_flops_per_token == expected_attention_flops_per_token
    assert estimate.flops_per_token == (
        expected_linear_flops_per_token + expected_attention_flops_per_token
    )
    assert estimate.flops_for_tokens(10) == 10 * estimate.flops_per_token
    assert estimate.tie_weights is tie_weights
    assert estimate.assumptions == (
        "one multiply-accumulate is two FLOPs",
        "backward costs twice the modeled forward matrix multiplications",
        "full-context layers execute sequence-length score and value products",
        "short-window layers use their declared maximum visible key span",
        "embedding lookup, normalization, activation, bias, softmax, dropout, "
        "loss, clipping, optimizer, and scheduler FLOPs are excluded",
    )

    with pytest.raises(FrozenInstanceError):
        estimate.flops_per_token = 0  # type: ignore[misc]


class _DelayedTracker(Tracker):
    def __init__(self, events: list[str], time: list[float]) -> None:
        self.events = events
        self.time = time
        self.records: list[tuple[dict[str, Any], int | None]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.events.append(f"tracker:{step}")
        self.records.append((metrics, step))
        self.time[0] += 50.0

    def log_config(self, config: dict[str, Any]) -> None:
        raise AssertionError("not used")

    def log_artifact(self, path: str, name: str, type: str) -> None:
        raise AssertionError("not used")

    def finish(self) -> None:
        pass


def test_training_steps_measure_actual_work_and_exclude_tracker_and_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = GPTConfig(
        vocab_size=16,
        seq_len=2,
        n_layer=1,
        n_head=1,
        n_embd=4,
        mlp_ratio=2,
    )
    train_config = TrainConfig(
        device_batch_size=1,
        total_batch_size_tokens=4,
        grad_accum_steps=2,
        max_steps=2,
        learning_rate=0.01,
        weight_decay=0.0,
        warmup_steps=0,
        warmdown_ratio=0.0,
    )
    model = GPT(model_config)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_lr_scheduler(optimizer, train_config)
    inputs = torch.tensor([[1, 2]])
    targets = torch.tensor([[2, -1]])
    events: list[str] = []
    current_time = [0.0]
    tracker = _DelayedTracker(events, current_time)
    real_forward = model.forward

    def forward(*args: Any, **kwargs: Any) -> Tensor:
        events.append("forward")
        current_time[0] += 1.0
        return real_forward(*args, **kwargs)

    def clock() -> float:
        events.append(f"clock:{current_time[0]}")
        return current_time[0]

    def reset_memory(device: str | torch.device) -> bool:
        assert torch.device(device) == torch.device("cpu")
        events.append("memory:reset")
        return True

    def collect_memory(
        device: str | torch.device,
    ) -> AcceleratorMemorySnapshot:
        assert torch.device(device) == torch.device("cpu")
        events.append("memory:collect")
        mib = 1024**2
        return AcceleratorMemorySnapshot(
            device=torch.device("cpu"),
            available=True,
            allocated_bytes=4 * mib,
            reserved_bytes=8 * mib,
            peak_allocated_bytes=6 * mib,
            peak_reserved_bytes=8 * mib,
            capacity_bytes=16 * mib,
        )

    def on_step(step: int, _result: object) -> None:
        events.append(f"callback:{step}")
        current_time[0] += 100.0

    monkeypatch.setattr(model, "forward", forward)
    peak_basis = PeakFlopsBasis(
        flops_per_second=1_000_000.0,
        description="synthetic explicit peak",
    )

    results = run_training_steps(
        model,
        [(inputs, targets)],
        optimizer,
        scheduler,
        max_steps=2,
        grad_accum_steps=2,
        grad_clip=train_config.grad_clip,
        device="cpu",
        tracker=tracker,
        log_every=1,
        on_step=on_step,
        initial_total_training_time_seconds=7.0,
        initial_total_training_flops=100.0,
        peak_flops_basis=peak_basis,
        clock=clock,
        reset_memory_peak=reset_memory,
        collect_memory=collect_memory,
    )

    flops = estimate_gpt_training_flops(model_config)
    step_flops = flops.flops_for_tokens(4)
    assert events == [
        "memory:reset",
        "clock:0.0",
        "forward",
        "forward",
        "clock:2.0",
        "memory:collect",
        "tracker:1",
        "callback:1",
        "memory:reset",
        "clock:152.0",
        "forward",
        "forward",
        "clock:154.0",
        "memory:collect",
        "tracker:2",
        "callback:2",
    ]
    assert len(results) == 2
    for index, result in enumerate(results, start=1):
        telemetry = result.telemetry
        assert telemetry is not None
        assert telemetry.processed_model_tokens == 4
        assert telemetry.supervised_target_tokens == 2
        assert telemetry.duration_seconds == 2.0
        assert telemetry.tokens_per_second == 2.0
        assert telemetry.step_flops == step_flops
        assert telemetry.total_training_flops == 100.0 + index * step_flops
        assert telemetry.total_training_time_seconds == 7.0 + index * 2.0
        assert telemetry.mfu == pytest.approx(step_flops / 2.0 / 1_000_000.0)
        assert telemetry.peak_flops_basis == peak_basis
        assert telemetry.peak_memory_mib == 6.0
        assert telemetry.flops_estimate == flops
        assert result.step_duration_seconds == telemetry.duration_seconds
        assert (
            result.total_training_time_seconds == telemetry.total_training_time_seconds
        )
        assert result.total_training_flops == telemetry.total_training_flops
        serialized = telemetry.to_dict()
        assert serialized["processed_model_tokens"] == 4
        assert serialized["supervised_target_tokens"] == 2
        assert serialized["peak_flops_basis"] == peak_basis.to_dict()
        assert serialized["flops_estimate"] == flops.to_dict()
        metrics, tracked_step = tracker.records[index - 1]
        assert tracked_step == index
        assert metrics == base_training_metrics(
            telemetry,
            loss=result.loss,
            learning_rate_multiplier=1.0,
            grad_norm=result.grad_norm,
            epoch=float(index * 2),
        )
        assert metrics == {
            "train/loss": result.loss,
            "train/lrm": 1.0,
            "train/dt": 2.0,
            "train/tok_per_sec": 2.0,
            "train/mfu": pytest.approx(step_flops / 2.0 / 1_000_000.0),
            "train/epoch": float(index * 2),
            "train/grad_norm": result.grad_norm,
            "train/peak_memory_mib": 6.0,
            "total_training_flops": 100.0 + index * step_flops,
            "total_training_time": 7.0 + index * 2.0,
        }


def test_cpu_training_omits_cuda_only_peak_memory() -> None:
    model_config = GPTConfig(
        vocab_size=8,
        seq_len=2,
        n_layer=1,
        n_head=1,
        n_embd=2,
        mlp_ratio=2,
    )
    train_config = TrainConfig(
        device_batch_size=1,
        total_batch_size_tokens=2,
        grad_accum_steps=1,
        max_steps=1,
        warmup_steps=0,
        warmdown_ratio=0.0,
    )
    model = GPT(model_config)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_lr_scheduler(optimizer, train_config)
    clock = iter((2.0, 3.0))

    result = run_training_steps(
        model,
        [(torch.tensor([[1, 2]]), torch.tensor([[2, 3]]))],
        optimizer,
        scheduler,
        max_steps=1,
        grad_accum_steps=1,
        grad_clip=train_config.grad_clip,
        device="cpu",
        clock=lambda: next(clock),
    )[0]

    assert result.telemetry is not None
    assert result.telemetry.peak_memory_mib is None
