"""Tests for single-device step mechanics and the phase-one text train path."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import scratch_llm.training as training
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.data import NextTokenDataset
from scratch_llm.model import GPT
from scratch_llm.optim import build_lr_scheduler, build_optimizer
from scratch_llm.run import prepare_run
from scratch_llm.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.tracking import JsonlTracker
from scratch_llm.training import (
    derive_grad_accum_steps,
    run_optimizer_step,
    run_training_steps,
    train_tiny_text,
)
from scratch_llm.utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_TEXT_PATH = PROJECT_ROOT / "data" / "fixtures" / "tiny.txt"
# At seed 1234, the fixed batch below must reach this mean cross-entropy in 60
# CPU steps. This is intentionally strict enough to catch a broken train path.
FIXED_BATCH_OVERFIT_MAX_LOSS = 1e-3


def _tiny_project_config(
    *,
    device: str = "cpu",
    max_steps: int = 12,
    output_dir: Path | None = None,
    seed: int = 123,
) -> ProjectConfig:
    seq_len = 16
    device_batch_size = 4
    return ProjectConfig(
        run=RunConfig(
            seed=seed,
            device=device,
            output_dir=str(output_dir or "runs/out"),
        ),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=device_batch_size,
            total_batch_size_tokens=device_batch_size * seq_len,
            grad_accum_steps="auto",
            max_steps=max_steps,
            learning_rate=0.01,
            weight_decay=0.0,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
    )


class _ValidationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_states: list[tuple[bool, bool, bool]] = []

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        self.forward_states.append(
            (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
                self.training,
            )
        )
        return F.mse_loss(inputs * self.weight, targets)


@pytest.mark.parametrize("training_mode", [True, False])
def test_validation_aggregates_by_target_count_without_mutating_training_state(
    training_mode: bool,
) -> None:
    model = _ValidationProbe()
    model.train(training_mode)
    model.weight.grad = torch.tensor(7.0)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1)
    parameter_before = model.weight.detach().clone()
    gradient_before = model.weight.grad.detach().clone()
    optimizer_before = deepcopy(optimizer.state_dict())
    scheduler_before = deepcopy(scheduler.state_dict())
    batches = [
        (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[0.0, 0.0]]),
        ),
        (
            torch.tensor([[3.0]]),
            torch.tensor([[0.0]]),
        ),
    ]

    loss = training.run_validation(model, batches, device="cpu")

    assert loss == pytest.approx(14.0 / 3.0)
    assert model.forward_states == [
        (False, True, False),
        (False, True, False),
    ]
    assert model.training is training_mode
    torch.testing.assert_close(model.weight, parameter_before)
    torch.testing.assert_close(model.weight.grad, gradient_before)
    assert optimizer.state_dict() == optimizer_before
    assert scheduler.state_dict() == scheduler_before


def test_tiny_text_smoke_writes_core_metrics_under_the_run_directory(
    tmp_path: Path,
) -> None:
    config = _tiny_project_config(
        max_steps=3,
        output_dir=tmp_path / "runs",
    )
    config.train.log_every = 1
    text = TINY_TEXT_PATH.read_text(encoding="utf-8")
    paths = prepare_run(config)
    destination = paths.run_dir / config.tracking.jsonl.path
    tracker = JsonlTracker(destination)

    try:
        result = train_tiny_text(text, config, tracker=tracker)
    finally:
        tracker.finish()

    records = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    batches_per_epoch = (
        len(
            NextTokenDataset(
                ByteTokenizer().encode(text),
                config.model.seq_len,
            )
        )
        // config.train.device_batch_size
    )

    assert destination == paths.metrics_dir / "metrics.jsonl"
    assert [record["step"] for record in records] == [1, 2, 3]
    assert all(record["record_type"] == "metrics" for record in records)
    assert len(records) == len(result.steps)
    for record, step_result in zip(records, result.steps, strict=True):
        metrics = record["metrics"]
        assert {
            "train/loss",
            "train/lrm",
            "train/dt",
            "train/grad_norm",
            "train/epoch",
        } <= metrics.keys()
        assert metrics["train/loss"] == pytest.approx(step_result.loss)
        assert metrics["train/lrm"] == pytest.approx(1.0)
        assert metrics["train/dt"] >= 0.0
        assert metrics["train/grad_norm"] == pytest.approx(step_result.grad_norm)
        assert metrics["train/epoch"] == pytest.approx(
            record["step"] / batches_per_epoch
        )


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


def test_cpu_training_loop_runs_forward_backward_clip_optimizer_and_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_seed(7)
    tokenizer = ByteTokenizer()
    dataset = NextTokenDataset(
        tokenizer.encode("abcd abcd abcd"),
        seq_len=4,
        vocab_size=tokenizer.get_vocab_size(),
    )
    batches = DataLoader(dataset, batch_size=2, shuffle=False)
    model = GPT(
        GPTConfig(
            vocab_size=tokenizer.get_vocab_size(),
            seq_len=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        )
    )
    train_config = TrainConfig(
        device_batch_size=2,
        total_batch_size_tokens=8,
        grad_accum_steps=1,
        max_steps=1,
        learning_rate=0.01,
        weight_decay=0.0,
        warmup_steps=0,
        warmdown_ratio=0.0,
    )
    optimizer = build_optimizer(model, train_config)
    scheduler = build_lr_scheduler(optimizer, train_config)
    events: list[str] = []
    original_forward = model.forward
    original_clip = training.clip_grad_norm_
    original_scheduler_step = scheduler.step

    def record_forward(*args: Any, **kwargs: Any) -> Tensor:
        events.append("forward")
        return original_forward(*args, **kwargs)

    def record_backward(gradient: Tensor) -> Tensor:
        events.append("backward")
        return gradient

    def record_clip(parameters: Iterable[Tensor], max_norm: float) -> Tensor:
        events.append("clip")
        return original_clip(parameters, max_norm)

    def record_optimizer_step(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        events.append("optimizer")

    def record_scheduler_step(*args: Any, **kwargs: Any) -> Any:
        events.append("scheduler")
        return original_scheduler_step(*args, **kwargs)

    monkeypatch.setattr(model, "forward", record_forward)
    monkeypatch.setattr(training, "clip_grad_norm_", record_clip)
    monkeypatch.setattr(scheduler, "step", record_scheduler_step)
    gradient_hook = model.token_embedding.weight.register_hook(record_backward)
    optimizer_hook = optimizer.register_step_post_hook(record_optimizer_step)

    results = run_training_steps(
        model,
        batches,
        optimizer,
        scheduler,
        max_steps=1,
        grad_accum_steps=1,
        grad_clip=train_config.grad_clip,
        device="cpu",
    )
    gradient_hook.remove()
    optimizer_hook.remove()

    assert events == ["forward", "backward", "clip", "optimizer", "scheduler"]
    assert len(results) == 1
    assert torch.isfinite(torch.tensor(results[0].loss))
    assert torch.isfinite(torch.tensor(results[0].grad_norm))
    assert scheduler.last_epoch == 1
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_fixed_seed_training_loss_decreases_on_the_tiny_text_fixture() -> None:
    config = _tiny_project_config()
    text = TINY_TEXT_PATH.read_text(encoding="utf-8")

    result = train_tiny_text(text, config)
    repeated_result = train_tiny_text(text, config)
    losses = [step.loss for step in result.steps]
    repeated_losses = [step.loss for step in repeated_result.steps]

    assert len(losses) == config.train.max_steps
    assert repeated_losses == losses
    assert losses[-1] < losses[0] * 0.7
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert isinstance(result.optimizer, AdamW)
    assert result.scheduler.last_epoch == config.train.max_steps
    assert all(
        parameter.device.type == "cpu" for parameter in result.model.parameters()
    )


def test_fixed_batch_reaches_the_documented_deterministic_overfit_threshold() -> None:
    set_seed(1234)
    tokenizer = ByteTokenizer()
    dataset = NextTokenDataset(
        tokenizer.encode("abcd efgh abcd efgh abcd efgh"),
        seq_len=8,
        vocab_size=tokenizer.get_vocab_size(),
    )
    inputs, targets = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    model = GPT(
        GPTConfig(
            vocab_size=tokenizer.get_vocab_size(),
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_ratio=2,
        )
    )
    train_config = TrainConfig(
        device_batch_size=2,
        total_batch_size_tokens=16,
        grad_accum_steps=1,
        max_steps=60,
        learning_rate=0.03,
        weight_decay=0.0,
        warmup_steps=0,
        warmdown_ratio=0.0,
    )
    optimizer = build_optimizer(model, train_config)
    scheduler = build_lr_scheduler(optimizer, train_config)

    run_training_steps(
        model,
        [(inputs, targets)],
        optimizer,
        scheduler,
        max_steps=train_config.max_steps,
        grad_accum_steps=1,
        grad_clip=train_config.grad_clip,
        device="cpu",
    )

    with torch.inference_mode():
        final_loss = model(inputs, targets).item()
    assert final_loss < FIXED_BATCH_OVERFIT_MAX_LOSS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tiny_text_training_runs_conditionally_on_cuda() -> None:
    config = _tiny_project_config(device="cuda", max_steps=1)

    result = train_tiny_text(
        TINY_TEXT_PATH.read_text(encoding="utf-8"),
        config,
    )

    assert len(result.steps) == 1
    assert torch.isfinite(torch.tensor(result.steps[0].loss))
    assert all(
        parameter.device.type == "cuda" for parameter in result.model.parameters()
    )
