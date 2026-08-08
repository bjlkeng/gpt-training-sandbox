"""CPU-safe coverage for the shared mixed-precision training policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from scratch_llm.training.precision import (
    PrecisionCheckpointState,
    PrecisionError,
    PrecisionPolicy,
    build_precision_policy,
)
from scratch_llm.training.loop import run_optimizer_step, run_training_steps
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.model import GPT
from scratch_llm.tracking import NullTracker
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.checkpoint import (
    ExactTrainingState,
    load_training_checkpoint,
    save_checkpoint,
)
from scratch_llm.training.rng_state import capture_training_rng_state
from scratch_llm.tokenization.tokenizer import ByteTokenizer, VOCAB_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeScaler:
    def __init__(self, events: list[str], *, skip_step: bool = False) -> None:
        self.events = events
        self.skip_step = skip_step
        self.remaining_skips = 1 if skip_step else 0
        self.scale_value = 8.0

    def is_enabled(self) -> bool:
        return True

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        self.events.append("scale")
        return loss * self.scale_value

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.events.append("unscale")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.events.append("scaler_step")
        if self.remaining_skips:
            self.remaining_skips -= 1
            return
        optimizer.step()

    def update(self) -> None:
        self.events.append("update")

    def state_dict(self) -> dict[str, Any]:
        return {"scale": self.scale_value}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.events.append("load")
        self.scale_value = float(state_dict["scale"])


class _GrowingFakeScaler(_FakeScaler):
    def update(self) -> None:
        super().update()
        self.scale_value *= 2.0


def _fake_float16_policy(
    events: list[str],
    *,
    skip_step: bool = False,
) -> PrecisionPolicy:
    return PrecisionPolicy(
        dtype="float16",
        device=torch.device("cuda"),
        autocast_dtype=torch.float16,
        scaler=_FakeScaler(events, skip_step=skip_step),
        autocast_factory=lambda **_: nullcontext(),
    )


def test_runtime_precision_matrix_is_explicit_and_actionable() -> None:
    float32 = build_precision_policy(dtype="float32", device="cpu")
    bfloat16 = build_precision_policy(dtype="bfloat16", device="cpu")

    assert float32.autocast_enabled is False
    assert float32.scaler_enabled is False
    assert bfloat16.autocast_enabled is True
    assert bfloat16.scaler_enabled is False
    with pytest.raises(PrecisionError, match="float16.*CUDA"):
        build_precision_policy(dtype="float16", device="cpu")


@pytest.mark.parametrize("skip_step", [False, True])
def test_float16_policy_orders_unscale_step_and_update(
    skip_step: bool,
) -> None:
    events: list[str] = []
    policy = _fake_float16_policy(events, skip_step=skip_step)
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = SGD([parameter], lr=0.1)
    original_step = optimizer.step

    def recorded_step(*args: Any, **kwargs: Any) -> Any:
        events.append("optimizer_step")
        return original_step(*args, **kwargs)

    optimizer.step = recorded_step  # type: ignore[method-assign]
    policy.backward(parameter.square())
    policy.unscale_(optimizer)
    applied = policy.step_and_update(optimizer)

    assert events == [
        "scale",
        "unscale",
        "scaler_step",
        *([] if skip_step else ["optimizer_step"]),
        "update",
    ]
    assert applied is (not skip_step)


def test_precision_checkpoint_state_round_trips_complete_scaler_state() -> None:
    events: list[str] = []
    policy = _fake_float16_policy(events)
    state = policy.checkpoint_state()

    assert state == PrecisionCheckpointState(
        dtype="float16",
        device_type="cuda",
        scaler_enabled=True,
        scaler_state={"scale": 8.0},
    )
    restored = _fake_float16_policy(events)
    restored.load_checkpoint_state(
        PrecisionCheckpointState(
            dtype="float16",
            device_type="cuda",
            scaler_enabled=True,
            scaler_state={"scale": 4.0},
        )
    )
    assert restored.checkpoint_state().scaler_state == {"scale": 4.0}
    assert events[-1] == "load"


def test_fake_float16_scaler_resume_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(device="cuda"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=2,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=2,
            grad_accum_steps=1,
            max_steps=2,
            learning_rate=0.01,
            weight_decay=0.0,
            warmup_steps=0,
            warmdown_ratio=0.0,
            dtype="float16",
        ),
    )
    batches = [
        (
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([[2, 3]], dtype=torch.long),
        ),
        (
            torch.tensor([[3, 4]], dtype=torch.long),
            torch.tensor([[4, 5]], dtype=torch.long),
        ),
    ]

    def build_runtime() -> tuple[
        GPT,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
        PrecisionPolicy,
    ]:
        torch.manual_seed(11)
        model = GPT(config.model)
        optimizer = build_optimizer(model, config.train)
        scheduler = build_lr_scheduler(optimizer, config.train)
        policy = PrecisionPolicy(
            dtype="float16",
            device=torch.device("cuda"),
            autocast_dtype=torch.float16,
            scaler=_GrowingFakeScaler([]),
            autocast_factory=lambda **_: nullcontext(),
        )
        return model, optimizer, scheduler, policy

    uninterrupted = build_runtime()
    run_training_steps(
        uninterrupted[0],
        batches,
        uninterrupted[1],
        uninterrupted[2],
        max_steps=2,
        grad_accum_steps=1,
        grad_clip=1.0,
        device="cpu",
        precision=uninterrupted[3],
    )

    interrupted = build_runtime()
    run_training_steps(
        interrupted[0],
        [batches[0]],
        interrupted[1],
        interrupted[2],
        max_steps=1,
        grad_accum_steps=1,
        grad_clip=1.0,
        device="cpu",
        precision=interrupted[3],
    )
    checkpoint_path = save_checkpoint(
        tmp_path / "scaled.pt",
        model=interrupted[0],
        optimizer=interrupted[1],
        scheduler=interrupted[2],
        config=config,
        step=1,
        tokenizer=ByteTokenizer(),
        continuation=ExactTrainingState(
            loader_format="precision_fixture",
            loader_state={"format": "precision_fixture", "position": 1},
            rng_state=capture_training_rng_state("cpu"),
            tracker_step=1,
            total_training_time_seconds=0.0,
            total_training_flops=0.0,
        ),
        precision=interrupted[3].checkpoint_state(),
    )
    resumed_policy = PrecisionPolicy(
        dtype="float16",
        device=torch.device("cuda"),
        autocast_dtype=torch.float16,
        scaler=_GrowingFakeScaler([]),
        autocast_factory=lambda **_: nullcontext(),
    )
    resumed = load_training_checkpoint(
        checkpoint_path,
        device="cpu",
        expected_precision=resumed_policy.checkpoint_state(),
    )
    assert resumed.precision is not None
    resumed_policy.load_checkpoint_state(resumed.precision)
    run_training_steps(
        resumed.model,
        [batches[1]],
        resumed.optimizer,
        resumed.scheduler,
        max_steps=2,
        grad_accum_steps=1,
        grad_clip=1.0,
        device="cpu",
        precision=resumed_policy,
    )

    def assert_nested_equal(actual: Any, expected: Any) -> None:
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, Mapping):
            assert set(actual) == set(expected)
            for key, value in expected.items():
                assert_nested_equal(actual[key], value)
        elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected, strict=True):
                assert_nested_equal(actual_item, expected_item)
        else:
            assert actual == expected

    assert_nested_equal(resumed.model.state_dict(), uninterrupted[0].state_dict())
    assert_nested_equal(
        resumed.optimizer.state_dict(),
        uninterrupted[1].state_dict(),
    )
    assert_nested_equal(
        resumed.scheduler.state_dict(),
        uninterrupted[2].state_dict(),
    )
    assert resumed_policy.checkpoint_state() == uninterrupted[3].checkpoint_state()


def test_shared_optimizer_boundary_unscales_before_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = _fake_float16_policy(events)
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = SGD([parameter], lr=0.1)
    original_step = optimizer.step

    def recorded_step(*args: Any, **kwargs: Any) -> Any:
        events.append("optimizer_step")
        return original_step(*args, **kwargs)

    def recorded_clip(
        parameters: Any,
        max_norm: float,
    ) -> torch.Tensor:
        events.append("clip")
        assert max_norm == 1.0
        assert parameter.grad is not None
        assert parameter.grad.item() == pytest.approx(4.0)
        return torch.tensor(4.0)

    optimizer.step = recorded_step  # type: ignore[method-assign]
    monkeypatch.setattr(
        "scratch_llm.training.loop.clip_grad_norm_",
        recorded_clip,
    )

    result = run_optimizer_step(
        optimizer,
        [parameter.square()],
        grad_accum_steps=1,
        grad_clip=1.0,
        precision=policy,
    )

    assert result.optimizer_step_applied is True
    assert result.grad_norm == 4.0
    assert events == [
        "scale",
        "unscale",
        "clip",
        "scaler_step",
        "optimizer_step",
        "update",
    ]


def test_skipped_scaled_attempt_does_not_advance_completed_step_state() -> None:
    events: list[str] = []
    policy = _fake_float16_policy(events, skip_step=True)

    class ScalarLossModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
        ) -> torch.Tensor:
            del targets
            return (self.weight * inputs.float()).square().mean()

    model = ScalarLossModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    completed_steps: list[int] = []
    clock_values = iter([0.0, 1.0, 2.0, 3.0])

    results = run_training_steps(
        model,
        [(torch.ones((1, 1), dtype=torch.long), torch.ones((1, 1)))],
        optimizer,
        scheduler,
        max_steps=1,
        grad_accum_steps=1,
        grad_clip=1.0,
        device="cpu",
        precision=policy,
        on_step=lambda step, _: completed_steps.append(step),
        clock=lambda: next(clock_values),
    )

    assert scheduler.last_epoch == 1
    assert completed_steps == [1]
    assert len(results) == 1
    assert results[0].optimizer_step_applied is True
    assert events.count("scale") == 2
    assert events.count("scaler_step") == 2
    assert events.count("update") == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_cuda_precision_smoke_records_finite_throughput_and_peak_memory(
    dtype: str,
) -> None:
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    class MetricsTracker(NullTracker):
        def __init__(self) -> None:
            self.records: list[dict[str, float | None]] = []

        def log(
            self,
            metrics: dict[str, float | None],
            step: int | None = None,
        ) -> None:
            assert step == 1
            self.records.append(metrics)

    torch.manual_seed(7)
    model_config = GPTConfig(
        vocab_size=64,
        seq_len=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        mlp_ratio=2,
    )
    train_config = TrainConfig(
        device_batch_size=2,
        total_batch_size_tokens=16,
        grad_accum_steps=1,
        max_steps=1,
        learning_rate=0.01,
        weight_decay=0.0,
        warmup_steps=0,
        warmdown_ratio=0.0,
        log_every=1,
        dtype=dtype,  # type: ignore[arg-type]
    )
    model = GPT(model_config).to("cuda")
    optimizer = build_optimizer(model, train_config)
    scheduler = build_lr_scheduler(optimizer, train_config)
    policy = build_precision_policy(
        dtype=train_config.dtype,
        device=torch.device("cuda"),
    )
    tracker = MetricsTracker()
    inputs = torch.randint(0, model_config.vocab_size, (2, 8))
    targets = torch.randint(0, model_config.vocab_size, (2, 8))

    results = run_training_steps(
        model,
        [(inputs, targets)],
        optimizer,
        scheduler,
        max_steps=1,
        grad_accum_steps=1,
        grad_clip=1.0,
        device="cuda",
        tracker=tracker,
        precision=policy,
    )

    assert torch.isfinite(torch.tensor(results[0].loss))
    assert torch.isfinite(torch.tensor(results[0].grad_norm))
    assert tracker.records[0]["train/tok_per_sec"] > 0  # type: ignore[operator]
    assert tracker.records[0]["train/peak_memory_mib"] > 0  # type: ignore[operator]


def test_readme_documents_precision_semantics_and_reproducible_smoke() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "float16` is CUDA-only" in readme
    assert "bfloat16` uses autocast without scaling" in readme
    assert "do not advance the scheduler, global step, or checkpoint cadence" in readme
    assert "tests/test_precision.py -k cuda_precision_smoke" in readme
    assert "checkpoint format version 7" in readme
