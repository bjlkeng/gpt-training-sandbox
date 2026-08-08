"""CPU coverage for the benchmark timing boundary around shared training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch

import scratch_llm.diagnostics.throughput_runtime as throughput_runtime
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.model import GPT
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.diagnostics.throughput_runtime import run_benchmark_training_steps


def test_fake_clock_measures_only_shared_optimizer_step_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig(
        run=RunConfig(name="cpu-benchmark", device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=265),
        model=GPTConfig(
            vocab_size=265,
            seq_len=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=4,
            grad_accum_steps=1,
            max_steps=10,
            warmup_steps=0,
            warmdown_ratio=0.0,
            dtype="bfloat16",
        ),
    )
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    clock_values: Iterator[float] = iter((10.0, 11.0, 20.0, 23.0))
    collected: list[str] = []
    observed_precision: list[tuple[str, bool, bool]] = []
    real_run_training_steps = throughput_runtime.run_training_steps

    def record_precision(*args: Any, **kwargs: Any):
        precision = kwargs["precision"]
        observed_precision.append(
            (
                precision.dtype,
                precision.autocast_enabled,
                precision.scaler_enabled,
            )
        )
        return real_run_training_steps(*args, **kwargs)

    monkeypatch.setattr(throughput_runtime, "run_training_steps", record_precision)

    def collect_memory(device: str | torch.device) -> AcceleratorMemorySnapshot:
        collected.append(str(device))
        return AcceleratorMemorySnapshot(
            device=torch.device(device),
            available=False,
            unavailable_reason="CPU has no CUDA allocator counters",
        )

    execution = run_benchmark_training_steps(
        config,
        model=model,
        batches=[(inputs, targets)],
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_steps=1,
        timed_steps=1,
        clock=lambda: next(clock_values),
        reset_memory_peak=lambda device: False,
        collect_memory=collect_memory,
    )

    assert [step.telemetry.duration_seconds for step in execution.steps] == [1.0, 3.0]  # type: ignore[union-attr]
    assert len(execution.memory_snapshots) == 2
    assert all(not snapshot.available for snapshot in execution.memory_snapshots)
    assert collected == ["cpu"]
    assert observed_precision == [("bfloat16", True, False)]
