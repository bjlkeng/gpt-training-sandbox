"""Protocol and publication tests for bounded pretraining benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scratch_llm.attention_backends import AttentionBackendSelection
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.throughput import (
    THROUGHPUT_BENCHMARK_FORMAT,
    THROUGHPUT_BENCHMARK_FORMAT_VERSION,
    BenchmarkExecution,
    ThroughputBenchmarkConflictError,
    build_throughput_benchmark,
    report_throughput_benchmark,
)
from scratch_llm.training.loop import OptimizerStepResult
from scratch_llm.training.compilation import CompileSelection
from scratch_llm.training.activation_checkpointing import (
    ActivationCheckpointSelection,
)
from scratch_llm.training.telemetry import (
    PeakFlopsBasis,
    TrainingStepTelemetry,
    estimate_gpt_training_flops,
)
from scratch_llm.tracking import NullTracker


_MIB = 1024**2
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, *, name: str = "benchmark") -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(
            name=name,
            device="cuda",
            output_dir=str(tmp_path / "runs"),
        ),
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
            mfu_peak_flops_per_second=1_000_000.0,
            mfu_peak_flops_basis="fake peak",
        ),
    )


def _execution(
    config: ProjectConfig,
    *,
    durations: tuple[float, ...],
    attention_selection: AttentionBackendSelection | None = None,
    compile_selection: CompileSelection | None = None,
    activation_checkpoint_selection: ActivationCheckpointSelection | None = None,
) -> BenchmarkExecution:
    flops = estimate_gpt_training_flops(config.model)
    basis = PeakFlopsBasis(1_000_000.0, "fake peak")
    total_time = 0.0
    total_flops = 0.0
    steps: list[OptimizerStepResult] = []
    snapshots: list[AcceleratorMemorySnapshot] = []
    for index, duration in enumerate(durations, start=1):
        total_time += duration
        step_flops = flops.flops_for_tokens(4)
        total_flops += step_flops
        peak_allocated = (index + 1) * _MIB
        peak_reserved = (index + 2) * _MIB
        telemetry = TrainingStepTelemetry(
            processed_model_tokens=4,
            supervised_target_tokens=3,
            duration_seconds=duration,
            tokens_per_second=4 / duration,
            step_flops=step_flops,
            total_training_flops=total_flops,
            total_training_time_seconds=total_time,
            mfu=step_flops / duration / basis.flops_per_second,
            peak_flops_basis=basis,
            peak_memory_mib=peak_allocated / _MIB,
            flops_estimate=flops,
        )
        steps.append(
            OptimizerStepResult(
                loss=1.0 / index,
                grad_norm=float(index),
                telemetry=telemetry,
            )
        )
        snapshots.append(
            AcceleratorMemorySnapshot(
                device=torch.device("cuda", 0),
                available=True,
                allocated_bytes=index * _MIB,
                reserved_bytes=(index + 1) * _MIB,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                capacity_bytes=24 * 1024 * _MIB,
            )
        )
    return BenchmarkExecution(
        steps=tuple(steps),
        memory_snapshots=tuple(snapshots),
        tokenizer_identity="tokenizer:fixture",
        manifest_identity="manifest:fixture",
        hardware_identity={"device_name": "Fake RTX 3090"},
        cuda_identity={"available": True, "runtime_version": "fixture"},
        pytorch_identity={"version": "fixture"},
        code_identity={"commit": "abc123", "tracked_dirty": False},
        attention_selection=attention_selection,
        compile_selection=compile_selection,
        activation_checkpoint_selection=activation_checkpoint_selection,
    )


def test_timed_aggregate_excludes_warmup_and_preserves_shared_telemetry(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    completed = build_throughput_benchmark(
        config,
        execution=_execution(config, durations=(10.0, 2.0, 4.0)),
        warmup_steps=1,
        timed_steps=2,
    )

    artifacts = report_throughput_benchmark(
        completed,
        run_dir=tmp_path / "run",
        tracker=NullTracker(),
    )

    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert payload["format"] == THROUGHPUT_BENCHMARK_FORMAT
    assert payload["format_version"] == THROUGHPUT_BENCHMARK_FORMAT_VERSION
    assert payload["protocol"]["warmup_steps"] == 1
    assert payload["protocol"]["timed_steps"] == 2
    assert payload["measurements"]["processed_model_tokens"] == 8
    assert payload["measurements"]["supervised_target_tokens"] == 6
    assert payload["measurements"]["elapsed_seconds"] == 6.0
    assert payload["measurements"]["tokens_per_second"] == pytest.approx(8 / 6)
    assert payload["measurements"]["training_flops"] == sum(
        step["step_flops"] for step in payload["timed_step_telemetry"]
    )
    assert all(
        "total_training_flops" not in step and "total_training_time_seconds" not in step
        for step in payload["timed_step_telemetry"]
    )
    assert payload["measurements"]["peak_allocated_mib"] == 4.0
    assert payload["measurements"]["peak_reserved_mib"] == 5.0
    assert payload["measurements"]["mfu_basis"] == {
        "description": "fake peak",
        "flops_per_second": 1_000_000.0,
    }
    assert payload["identities"]["config"].startswith("sha256:")
    assert payload["identities"]["tokenizer"] == "tokenizer:fixture"
    assert payload["identities"]["manifest"] == "manifest:fixture"
    assert payload["optimization_state"]["attention"] == {
        "effective_backend": "manual",
        "fallback_reason": None,
        "provider": None,
        "provider_version": None,
        "requested_backend": "manual",
    }
    assert payload["resource_estimate"]["memory"]["classification"] == (
        "conservative_estimate_not_observed"
    )
    assert (
        payload["resource_estimate_delta"]["observed"]["peak_allocated_bytes"]
        == 4 * _MIB
    )


def test_actual_attention_fallback_enters_report_and_protocol_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    execution = _execution(config, durations=(1.0, 2.0))
    direct = build_throughput_benchmark(
        config,
        execution=execution,
        warmup_steps=1,
        timed_steps=1,
    )
    fallback_selection = AttentionBackendSelection(
        requested_backend="flash",
        effective_backend="sdpa",
        fallback_reason="flash_dependency_unavailable",
    )
    fallback = build_throughput_benchmark(
        config,
        execution=_execution(
            config,
            durations=(1.0, 2.0),
            attention_selection=fallback_selection,
        ),
        warmup_steps=1,
        timed_steps=1,
    )

    assert fallback.payload["optimization_state"]["attention"] == (
        fallback_selection.to_dict()
    )
    assert fallback.protocol_identity != direct.protocol_identity


def test_compile_startup_and_effective_state_enter_report_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    direct = build_throughput_benchmark(
        config,
        execution=_execution(config, durations=(1.0, 2.0)),
        warmup_steps=1,
        timed_steps=1,
    )
    compile_selection = CompileSelection(
        requested=True,
        effective=True,
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
        compile_duration_seconds=1.25,
    )
    compiled = build_throughput_benchmark(
        config,
        execution=_execution(
            config,
            durations=(9.0, 2.0),
            compile_selection=compile_selection,
        ),
        warmup_steps=1,
        timed_steps=1,
    )

    assert compiled.payload["optimization_state"]["compile"] == (
        compile_selection.to_dict()
    )
    assert compiled.payload["measurements"]["elapsed_seconds"] == 2.0
    assert compiled.protocol_identity != direct.protocol_identity


def test_activation_checkpoint_state_enters_report_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    direct = build_throughput_benchmark(
        config,
        execution=_execution(config, durations=(1.0, 2.0)),
        warmup_steps=1,
        timed_steps=1,
    )
    selection = ActivationCheckpointSelection(requested=True, effective=True)
    checkpointed = build_throughput_benchmark(
        config,
        execution=_execution(
            config,
            durations=(1.0, 2.0),
            activation_checkpoint_selection=selection,
        ),
        warmup_steps=1,
        timed_steps=1,
    )

    assert (
        checkpointed.payload["optimization_state"]["activation_checkpointing"]
        == selection.to_dict()
    )
    assert checkpointed.protocol_identity != direct.protocol_identity


def test_atomic_report_failure_preserves_prior_complete_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    first = build_throughput_benchmark(
        config,
        execution=_execution(config, durations=(1.0, 2.0)),
        warmup_steps=1,
        timed_steps=1,
    )
    report = report_throughput_benchmark(
        first,
        run_dir=run_dir,
        tracker=NullTracker(),
    ).report_path
    stable = report.read_bytes()
    replacement = build_throughput_benchmark(
        config,
        execution=_execution(config, durations=(1.0, 3.0)),
        warmup_steps=1,
        timed_steps=1,
    )

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("atomic replace failed")

    monkeypatch.setattr("scratch_llm.utils.os.replace", fail_replace)
    with pytest.raises(OSError, match="atomic replace failed"):
        report_throughput_benchmark(
            replacement,
            run_dir=run_dir,
            tracker=NullTracker(),
        )

    assert report.read_bytes() == stable
    assert not list(report.parent.glob(".throughput_benchmark.json.*.tmp"))


def test_existing_report_rejects_a_different_protocol_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    completed = build_throughput_benchmark(
        config,
        execution=_execution(config, durations=(1.0, 2.0)),
        warmup_steps=1,
        timed_steps=1,
    )
    report = report_throughput_benchmark(
        completed,
        run_dir=run_dir,
        tracker=NullTracker(),
    ).report_path
    stable = report.read_bytes()
    changed_config = _config(tmp_path, name="different-config")
    changed = build_throughput_benchmark(
        changed_config,
        execution=_execution(changed_config, durations=(1.0, 2.0)),
        warmup_steps=1,
        timed_steps=1,
    )

    with pytest.raises(ThroughputBenchmarkConflictError, match="protocol identity"):
        report_throughput_benchmark(
            changed,
            run_dir=run_dir,
            tracker=NullTracker(),
        )

    assert report.read_bytes() == stable


def test_rtx_3090_guide_documents_complete_ordered_workflow() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for command in (
        "scripts.download_climbmix",
        "scripts.train_tokenizer",
        "scripts.prepare_data",
        "scripts.pretrain",
        "scripts.eval_base",
        "scripts.benchmark_pretrain",
        "scripts.compare_runs",
    ):
        assert command in readme
    for preset_run in (
        "base-smoke-3090",
        "tiny-20m-3090",
        "small-45m-3090",
        "base-smoke-throughput",
        "tiny-20m-throughput",
        "small-45m-throughput",
    ):
        assert preset_run in readme
    for artifact in (
        "metrics/throughput_benchmark.json",
        "metrics/base_eval.json",
        "metrics/base_samples.md",
        "comparison.json",
        "comparison.md",
    ):
        assert artifact in readme
    ordered_oom_fields = (
        "1. `train.device_batch_size`",
        "2. `model.seq_len`",
        "3. `model.n_embd`",
        "4. `model.n_layer`",
    )
    assert list(map(readme.index, ordered_oom_fields)) == sorted(
        map(readme.index, ordered_oom_fields)
    )
    assert "never a promise that a configuration will fit" in readme
    assert "AMP, `GradScaler`" in readme
    assert "activation checkpointing" in readme
    assert "`torch.compile`" in readme
    assert "FlashAttention" in readme
    assert "Phase 12" in readme
