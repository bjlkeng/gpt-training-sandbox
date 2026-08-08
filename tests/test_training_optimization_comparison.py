"""Identity-safe Phase 12 training-throughput comparison contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.compare_training_benchmarks import main as compare_main
from scratch_llm.attention_backends import AttentionBackendSelection
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.diagnostics.throughput import (
    BenchmarkExecution,
    build_throughput_benchmark,
)
from scratch_llm.diagnostics.throughput_comparison import (
    TRAINING_OPTIMIZATION_COMPARISON_FORMAT,
    TRAINING_OPTIMIZATION_COMPARISON_FORMAT_VERSION,
    TrainingOptimizationComparisonError,
    compare_training_benchmarks,
)
from scratch_llm.training.activation_checkpointing import (
    ActivationCheckpointSelection,
)
from scratch_llm.training.compilation import CompileSelection
from scratch_llm.training.loop import OptimizerStepResult
from scratch_llm.training.telemetry import (
    PeakFlopsBasis,
    TrainingStepTelemetry,
    estimate_gpt_training_flops,
)
from scratch_llm.utils import save_json


_MIB = 1024**2


def _config(
    tmp_path: Path,
    optimization: str,
    *,
    learning_rate: float = 0.0003,
) -> ProjectConfig:
    model_values: dict[str, object] = {
        "vocab_size": 265,
        "seq_len": 4,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "mlp_ratio": 2,
        "dropout": 0,
    }
    train_values: dict[str, object] = {
        "device_batch_size": 1,
        "total_batch_size_tokens": 4,
        "grad_accum_steps": 1,
        "max_steps": 10,
        "learning_rate": learning_rate,
        "warmup_steps": 0,
        "warmdown_ratio": 0,
        "mfu_peak_flops_per_second": 1_000_000.0,
        "mfu_peak_flops_basis": "fixture peak",
    }
    if optimization == "amp":
        train_values["dtype"] = "bfloat16"
    elif optimization == "sdpa":
        model_values["attention_backend"] = "sdpa"
    elif optimization == "flash":
        model_values["attention_backend"] = "flash"
    elif optimization == "compile":
        train_values["compile"] = True
    elif optimization == "activation_checkpointing":
        train_values["activation_checkpointing"] = True
    elif optimization == "combined":
        model_values["attention_backend"] = "sdpa"
        train_values.update(
            {
                "dtype": "bfloat16",
                "compile": True,
                "activation_checkpointing": True,
            }
        )
    elif optimization != "baseline":
        raise AssertionError(f"unknown fixture optimization {optimization}")
    return ProjectConfig(
        run=RunConfig(
            name=f"benchmark-{optimization}",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tokenizer=TokenizerConfig(type="byte", vocab_size=265),
        model=GPTConfig(**model_values),  # type: ignore[arg-type]
        train=TrainConfig(**train_values),  # type: ignore[arg-type]
    )


def _execution(
    config: ProjectConfig,
    *,
    tokens_per_second: float,
    peak_allocated_mib: int,
    hardware_name: str = "fixture-device",
    attention: AttentionBackendSelection | None = None,
    compile_selection: CompileSelection | None = None,
    activation: ActivationCheckpointSelection | None = None,
) -> BenchmarkExecution:
    flops = estimate_gpt_training_flops(config.model)
    basis = PeakFlopsBasis(1_000_000.0, "fixture peak")
    steps = []
    snapshots = []
    total_time = 0.0
    total_flops = 0
    for index in range(2):
        duration = 4 / tokens_per_second
        total_time += duration
        step_flops = flops.flops_for_tokens(4)
        total_flops += step_flops
        steps.append(
            OptimizerStepResult(
                loss=1.0,
                grad_norm=1.0,
                telemetry=TrainingStepTelemetry(
                    processed_model_tokens=4,
                    supervised_target_tokens=4,
                    duration_seconds=duration,
                    tokens_per_second=tokens_per_second,
                    step_flops=step_flops,
                    total_training_flops=total_flops,
                    total_training_time_seconds=total_time,
                    mfu=step_flops / duration / basis.flops_per_second,
                    peak_flops_basis=basis,
                    peak_memory_mib=float(peak_allocated_mib),
                    flops_estimate=flops,
                ),
            )
        )
        snapshots.append(
            AcceleratorMemorySnapshot(
                device=torch.device("cuda", 0),
                available=True,
                allocated_bytes=(peak_allocated_mib - 2) * _MIB,
                reserved_bytes=(peak_allocated_mib - 1) * _MIB,
                peak_allocated_bytes=peak_allocated_mib * _MIB,
                peak_reserved_bytes=(peak_allocated_mib + 1) * _MIB,
                capacity_bytes=24 * 1024 * _MIB,
            )
        )
    return BenchmarkExecution(
        steps=tuple(steps),
        memory_snapshots=tuple(snapshots),
        tokenizer_identity="tokenizer:fixture",
        manifest_identity="manifest:fixture",
        hardware_identity={"device": "cuda:0", "device_name": hardware_name},
        cuda_identity={"available": True, "compiled_version": "fixture"},
        pytorch_identity={"version": "fixture"},
        code_identity={"commit": "abc123", "tracked_dirty": False},
        attention_selection=attention,
        compile_selection=compile_selection,
        activation_checkpoint_selection=activation,
        precision_selection={
            "autocast_enabled": config.train.dtype != "float32",
            "device_type": "cpu",
            "effective_dtype": config.train.dtype,
            "requested_dtype": config.train.dtype,
            "scaler_enabled": config.train.dtype == "float16",
        },
    )


def _write_report(
    tmp_path: Path,
    optimization: str,
    *,
    tokens_per_second: float,
    peak_allocated_mib: int,
    hardware_name: str = "fixture-device",
    flash_fallback: bool = False,
    learning_rate: float = 0.0003,
) -> Path:
    config = _config(tmp_path, optimization, learning_rate=learning_rate)
    requested_backend = config.model.attention_backend
    attention = AttentionBackendSelection(
        requested_backend=requested_backend,
        effective_backend=(
            "sdpa" if optimization == "flash" and flash_fallback else requested_backend
        ),
        fallback_reason=(
            "flash_dependency_unavailable"
            if optimization == "flash" and flash_fallback
            else None
        ),
    )
    compile_selection = CompileSelection(
        requested=config.train.compile,
        effective=config.train.compile,
        backend=config.train.compile_backend,
        mode=config.train.compile_mode,
        fullgraph=config.train.compile_fullgraph,
        dynamic=config.train.compile_dynamic,
        compile_duration_seconds=0.5 if config.train.compile else 0,
    )
    activation = ActivationCheckpointSelection(
        requested=config.train.activation_checkpointing,
        effective=config.train.activation_checkpointing,
    )
    completed = build_throughput_benchmark(
        config,
        execution=_execution(
            config,
            tokens_per_second=tokens_per_second,
            peak_allocated_mib=peak_allocated_mib,
            hardware_name=hardware_name,
            attention=attention,
            compile_selection=compile_selection,
            activation=activation,
        ),
        warmup_steps=1,
        timed_steps=1,
    )
    return save_json(completed.to_dict(), tmp_path / f"{optimization}.json")


def _reports(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    baseline = _write_report(
        tmp_path,
        "baseline",
        tokens_per_second=100,
        peak_allocated_mib=100,
    )
    variants = {
        "amp": _write_report(
            tmp_path,
            "amp",
            tokens_per_second=125,
            peak_allocated_mib=80,
        ),
        "sdpa": _write_report(
            tmp_path,
            "sdpa",
            tokens_per_second=150,
            peak_allocated_mib=90,
        ),
        "flash": _write_report(
            tmp_path,
            "flash",
            tokens_per_second=145,
            peak_allocated_mib=88,
            flash_fallback=True,
        ),
        "compile": _write_report(
            tmp_path,
            "compile",
            tokens_per_second=130,
            peak_allocated_mib=95,
        ),
        "activation_checkpointing": _write_report(
            tmp_path,
            "activation_checkpointing",
            tokens_per_second=75,
            peak_allocated_mib=60,
        ),
        "combined": _write_report(
            tmp_path,
            "combined",
            tokens_per_second=170,
            peak_allocated_mib=65,
        ),
    }
    return baseline, variants


def test_benchmark_reports_comparison_identities_and_precision_state(
    tmp_path: Path,
) -> None:
    baseline, variants = _reports(tmp_path)
    baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
    amp_payload = json.loads(variants["amp"].read_text(encoding="utf-8"))

    assert baseline_payload["optimization_state"]["precision"] == {
        "autocast_enabled": False,
        "device_type": "cpu",
        "effective_dtype": "float32",
        "requested_dtype": "float32",
        "scaler_enabled": False,
    }
    assert amp_payload["optimization_state"]["precision"]["effective_dtype"] == (
        "bfloat16"
    )
    assert (
        baseline_payload["identities"]["config"] != amp_payload["identities"]["config"]
    )
    assert baseline_payload["identities"]["model"] == amp_payload["identities"]["model"]
    assert baseline_payload["identities"]["data"] == amp_payload["identities"]["data"]
    assert (
        baseline_payload["identities"]["workload"]
        == amp_payload["identities"]["workload"]
    )


def test_comparison_manifest_covers_all_variants_and_exact_deltas(
    tmp_path: Path,
) -> None:
    baseline, variants = _reports(tmp_path)

    artifacts = compare_training_benchmarks(
        baseline,
        variants,
        output_dir=tmp_path / "comparison",
    )
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))

    assert payload["format"] == TRAINING_OPTIMIZATION_COMPARISON_FORMAT
    assert payload["format_version"] == (
        TRAINING_OPTIMIZATION_COMPARISON_FORMAT_VERSION
    )
    assert [
        entry["declared_optimization"] for entry in payload["manifest"]["variants"]
    ] == [
        "amp",
        "sdpa",
        "flash",
        "compile",
        "activation_checkpointing",
        "combined",
    ]
    amp = payload["variants"]["amp"]
    assert amp["deltas"]["tokens_per_second"] == {
        "absolute": 25.0,
        "relative_fraction": 0.25,
        "relative_percent": 25.0,
    }
    assert amp["deltas"]["peak_allocated_bytes"] == {
        "absolute": -20 * _MIB,
        "relative_fraction": -0.2,
        "relative_percent": -20.0,
    }
    assert amp["deltas"]["peak_allocated_mib"]["absolute"] == -20.0
    assert payload["identity_comparison"]["config"]["all_equal"] is False
    for field in (
        "code",
        "cuda",
        "data",
        "hardware",
        "manifest",
        "model",
        "pytorch",
        "tokenizer",
        "workload",
    ):
        assert payload["identity_comparison"][field]["all_equal"] is True
    assert artifacts.markdown_path.read_text(encoding="utf-8").startswith(
        "# Training optimization comparison\n"
    )


def test_flash_fallback_is_never_labeled_as_flash_execution(tmp_path: Path) -> None:
    baseline, variants = _reports(tmp_path)

    artifacts = compare_training_benchmarks(
        baseline,
        {"flash": variants["flash"]},
        output_dir=tmp_path / "comparison",
    )
    flash = json.loads(artifacts.json_path.read_text(encoding="utf-8"))["variants"][
        "flash"
    ]

    assert flash["declared_optimization"] == "flash"
    assert flash["requested_state"]["attention"]["requested_backend"] == "flash"
    assert flash["effective_state"]["attention_backend"] == "sdpa"
    assert flash["fallback"] == {
        "occurred": True,
        "reasons": ["flash_dependency_unavailable"],
    }
    assert flash["result_label"] == "flash→sdpa fallback"
    assert flash["result_label"] != "flash"


@pytest.mark.parametrize(
    "mismatch",
    ["hardware", "workload", "processed_tokens", "protocol"],
)
def test_comparison_rejects_uncontrolled_identity_or_work_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    baseline = _write_report(
        tmp_path,
        "baseline",
        tokens_per_second=100,
        peak_allocated_mib=100,
    )
    variant = _write_report(
        tmp_path,
        "amp",
        tokens_per_second=110,
        peak_allocated_mib=90,
        hardware_name="other-device" if mismatch == "hardware" else "fixture-device",
        learning_rate=0.0004 if mismatch == "workload" else 0.0003,
    )
    payload = json.loads(variant.read_text(encoding="utf-8"))
    if mismatch == "processed_tokens":
        payload["measurements"]["processed_model_tokens"] += 1
    elif mismatch == "protocol":
        payload["protocol"]["timed_steps"] += 1
    save_json(payload, variant)

    with pytest.raises(TrainingOptimizationComparisonError, match=mismatch):
        compare_training_benchmarks(
            baseline,
            {"amp": variant},
            output_dir=tmp_path / "comparison",
        )


def test_declared_variant_must_match_exact_requested_switch_set(
    tmp_path: Path,
) -> None:
    baseline, variants = _reports(tmp_path)

    with pytest.raises(TrainingOptimizationComparisonError, match="declared.*sdpa"):
        compare_training_benchmarks(
            baseline,
            {"sdpa": variants["amp"]},
            output_dir=tmp_path / "wrong-declaration",
        )
    with pytest.raises(TrainingOptimizationComparisonError, match="combined"):
        compare_training_benchmarks(
            baseline,
            {"combined": variants["amp"]},
            output_dir=tmp_path / "wrong-combined",
        )


def test_comparison_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    baseline, variants = _reports(tmp_path)
    output_dir = tmp_path / "cli-comparison"

    result = compare_main(
        [
            "--baseline",
            str(baseline),
            "--variant",
            f"amp={variants['amp']}",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert (output_dir / "training_optimization_comparison.json").is_file()
    assert (output_dir / "training_optimization_comparison.md").is_file()
