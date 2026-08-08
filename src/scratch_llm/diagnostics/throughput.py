"""Immutable aggregation and atomic reporting for pretraining throughput runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scratch_llm.attention_backends import AttentionBackendSelection
from scratch_llm.training.compilation import CompileSelection
from scratch_llm.training.activation_checkpointing import (
    ActivationCheckpointSelection,
)
from scratch_llm._validation import (
    JsonValueValidator,
    require_finite_non_negative_real,
    require_non_empty_string,
    require_positive_integer,
)
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.config import ProjectConfig
from scratch_llm.identity import canonical_json_identity, project_config_identity
from scratch_llm.diagnostics.resource_estimation import (
    compare_memory_estimate,
    estimate_training_resources,
)
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.training.loop import OptimizerStepResult
from scratch_llm.training.telemetry import TrainingStepTelemetry
from scratch_llm.utils import load_json, save_json


THROUGHPUT_BENCHMARK_FORMAT: Final = "scratch_llm_throughput_benchmark"
THROUGHPUT_BENCHMARK_FORMAT_VERSION: Final = 1
THROUGHPUT_BENCHMARK_PROTOCOL_ID: Final = "production_pretraining_optimizer_steps_v1"
_REPORT_RELATIVE_PATH = Path("metrics/throughput_benchmark.json")
_IDENTITY_FIELDS = (
    "code_identity",
    "cuda_identity",
    "hardware_identity",
    "pytorch_identity",
)


class ThroughputBenchmarkConflictError(RuntimeError):
    """An existing canonical report belongs to another benchmark protocol."""


_JSON_VALUES: Final = JsonValueValidator(ValueError)
_EXISTING_JSON_VALUES: Final = JsonValueValidator(ThroughputBenchmarkConflictError)


@dataclass(frozen=True)
class BenchmarkExecution:
    """Raw shared-step results plus the immutable environment that produced them."""

    steps: tuple[OptimizerStepResult, ...]
    memory_snapshots: tuple[AcceleratorMemorySnapshot, ...]
    tokenizer_identity: str
    manifest_identity: str
    hardware_identity: Mapping[str, object]
    cuda_identity: Mapping[str, object]
    pytorch_identity: Mapping[str, object]
    code_identity: Mapping[str, object]
    attention_selection: AttentionBackendSelection | None = None
    compile_selection: CompileSelection | None = None
    activation_checkpoint_selection: ActivationCheckpointSelection | None = None
    precision_selection: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("steps must be a non-empty tuple")
        if not all(isinstance(step, OptimizerStepResult) for step in self.steps):
            raise TypeError("steps must contain only OptimizerStepResult values")
        if any(step.telemetry is None for step in self.steps):
            raise ValueError("every benchmark step must expose training telemetry")
        if (
            not isinstance(self.memory_snapshots, tuple)
            or len(self.memory_snapshots) != len(self.steps)
            or not all(
                isinstance(snapshot, AcceleratorMemorySnapshot)
                for snapshot in self.memory_snapshots
            )
        ):
            raise ValueError("memory_snapshots must align one-for-one with steps")
        for name in ("tokenizer_identity", "manifest_identity"):
            require_non_empty_string(getattr(self, name), name=name)
        for field in _IDENTITY_FIELDS:
            object.__setattr__(
                self,
                field,
                MappingProxyType(_json_object(getattr(self, field), label=field)),
            )
        if self.attention_selection is not None and not isinstance(
            self.attention_selection,
            AttentionBackendSelection,
        ):
            raise TypeError(
                "attention_selection must be an AttentionBackendSelection or None"
            )
        if self.compile_selection is not None and not isinstance(
            self.compile_selection,
            CompileSelection,
        ):
            raise TypeError("compile_selection must be a CompileSelection or None")
        if self.activation_checkpoint_selection is not None and not isinstance(
            self.activation_checkpoint_selection,
            ActivationCheckpointSelection,
        ):
            raise TypeError(
                "activation_checkpoint_selection must be an "
                "ActivationCheckpointSelection or None"
            )
        if self.precision_selection is not None:
            object.__setattr__(
                self,
                "precision_selection",
                MappingProxyType(
                    _json_object(
                        dict(self.precision_selection),
                        label="precision_selection",
                    )
                ),
            )


@dataclass(frozen=True)
class CompletedThroughputBenchmark:
    """One fully validated and JSON-compatible benchmark report."""

    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        canonical = _json_object(self.payload, label="benchmark payload")
        if canonical.get("format") != THROUGHPUT_BENCHMARK_FORMAT:
            raise ValueError("benchmark payload format is invalid")
        if canonical.get("format_version") != THROUGHPUT_BENCHMARK_FORMAT_VERSION:
            raise ValueError("benchmark payload version is invalid")
        if canonical.get("status") != "completed":
            raise ValueError("benchmark payload must be completed")
        object.__setattr__(self, "payload", MappingProxyType(canonical))

    @property
    def protocol_identity(self) -> str:
        value = self.payload["protocol_identity"]
        assert isinstance(value, str)
        return value

    def to_dict(self) -> dict[str, object]:
        """Return a detached canonical JSON object."""

        return _json_object(dict(self.payload), label="benchmark payload")


@dataclass(frozen=True)
class ThroughputBenchmarkArtifacts:
    """The installed report and its complete immutable domain result."""

    report_path: Path
    completed: CompletedThroughputBenchmark


def build_throughput_benchmark(
    config: ProjectConfig,
    *,
    execution: BenchmarkExecution,
    warmup_steps: int,
    timed_steps: int,
) -> CompletedThroughputBenchmark:
    """Aggregate only timed shared-step telemetry into the benchmark schema."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    if not isinstance(execution, BenchmarkExecution):
        raise TypeError("execution must be a BenchmarkExecution")
    config.validate()
    warmup_steps = require_positive_integer(warmup_steps, name="warmup_steps")
    timed_steps = require_positive_integer(timed_steps, name="timed_steps")
    expected_steps = warmup_steps + timed_steps
    if len(execution.steps) != expected_steps:
        raise ValueError(
            f"execution contains {len(execution.steps)} steps; expected {expected_steps}"
        )

    timed_results = execution.steps[warmup_steps:]
    timed_snapshots = execution.memory_snapshots[warmup_steps:]
    telemetry = tuple(_require_telemetry(result) for result in timed_results)
    memory = _aggregate_memory(timed_snapshots)
    resource_estimate = estimate_training_resources(config)
    identities = {
        "code": dict(execution.code_identity),
        "config": project_config_identity(config),
        "cuda": dict(execution.cuda_identity),
        "data": canonical_json_identity(config.data.to_dict()),
        "hardware": dict(execution.hardware_identity),
        "manifest": execution.manifest_identity,
        "model": canonical_json_identity(config.model.parameter_compatibility_dict()),
        "pytorch": dict(execution.pytorch_identity),
        "tokenizer": execution.tokenizer_identity,
        "workload": _training_workload_identity(config),
    }
    protocol = {
        "excluded_work": [
            "artifact and tokenizer loading",
            "cold torch.compile graph construction (warmup; reported separately)",
            "data-loader planning",
            "model and optimizer construction",
            "validation and sampling",
            "checkpoint I/O",
            "peak-memory reset and collection",
            "Tracker fan-out",
            "report construction and writing",
        ],
        "id": THROUGHPUT_BENCHMARK_PROTOCOL_ID,
        "included_work": [
            "training-batch retrieval and device transfer",
            "forward and backward passes",
            "gradient clipping",
            "optimizer update and gradient clearing",
            "scheduler update",
        ],
        "timed_steps": timed_steps,
        "timing_source": "TrainingStepTelemetry.duration_seconds",
        "version": 1,
        "warmup_steps": warmup_steps,
    }
    selection = execution.attention_selection or AttentionBackendSelection(
        requested_backend=config.model.attention_backend,
        effective_backend=config.model.attention_backend,
    )
    compile_selection = execution.compile_selection or CompileSelection(
        requested=config.train.compile,
        effective=config.train.compile,
        backend=config.train.compile_backend,
        mode=config.train.compile_mode,
        fullgraph=config.train.compile_fullgraph,
        dynamic=config.train.compile_dynamic,
    )
    activation_checkpoint_selection = (
        execution.activation_checkpoint_selection
        or ActivationCheckpointSelection(
            requested=config.train.activation_checkpointing,
            effective=config.train.activation_checkpointing,
        )
    )
    precision_selection = _precision_selection(config, execution.precision_selection)
    optimization_state = {
        "activation_checkpointing": activation_checkpoint_selection.to_dict(),
        "attention": selection.to_dict(),
        "compile": compile_selection.to_dict(),
        "precision": precision_selection,
    }
    protocol_identity = _payload_identity(
        {
            "identities": identities,
            "optimization_state": optimization_state,
            "protocol": protocol,
        }
    )
    payload: dict[str, object] = {
        "format": THROUGHPUT_BENCHMARK_FORMAT,
        "format_version": THROUGHPUT_BENCHMARK_FORMAT_VERSION,
        "identities": identities,
        "measurements": _aggregate_measurements(telemetry, memory=memory),
        "optimization_state": optimization_state,
        "protocol": protocol,
        "protocol_identity": protocol_identity,
        "resource_estimate": resource_estimate.to_dict(),
        "resource_estimate_delta": compare_memory_estimate(
            resource_estimate.memory,
            memory,
        ).to_dict(),
        "status": "completed",
        "timed_step_telemetry": [
            _timed_step_payload(
                result,
                step_telemetry,
                benchmark_step=benchmark_step,
                optimizer_step=warmup_steps + benchmark_step,
            )
            for benchmark_step, (result, step_telemetry) in enumerate(
                zip(timed_results, telemetry, strict=True),
                start=1,
            )
        ],
    }
    return CompletedThroughputBenchmark(payload)


def report_throughput_benchmark(
    completed: CompletedThroughputBenchmark,
    *,
    run_dir: str | Path,
    tracker: Tracker,
) -> ThroughputBenchmarkArtifacts:
    """Atomically install one report, then fan out finalized scalar metadata."""

    if not isinstance(completed, CompletedThroughputBenchmark):
        raise TypeError("completed must be a CompletedThroughputBenchmark")
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")
    report_path = Path(run_dir) / _REPORT_RELATIVE_PATH
    payload = completed.to_dict()
    _validate_existing_protocol(report_path, completed.protocol_identity)
    report_path = save_json(payload, report_path)

    measurements = payload["measurements"]
    assert isinstance(measurements, dict)
    metrics = {
        "benchmark/elapsed_seconds": measurements["elapsed_seconds"],
        "benchmark/mfu": measurements["mfu"],
        "benchmark/peak_memory_mib": measurements["peak_allocated_mib"],
        "benchmark/supervised_tokens": measurements["supervised_target_tokens"],
        "benchmark/tokens_per_sec": measurements["tokens_per_second"],
        "benchmark/training_flops": measurements["training_flops"],
    }
    event_prefix = f"throughput-benchmark:{_payload_identity(payload)}"
    if isinstance(tracker, RunTracker):
        tracker.log_once(metrics, event_id=f"{event_prefix}:metrics")
        tracker.log_artifact_once(
            _REPORT_RELATIVE_PATH.as_posix(),
            "throughput_benchmark",
            "benchmark",
            event_id=f"{event_prefix}:artifact",
        )
    else:
        tracker.log(metrics)
        tracker.log_artifact(
            _REPORT_RELATIVE_PATH.as_posix(),
            "throughput_benchmark",
            "benchmark",
        )
    return ThroughputBenchmarkArtifacts(
        report_path=report_path,
        completed=completed,
    )


def _require_telemetry(result: OptimizerStepResult) -> TrainingStepTelemetry:
    telemetry = result.telemetry
    if not isinstance(telemetry, TrainingStepTelemetry):
        raise ValueError("benchmark optimizer steps must expose training telemetry")
    return telemetry


def _timed_step_payload(
    result: OptimizerStepResult,
    telemetry: TrainingStepTelemetry,
    *,
    benchmark_step: int,
    optimizer_step: int,
) -> dict[str, object]:
    payload = telemetry.to_dict()
    payload.pop("total_training_flops")
    payload.pop("total_training_time_seconds")
    return {
        "benchmark_step": benchmark_step,
        "grad_norm": require_finite_non_negative_real(
            result.grad_norm,
            name="grad_norm",
        ),
        "loss": require_finite_non_negative_real(result.loss, name="loss"),
        "optimizer_step": optimizer_step,
        **payload,
    }


def _aggregate_measurements(
    telemetry: tuple[TrainingStepTelemetry, ...],
    *,
    memory: AcceleratorMemorySnapshot,
) -> dict[str, object]:
    processed_tokens = sum(step.processed_model_tokens for step in telemetry)
    supervised_tokens = sum(step.supervised_target_tokens for step in telemetry)
    elapsed_seconds = sum(step.duration_seconds for step in telemetry)
    training_flops = sum(step.step_flops for step in telemetry)
    bases = {step.peak_flops_basis for step in telemetry}
    if len(bases) != 1:
        raise ValueError("timed steps must use one MFU peak basis")
    basis = next(iter(bases))
    mfu = (
        None
        if basis is None
        else training_flops / elapsed_seconds / basis.flops_per_second
    )
    return {
        "elapsed_seconds": elapsed_seconds,
        "mfu": mfu,
        "mfu_basis": None if basis is None else basis.to_dict(),
        "peak_allocated_bytes": memory.peak_allocated_bytes,
        "peak_allocated_mib": memory.peak_allocated_mib,
        "peak_reserved_bytes": memory.peak_reserved_bytes,
        "peak_reserved_mib": memory.peak_reserved_mib,
        "processed_model_tokens": processed_tokens,
        "supervised_target_tokens": supervised_tokens,
        "tokens_per_second": processed_tokens / elapsed_seconds,
        "training_flops": training_flops,
    }


def _aggregate_memory(
    snapshots: tuple[AcceleratorMemorySnapshot, ...],
) -> AcceleratorMemorySnapshot:
    available = {snapshot.available for snapshot in snapshots}
    if len(available) != 1:
        raise ValueError("timed memory snapshots cannot mix available and unavailable")
    devices = {snapshot.device for snapshot in snapshots}
    if len(devices) != 1:
        raise ValueError("timed memory snapshots must use one device")
    if not snapshots[0].available:
        reasons = {snapshot.unavailable_reason for snapshot in snapshots}
        return AcceleratorMemorySnapshot(
            device=snapshots[0].device,
            available=False,
            unavailable_reason="; ".join(sorted(str(reason) for reason in reasons)),
        )

    last = snapshots[-1]
    capacities = {snapshot.capacity_bytes for snapshot in snapshots}
    if len(capacities) != 1:
        raise ValueError("timed memory snapshots must use one device capacity")
    return AcceleratorMemorySnapshot(
        device=last.device,
        available=True,
        allocated_bytes=last.allocated_bytes,
        reserved_bytes=last.reserved_bytes,
        peak_allocated_bytes=max(
            _present(snapshot.peak_allocated_bytes) for snapshot in snapshots
        ),
        peak_reserved_bytes=max(
            _present(snapshot.peak_reserved_bytes) for snapshot in snapshots
        ),
        capacity_bytes=last.capacity_bytes,
    )


def _present(value: int | None) -> int:
    if value is None:  # pragma: no cover - available snapshots guarantee values.
        raise ValueError("available memory snapshot omitted a counter")
    return value


def _validate_existing_protocol(path: Path, expected_identity: str) -> None:
    try:
        existing = load_json(path)
    except FileNotFoundError:
        return
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ThroughputBenchmarkConflictError(
            f"existing throughput benchmark cannot be validated: {error}"
        ) from error
    value = _EXISTING_JSON_VALUES.require_object(
        existing,
        label=f"existing throughput benchmark {path}",
    )
    reconstructed_identity = _payload_identity(
        {
            "identities": value.get("identities"),
            "optimization_state": value.get("optimization_state"),
            "protocol": value.get("protocol"),
        }
    )
    if (
        value.get("format") != THROUGHPUT_BENCHMARK_FORMAT
        or value.get("format_version") != THROUGHPUT_BENCHMARK_FORMAT_VERSION
        or value.get("status") != "completed"
        or value.get("protocol_identity") != reconstructed_identity
        or value.get("protocol_identity") != expected_identity
    ):
        raise ThroughputBenchmarkConflictError(
            f"{path} belongs to a different benchmark protocol identity"
        )


def _json_object(value: object, *, label: str) -> dict[str, object]:
    try:
        canonical = json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-compatible: {error}") from error
    return _JSON_VALUES.require_object(canonical, label=label)


def _payload_identity(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _precision_selection(
    config: ProjectConfig,
    observed: Mapping[str, object] | None,
) -> dict[str, object]:
    expected = {
        "autocast_enabled": config.train.dtype != "float32",
        "device_type": config.run.device.partition(":")[0],
        "effective_dtype": config.train.dtype,
        "requested_dtype": config.train.dtype,
        "scaler_enabled": config.train.dtype == "float16",
    }
    if observed is None:
        return expected
    normalized = _json_object(dict(observed), label="precision_selection")
    if normalized != expected:
        raise ValueError(
            "benchmark precision selection does not match the requested runtime: "
            f"expected {expected}, got {normalized}"
        )
    return normalized


def _training_workload_identity(config: ProjectConfig) -> str:
    model = config.model.to_dict()
    model.pop("attention_backend")
    train = config.train.to_dict()
    for field_name in (
        "activation_checkpointing",
        "compile",
        "dtype",
    ):
        train.pop(field_name)
    return canonical_json_identity(
        {
            "data": config.data.to_dict(),
            "model": model,
            "run": {
                "device": config.run.device,
                "seed": config.run.seed,
            },
            "tokenizer": config.tokenizer.to_dict(),
            "train": train,
        }
    )


__all__ = [
    "THROUGHPUT_BENCHMARK_FORMAT",
    "THROUGHPUT_BENCHMARK_FORMAT_VERSION",
    "THROUGHPUT_BENCHMARK_PROTOCOL_ID",
    "BenchmarkExecution",
    "CompletedThroughputBenchmark",
    "ThroughputBenchmarkArtifacts",
    "ThroughputBenchmarkConflictError",
    "build_throughput_benchmark",
    "report_throughput_benchmark",
]
