"""Identity-safe comparison of completed training-throughput benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Final

from scratch_llm.identity import canonical_json_identity
from scratch_llm.diagnostics.throughput import (
    THROUGHPUT_BENCHMARK_FORMAT,
    THROUGHPUT_BENCHMARK_FORMAT_VERSION,
)
from scratch_llm.utils import atomic_write, save_json


TRAINING_OPTIMIZATION_COMPARISON_FORMAT: Final = (
    "scratch_llm_training_optimization_comparison"
)
TRAINING_OPTIMIZATION_COMPARISON_FORMAT_VERSION: Final = 1
TRAINING_OPTIMIZATION_VARIANTS: Final = (
    "amp",
    "sdpa",
    "flash",
    "compile",
    "activation_checkpointing",
    "combined",
)
_IDENTITY_FIELDS: Final = (
    "code",
    "config",
    "cuda",
    "data",
    "hardware",
    "manifest",
    "model",
    "pytorch",
    "tokenizer",
    "workload",
)
_REQUIRED_EQUAL_IDENTITIES: Final = frozenset(_IDENTITY_FIELDS) - {"config"}
_REQUIRED_EQUAL_MEASUREMENTS: Final = (
    "mfu_basis",
    "processed_model_tokens",
    "supervised_target_tokens",
    "training_flops",
)
_JSON_NAME: Final = "training_optimization_comparison.json"
_MARKDOWN_NAME: Final = "training_optimization_comparison.md"


class TrainingOptimizationComparisonError(ValueError):
    """One or more benchmark reports cannot be compared safely."""


@dataclass(frozen=True)
class TrainingOptimizationComparisonArtifacts:
    """Canonical comparison reports installed by one offline operation."""

    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class _BenchmarkSnapshot:
    path: Path
    payload: dict[str, object]
    identities: dict[str, object]
    measurements: dict[str, object]
    optimization_state: dict[str, object]
    protocol: dict[str, object]
    protocol_identity: str


def compare_training_benchmarks(
    baseline_report: str | os.PathLike[str],
    variants: Mapping[str, str | os.PathLike[str]],
    *,
    output_dir: str | os.PathLike[str],
) -> TrainingOptimizationComparisonArtifacts:
    """Validate matched reports and atomically write JSON and Markdown results."""

    if not isinstance(variants, Mapping) or not variants:
        raise TrainingOptimizationComparisonError(
            "variants must be a non-empty mapping of optimization names to reports"
        )
    if not all(isinstance(name, str) for name in variants):
        raise TrainingOptimizationComparisonError(
            "declared optimization names must be strings"
        )
    unknown = sorted(set(variants) - set(TRAINING_OPTIMIZATION_VARIANTS))
    if unknown:
        raise TrainingOptimizationComparisonError(
            f"unknown declared optimization variants: {unknown}"
        )

    baseline = _load_benchmark(baseline_report, label="baseline")
    _validate_baseline(baseline)
    ordered_names = tuple(
        name for name in TRAINING_OPTIMIZATION_VARIANTS if name in variants
    )
    snapshots = {
        name: _load_benchmark(variants[name], label=f"variant {name}")
        for name in ordered_names
    }
    for name, snapshot in snapshots.items():
        _validate_declared_variant(name, snapshot)
        _validate_comparability(baseline, snapshot, declared=name)

    identity_comparison = _identity_comparison(baseline, snapshots)
    variant_results = {
        name: _variant_result(name, baseline, snapshots[name]) for name in ordered_names
    }
    payload: dict[str, object] = {
        "baseline": _baseline_result(baseline),
        "format": TRAINING_OPTIMIZATION_COMPARISON_FORMAT,
        "format_version": TRAINING_OPTIMIZATION_COMPARISON_FORMAT_VERSION,
        "identity_comparison": identity_comparison,
        "manifest": {
            "baseline": {
                "path": str(baseline.path),
                "protocol_identity": baseline.protocol_identity,
            },
            "variants": [
                {
                    "declared_optimization": name,
                    "path": str(snapshots[name].path),
                    "protocol_identity": snapshots[name].protocol_identity,
                }
                for name in ordered_names
            ],
        },
        "protocol": baseline.protocol,
        "status": "completed",
        "variants": variant_results,
    }
    destination = Path(output_dir)
    markdown = render_training_optimization_comparison(payload)
    markdown_path = atomic_write(destination / _MARKDOWN_NAME, markdown)
    json_path = save_json(payload, destination / _JSON_NAME)
    return TrainingOptimizationComparisonArtifacts(
        json_path=json_path,
        markdown_path=markdown_path,
    )


def render_training_optimization_comparison(payload: Mapping[str, object]) -> str:
    """Render one compact human-readable view of a validated comparison."""

    baseline = _require_object(payload.get("baseline"), label="baseline")
    variants = _require_object(payload.get("variants"), label="variants")
    identities = _require_object(
        payload.get("identity_comparison"),
        label="identity_comparison",
    )
    baseline_measurements = _require_object(
        baseline.get("measurements"),
        label="baseline measurements",
    )
    lines = [
        "# Training optimization comparison",
        "",
        "All rows use the same bounded production optimizer-step timing boundary. ",
        "Cold compiler startup is reported separately and is not included in tokens/sec.",
        "",
        "## Results",
        "",
        "| Variant | Observed result | Tokens/sec | Δ tokens/sec | Peak MiB | Δ peak MiB | Compile startup (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        _result_row("baseline", baseline),
    ]
    for name in TRAINING_OPTIMIZATION_VARIANTS:
        if name in variants:
            result = _require_object(variants[name], label=f"variant {name}")
            lines.append(_result_row(name, result))
    lines.extend(
        [
            "",
            "## Identity checks",
            "",
            "| Identity | Equal across reports | Required |",
            "| --- | --- | --- |",
        ]
    )
    for field in _IDENTITY_FIELDS:
        comparison = _require_object(identities[field], label=f"identity {field}")
        lines.append(
            f"| `{field}` | {_yes_no(comparison['all_equal'])} | "
            f"{_yes_no(comparison['required_equal'])} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Warmup steps: {payload['protocol']['warmup_steps']}",  # type: ignore[index]
            f"- Timed steps: {payload['protocol']['timed_steps']}",  # type: ignore[index]
            f"- Baseline processed tokens: {baseline_measurements['processed_model_tokens']}",
            "- Requested and effective states are preserved in the JSON report; a fallback row is labeled with the backend that actually ran.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_benchmark(
    report: str | os.PathLike[str],
    *,
    label: str,
) -> _BenchmarkSnapshot:
    path = Path(report).resolve()
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(
                source,
                parse_constant=lambda token: _reject_json_constant(token, path),
            )
    except TrainingOptimizationComparisonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingOptimizationComparisonError(
            f"cannot load {label} benchmark {path}: {error}"
        ) from error
    payload = _require_object(value, label=f"{label} benchmark")
    if payload.get("format") != THROUGHPUT_BENCHMARK_FORMAT:
        raise TrainingOptimizationComparisonError(
            f"{label} benchmark has an unsupported format"
        )
    if payload.get("format_version") != THROUGHPUT_BENCHMARK_FORMAT_VERSION:
        raise TrainingOptimizationComparisonError(
            f"{label} benchmark has an unsupported format version"
        )
    if payload.get("status") != "completed":
        raise TrainingOptimizationComparisonError(
            f"{label} benchmark must have completed status"
        )
    identities = _require_object(payload.get("identities"), label=f"{label} identities")
    missing_identities = sorted(set(_IDENTITY_FIELDS) - set(identities))
    if missing_identities:
        raise TrainingOptimizationComparisonError(
            f"{label} identities are missing {missing_identities}"
        )
    measurements = _require_object(
        payload.get("measurements"),
        label=f"{label} measurements",
    )
    optimization_state = _require_object(
        payload.get("optimization_state"),
        label=f"{label} optimization_state",
    )
    for field in ("activation_checkpointing", "attention", "compile", "precision"):
        _require_object(
            optimization_state.get(field),
            label=f"{label} optimization_state.{field}",
        )
    _validate_optimization_state(optimization_state, label=label)
    protocol = _require_object(payload.get("protocol"), label=f"{label} protocol")
    protocol_identity = _require_string(
        payload.get("protocol_identity"),
        label=f"{label} protocol identity",
    )
    reconstructed_identity = canonical_json_identity(
        {
            "identities": identities,
            "optimization_state": optimization_state,
            "protocol": protocol,
        }
    )
    if protocol_identity != reconstructed_identity:
        raise TrainingOptimizationComparisonError(
            f"{label} protocol identity does not match report contents"
        )
    _validate_measurements(measurements, label=label)
    _validate_protocol(protocol, label=label)
    return _BenchmarkSnapshot(
        path=path,
        payload=payload,
        identities=identities,
        measurements=measurements,
        optimization_state=optimization_state,
        protocol=protocol,
        protocol_identity=protocol_identity,
    )


def _validate_baseline(snapshot: _BenchmarkSnapshot) -> None:
    state = snapshot.optimization_state
    precision = _state_object(state, "precision")
    attention = _state_object(state, "attention")
    compile_state = _state_object(state, "compile")
    activation = _state_object(state, "activation_checkpointing")
    expected = {
        "precision requested_dtype": (precision.get("requested_dtype"), "float32"),
        "precision effective_dtype": (precision.get("effective_dtype"), "float32"),
        "attention requested_backend": (attention.get("requested_backend"), "manual"),
        "attention effective_backend": (attention.get("effective_backend"), "manual"),
        "compile requested": (compile_state.get("requested"), False),
        "compile effective": (compile_state.get("effective"), False),
        "activation checkpointing requested": (activation.get("requested"), False),
        "activation checkpointing effective": (activation.get("effective"), False),
    }
    mismatches = [
        f"{field}={actual!r} (expected {wanted!r})"
        for field, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise TrainingOptimizationComparisonError(
            "baseline must be float32/manual without compile or activation "
            f"checkpointing: {', '.join(mismatches)}"
        )


def _validate_declared_variant(name: str, snapshot: _BenchmarkSnapshot) -> None:
    switches = _requested_switches(snapshot.optimization_state)
    if name == "combined":
        if len(switches) < 2:
            raise TrainingOptimizationComparisonError(
                "declared combined variant must request at least two optimization "
                f"switches; observed {sorted(switches)}"
            )
        return
    if switches != {name}:
        raise TrainingOptimizationComparisonError(
            f"declared {name} variant must request exactly the {name} switch; "
            f"observed {sorted(switches)}"
        )


def _requested_switches(state: Mapping[str, object]) -> set[str]:
    precision = _state_object(state, "precision")
    attention = _state_object(state, "attention")
    compile_state = _state_object(state, "compile")
    activation = _state_object(state, "activation_checkpointing")
    switches: set[str] = set()
    dtype = precision.get("requested_dtype")
    if dtype != "float32":
        if dtype not in {"float16", "bfloat16"}:
            raise TrainingOptimizationComparisonError(
                f"unsupported requested precision dtype {dtype!r}"
            )
        switches.add("amp")
    backend = attention.get("requested_backend")
    if backend != "manual":
        if backend not in {"sdpa", "flash"}:
            raise TrainingOptimizationComparisonError(
                f"unsupported requested attention backend {backend!r}"
            )
        assert isinstance(backend, str)
        switches.add(backend)
    if compile_state.get("requested") is True:
        switches.add("compile")
    elif compile_state.get("requested") is not False:
        raise TrainingOptimizationComparisonError(
            "compile requested state must be a boolean"
        )
    if activation.get("requested") is True:
        switches.add("activation_checkpointing")
    elif activation.get("requested") is not False:
        raise TrainingOptimizationComparisonError(
            "activation checkpointing requested state must be a boolean"
        )
    return switches


def _validate_comparability(
    baseline: _BenchmarkSnapshot,
    variant: _BenchmarkSnapshot,
    *,
    declared: str,
) -> None:
    for field in _REQUIRED_EQUAL_IDENTITIES:
        if baseline.identities[field] != variant.identities[field]:
            raise TrainingOptimizationComparisonError(
                f"{declared} variant {field} identity does not match baseline"
            )
    if baseline.protocol != variant.protocol:
        raise TrainingOptimizationComparisonError(
            f"{declared} variant protocol does not match baseline"
        )
    for field in _REQUIRED_EQUAL_MEASUREMENTS:
        if baseline.measurements.get(field) != variant.measurements.get(field):
            label = "processed_tokens" if field == "processed_model_tokens" else field
            raise TrainingOptimizationComparisonError(
                f"{declared} variant {label} does not match baseline"
            )


def _identity_comparison(
    baseline: _BenchmarkSnapshot,
    variants: Mapping[str, _BenchmarkSnapshot],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in _IDENTITY_FIELDS:
        baseline_value = baseline.identities[field]
        variant_values = {
            name: snapshot.identities[field] for name, snapshot in variants.items()
        }
        result[field] = {
            "all_equal": all(
                value == baseline_value for value in variant_values.values()
            ),
            "baseline": baseline_value,
            "required_equal": field in _REQUIRED_EQUAL_IDENTITIES,
            "variants": variant_values,
        }
    return result


def _baseline_result(snapshot: _BenchmarkSnapshot) -> dict[str, object]:
    return {
        "measurements": _result_measurements(snapshot),
        "optimization_state": snapshot.optimization_state,
        "result_label": "float32/manual baseline",
        "startup": _startup(snapshot.optimization_state),
    }


def _variant_result(
    name: str,
    baseline: _BenchmarkSnapshot,
    variant: _BenchmarkSnapshot,
) -> dict[str, object]:
    fallback_reasons = _fallback_reasons(variant.optimization_state)
    measurements = _result_measurements(variant)
    return {
        "declared_optimization": name,
        "deltas": {
            "peak_allocated_bytes": _delta(
                baseline.measurements["peak_allocated_bytes"],
                variant.measurements["peak_allocated_bytes"],
                label="peak allocated bytes",
            ),
            "peak_allocated_mib": _delta(
                baseline.measurements["peak_allocated_mib"],
                variant.measurements["peak_allocated_mib"],
                label="peak allocated MiB",
            ),
            "tokens_per_second": _delta(
                baseline.measurements["tokens_per_second"],
                variant.measurements["tokens_per_second"],
                label="tokens per second",
            ),
        },
        "effective_state": _effective_state(variant.optimization_state),
        "fallback": {
            "occurred": bool(fallback_reasons),
            "reasons": fallback_reasons,
        },
        "measurements": measurements,
        "optimization_state": variant.optimization_state,
        "requested_state": _requested_state(variant.optimization_state),
        "result_label": _result_label(
            name, variant.optimization_state, fallback_reasons
        ),
        "startup": _startup(variant.optimization_state),
    }


def _result_measurements(snapshot: _BenchmarkSnapshot) -> dict[str, object]:
    measurements = snapshot.measurements
    return {
        "elapsed_seconds": measurements["elapsed_seconds"],
        "mfu": measurements["mfu"],
        "mfu_basis": measurements["mfu_basis"],
        "peak_allocated_bytes": measurements["peak_allocated_bytes"],
        "peak_allocated_mib": measurements["peak_allocated_mib"],
        "processed_model_tokens": measurements["processed_model_tokens"],
        "supervised_target_tokens": measurements["supervised_target_tokens"],
        "tokens_per_second": measurements["tokens_per_second"],
        "training_flops": measurements["training_flops"],
    }


def _requested_state(state: Mapping[str, object]) -> dict[str, object]:
    precision = _state_object(state, "precision")
    attention = _state_object(state, "attention")
    compile_state = _state_object(state, "compile")
    activation = _state_object(state, "activation_checkpointing")
    return {
        "activation_checkpointing": {
            "requested": activation["requested"],
        },
        "attention": {
            "requested_backend": attention["requested_backend"],
        },
        "compile": {
            field: compile_state[field]
            for field in ("backend", "dynamic", "fullgraph", "mode", "requested")
        },
        "precision": {
            "requested_dtype": precision["requested_dtype"],
        },
    }


def _effective_state(state: Mapping[str, object]) -> dict[str, object]:
    precision = _state_object(state, "precision")
    attention = _state_object(state, "attention")
    compile_state = _state_object(state, "compile")
    activation = _state_object(state, "activation_checkpointing")
    return {
        "activation_checkpointing": activation["effective"],
        "attention_backend": attention["effective_backend"],
        "compile": compile_state["effective"],
        "precision": {
            field: precision[field]
            for field in (
                "autocast_enabled",
                "device_type",
                "effective_dtype",
                "scaler_enabled",
            )
        },
    }


def _fallback_reasons(state: Mapping[str, object]) -> list[object]:
    precision = _state_object(state, "precision")
    attention = _state_object(state, "attention")
    compile_state = _state_object(state, "compile")
    activation = _state_object(state, "activation_checkpointing")
    reasons: list[object] = []
    attention_reason = attention.get("fallback_reason")
    if attention_reason is not None:
        reasons.append(attention_reason)
    elif attention.get("requested_backend") != attention.get("effective_backend"):
        reasons.append("attention_backend_not_effective")
    compile_reason = compile_state.get("fallback_reason")
    if compile_reason is not None:
        reasons.append(compile_reason)
    elif compile_state.get("requested") != compile_state.get("effective"):
        reasons.append("compile_not_effective")
    if activation.get("requested") != activation.get("effective"):
        reasons.append("activation_checkpointing_not_effective")
    if precision.get("requested_dtype") != precision.get("effective_dtype"):
        reasons.append("precision_not_effective")
    return list(dict.fromkeys(reasons))


def _result_label(
    name: str,
    state: Mapping[str, object],
    fallback_reasons: list[object],
) -> str:
    precision = _state_object(state, "precision")
    attention = _state_object(state, "attention")
    compile_state = _state_object(state, "compile")
    activation = _state_object(state, "activation_checkpointing")
    requested_backend = attention["requested_backend"]
    effective_backend = attention["effective_backend"]
    if name in {"sdpa", "flash"}:
        if requested_backend != effective_backend or fallback_reasons:
            return f"{requested_backend}→{effective_backend} fallback"
        return str(effective_backend)
    if name == "amp":
        effective_dtype = precision["effective_dtype"]
        if fallback_reasons:
            return f"AMP→{effective_dtype} fallback"
        return f"{effective_dtype} AMP"
    if name == "compile":
        return (
            "torch.compile"
            if compile_state["effective"] is True and not fallback_reasons
            else "torch.compile→eager fallback"
        )
    if name == "activation_checkpointing":
        return (
            "activation checkpointing"
            if activation["effective"] is True and not fallback_reasons
            else "activation checkpointing→disabled fallback"
        )
    effective_parts = [str(precision["effective_dtype"]), str(effective_backend)]
    effective_parts.append(
        "compiled" if compile_state["effective"] is True else "eager"
    )
    effective_parts.append(
        "checkpointed" if activation["effective"] is True else "uncheckpointed"
    )
    suffix = " fallback" if fallback_reasons else ""
    return "combined: " + "/".join(effective_parts) + suffix


def _startup(state: Mapping[str, object]) -> dict[str, object]:
    compile_state = _state_object(state, "compile")
    return {
        "compile_duration_seconds": compile_state["compile_duration_seconds"],
        "excluded_from_timed_measurements": True,
    }


def _delta(baseline: object, variant: object, *, label: str) -> dict[str, object]:
    if baseline is None or variant is None:
        return {
            "absolute": None,
            "relative_fraction": None,
            "relative_percent": None,
            "unavailable_reason": f"{label} was unavailable in one or both reports",
        }
    baseline_number = _require_finite_number(baseline, label=f"baseline {label}")
    variant_number = _require_finite_number(variant, label=f"variant {label}")
    absolute = variant_number - baseline_number
    if isinstance(baseline, int) and isinstance(variant, int):
        absolute_value: int | float = int(absolute)
    else:
        absolute_value = absolute
    if baseline_number == 0:
        return {
            "absolute": absolute_value,
            "relative_fraction": None,
            "relative_percent": None,
            "unavailable_reason": f"baseline {label} is zero",
        }
    relative = absolute / baseline_number
    return {
        "absolute": absolute_value,
        "relative_fraction": relative,
        "relative_percent": relative * 100,
    }


def _validate_measurements(measurements: Mapping[str, object], *, label: str) -> None:
    for field in (
        "elapsed_seconds",
        "processed_model_tokens",
        "supervised_target_tokens",
        "tokens_per_second",
        "training_flops",
    ):
        if field not in measurements:
            raise TrainingOptimizationComparisonError(
                f"{label} measurements are missing {field}"
            )
        numeric = _require_finite_number(
            measurements[field],
            label=f"{label} {field}",
        )
        if numeric <= 0:
            raise TrainingOptimizationComparisonError(
                f"{label} {field} must be greater than zero"
            )
    for field in ("peak_allocated_bytes", "peak_allocated_mib"):
        if field not in measurements:
            raise TrainingOptimizationComparisonError(
                f"{label} measurements are missing {field}"
            )
        value = measurements[field]
        if (
            value is not None
            and _require_finite_number(
                value,
                label=f"{label} {field}",
            )
            < 0
        ):
            raise TrainingOptimizationComparisonError(
                f"{label} {field} must be non-negative"
            )
    for field in ("mfu", "mfu_basis"):
        if field not in measurements:
            raise TrainingOptimizationComparisonError(
                f"{label} measurements are missing {field}"
            )
    if measurements["mfu"] is not None:
        _require_finite_number(measurements["mfu"], label=f"{label} mfu")
        basis = _require_object(
            measurements["mfu_basis"],
            label=f"{label} mfu_basis",
        )
        if (
            _require_finite_number(
                basis.get("flops_per_second"),
                label=f"{label} mfu_basis flops_per_second",
            )
            <= 0
        ):
            raise TrainingOptimizationComparisonError(
                f"{label} mfu_basis flops_per_second must be greater than zero"
            )
        _require_string(
            basis.get("description"),
            label=f"{label} mfu_basis description",
        )
    elif measurements["mfu_basis"] is not None:
        raise TrainingOptimizationComparisonError(
            f"{label} mfu_basis must be null when mfu is unavailable"
        )


def _validate_optimization_state(
    state: Mapping[str, object],
    *,
    label: str,
) -> None:
    precision = _state_object(state, "precision")
    for field in ("autocast_enabled", "scaler_enabled"):
        _require_boolean(
            precision.get(field),
            label=f"{label} precision {field}",
        )
    _require_string(
        precision.get("device_type"),
        label=f"{label} precision device_type",
    )
    for field in ("requested_dtype", "effective_dtype"):
        dtype = precision.get(field)
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise TrainingOptimizationComparisonError(
                f"{label} precision {field} is unsupported: {dtype!r}"
            )
    expected_autocast = precision["effective_dtype"] != "float32"
    if precision["autocast_enabled"] is not expected_autocast:
        raise TrainingOptimizationComparisonError(
            f"{label} precision autocast state disagrees with effective dtype"
        )
    expected_scaler = precision["effective_dtype"] == "float16"
    if precision["scaler_enabled"] is not expected_scaler:
        raise TrainingOptimizationComparisonError(
            f"{label} precision scaler state disagrees with effective dtype"
        )

    attention = _state_object(state, "attention")
    for field in ("requested_backend", "effective_backend"):
        backend = attention.get(field)
        if backend not in {"manual", "sdpa", "flash"}:
            raise TrainingOptimizationComparisonError(
                f"{label} attention {field} is unsupported: {backend!r}"
            )
    _require_optional_string(
        attention.get("fallback_reason"),
        label=f"{label} attention fallback_reason",
    )

    compile_state = _state_object(state, "compile")
    for field in ("requested", "effective", "dynamic", "fullgraph"):
        _require_boolean(
            compile_state.get(field),
            label=f"{label} compile {field}",
        )
    for field in ("backend", "mode"):
        _require_string(
            compile_state.get(field),
            label=f"{label} compile {field}",
        )
    compile_duration = _require_finite_number(
        compile_state.get("compile_duration_seconds"),
        label=f"{label} compile duration",
    )
    if compile_duration < 0:
        raise TrainingOptimizationComparisonError(
            f"{label} compile duration must be non-negative"
        )
    _require_optional_string(
        compile_state.get("fallback_reason"),
        label=f"{label} compile fallback_reason",
    )

    activation = _state_object(state, "activation_checkpointing")
    for field in ("requested", "effective", "block_boundary", "use_reentrant"):
        _require_boolean(
            activation.get(field),
            label=f"{label} activation checkpointing {field}",
        )


def _validate_protocol(protocol: Mapping[str, object], *, label: str) -> None:
    for field in ("id", "timing_source"):
        _require_string(protocol.get(field), label=f"{label} protocol {field}")
    for field in ("warmup_steps", "timed_steps"):
        value = protocol.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TrainingOptimizationComparisonError(
                f"{label} protocol {field} must be a positive integer"
            )
    for field in ("included_work", "excluded_work"):
        value = protocol.get(field)
        if not isinstance(value, list) or not value:
            raise TrainingOptimizationComparisonError(
                f"{label} protocol {field} must be a non-empty list"
            )


def _result_row(name: str, result: Mapping[str, object]) -> str:
    measurements = _require_object(
        result.get("measurements"), label=f"{name} measurements"
    )
    startup = _require_object(result.get("startup"), label=f"{name} startup")
    if name == "baseline":
        throughput_delta = "—"
        memory_delta = "—"
    else:
        deltas = _require_object(result.get("deltas"), label=f"{name} deltas")
        throughput_delta = _format_delta(
            _require_object(deltas["tokens_per_second"], label="throughput delta")
        )
        memory_delta = _format_delta(
            _require_object(deltas["peak_allocated_mib"], label="memory delta")
        )
    return (
        f"| `{name}` | {_markdown_cell(result['result_label'])} | "
        f"{_format_number(measurements['tokens_per_second'])} | {throughput_delta} | "
        f"{_format_number(measurements['peak_allocated_mib'])} | {memory_delta} | "
        f"{_format_number(startup['compile_duration_seconds'])} |"
    )


def _format_delta(delta: Mapping[str, object]) -> str:
    absolute = delta.get("absolute")
    percent = delta.get("relative_percent")
    if absolute is None or percent is None:
        return "unavailable"
    absolute_number = _require_finite_number(absolute, label="absolute delta")
    percent_number = _require_finite_number(percent, label="relative delta")
    return f"{absolute_number:+.3f} ({percent_number:+.2f}%)"


def _format_number(value: object) -> str:
    if value is None:
        return "unavailable"
    return f"{_require_finite_number(value, label='report value'):.3f}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _yes_no(value: object) -> str:
    return "yes" if value is True else "no"


def _state_object(state: Mapping[str, object], field: str) -> dict[str, object]:
    return _require_object(state.get(field), label=f"optimization_state.{field}")


def _require_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TrainingOptimizationComparisonError(f"{label} must be an object")
    return value if isinstance(value, dict) else dict(value)


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingOptimizationComparisonError(f"{label} must be a non-empty string")
    return value


def _require_optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, label=label)


def _require_boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingOptimizationComparisonError(f"{label} must be a boolean")
    return value


def _require_finite_number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingOptimizationComparisonError(f"{label} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TrainingOptimizationComparisonError(f"{label} must be finite")
    return numeric


def _reject_json_constant(token: str, path: Path) -> None:
    raise TrainingOptimizationComparisonError(
        f"benchmark {path} contains non-finite JSON value {token}"
    )


__all__ = [
    "TRAINING_OPTIMIZATION_COMPARISON_FORMAT",
    "TRAINING_OPTIMIZATION_COMPARISON_FORMAT_VERSION",
    "TRAINING_OPTIMIZATION_VARIANTS",
    "TrainingOptimizationComparisonArtifacts",
    "TrainingOptimizationComparisonError",
    "compare_training_benchmarks",
    "render_training_optimization_comparison",
]
