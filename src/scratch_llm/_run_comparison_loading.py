"""Strict offline loading for local training-run comparison artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
from typing import Any, Final

from scratch_llm._run_comparison_model import (
    STEP_METRICS,
    RunComparisonError,
    RunSnapshot,
)
from scratch_llm._validation import (
    JsonValueValidator,
    require_finite_non_negative_real,
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.bpb import BaseValidationResult
from scratch_llm.config import ProjectConfig, load_config
from scratch_llm.full_document_bpb import FULL_DOCUMENT_PROTOCOL_ID
from scratch_llm.nanochat_bpb import NANOCHAT_COMPAT_PROTOCOL_ID
from scratch_llm.utils import load_json


_BASE_EVALUATION_FORMAT = "scratch_llm_base_evaluation"
_BASE_EVALUATION_VERSION = 2
_SUMMARY_KEYS = frozenset(
    {"latest_metrics", "latest_step", "run", "schema_version", "status"}
)
_SUMMARY_RUN_KEYS = frozenset({"name", "output_dir", "stage"})
_CONFIG_RECORD_KEYS = frozenset({"config", "record_type"})
_METRICS_RECORD_REQUIRED_KEYS = frozenset({"metrics", "record_type"})
_METRICS_RECORD_OPTIONAL_KEYS = frozenset({"event_id", "step"})
_ARTIFACT_RECORD_REQUIRED_KEYS = frozenset({"name", "path", "record_type", "type"})
_ARTIFACT_RECORD_OPTIONAL_KEYS = frozenset({"event_id"})
_BASE_EVALUATION_REQUIRED_KEYS = frozenset(
    {
        "bounded",
        "completed_modes",
        "format",
        "format_version",
        "identities",
        "max_per_task",
        "requested_modes",
        "results",
        "run_kind",
        "status",
    }
)
_BASE_EVALUATION_OPTIONAL_KEYS = frozenset({"core", "samples"})
_BASE_IDENTITY_KEYS = frozenset(
    {"checkpoint", "config", "tokenizer", "validation_manifest"}
)
_BASE_CHECKPOINT_IDENTITY_KEYS = frozenset({"identity", "step"})
_BASE_RESULT_KEYS = frozenset(
    {
        "bpb",
        "checkpoint_identity",
        "counted_target_bytes",
        "counted_target_tokens",
        "processed_model_tokens",
        "protocol_id",
        "protocol_version",
        "reference_commit",
        "reference_config",
        "source_byte_retention",
        "source_bytes",
        "source_documents",
        "source_token_retention",
        "source_tokens",
        "tokenizer_identity",
        "total_nats",
        "unique_source_bytes",
        "unique_source_tokens",
        "validation_manifest_identity",
    }
)
_JSON_VALUES: Final = JsonValueValidator(RunComparisonError)


def load_run_snapshots(
    run_dirs: Sequence[str | os.PathLike[str]],
) -> tuple[RunSnapshot, ...]:
    """Load two or more unique runs and return them in stable name order."""

    if isinstance(run_dirs, (str, bytes)) or not isinstance(run_dirs, Sequence):
        raise TypeError("run_dirs must be a sequence of local directories")
    if len(run_dirs) < 2:
        raise RunComparisonError("run comparison requires at least two run directories")
    paths = tuple(Path(path).resolve() for path in run_dirs)
    if len(set(paths)) != len(paths):
        raise RunComparisonError("run comparison directories must be unique")
    snapshots = tuple(_load_snapshot(path) for path in paths)
    names = [snapshot.name for snapshot in snapshots]
    if len(set(names)) != len(names):
        raise RunComparisonError("compared runs must have unique configured names")
    return tuple(sorted(snapshots, key=lambda snapshot: snapshot.name))


def _load_snapshot(path: Path) -> RunSnapshot:
    if not path.is_dir():
        raise RunComparisonError(f"run directory does not exist: {path}")
    try:
        config = load_config(path / "config.yaml")
        summary = _load_summary(path / "metrics" / "summary.json", config=config)
        training_metrics, observed_latest_step = _load_training_metrics(
            path / "metrics" / "metrics.jsonl",
            config=config,
        )
        if summary["latest_step"] != observed_latest_step:
            raise RunComparisonError(
                f"run summary latest_step does not match metrics JSONL in {path}"
            )
        base_evaluation = _load_base_evaluation(path / "metrics" / "base_eval.json")
    except RunComparisonError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise RunComparisonError(f"invalid run {path}: {error}") from error
    return RunSnapshot(
        path=path,
        name=config.run.name,
        config=config,
        summary=summary,
        training_metrics=training_metrics,
        base_evaluation=base_evaluation,
    )


def _load_summary(path: Path, *, config: ProjectConfig) -> dict[str, Any]:
    try:
        raw = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunComparisonError(f"invalid run summary {path}: {error}") from error
    value = _JSON_VALUES.require_object(
        raw,
        label=f"run summary {path}",
        expected_keys=_SUMMARY_KEYS,
    )
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise RunComparisonError(f"unsupported run summary version in {path}")
    run = _JSON_VALUES.require_object(
        value["run"],
        label=f"run summary identity {path}",
        expected_keys=_SUMMARY_RUN_KEYS,
    )
    if run["name"] != config.run.name:
        raise RunComparisonError(f"run summary name does not match config in {path}")
    for field in _SUMMARY_RUN_KEYS:
        try:
            require_non_empty_string(run[field], name=f"run summary {field}")
        except (TypeError, ValueError) as error:
            raise RunComparisonError(
                f"invalid run summary identity in {path}"
            ) from error
    if Path(str(run["output_dir"])).resolve() != path.parent.parent.resolve():
        raise RunComparisonError(
            f"run summary output_dir does not match its run directory in {path}"
        )
    if value["status"] not in ("running", "completed", "failed"):
        raise RunComparisonError(f"invalid run summary status in {path}")
    latest_step = value["latest_step"]
    if latest_step is not None:
        try:
            require_non_negative_integer(latest_step, name="run summary latest_step")
        except (TypeError, ValueError) as error:
            raise RunComparisonError(
                f"invalid run summary latest_step in {path}"
            ) from error
    latest_metrics = _JSON_VALUES.require_object(
        value["latest_metrics"],
        label=f"run summary latest_metrics {path}",
    )
    _validate_scalar_metrics(latest_metrics, label=str(path))
    return value


def _load_training_metrics(
    path: Path,
    *,
    config: ProjectConfig,
) -> tuple[dict[int, dict[str, Any]], int | None]:
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise RunComparisonError(
            f"could not read training metrics {path}: {error}"
        ) from error
    if contents and not contents.endswith(b"\n"):
        raise RunComparisonError(
            f"{path} must contain complete newline-delimited records"
        )
    try:
        lines = contents.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RunComparisonError(f"{path} is not valid UTF-8: {error}") from error

    prior_step: int | None = None
    training: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        label = f"{path}:{line_number}"
        record = _json_object(line, label=label)
        record_type = record.get("record_type")
        if record_type == "config":
            if line_number != 1 or frozenset(record) != _CONFIG_RECORD_KEYS:
                raise RunComparisonError(
                    f"{path} must contain exactly one resolved configuration first"
                )
            if record["config"] != config.to_dict():
                raise RunComparisonError(
                    f"resolved configuration in {path} does not match config.yaml"
                )
            continue
        if line_number == 1:
            raise RunComparisonError(
                f"{path} must contain exactly one resolved configuration first"
            )
        if record_type == "artifact":
            _validate_artifact_record(record, label=label)
            continue
        if record_type != "metrics":
            raise RunComparisonError(f"unknown tracking record type at {label}")
        if not _record_keys_match(
            frozenset(record),
            required=_METRICS_RECORD_REQUIRED_KEYS,
            optional=_METRICS_RECORD_OPTIONAL_KEYS,
        ):
            raise RunComparisonError(f"invalid metrics record schema at {label}")
        _validate_event_id(record, label=label)
        metrics = _JSON_VALUES.require_object(
            record["metrics"],
            label=f"metrics at {label}",
        )
        _validate_scalar_metrics(metrics, label=label)
        step = record.get("step")
        if step is None:
            continue
        try:
            step = require_non_negative_integer(step, name="metrics step")
        except (TypeError, ValueError) as error:
            raise RunComparisonError(f"invalid metrics step at {label}") from error
        if prior_step is not None and step < prior_step:
            raise RunComparisonError(
                f"metrics steps move backwards at {label}: {prior_step} to {step}"
            )
        prior_step = step
        selected = {name: metrics[name] for name in STEP_METRICS if name in metrics}
        if not selected:
            continue
        existing = training.setdefault(
            step,
            {name: None for name in STEP_METRICS},
        )
        for name, metric in selected.items():
            try:
                normalized_metric = require_finite_non_negative_real(
                    metric,
                    name=f"training metric {name!r}",
                )
            except (TypeError, ValueError) as error:
                raise RunComparisonError(
                    f"training metric {name!r} must be a finite non-negative "
                    f"number at {label}"
                ) from error
            prior_metric = existing[name]
            if prior_metric is not None and prior_metric != normalized_metric:
                raise RunComparisonError(
                    f"conflicting training metric {name!r} for step {step} in {path}"
                )
            existing[name] = normalized_metric
    if not lines:
        raise RunComparisonError(
            f"{path} must contain exactly one resolved configuration first"
        )
    return training, prior_step


def _load_base_evaluation(path: Path) -> dict[str, Any] | None:
    try:
        raw = load_json(path)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunComparisonError(f"invalid base evaluation {path}: {error}") from error
    value = _JSON_VALUES.require_object(raw, label=f"base evaluation {path}")
    if not _record_keys_match(
        frozenset(value),
        required=_BASE_EVALUATION_REQUIRED_KEYS,
        optional=_BASE_EVALUATION_OPTIONAL_KEYS,
    ):
        raise RunComparisonError(f"invalid base evaluation schema in {path}")
    if (
        value.get("format") != _BASE_EVALUATION_FORMAT
        or value.get("format_version") != _BASE_EVALUATION_VERSION
        or isinstance(value.get("format_version"), bool)
    ):
        raise RunComparisonError(f"unsupported base evaluation schema in {path}")
    if value.get("status") != "completed":
        raise RunComparisonError(f"base evaluation is not completed in {path}")
    requested = _JSON_VALUES.require_list(
        value.get("requested_modes"),
        label=f"base evaluation requested_modes {path}",
        non_empty=True,
    )
    completed = value.get("completed_modes")
    if (
        any(mode not in ("bpb", "core", "sample") for mode in requested)
        or len(set(requested)) != len(requested)
        or completed != requested
    ):
        raise RunComparisonError(f"base evaluation modes are incomplete in {path}")
    for mode, field in (("core", "core"), ("sample", "samples")):
        if (mode in requested) != (field in value):
            raise RunComparisonError(
                f"base evaluation {field} payload does not match modes in {path}"
            )
        if field in value:
            _JSON_VALUES.require_object(
                value[field],
                label=f"base evaluation {field} payload {path}",
            )
    _validate_base_evaluation_context(value, requested=requested, path=path)
    results = _JSON_VALUES.require_object(
        value.get("results"),
        label=f"base evaluation results {path}",
    )
    expected_protocols = (
        {NANOCHAT_COMPAT_PROTOCOL_ID, FULL_DOCUMENT_PROTOCOL_ID}
        if "bpb" in requested
        else set()
    )
    if set(results) != expected_protocols:
        raise RunComparisonError(
            f"base evaluation BPB results do not match requested modes in {path}"
        )
    parsed_results = {
        protocol_id: _parse_base_result(
            raw_result,
            protocol_id=protocol_id,
            path=path,
        )
        for protocol_id, raw_result in results.items()
    }
    identities = value["identities"]
    assert isinstance(identities, dict)
    checkpoint = identities["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint_identity = checkpoint["identity"]
    for result in parsed_results.values():
        if (
            result["checkpoint_identity"] != checkpoint_identity
            or result["tokenizer_identity"] != identities["tokenizer"]
            or result["validation_manifest_identity"]
            != identities["validation_manifest"]
        ):
            raise RunComparisonError(
                f"base evaluation result identities do not match report in {path}"
            )
    normalized = dict(value)
    normalized["results"] = parsed_results
    return normalized


def _validate_base_evaluation_context(
    value: Mapping[str, Any],
    *,
    requested: list[object],
    path: Path,
) -> None:
    bounded = value["bounded"]
    run_kind = value["run_kind"]
    max_per_task = value["max_per_task"]
    if not isinstance(bounded, bool) or run_kind not in ("bounded", "full"):
        raise RunComparisonError(f"invalid base evaluation run kind in {path}")
    if bounded != (run_kind == "bounded"):
        raise RunComparisonError(f"inconsistent base evaluation run kind in {path}")
    if bounded:
        try:
            require_positive_integer(
                max_per_task,
                name="base evaluation max_per_task",
            )
        except (TypeError, ValueError) as error:
            raise RunComparisonError(
                f"invalid base evaluation max_per_task in {path}"
            ) from error
    elif max_per_task is not None:
        raise RunComparisonError(
            f"full base evaluation cannot set max_per_task in {path}"
        )

    identities = _JSON_VALUES.require_object(
        value["identities"],
        label=f"base evaluation identities {path}",
        expected_keys=_BASE_IDENTITY_KEYS,
    )
    checkpoint = _JSON_VALUES.require_object(
        identities["checkpoint"],
        label=f"base evaluation checkpoint identity {path}",
        expected_keys=_BASE_CHECKPOINT_IDENTITY_KEYS,
    )
    try:
        require_non_empty_string(
            checkpoint["identity"],
            name="base evaluation checkpoint identity",
        )
        require_non_negative_integer(
            checkpoint["step"],
            name="base evaluation checkpoint step",
        )
        require_non_empty_string(
            identities["config"],
            name="base evaluation config identity",
        )
        require_non_empty_string(
            identities["tokenizer"],
            name="base evaluation tokenizer identity",
        )
    except (TypeError, ValueError) as error:
        raise RunComparisonError(
            f"invalid base evaluation identities in {path}"
        ) from error
    manifest_identity = identities["validation_manifest"]
    if "bpb" in requested:
        try:
            require_non_empty_string(
                manifest_identity,
                name="base evaluation validation manifest identity",
            )
        except (TypeError, ValueError) as error:
            raise RunComparisonError(
                f"invalid validation manifest identity in {path}"
            ) from error
    elif manifest_identity is not None:
        raise RunComparisonError(
            f"non-BPB evaluation cannot set a validation manifest in {path}"
        )


def _parse_base_result(
    value: object,
    *,
    protocol_id: str,
    path: Path,
) -> dict[str, Any]:
    result_payload = _JSON_VALUES.require_object(
        value,
        label=f"{protocol_id!r} result {path}",
        expected_keys=_BASE_RESULT_KEYS,
    )
    try:
        result = BaseValidationResult(**result_payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise RunComparisonError(
            f"invalid {protocol_id!r} result in {path}: {error}"
        ) from error
    if result.protocol_id != protocol_id:
        raise RunComparisonError(
            f"result key {protocol_id!r} does not match its protocol id in {path}"
        )
    return result.to_dict()


def _json_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_JSON_VALUES.duplicate_object_hook(label=label),
        )
    except json.JSONDecodeError as error:
        raise RunComparisonError(f"invalid JSON at {label}: {error.msg}") from error
    return _JSON_VALUES.require_object(parsed, label=f"JSON record at {label}")


def _record_keys_match(
    keys: frozenset[str],
    *,
    required: frozenset[str],
    optional: frozenset[str],
) -> bool:
    return required <= keys <= required | optional


def _validate_event_id(record: Mapping[str, Any], *, label: str) -> None:
    if "event_id" not in record:
        return
    try:
        require_non_empty_string(record["event_id"], name="event_id")
    except (TypeError, ValueError) as error:
        raise RunComparisonError(
            f"event_id must be a non-empty string at {label}"
        ) from error


def _validate_artifact_record(record: Mapping[str, Any], *, label: str) -> None:
    if not _record_keys_match(
        frozenset(record),
        required=_ARTIFACT_RECORD_REQUIRED_KEYS,
        optional=_ARTIFACT_RECORD_OPTIONAL_KEYS,
    ):
        raise RunComparisonError(f"invalid artifact record schema at {label}")
    _validate_event_id(record, label=label)
    for field in ("name", "path", "type"):
        try:
            require_non_empty_string(record[field], name=f"artifact {field}")
        except (TypeError, ValueError) as error:
            raise RunComparisonError(
                f"artifact {field} must be a non-empty string at {label}"
            ) from error


def _validate_scalar_metrics(value: Mapping[str, object], *, label: str) -> None:
    for key, metric in value.items():
        if not (metric is None or isinstance(metric, (bool, int, float, str))):
            raise RunComparisonError(f"metric {key!r} is not scalar in {label}")
        if isinstance(metric, float) and not math.isfinite(metric):
            raise RunComparisonError(f"metric {key!r} is not finite in {label}")


__all__ = ["load_run_snapshots"]
