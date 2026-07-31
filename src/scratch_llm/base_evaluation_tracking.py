"""Tracking adapters for completed base BPB and fixed-sampling results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from scratch_llm.base_sampling import (
    BaseSamplesResult,
    write_base_samples_markdown,
)
from scratch_llm.best_checkpoint import (
    BEST_CHECKPOINT_RANKING_PROTOCOL_ID,
    PeriodicValidationResult,
    ValidationCheckpointState,
)
from scratch_llm.bpb import BaseValidationResult
from scratch_llm.full_document_bpb import (
    FULL_DOCUMENT_EVAL_METRIC,
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_TRAIN_METRIC,
)
from scratch_llm.nanochat_bpb import (
    NANOCHAT_COMPAT_EVAL_METRIC,
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_TRAIN_METRIC,
)
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.utils import save_json


BASE_EVALUATION_REPORT_FORMAT: Final = "scratch_llm_base_evaluation"
BASE_EVALUATION_REPORT_FORMAT_VERSION: Final = 1
BASE_EVALUATION_ARTIFACT_NAME: Final = "base_eval"
BASE_SAMPLES_ARTIFACT_NAME: Final = "base_samples"
BASE_EVALUATION_ARTIFACT_TYPE: Final = "evaluation"
NANOCHAT_MINIMUM_TRAIN_METRIC: Final = "min_val_bpb"
FULL_DOCUMENT_MINIMUM_TRAIN_METRIC: Final = "min_val_bpb_full_documents"
NANOCHAT_SOURCE_BYTE_RETENTION_METRIC: Final = (
    "eval/val_bpb_nanochat_source_byte_retention"
)
FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC: Final = (
    "eval/val_bpb_full_document_source_byte_retention"
)
BASE_SAMPLE_THROUGHPUT_METRIC: Final = "eval/sample_tokens_per_sec"
_BASE_EVALUATION_RELATIVE_PATH = Path("metrics/base_eval.json")
_BASE_SAMPLES_RELATIVE_PATH = Path("metrics/base_samples.md")


@dataclass(frozen=True)
class TrackedBaseEvaluation:
    """Completed standalone BPB metrics and canonical report path."""

    metrics: Mapping[str, float]
    report_path: Path


@dataclass(frozen=True)
class TrackedBaseSamples:
    """Completed fixed-sample throughput and canonical Markdown path."""

    metrics: Mapping[str, float]
    markdown_path: Path


def track_periodic_base_validation(
    validation: PeriodicValidationResult,
    state: ValidationCheckpointState,
    *,
    tracker: Tracker,
    step: int,
) -> dict[str, float]:
    """Log one accepted dual-protocol validation on its optimizer step."""

    compatibility, full_document = _complete_protocol_results(validation)
    if not isinstance(state, ValidationCheckpointState):
        raise TypeError(
            f"state must be a ValidationCheckpointState, got {type(state).__name__}"
        )
    _require_tracker(tracker)
    step = _non_negative_integer(step, name="step")
    if state.ranking_protocol_id != BEST_CHECKPOINT_RANKING_PROTOCOL_ID:
        raise ValueError(
            "validation state ranking protocol does not match "
            f"{BEST_CHECKPOINT_RANKING_PROTOCOL_ID!r}"
        )
    if state.validation_step != step:
        raise ValueError(
            f"validation state validation step {state.validation_step} "
            f"does not match tracking step {step}"
        )
    if state.validation_identity != validation.validation_identity:
        raise ValueError("validation state identity does not match the result pair")
    if state.current_compatibility_bpb != compatibility.bpb:
        raise ValueError(
            "validation state current compatibility BPB does not match the result"
        )
    if state.current_full_document_bpb != full_document.bpb:
        raise ValueError(
            "validation state current full-document BPB does not match the result"
        )
    minimum_full_document_bpb = state.minimum_full_document_bpb
    if minimum_full_document_bpb is None:
        raise ValueError(
            "complete periodic validation requires a full-document minimum"
        )
    metrics = {
        NANOCHAT_COMPAT_TRAIN_METRIC: compatibility.bpb,
        NANOCHAT_MINIMUM_TRAIN_METRIC: state.minimum_compatibility_bpb,
        FULL_DOCUMENT_TRAIN_METRIC: full_document.bpb,
        FULL_DOCUMENT_MINIMUM_TRAIN_METRIC: minimum_full_document_bpb,
    }
    tracker.log(metrics, step=step)
    return metrics


def report_standalone_base_evaluation(
    validation: PeriodicValidationResult,
    *,
    tracker: Tracker,
    run_dir: str | Path,
) -> TrackedBaseEvaluation:
    """Atomically write and track one completed standalone dual-BPB result."""

    compatibility, full_document = _complete_protocol_results(validation)
    _require_tracker(tracker)
    resolved_run_dir = _run_directory(run_dir)
    payload = _base_evaluation_payload(
        compatibility,
        full_document,
    )
    report_path = save_json(
        payload,
        resolved_run_dir / _BASE_EVALUATION_RELATIVE_PATH,
    )
    metrics = {
        NANOCHAT_COMPAT_EVAL_METRIC: compatibility.bpb,
        FULL_DOCUMENT_EVAL_METRIC: full_document.bpb,
        NANOCHAT_SOURCE_BYTE_RETENTION_METRIC: (compatibility.source_byte_retention),
        FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC: (
            full_document.source_byte_retention
        ),
    }
    event_prefix = f"base-evaluation:{_payload_identity(payload)}"
    _log_metrics_once(
        tracker,
        metrics,
        event_id=f"{event_prefix}:metrics",
    )
    _log_artifact_once(
        tracker,
        _BASE_EVALUATION_RELATIVE_PATH.as_posix(),
        name=BASE_EVALUATION_ARTIFACT_NAME,
        event_id=f"{event_prefix}:artifact:base_eval.json",
    )
    return TrackedBaseEvaluation(
        metrics=MappingProxyType(metrics),
        report_path=report_path,
    )


def report_base_samples(
    samples: BaseSamplesResult,
    *,
    tracker: Tracker,
    run_dir: str | Path,
) -> TrackedBaseSamples:
    """Atomically write and track only the frozen public base-sample suite."""

    if not isinstance(samples, BaseSamplesResult):
        raise TypeError(
            f"samples must be a BaseSamplesResult, got {type(samples).__name__}"
        )
    _require_tracker(tracker)
    resolved_run_dir = _run_directory(run_dir)
    markdown_path = write_base_samples_markdown(
        samples,
        resolved_run_dir / "metrics",
    )
    sampled_tokens = sum(sample.sampled_token_count for sample in samples.samples)
    elapsed_seconds = sum(sample.elapsed_seconds for sample in samples.samples)
    metrics = {
        BASE_SAMPLE_THROUGHPUT_METRIC: sampled_tokens / elapsed_seconds,
    }
    event_prefix = f"base-samples:{_payload_identity(samples.to_dict())}"
    _log_metrics_once(
        tracker,
        metrics,
        event_id=f"{event_prefix}:metrics",
    )
    _log_artifact_once(
        tracker,
        _BASE_SAMPLES_RELATIVE_PATH.as_posix(),
        name=BASE_SAMPLES_ARTIFACT_NAME,
        event_id=f"{event_prefix}:artifact:base_samples.md",
    )
    return TrackedBaseSamples(
        metrics=MappingProxyType(metrics),
        markdown_path=markdown_path,
    )


def _base_evaluation_payload(
    compatibility: BaseValidationResult,
    full_document: BaseValidationResult,
) -> dict[str, object]:
    return {
        "format": BASE_EVALUATION_REPORT_FORMAT,
        "format_version": BASE_EVALUATION_REPORT_FORMAT_VERSION,
        "results": {
            NANOCHAT_COMPAT_PROTOCOL_ID: compatibility.to_dict(),
            FULL_DOCUMENT_PROTOCOL_ID: full_document.to_dict(),
        },
    }


def _complete_protocol_results(
    validation: PeriodicValidationResult,
) -> tuple[BaseValidationResult, BaseValidationResult]:
    if not isinstance(validation, PeriodicValidationResult):
        raise TypeError(
            "validation must be a PeriodicValidationResult, "
            f"got {type(validation).__name__}"
        )
    full_document = validation.full_document
    if full_document is None:
        raise ValueError(
            "standalone and tracked validation require both "
            f"{NANOCHAT_COMPAT_PROTOCOL_ID!r} and {FULL_DOCUMENT_PROTOCOL_ID!r}"
        )
    return validation.compatibility, full_document


def _run_directory(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"run_dir must be a string or Path, got {type(value).__name__}")
    path = Path(value)
    if not str(path):
        raise ValueError("run_dir must not be empty")
    return path


def _require_tracker(tracker: Tracker) -> None:
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")


def _log_metrics_once(
    tracker: Tracker,
    metrics: dict[str, Any],
    *,
    event_id: str,
) -> None:
    if isinstance(tracker, RunTracker):
        tracker.log_once(metrics, event_id=event_id)
    else:
        tracker.log(metrics)


def _log_artifact_once(
    tracker: Tracker,
    path: str,
    *,
    name: str,
    event_id: str,
) -> None:
    if isinstance(tracker, RunTracker):
        tracker.log_artifact_once(
            path,
            name,
            BASE_EVALUATION_ARTIFACT_TYPE,
            event_id=event_id,
        )
    else:
        tracker.log_artifact(
            path,
            name,
            BASE_EVALUATION_ARTIFACT_TYPE,
        )


def _payload_identity(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _non_negative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "BASE_EVALUATION_ARTIFACT_NAME",
    "BASE_EVALUATION_ARTIFACT_TYPE",
    "BASE_EVALUATION_REPORT_FORMAT",
    "BASE_EVALUATION_REPORT_FORMAT_VERSION",
    "BASE_SAMPLE_THROUGHPUT_METRIC",
    "BASE_SAMPLES_ARTIFACT_NAME",
    "FULL_DOCUMENT_MINIMUM_TRAIN_METRIC",
    "FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC",
    "NANOCHAT_MINIMUM_TRAIN_METRIC",
    "NANOCHAT_SOURCE_BYTE_RETENTION_METRIC",
    "TrackedBaseEvaluation",
    "TrackedBaseSamples",
    "report_base_samples",
    "report_standalone_base_evaluation",
    "track_periodic_base_validation",
]
