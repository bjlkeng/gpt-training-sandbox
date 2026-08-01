"""Tracking adapters for completed base BPB, sampling, and CORE results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from scratch_llm._validation import require_non_negative_integer
from scratch_llm.evaluation.base import (
    BaseEvaluationContext,
    CompletedBaseEvaluation,
)
from scratch_llm.evaluation.sampling import (
    BaseSamplesResult,
    write_base_samples_markdown,
)
from scratch_llm.best_checkpoint import (
    BEST_CHECKPOINT_RANKING_PROTOCOL_ID,
    PeriodicValidationResult,
    ValidationCheckpointState,
)
from scratch_llm.evaluation.bpb import BaseValidationResult
from scratch_llm.evaluation.core.reporting import (
    CORE_COMPARISON_FILENAME,
    render_core_comparison_markdown,
    write_core_comparison_markdown,
)
from scratch_llm.evaluation.core.tracking import (
    CORE_COMPARISON_ARTIFACT_NAME,
    CoreMetricValue,
    core_evaluation_metrics,
)
from scratch_llm.evaluation.full_document_bpb import (
    FULL_DOCUMENT_EVAL_METRIC,
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_TRAIN_METRIC,
)
from scratch_llm.evaluation.nanochat_bpb import (
    NANOCHAT_COMPAT_EVAL_METRIC,
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_TRAIN_METRIC,
)
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.utils import save_json


BASE_EVALUATION_REPORT_FORMAT: Final = "scratch_llm_base_evaluation"
BASE_EVALUATION_REPORT_FORMAT_VERSION: Final = 3
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
_CORE_COMPARISON_RELATIVE_PATH = Path("metrics") / CORE_COMPARISON_FILENAME


@dataclass(frozen=True)
class TrackedBaseEvaluation:
    """Completed standalone metrics and canonical artifact paths."""

    metrics: Mapping[str, CoreMetricValue]
    report_path: Path
    sample_markdown_path: Path | None = None
    core_comparison_path: Path | None = None


@dataclass(frozen=True)
class TrackedBaseSamples:
    """Completed fixed-sample throughput and canonical Markdown path."""

    metrics: Mapping[str, float]
    markdown_path: Path


class BaseEvaluationReportConflictError(RuntimeError):
    """A completed run already owns a different canonical evaluation report."""


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
    step = require_non_negative_integer(step, name="step")
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
    """Publish a dual-BPB result through the complete standalone contract."""

    compatibility, full_document = _complete_protocol_results(validation)
    return report_completed_base_evaluation(
        CompletedBaseEvaluation(
            context=BaseEvaluationContext(
                checkpoint_identity=compatibility.checkpoint_identity,
                checkpoint_step=0,
                config_identity="legacy:unspecified",
                tokenizer_identity=compatibility.tokenizer_identity,
                validation_manifest_identity=(
                    compatibility.validation_manifest_identity
                ),
                run_kind="full",
                max_per_task=None,
            ),
            requested_modes=("bpb",),
            completed_modes=("bpb",),
            validation=PeriodicValidationResult(
                compatibility=compatibility,
                full_document=full_document,
            ),
            samples=None,
            core_result=None,
        ),
        tracker=tracker,
        run_dir=run_dir,
    )


def report_completed_base_evaluation(
    completed: CompletedBaseEvaluation,
    *,
    tracker: Tracker,
    run_dir: str | Path,
) -> TrackedBaseEvaluation:
    """Publish one all-modes completion marker and its finalized tracker fan-out."""

    if not isinstance(completed, CompletedBaseEvaluation):
        raise TypeError(
            "completed must be a CompletedBaseEvaluation, got "
            f"{type(completed).__name__}"
        )
    _require_tracker(tracker)
    resolved_run_dir = _run_directory(run_dir)
    payload = _base_evaluation_payload(completed)
    metrics = _completed_metrics(completed)
    report_path = resolved_run_dir / _BASE_EVALUATION_RELATIVE_PATH
    report_exists = _require_compatible_existing_report(report_path, payload)

    sample_markdown_path: Path | None = None
    if completed.samples is not None:
        sample_markdown_path = write_base_samples_markdown(
            completed.samples,
            resolved_run_dir / "metrics",
        )
    core_comparison_path: Path | None = None
    core_comparison_existed = False
    if completed.core_result is not None:
        core_comparison_path = resolved_run_dir / _CORE_COMPARISON_RELATIVE_PATH
        expected_markdown = render_core_comparison_markdown(completed.core_result)
        core_comparison_existed = _require_compatible_existing_text(
            core_comparison_path,
            expected_markdown,
            label="CORE comparison report",
        )
        if not core_comparison_existed:
            core_comparison_path = write_core_comparison_markdown(
                completed.core_result,
                resolved_run_dir / "metrics",
            )
    try:
        if not report_exists:
            report_path = save_json(payload, report_path)
    except BaseException:
        if core_comparison_path is not None and not core_comparison_existed:
            core_comparison_path.unlink(missing_ok=True)
        raise

    event_prefix = f"base-evaluation:{_payload_identity(payload)}"
    if metrics:
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
    if sample_markdown_path is not None:
        _log_artifact_once(
            tracker,
            _BASE_SAMPLES_RELATIVE_PATH.as_posix(),
            name=BASE_SAMPLES_ARTIFACT_NAME,
            event_id=f"{event_prefix}:artifact:base_samples.md",
        )
    if core_comparison_path is not None:
        _log_artifact_once(
            tracker,
            _CORE_COMPARISON_RELATIVE_PATH.as_posix(),
            name=CORE_COMPARISON_ARTIFACT_NAME,
            event_id=f"{event_prefix}:artifact:{CORE_COMPARISON_FILENAME}",
        )
    return TrackedBaseEvaluation(
        metrics=MappingProxyType(metrics),
        report_path=report_path,
        sample_markdown_path=sample_markdown_path,
        core_comparison_path=core_comparison_path,
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
    completed: CompletedBaseEvaluation,
) -> dict[str, object]:
    context = completed.context
    payload: dict[str, object] = {
        "bounded": context.run_kind == "bounded",
        "completed_modes": list(completed.completed_modes),
        "format": BASE_EVALUATION_REPORT_FORMAT,
        "format_version": BASE_EVALUATION_REPORT_FORMAT_VERSION,
        "identities": {
            "checkpoint": {
                "identity": context.checkpoint_identity,
                "step": context.checkpoint_step,
            },
            "config": context.config_identity,
            "tokenizer": context.tokenizer_identity,
            "validation_manifest": context.validation_manifest_identity,
        },
        "max_per_task": context.max_per_task,
        "requested_modes": list(completed.requested_modes),
        "results": {},
        "run_kind": context.run_kind,
        "status": "completed",
    }
    if completed.validation is not None:
        compatibility, full_document = _complete_protocol_results(completed.validation)
        payload["results"] = {
            NANOCHAT_COMPAT_PROTOCOL_ID: compatibility.to_dict(),
            FULL_DOCUMENT_PROTOCOL_ID: full_document.to_dict(),
        }
    if completed.samples is not None:
        samples = completed.samples
        payload["samples"] = {
            "artifact_path": _BASE_SAMPLES_RELATIVE_PATH.as_posix(),
            "completion_reasons": {
                "max_new_tokens": sum(
                    sample.completion_reason == "max_new_tokens"
                    for sample in samples.samples
                ),
                "stop_token": sum(
                    sample.completion_reason == "stop_token"
                    for sample in samples.samples
                ),
            },
            "generation": samples.config.to_dict(),
            "generation_identity": samples.generation_identity,
            "prompt_set_identity": samples.prompt_set_identity,
            "result": samples.to_dict(),
            "sample_count": len(samples.samples),
            "sampled_token_count": sum(
                sample.sampled_token_count for sample in samples.samples
            ),
        }
    if completed.core_result is not None:
        core_payload = completed.core_result.to_dict()
        core_payload["comparison_artifact_path"] = (
            _CORE_COMPARISON_RELATIVE_PATH.as_posix()
        )
        payload["core"] = core_payload
    return payload


def _completed_metrics(
    completed: CompletedBaseEvaluation,
) -> dict[str, CoreMetricValue]:
    metrics: dict[str, CoreMetricValue] = {}
    if completed.validation is not None:
        compatibility, full_document = _complete_protocol_results(completed.validation)
        metrics.update(
            {
                NANOCHAT_COMPAT_EVAL_METRIC: compatibility.bpb,
                FULL_DOCUMENT_EVAL_METRIC: full_document.bpb,
                NANOCHAT_SOURCE_BYTE_RETENTION_METRIC: (
                    compatibility.source_byte_retention
                ),
                FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC: (
                    full_document.source_byte_retention
                ),
            }
        )
    if completed.samples is not None:
        sampled_tokens = sum(
            sample.sampled_token_count for sample in completed.samples.samples
        )
        elapsed_seconds = sum(
            sample.elapsed_seconds for sample in completed.samples.samples
        )
        metrics[BASE_SAMPLE_THROUGHPUT_METRIC] = sampled_tokens / elapsed_seconds
    if completed.core_result is not None:
        metrics.update(core_evaluation_metrics(completed.core_result))
    return metrics


def _require_compatible_existing_report(
    path: Path,
    expected: dict[str, object],
) -> bool:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BaseEvaluationReportConflictError(
            f"existing base evaluation report cannot be validated: {error}"
        ) from error
    if existing != expected:
        raise BaseEvaluationReportConflictError(
            f"{path} already contains a different completed evaluation"
        )
    return True


def _require_compatible_existing_text(
    path: Path,
    expected: str,
    *,
    label: str,
) -> bool:
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except (OSError, UnicodeError) as error:
        raise BaseEvaluationReportConflictError(
            f"existing {label} cannot be validated: {error}"
        ) from error
    if existing != expected:
        raise BaseEvaluationReportConflictError(
            f"{path} already contains a different completed {label}"
        )
    return True


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


__all__ = [
    "BASE_EVALUATION_ARTIFACT_NAME",
    "BASE_EVALUATION_ARTIFACT_TYPE",
    "BASE_EVALUATION_REPORT_FORMAT",
    "BASE_EVALUATION_REPORT_FORMAT_VERSION",
    "BASE_SAMPLE_THROUGHPUT_METRIC",
    "BASE_SAMPLES_ARTIFACT_NAME",
    "BaseEvaluationReportConflictError",
    "FULL_DOCUMENT_MINIMUM_TRAIN_METRIC",
    "FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC",
    "NANOCHAT_MINIMUM_TRAIN_METRIC",
    "NANOCHAT_SOURCE_BYTE_RETENTION_METRIC",
    "TrackedBaseEvaluation",
    "TrackedBaseSamples",
    "report_base_samples",
    "report_completed_base_evaluation",
    "report_standalone_base_evaluation",
    "track_periodic_base_validation",
]
