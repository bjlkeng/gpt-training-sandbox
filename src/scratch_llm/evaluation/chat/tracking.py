"""Tracker fan-out for an already-written immutable chat evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scratch_llm.evaluation.chat.reporting import (
    CHAT_EVALUATION_REPORT_RELATIVE_PATH,
    CompletedChatEvaluation,
)
from scratch_llm.identity import canonical_json_identity, file_identity
from scratch_llm.tracking import RunTracker, Tracker


CHATCORE_METRIC: Final = "sft/chatcore_metric"
CHATCORE_CAT_METRIC: Final = "sft/chatcore_cat"
CHATCORE_TASK_METRIC_PREFIX: Final = "sft/chatcore"
CHAT_EVALUATION_ARTIFACT_NAME: Final = "chat_eval"
CHAT_EVALUATION_ARTIFACT_TYPE: Final = "evaluation"


class ChatEvaluationTrackingError(RuntimeError):
    """A report cannot be verified as the completed result tracker input."""


@dataclass(frozen=True, slots=True)
class TrackedChatEvaluation:
    """Published scalar values and the verified canonical artifact identity."""

    metrics: Mapping[str, float]
    report_path: Path
    artifact_identity: str


def chat_evaluation_metrics(
    completed: CompletedChatEvaluation,
) -> dict[str, float]:
    """Return full public names or scope-qualified non-full task values."""

    if not isinstance(completed, CompletedChatEvaluation):
        raise TypeError("completed must be a CompletedChatEvaluation")
    if completed.settings.full:
        if completed.chatcore_metric is None or completed.chatcore_cat is None:
            raise ChatEvaluationTrackingError(
                "a full chat evaluation is missing its ChatCORE aggregates"
            )
        metrics = {
            CHATCORE_METRIC: completed.chatcore_metric,
            CHATCORE_CAT_METRIC: completed.chatcore_cat,
        }
        prefix = CHATCORE_TASK_METRIC_PREFIX
    else:
        metrics = {}
        prefix = f"{CHATCORE_TASK_METRIC_PREFIX}/{completed.settings.kind}"
    metrics.update(
        {
            f"{prefix}/{task.task_name}": task.centered_score
            for task in completed.chatcore.tasks
        }
    )
    return metrics


def track_completed_chat_evaluation(
    completed: CompletedChatEvaluation,
    *,
    report_path: str | Path,
    tracker: Tracker,
    run_dir: str | Path,
) -> TrackedChatEvaluation:
    """Verify and register the reporter-owned artifact, then fan out metrics once."""

    if not isinstance(completed, CompletedChatEvaluation):
        raise TypeError("completed must be a CompletedChatEvaluation")
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")
    resolved_run_dir = Path(run_dir).resolve()
    expected_path = resolved_run_dir / CHAT_EVALUATION_REPORT_RELATIVE_PATH
    provided_path = Path(report_path)
    try:
        resolved_report_path = provided_path.resolve(strict=True)
    except OSError as error:
        raise ChatEvaluationTrackingError(
            f"chat evaluation report cannot be read: {provided_path}"
        ) from error
    if (
        resolved_report_path != expected_path
        or provided_path.is_symlink()
        or not resolved_report_path.is_file()
    ):
        raise ChatEvaluationTrackingError(
            "report_path must be the regular run-relative metrics/chat_eval.json"
        )

    expected_payload = completed.to_dict()
    try:
        actual_payload = json.loads(resolved_report_path.read_text(encoding="utf-8"))
        matches = canonical_json_identity(actual_payload) == canonical_json_identity(
            expected_payload
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise ChatEvaluationTrackingError(
            f"chat evaluation report cannot be verified: {resolved_report_path}"
        ) from error
    if not matches:
        raise ChatEvaluationTrackingError(
            "chat evaluation report does not match the immutable result"
        )

    artifact_identity = file_identity(resolved_report_path)
    metrics = chat_evaluation_metrics(completed)
    result_identity = expected_payload["result_identity"]
    if not isinstance(result_identity, str):  # pragma: no cover - report invariant.
        raise ChatEvaluationTrackingError("chat evaluation result identity is invalid")
    event_prefix = f"chat-evaluation:{result_identity}"
    _log_metrics_once(
        tracker,
        metrics,
        step=completed.checkpoint_step,
        event_id=f"{event_prefix}:metrics",
    )
    _log_artifact_once(
        tracker,
        event_id=f"{event_prefix}:artifact:chat_eval.json",
    )
    return TrackedChatEvaluation(
        metrics=MappingProxyType(metrics),
        report_path=resolved_report_path,
        artifact_identity=artifact_identity,
    )


def _log_metrics_once(
    tracker: Tracker,
    metrics: dict[str, float],
    *,
    step: int,
    event_id: str,
) -> None:
    if isinstance(tracker, RunTracker):
        tracker.log_once(metrics, event_id=event_id, step=step)
    else:
        tracker.log(metrics, step=step)


def _log_artifact_once(tracker: Tracker, *, event_id: str) -> None:
    path = CHAT_EVALUATION_REPORT_RELATIVE_PATH.as_posix()
    if isinstance(tracker, RunTracker):
        tracker.log_artifact_once(
            path,
            CHAT_EVALUATION_ARTIFACT_NAME,
            CHAT_EVALUATION_ARTIFACT_TYPE,
            event_id=event_id,
        )
    else:
        tracker.log_artifact(
            path,
            CHAT_EVALUATION_ARTIFACT_NAME,
            CHAT_EVALUATION_ARTIFACT_TYPE,
        )


__all__ = [
    "CHATCORE_CAT_METRIC",
    "CHATCORE_METRIC",
    "CHATCORE_TASK_METRIC_PREFIX",
    "CHAT_EVALUATION_ARTIFACT_NAME",
    "CHAT_EVALUATION_ARTIFACT_TYPE",
    "ChatEvaluationTrackingError",
    "TrackedChatEvaluation",
    "chat_evaluation_metrics",
    "track_completed_chat_evaluation",
]
