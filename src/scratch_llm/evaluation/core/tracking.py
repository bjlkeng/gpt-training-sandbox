"""Stable Tracker metric names and values for completed CORE evaluation."""

from __future__ import annotations

import re
from typing import Final, TypeAlias

from scratch_llm.evaluation.core.results import CoreEvaluationResult


CoreMetricValue: TypeAlias = float | int | str | None

CORE_COMPARISON_ARTIFACT_NAME: Final = "core_comparison"
CORE_EVAL_METRIC: Final = "eval/core_metric"
CORE_RUN_KIND_METRIC: Final = "eval/core_run_kind"
CORE_MAX_PER_TASK_METRIC: Final = "eval/core_max_per_task"
CORE_TASK_COUNT_METRIC: Final = "eval/core_task_count"
CORE_TASK_CENTERED_METRIC_PREFIX: Final = "eval/core"
CORE_TASK_ACCURACY_METRIC_PREFIX: Final = "eval/core_accuracy"
CORE_TASK_RANDOM_BASELINE_METRIC_PREFIX: Final = "eval/core_random_baseline_percent"
CORE_TASK_CORRECT_COUNT_METRIC_PREFIX: Final = "eval/core_correct_examples"
CORE_TASK_EVALUATED_COUNT_METRIC_PREFIX: Final = "eval/core_evaluated_examples"
CORE_TASK_AVAILABLE_COUNT_METRIC_PREFIX: Final = "eval/core_available_examples"


def core_evaluation_metrics(
    result: CoreEvaluationResult,
) -> dict[str, CoreMetricValue]:
    """Return collision-free aggregate, scope, centered, and raw task metrics."""

    if not isinstance(result, CoreEvaluationResult):
        raise TypeError("result must be a CoreEvaluationResult")
    metrics: dict[str, CoreMetricValue] = {
        CORE_EVAL_METRIC: result.core_metric,
        CORE_RUN_KIND_METRIC: result.run_kind,
        CORE_MAX_PER_TASK_METRIC: result.max_per_task,
        CORE_TASK_COUNT_METRIC: len(result.tasks),
    }
    labels_by_slug: dict[str, str] = {}
    for task in result.tasks:
        slug = _normalize_task_label(task.label)
        previous = labels_by_slug.get(slug)
        if previous is not None:
            raise ValueError(
                "CORE task metric labels collide after normalization: "
                f"{previous!r} and {task.label!r} both map to {slug!r}"
            )
        labels_by_slug[slug] = task.label
        metrics.update(
            {
                f"{CORE_TASK_CENTERED_METRIC_PREFIX}/{slug}": task.centered_score,
                f"{CORE_TASK_ACCURACY_METRIC_PREFIX}/{slug}": task.accuracy,
                f"{CORE_TASK_RANDOM_BASELINE_METRIC_PREFIX}/{slug}": (
                    task.random_baseline_percent
                ),
                f"{CORE_TASK_CORRECT_COUNT_METRIC_PREFIX}/{slug}": (
                    task.correct_examples
                ),
                f"{CORE_TASK_EVALUATED_COUNT_METRIC_PREFIX}/{slug}": (
                    task.evaluated_examples
                ),
                f"{CORE_TASK_AVAILABLE_COUNT_METRIC_PREFIX}/{slug}": (
                    task.available_examples
                ),
            }
        )
    return metrics


def _normalize_task_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not normalized:
        raise ValueError(f"CORE task label {label!r} has no metric-safe characters")
    return normalized


__all__ = [
    "CORE_COMPARISON_ARTIFACT_NAME",
    "CORE_EVAL_METRIC",
    "CORE_MAX_PER_TASK_METRIC",
    "CORE_RUN_KIND_METRIC",
    "CORE_TASK_ACCURACY_METRIC_PREFIX",
    "CORE_TASK_AVAILABLE_COUNT_METRIC_PREFIX",
    "CORE_TASK_CENTERED_METRIC_PREFIX",
    "CORE_TASK_CORRECT_COUNT_METRIC_PREFIX",
    "CORE_TASK_COUNT_METRIC",
    "CORE_TASK_EVALUATED_COUNT_METRIC_PREFIX",
    "CORE_TASK_RANDOM_BASELINE_METRIC_PREFIX",
    "CoreMetricValue",
    "core_evaluation_metrics",
]
