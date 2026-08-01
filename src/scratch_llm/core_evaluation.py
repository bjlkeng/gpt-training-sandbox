"""Typed results and arithmetic for the pinned nanochat CORE protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_finite_real,
    require_finite_unit_interval,
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.nanochat_bpb import NANOCHAT_REFERENCE_COMMIT


CORE_PROTOCOL_ID: Final = "nanochat_core_v1"
CORE_PROTOCOL_VERSION: Final = 1
CORE_REFERENCE_COMMIT: Final = NANOCHAT_REFERENCE_COMMIT
CORE_REFERENCE_FILE_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "nanochat/core_eval.py": (
            "9303997ff17fa48ad5e5730f33b7e79a687cc2a1fdad928666bb63dc3514c51d"
        ),
        "scripts/base_eval.py": (
            "644083a99ded710f677b25168ca76fd0416dc7ef3890d346a72f76a20bb522fe"
        ),
    }
)
CoreTaskType: TypeAlias = Literal[
    "multiple_choice",
    "schema",
    "language_modeling",
]
CoreEvaluationRunKind: TypeAlias = Literal["bounded", "full"]
_CORE_TASK_TYPES = frozenset({"multiple_choice", "schema", "language_modeling"})


class CoreEvaluationError(ValueError):
    """A CORE bundle, request, or result violates the pinned protocol."""


@dataclass(frozen=True)
class CoreTaskResult:
    """Validated counts and centered score for one configured task."""

    label: str
    task_type: CoreTaskType
    num_fewshot: int
    random_baseline_percent: float
    correct_examples: int
    evaluated_examples: int
    available_examples: int
    elapsed_seconds: float
    data_identity: str

    def __post_init__(self) -> None:
        require_non_empty_string(self.label, name="label")
        if self.task_type not in _CORE_TASK_TYPES:
            raise ValueError(f"unsupported CORE task type {self.task_type!r}")
        require_non_negative_integer(self.num_fewshot, name="num_fewshot")
        require_positive_integer(self.evaluated_examples, name="evaluated_examples")
        require_positive_integer(self.available_examples, name="available_examples")
        require_non_negative_integer(self.correct_examples, name="correct_examples")
        if self.correct_examples > self.evaluated_examples:
            raise ValueError("correct_examples must not exceed evaluated_examples")
        if self.evaluated_examples > self.available_examples:
            raise ValueError("evaluated_examples must not exceed available_examples")
        centered_core_score(self.accuracy, self.random_baseline_percent)
        require_finite_non_negative_real(
            self.elapsed_seconds,
            name="elapsed_seconds",
        )
        require_non_empty_string(self.data_identity, name="data_identity")

    @property
    def accuracy(self) -> float:
        """Return exact-match accuracy derived from integer counts."""

        return self.correct_examples / self.evaluated_examples

    @property
    def centered_score(self) -> float:
        """Return this task accuracy centered against its random baseline."""

        return centered_core_score(self.accuracy, self.random_baseline_percent)

    def to_dict(self) -> dict[str, object]:
        """Return the stable public representation of one task result."""

        return {
            "accuracy": self.accuracy,
            "available_examples": self.available_examples,
            "centered_score": self.centered_score,
            "correct_examples": self.correct_examples,
            "data_identity": self.data_identity,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluated_examples": self.evaluated_examples,
            "num_fewshot": self.num_fewshot,
            "random_baseline_percent": self.random_baseline_percent,
            "task_type": self.task_type,
        }


@dataclass(frozen=True)
class CoreReferenceComparison:
    """One pinned reference aggregate used only for rough comparison."""

    model_id: str
    core_metric: float

    def __post_init__(self) -> None:
        require_non_empty_string(self.model_id, name="model_id")
        require_finite_real(self.core_metric, name="core_metric")


@dataclass(frozen=True)
class CoreEvaluationResult:
    """One complete full or deterministically bounded CORE run."""

    checkpoint_identity: str
    tokenizer_identity: str
    bundle_identity: str
    config_identity: str
    metadata_identity: str
    run_kind: CoreEvaluationRunKind
    max_per_task: int | None
    tasks: tuple[CoreTaskResult, ...]
    references: tuple[CoreReferenceComparison, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_identity",
            "tokenizer_identity",
            "bundle_identity",
            "config_identity",
            "metadata_identity",
        ):
            require_non_empty_string(getattr(self, name), name=name)
        if self.run_kind not in ("bounded", "full"):
            raise ValueError("run_kind must be 'bounded' or 'full'")
        if self.run_kind == "full":
            if self.max_per_task is not None:
                raise ValueError("full CORE results must not set max_per_task")
        elif self.max_per_task is None:
            raise ValueError("bounded CORE results require max_per_task")
        else:
            require_positive_integer(self.max_per_task, name="max_per_task")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ValueError("tasks must be a non-empty tuple")
        if any(not isinstance(task, CoreTaskResult) for task in self.tasks):
            raise TypeError("tasks must contain only CoreTaskResult values")
        task_labels = tuple(task.label for task in self.tasks)
        if len(set(task_labels)) != len(task_labels):
            raise ValueError("CORE task labels must be unique")
        if not isinstance(self.references, tuple) or not self.references:
            raise ValueError("references must be a non-empty tuple")
        if any(
            not isinstance(reference, CoreReferenceComparison)
            for reference in self.references
        ):
            raise TypeError(
                "references must contain only CoreReferenceComparison values"
            )
        reference_ids = tuple(reference.model_id for reference in self.references)
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("CORE reference model ids must be unique")
        require_finite_non_negative_real(
            self.elapsed_seconds,
            name="elapsed_seconds",
        )

    @property
    def core_metric(self) -> float:
        """Return the unweighted mean centered score across configured tasks."""

        return sum(task.centered_score for task in self.tasks) / len(self.tasks)

    def to_dict(self) -> dict[str, object]:
        """Return complete protocol provenance, task results, and comparisons."""

        comparable = self.run_kind == "full"
        return {
            "bundle": {
                "config_identity": self.config_identity,
                "identity": self.bundle_identity,
                "metadata_identity": self.metadata_identity,
            },
            "comparison": {
                "comparable": comparable,
                "note": (
                    "Full-run deltas use the same pinned 22-task bundle."
                    if comparable
                    else "Bounded max-per-task results are estimates and are not "
                    "ranked against full reference runs."
                ),
                "references": {
                    reference.model_id: {
                        "core_metric": reference.core_metric,
                        "delta": (
                            self.core_metric - reference.core_metric
                            if comparable
                            else None
                        ),
                    }
                    for reference in self.references
                },
            },
            "core_metric": self.core_metric,
            "elapsed_seconds": self.elapsed_seconds,
            "identities": {
                "checkpoint": self.checkpoint_identity,
                "tokenizer": self.tokenizer_identity,
            },
            "protocol_id": CORE_PROTOCOL_ID,
            "protocol_version": CORE_PROTOCOL_VERSION,
            "reference_commit": CORE_REFERENCE_COMMIT,
            "reference_files": dict(CORE_REFERENCE_FILE_SHA256),
            "scope": {
                "available_examples": sum(
                    task.available_examples for task in self.tasks
                ),
                "bounded": self.run_kind == "bounded",
                "evaluated_examples": sum(
                    task.evaluated_examples for task in self.tasks
                ),
                "max_per_task": self.max_per_task,
                "run_kind": self.run_kind,
                "task_count": len(self.tasks),
            },
            "task_order": [task.label for task in self.tasks],
            "tasks": {task.label: task.to_dict() for task in self.tasks},
        }


def centered_core_score(
    accuracy: float,
    random_baseline_percent: float,
) -> float:
    """Center one task accuracy against its percentage random baseline."""

    accuracy = require_finite_unit_interval(accuracy, name="accuracy")
    random_baseline_percent = require_finite_real(
        random_baseline_percent,
        name="random_baseline_percent",
    )
    if not 0 <= random_baseline_percent < 100:
        raise ValueError("random_baseline_percent must be in [0, 100)")
    random_baseline = 0.01 * random_baseline_percent
    return (accuracy - random_baseline) / (1.0 - random_baseline)


__all__ = [
    "CORE_PROTOCOL_ID",
    "CORE_PROTOCOL_VERSION",
    "CORE_REFERENCE_COMMIT",
    "CORE_REFERENCE_FILE_SHA256",
    "CoreEvaluationError",
    "CoreEvaluationResult",
    "CoreEvaluationRunKind",
    "CoreReferenceComparison",
    "CoreTaskResult",
    "CoreTaskType",
    "centered_core_score",
]
