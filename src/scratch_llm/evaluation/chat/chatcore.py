"""Immutable normalized task scores and arithmetic for ChatCORE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias, cast

from scratch_llm._validation import (
    JsonValueValidator,
    require_finite_unit_interval,
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.identity import canonical_json_identity


ChatCoreTaskName: TypeAlias = Literal[
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
]
ChatCoreRunKind: TypeAlias = Literal["bounded", "full"]

CHATCORE_PROTOCOL_ID: Final = "nanochat_chatcore_v1"
CHATCORE_PROTOCOL_VERSION: Final = 1
CHATCORE_TASK_ORDER: Final[tuple[ChatCoreTaskName, ...]] = (
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
)
CHATCORE_CATEGORICAL_TASKS: Final[tuple[ChatCoreTaskName, ...]] = CHATCORE_TASK_ORDER[
    :3
]
CHATCORE_BASELINE_ACCURACIES: Final[Mapping[ChatCoreTaskName, float]] = (
    MappingProxyType(
        {
            "ARC-Easy": 0.25,
            "ARC-Challenge": 0.25,
            "MMLU": 0.25,
            "GSM8K": 0.0,
            "HumanEval": 0.0,
        }
    )
)

_TASK_INDEX = {task_name: index for index, task_name in enumerate(CHATCORE_TASK_ORDER)}
_TASK_SOURCE_FIELDS = frozenset(
    {
        "accuracy",
        "checkpoint_identity",
        "max_problems",
        "passed_count",
        "renderer_identity",
        "run_kind",
        "status",
        "task_name",
        "tokenizer_identity",
        "total_count",
    }
)
_TASK_FIELDS = _TASK_SOURCE_FIELDS | {"baseline_accuracy", "centered_score"}
_RESULT_FIELDS = frozenset(
    {
        "chatcore_cat",
        "chatcore_metric",
        "complete",
        "missing_tasks",
        "protocol_id",
        "protocol_version",
        "scope",
        "tasks",
    }
)


class ChatCoreEvaluationError(ValueError):
    """Task scores cannot produce a trustworthy ChatCORE result."""


_VALUES = JsonValueValidator(ChatCoreEvaluationError)


@dataclass(frozen=True, slots=True)
class ChatCoreTaskResult:
    """One completed task score normalized for aggregate-only evaluation."""

    task_name: ChatCoreTaskName
    status: Literal["completed"]
    passed_count: int
    total_count: int
    accuracy: float
    checkpoint_identity: str
    tokenizer_identity: str
    renderer_identity: str
    run_kind: ChatCoreRunKind
    max_problems: int | None

    def __post_init__(self) -> None:
        if self.task_name not in CHATCORE_TASK_ORDER:
            raise ChatCoreEvaluationError(
                f"task_name must be a canonical task name, got {self.task_name!r}"
            )
        if self.status != "completed":
            raise ChatCoreEvaluationError(
                "failed task results cannot be included in ChatCORE"
            )
        try:
            passed_count = require_non_negative_integer(
                self.passed_count,
                name="passed_count",
            )
            total_count = require_positive_integer(
                self.total_count,
                name="total_count",
            )
            accuracy = require_finite_unit_interval(self.accuracy, name="accuracy")
            for name in (
                "checkpoint_identity",
                "tokenizer_identity",
                "renderer_identity",
            ):
                require_non_empty_string(getattr(self, name), name=name)
        except (TypeError, ValueError) as error:
            raise ChatCoreEvaluationError(str(error)) from error
        if passed_count > total_count:
            raise ChatCoreEvaluationError("passed_count must not exceed total_count")
        if accuracy != passed_count / total_count:
            raise ChatCoreEvaluationError(
                "accuracy must equal passed_count / total_count"
            )
        if self.run_kind == "full":
            if self.max_problems is not None:
                raise ChatCoreEvaluationError(
                    "full task results must not set max_problems"
                )
        elif self.run_kind == "bounded":
            try:
                require_positive_integer(self.max_problems, name="max_problems")
            except (TypeError, ValueError) as error:
                raise ChatCoreEvaluationError(str(error)) from error
        else:
            raise ChatCoreEvaluationError("run_kind must be 'bounded' or 'full'")
        object.__setattr__(self, "accuracy", accuracy)

    @property
    def baseline_accuracy(self) -> float:
        """Return the fixed random baseline for this task."""

        return CHATCORE_BASELINE_ACCURACIES[self.task_name]

    @property
    def centered_score(self) -> float:
        """Center accuracy without clipping values below random chance."""

        baseline = self.baseline_accuracy
        return (self.accuracy - baseline) / (1.0 - baseline)

    def to_dict(self) -> dict[str, object]:
        """Return exact raw inputs plus their two derived score values."""

        return {
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "centered_score": self.centered_score,
            "checkpoint_identity": self.checkpoint_identity,
            "max_problems": self.max_problems,
            "passed_count": self.passed_count,
            "renderer_identity": self.renderer_identity,
            "run_kind": self.run_kind,
            "status": self.status,
            "task_name": self.task_name,
            "tokenizer_identity": self.tokenizer_identity,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> ChatCoreTaskResult:
        """Rebuild one task only when its exact derived values agree."""

        data = _VALUES.require_object(
            value,
            label="ChatCORE task result",
            expected_keys=_TASK_FIELDS,
            schema_label="the current format",
        )
        source = {name: data[name] for name in _TASK_SOURCE_FIELDS}
        try:
            result = cls(**cast(Any, source))
        except ChatCoreEvaluationError:
            raise
        except (TypeError, ValueError) as error:
            raise ChatCoreEvaluationError(
                f"invalid ChatCORE task result: {error}"
            ) from error
        _require_exact_json(result.to_dict(), data)
        return result


@dataclass(frozen=True, slots=True)
class ChatCoreResult:
    """Canonical complete or partial ChatCORE task collection."""

    tasks: tuple[ChatCoreTaskResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ChatCoreEvaluationError("tasks must be a non-empty tuple")
        if any(not isinstance(task, ChatCoreTaskResult) for task in self.tasks):
            raise TypeError("tasks must contain only ChatCoreTaskResult values")
        task_names = tuple(task.task_name for task in self.tasks)
        if len(set(task_names)) != len(task_names):
            raise ChatCoreEvaluationError("duplicate ChatCORE task results")

        expected_context = _execution_context(self.tasks[0])
        if any(_execution_context(task) != expected_context for task in self.tasks[1:]):
            raise ChatCoreEvaluationError(
                "all task results must use the same execution context"
            )
        object.__setattr__(
            self,
            "tasks",
            tuple(sorted(self.tasks, key=lambda task: _TASK_INDEX[task.task_name])),
        )

    @property
    def checkpoint_identity(self) -> str:
        return self.tasks[0].checkpoint_identity

    @property
    def tokenizer_identity(self) -> str:
        return self.tasks[0].tokenizer_identity

    @property
    def renderer_identity(self) -> str:
        return self.tasks[0].renderer_identity

    @property
    def run_kind(self) -> ChatCoreRunKind:
        return self.tasks[0].run_kind

    @property
    def max_problems(self) -> int | None:
        return self.tasks[0].max_problems

    @property
    def missing_tasks(self) -> tuple[ChatCoreTaskName, ...]:
        """Return canonical tasks absent from this partial result."""

        present = {task.task_name for task in self.tasks}
        return tuple(
            task_name for task_name in CHATCORE_TASK_ORDER if task_name not in present
        )

    @property
    def complete(self) -> bool:
        return not self.missing_tasks

    @property
    def chatcore_metric(self) -> float | None:
        """Return the five-task centered mean only for a complete run."""

        if not self.complete:
            return None
        return math.fsum(task.centered_score for task in self.tasks) / len(
            CHATCORE_TASK_ORDER
        )

    @property
    def chatcore_cat(self) -> float | None:
        """Return the categorical centered mean only for a complete run."""

        if not self.complete:
            return None
        return math.fsum(
            task.centered_score
            for task in self.tasks
            if task.task_name in CHATCORE_CATEGORICAL_TASKS
        ) / len(CHATCORE_CATEGORICAL_TASKS)

    def to_dict(self) -> dict[str, object]:
        """Return the exact content-free aggregate representation."""

        return {
            "chatcore_cat": self.chatcore_cat,
            "chatcore_metric": self.chatcore_metric,
            "complete": self.complete,
            "missing_tasks": list(self.missing_tasks),
            "protocol_id": CHATCORE_PROTOCOL_ID,
            "protocol_version": CHATCORE_PROTOCOL_VERSION,
            "scope": {
                "bounded": self.run_kind == "bounded",
                "max_problems": self.max_problems,
                "run_kind": self.run_kind,
                "task_count": len(self.tasks),
            },
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, value: object) -> ChatCoreResult:
        """Rebuild one result only when its exact derived schema agrees."""

        data = _VALUES.require_object(
            value,
            label="ChatCORE result",
            expected_keys=_RESULT_FIELDS,
            schema_label="the current format",
        )
        protocol_id = _VALUES.require_string(
            data["protocol_id"],
            label="protocol_id",
        )
        protocol_version = _VALUES.require_integer(
            data["protocol_version"],
            label="protocol_version",
            minimum=1,
        )
        if (
            protocol_id != CHATCORE_PROTOCOL_ID
            or protocol_version != CHATCORE_PROTOCOL_VERSION
        ):
            raise ChatCoreEvaluationError("unsupported ChatCORE protocol")
        raw_tasks = _VALUES.require_list(
            data["tasks"],
            label="tasks",
            non_empty=True,
        )
        result = cls(tuple(ChatCoreTaskResult.from_dict(task) for task in raw_tasks))
        _require_exact_json(result.to_dict(), data)
        return result


def _execution_context(
    task: ChatCoreTaskResult,
) -> tuple[str, str, str, ChatCoreRunKind, int | None]:
    return (
        task.checkpoint_identity,
        task.tokenizer_identity,
        task.renderer_identity,
        task.run_kind,
        task.max_problems,
    )


def _require_exact_json(expected: object, actual: object) -> None:
    """Reject type changes and non-JSON values as well as unequal payloads."""

    try:
        matches = canonical_json_identity(expected) == canonical_json_identity(actual)
    except (TypeError, ValueError) as error:
        raise ChatCoreEvaluationError(
            "serialized ChatCORE result must contain only finite JSON values"
        ) from error
    if not matches:
        raise ChatCoreEvaluationError(
            "serialized ChatCORE result does not match its derived values"
        )


__all__ = [
    "CHATCORE_BASELINE_ACCURACIES",
    "CHATCORE_CATEGORICAL_TASKS",
    "CHATCORE_PROTOCOL_ID",
    "CHATCORE_PROTOCOL_VERSION",
    "CHATCORE_TASK_ORDER",
    "ChatCoreEvaluationError",
    "ChatCoreResult",
    "ChatCoreRunKind",
    "ChatCoreTaskName",
    "ChatCoreTaskResult",
]
