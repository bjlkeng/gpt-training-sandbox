"""Tests for immutable ChatCORE task scores and centered aggregates."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math
from typing import Any, cast

import pytest

from scratch_llm.evaluation.chat.chatcore import (
    CHATCORE_BASELINE_ACCURACIES,
    CHATCORE_CATEGORICAL_TASKS,
    CHATCORE_TASK_ORDER,
    ChatCoreEvaluationError,
    ChatCoreResult,
    ChatCoreRunKind,
    ChatCoreTaskName,
    ChatCoreTaskResult,
)


_CHECKPOINT_IDENTITY = "sha256:" + "1" * 64
_TOKENIZER_IDENTITY = "sha256:" + "2" * 64
_RENDERER_IDENTITY = "renderer-v1"


def _task(
    task_name: str,
    passed: int,
    total: int = 100,
    *,
    run_kind: str = "full",
    max_problems: int | None = None,
    checkpoint_identity: str = _CHECKPOINT_IDENTITY,
    tokenizer_identity: str = _TOKENIZER_IDENTITY,
    renderer_identity: str = _RENDERER_IDENTITY,
) -> ChatCoreTaskResult:
    return ChatCoreTaskResult(
        task_name=cast(ChatCoreTaskName, task_name),
        status="completed",
        passed_count=passed,
        total_count=total,
        accuracy=passed / total,
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer_identity,
        renderer_identity=renderer_identity,
        run_kind=cast(ChatCoreRunKind, run_kind),
        max_problems=max_problems,
    )


def _complete_results(
    passed_by_task: dict[str, int],
    *,
    run_kind: str = "full",
    max_problems: int | None = None,
) -> tuple[ChatCoreTaskResult, ...]:
    return tuple(
        _task(
            task_name,
            passed_by_task[task_name],
            run_kind=run_kind,
            max_problems=max_problems,
        )
        for task_name in CHATCORE_TASK_ORDER
    )


def test_fixed_task_order_baselines_and_random_baseline_aggregates() -> None:
    assert CHATCORE_TASK_ORDER == (
        "ARC-Easy",
        "ARC-Challenge",
        "MMLU",
        "GSM8K",
        "HumanEval",
    )
    assert CHATCORE_CATEGORICAL_TASKS == CHATCORE_TASK_ORDER[:3]
    assert dict(CHATCORE_BASELINE_ACCURACIES) == {
        "ARC-Easy": 0.25,
        "ARC-Challenge": 0.25,
        "MMLU": 0.25,
        "GSM8K": 0.0,
        "HumanEval": 0.0,
    }

    result = ChatCoreResult(
        _complete_results(
            {
                "ARC-Easy": 25,
                "ARC-Challenge": 25,
                "MMLU": 25,
                "GSM8K": 0,
                "HumanEval": 0,
            }
        )
    )

    assert result.complete is True
    assert result.missing_tasks == ()
    assert result.chatcore_metric == 0.0
    assert result.chatcore_cat == 0.0


def test_perfect_below_baseline_and_mixed_scores_use_unclipped_centering() -> None:
    perfect = ChatCoreResult(
        _complete_results({task_name: 100 for task_name in CHATCORE_TASK_ORDER})
    )
    below = ChatCoreResult(
        _complete_results({task_name: 0 for task_name in CHATCORE_TASK_ORDER})
    )
    mixed = ChatCoreResult(
        _complete_results(
            {
                "ARC-Easy": 50,
                "ARC-Challenge": 25,
                "MMLU": 10,
                "GSM8K": 50,
                "HumanEval": 100,
            }
        )
    )

    assert perfect.chatcore_metric == 1.0
    assert perfect.chatcore_cat == 1.0
    assert below.chatcore_metric == pytest.approx(-0.2)
    assert below.chatcore_cat == pytest.approx(-1 / 3)
    assert mixed.tasks[0].centered_score == pytest.approx(1 / 3)
    assert mixed.tasks[2].centered_score == pytest.approx(-0.2)
    assert mixed.chatcore_metric == pytest.approx(49 / 150)
    assert mixed.chatcore_cat == pytest.approx(2 / 45)


def test_input_order_is_canonical_and_result_is_frozen() -> None:
    tasks = _complete_results({task_name: 50 for task_name in CHATCORE_TASK_ORDER})

    ordered = ChatCoreResult(tasks)
    reversed_input = ChatCoreResult(tuple(reversed(tasks)))

    assert reversed_input == ordered
    assert tuple(task.task_name for task in reversed_input.tasks) == CHATCORE_TASK_ORDER
    assert reversed_input.to_dict() == ordered.to_dict()
    with pytest.raises(FrozenInstanceError):
        reversed_input.tasks = ()  # type: ignore[misc]


def test_complete_bounded_result_is_explicitly_labeled_and_serializes_exactly() -> None:
    result = ChatCoreResult(
        _complete_results(
            {task_name: 5 for task_name in CHATCORE_TASK_ORDER},
            run_kind="bounded",
            max_problems=10,
        )
    )

    payload = result.to_dict()

    assert payload["scope"] == {
        "bounded": True,
        "max_problems": 10,
        "run_kind": "bounded",
        "task_count": 5,
    }
    assert payload["complete"] is True
    assert ChatCoreResult.from_dict(json.loads(json.dumps(payload))) == result


def test_missing_tasks_produce_partial_result_with_no_aggregate() -> None:
    result = ChatCoreResult(
        tuple(
            _task(task_name, 5, 10, run_kind="bounded", max_problems=10)
            for task_name in CHATCORE_CATEGORICAL_TASKS
        )
    )

    assert result.complete is False
    assert result.missing_tasks == ("GSM8K", "HumanEval")
    assert result.chatcore_metric is None
    assert result.chatcore_cat is None
    assert result.to_dict()["chatcore_metric"] is None
    assert result.to_dict()["chatcore_cat"] is None
    assert result.to_dict()["complete"] is False
    assert ChatCoreResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"task_name": "unknown"}, "canonical task name"),
        ({"status": "failed"}, "failed task"),
        ({"total_count": 0, "accuracy": 0.0}, "total_count must be positive"),
        ({"accuracy": 0.4}, "accuracy must equal passed_count / total_count"),
        ({"accuracy": math.nan}, "accuracy must be finite"),
        ({"accuracy": math.inf}, "accuracy must be finite"),
    ],
)
def test_invalid_raw_task_results_are_rejected(
    change: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ChatCoreEvaluationError, match=match):
        replace(_task("ARC-Easy", 50), **change)


def test_duplicate_tasks_and_mismatched_execution_contexts_are_rejected() -> None:
    task = _task("ARC-Easy", 50)
    with pytest.raises(ChatCoreEvaluationError, match="duplicate"):
        ChatCoreResult((task, task))

    mismatches = (
        replace(task, task_name="MMLU", checkpoint_identity="other-checkpoint"),
        replace(task, task_name="MMLU", tokenizer_identity="other-tokenizer"),
        replace(task, task_name="MMLU", renderer_identity="other-renderer"),
        replace(task, task_name="MMLU", run_kind="bounded", max_problems=10),
    )
    for mismatch in mismatches:
        with pytest.raises(ChatCoreEvaluationError, match="same execution context"):
            ChatCoreResult((task, mismatch))

    bounded = _task("ARC-Easy", 5, 10, run_kind="bounded", max_problems=10)
    with pytest.raises(ChatCoreEvaluationError, match="same execution context"):
        ChatCoreResult(
            (
                bounded,
                replace(bounded, task_name="MMLU", max_problems=20),
            )
        )


def test_serialized_results_reject_derived_or_schema_tampering() -> None:
    result = ChatCoreResult(
        _complete_results({task_name: 50 for task_name in CHATCORE_TASK_ORDER})
    )
    payload = result.to_dict()
    payload["chatcore_metric"] = 0.0
    with pytest.raises(ChatCoreEvaluationError, match="derived values"):
        ChatCoreResult.from_dict(payload)

    payload = result.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ChatCoreEvaluationError, match="fields do not match"):
        ChatCoreResult.from_dict(payload)

    payload = result.to_dict()
    payload["complete"] = 1
    with pytest.raises(ChatCoreEvaluationError, match="derived values"):
        ChatCoreResult.from_dict(payload)
