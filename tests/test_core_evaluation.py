"""Tests for the pinned nanochat CORE evaluation domain."""

from __future__ import annotations

import pytest

from scratch_llm.core_evaluation import centered_core_score
from scratch_llm.core_evaluation import (
    CoreEvaluationResult,
    CoreReferenceComparison,
    CoreTaskResult,
)


@pytest.mark.parametrize(
    ("accuracy", "random_baseline_percent", "expected"),
    [
        (0.25, 25.0, 0.0),
        (0.5, 25.0, 1.0 / 3.0),
        (0.125, 0.0, 0.125),
    ],
)
def test_centered_core_score_matches_the_pinned_formula(
    accuracy: float,
    random_baseline_percent: float,
    expected: float,
) -> None:
    assert centered_core_score(accuracy, random_baseline_percent) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("accuracy", "random_baseline_percent"),
    [(-0.1, 25.0), (1.1, 25.0), (0.5, -1.0), (0.5, 100.0)],
)
def test_centered_core_score_rejects_invalid_inputs(
    accuracy: float,
    random_baseline_percent: float,
) -> None:
    with pytest.raises(ValueError):
        centered_core_score(accuracy, random_baseline_percent)


def _task_result(label: str, *, correct: int, baseline: float) -> CoreTaskResult:
    return CoreTaskResult(
        label=label,
        task_type="multiple_choice",
        num_fewshot=0,
        random_baseline_percent=baseline,
        correct_examples=correct,
        evaluated_examples=4,
        available_examples=8,
        elapsed_seconds=1.5,
        data_identity=f"sha256:{label:0<64}"[:71],
    )


def test_core_result_derives_task_accuracy_centering_and_aggregate() -> None:
    first = _task_result("first", correct=2, baseline=25.0)
    second = _task_result("second", correct=3, baseline=50.0)
    result = CoreEvaluationResult(
        checkpoint_identity="sha256:" + "1" * 64,
        tokenizer_identity="sha256:" + "2" * 64,
        bundle_identity="sha256:" + "3" * 64,
        config_identity="sha256:" + "4" * 64,
        metadata_identity="sha256:" + "5" * 64,
        run_kind="full",
        max_per_task=None,
        tasks=(first, second),
        references=(CoreReferenceComparison("reference", 0.2),),
        elapsed_seconds=3.0,
    )

    assert first.accuracy == 0.5
    assert first.centered_score == pytest.approx(1.0 / 3.0)
    assert second.centered_score == 0.5
    assert result.core_metric == pytest.approx(5.0 / 12.0)
    payload = result.to_dict()
    assert payload["task_order"] == ["first", "second"]
    assert payload["comparison"]["comparable"] is True
    assert payload["comparison"]["references"]["reference"]["delta"] == pytest.approx(
        result.core_metric - 0.2
    )


def test_bounded_core_result_has_no_reference_delta() -> None:
    result = CoreEvaluationResult(
        checkpoint_identity="checkpoint",
        tokenizer_identity="tokenizer",
        bundle_identity="bundle",
        config_identity="config",
        metadata_identity="metadata",
        run_kind="bounded",
        max_per_task=4,
        tasks=(_task_result("first", correct=1, baseline=25.0),),
        references=(CoreReferenceComparison("reference", 0.2),),
        elapsed_seconds=1.0,
    )

    comparison = result.to_dict()["comparison"]
    assert comparison["comparable"] is False
    assert comparison["references"]["reference"]["delta"] is None
