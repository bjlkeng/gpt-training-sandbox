"""Tests for the immutable, deterministic chat-evaluation report."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from scratch_llm.chat.rendering import CHAT_RENDERER_ID
from scratch_llm.evaluation.chat.categorical import CategoricalTaskResult
from scratch_llm.evaluation.chat.diagnostics import (
    CodePromptDiagnostic,
    FixedSFTDiagnostics,
    JSONPromptDiagnostic,
)
from scratch_llm.evaluation.chat.generative import (
    GenerativeEvaluationConfig,
    GenerativeProblemResult,
    GenerativeSampleResult,
    GenerativeTaskResult,
)
from scratch_llm.evaluation.chat.reporting import (
    CHAT_EVALUATION_REPORT_FORMAT,
    CHAT_EVALUATION_REPORT_FORMAT_VERSION,
    ChatEvaluationError,
    ChatEvaluationReportConflictError,
    ChatEvaluationSettings,
    CompletedChatEvaluation,
    write_chat_evaluation_report,
)
from scratch_llm.evaluation.sft_sampling import FixedSFTSamplingConfig
from scratch_llm.evaluation.sft_sampling import FIXED_SFT_PROMPT_SET_IDENTITY


_CHECKPOINT = "sha256:" + "1" * 64
_TOKENIZER = "sha256:" + "2" * 64


def _categorical(
    task_name: str, *, elapsed_seconds: float = 1.0
) -> CategoricalTaskResult:
    return CategoricalTaskResult(
        task_name=task_name,
        checkpoint_identity=_CHECKPOINT,
        tokenizer_identity=_TOKENIZER,
        source_identity=f"source:{task_name}",
        dataset_identity=f"dataset:{task_name}",
        order_identity=f"order:{task_name}",
        run_kind="full",
        max_problems=None,
        passed_count=1,
        evaluated_count=2,
        available_count=2,
        elapsed_seconds=elapsed_seconds,
    )


def _generative(task_name: str, *, passed: bool) -> GenerativeTaskResult:
    outcome = "passed" if task_name == "HumanEval" and passed else None
    if task_name == "HumanEval" and not passed:
        outcome = "test_failure"
    sample = GenerativeSampleResult(
        problem_index=0,
        sample_index=0,
        seed=7,
        passed=passed,
        generated_token_count=1,
        sampled_token_count=1,
        completion_reason="max_new_tokens",
        stop_token_id=None,
        completion_identity=f"completion:{task_name}",
        score_outcome=outcome,
    )
    problem = GenerativeProblemResult(
        problem_index=0,
        problem_identity=f"problem:{task_name}",
        source_row=3,
        passed=passed,
        samples=(sample,),
    )
    return GenerativeTaskResult(
        task_name=task_name,
        checkpoint_identity=_CHECKPOINT,
        tokenizer_identity=_TOKENIZER,
        source_identity=f"source:{task_name}",
        dataset_identity=f"dataset:{task_name}",
        order_identity=f"order:{task_name}",
        run_kind="full",
        max_problems=None,
        available_count=1,
        assistant_end_token_id=263,
        bos_token_id=264,
        config=GenerativeEvaluationConfig(
            num_samples=1,
            max_new_tokens=1,
            temperature=0,
            top_k=1,
            seed=7,
        ),
        problems=(problem,),
        scoring_identity=("executor:v1" if task_name == "HumanEval" else None),
    )


def _diagnostics() -> FixedSFTDiagnostics:
    return FixedSFTDiagnostics(
        checkpoint_identity=_CHECKPOINT,
        tokenizer_identity=_TOKENIZER,
        renderer_identity=CHAT_RENDERER_ID,
        prompt_set_identity=FIXED_SFT_PROMPT_SET_IDENTITY,
        generation_identity="fixed-generation:v1",
        sample_count=5,
        assistant_end_stop_count=4,
        bos_safety_stop_count=0,
        max_token_count=1,
        visible_token_mean=2.0,
        visible_token_min=1,
        visible_token_max=3,
        empty_response_count=0,
        json_prompt=JSONPromptDiagnostic(True, True, True, True, True, True),
        code_prompt=CodePromptDiagnostic("plain_code", 0),
    )


def _settings(
    *tasks: str,
    max_problems: int | None = None,
) -> ChatEvaluationSettings:
    return ChatEvaluationSettings(
        task_names=tasks,
        batch_size=2,
        max_problems=max_problems,
        generation=GenerativeEvaluationConfig(
            num_samples=1,
            max_new_tokens=1,
            temperature=0,
            top_k=1,
            seed=7,
        ),
        fixed_sampling=FixedSFTSamplingConfig(
            max_new_tokens=1,
            temperature=0,
            top_k=1,
            seed=7,
        ),
        allow_generated_code_execution="HumanEval" in tasks,
        executor_identity=("executor:v1" if "HumanEval" in tasks else None),
    )


def _completed(*, categorical_elapsed: float = 1.0) -> CompletedChatEvaluation:
    tasks = (
        _categorical("ARC-Easy", elapsed_seconds=categorical_elapsed),
        _categorical("ARC-Challenge"),
        _categorical("MMLU"),
        _generative("GSM8K", passed=True),
        _generative("HumanEval", passed=False),
    )
    return CompletedChatEvaluation(
        config_identity="config:v1",
        checkpoint_identity=_CHECKPOINT,
        checkpoint_step=11,
        tokenizer_identity=_TOKENIZER,
        settings=_settings(*(task.task_name for task in tasks)),
        task_results=tasks,
        diagnostics=_diagnostics(),
    )


def test_full_report_is_canonical_content_free_and_derived_from_exact_counts() -> None:
    completed = _completed()

    payload = completed.to_dict()

    assert completed.chatcore_metric == pytest.approx(2 / 5)
    assert completed.chatcore_cat == pytest.approx(1 / 3)
    assert payload["format"] == CHAT_EVALUATION_REPORT_FORMAT
    assert payload["format_version"] == CHAT_EVALUATION_REPORT_FORMAT_VERSION
    assert payload["status"] == "completed"
    assert payload["scope"] == {
        "bounded": False,
        "full": True,
        "kind": "full",
        "max_problems": None,
        "missing_tasks": [],
        "selected_tasks": [
            "ARC-Easy",
            "ARC-Challenge",
            "MMLU",
            "GSM8K",
            "HumanEval",
        ],
        "task_count": 5,
    }
    assert payload["chatcore"]["chatcore_metric"] == pytest.approx(2 / 5)
    assert payload["chatcore"]["chatcore_cat"] == pytest.approx(1 / 3)
    assert [task["score"]["task_name"] for task in payload["tasks"]] == [
        "ARC-Easy",
        "ARC-Challenge",
        "MMLU",
        "GSM8K",
        "HumanEval",
    ]
    assert payload["tasks"][0]["score"]["passed_count"] == 1
    assert payload["tasks"][0]["score"]["baseline_accuracy"] == 0.25
    assert payload["tasks"][0]["score"]["centered_score"] == pytest.approx(1 / 3)
    assert payload["tasks"][3]["evaluation_type"] == "generative"
    assert payload["tasks"][4]["evaluation_type"] == "code_execution"
    serialized = json.dumps(payload, sort_keys=True)
    assert "elapsed_seconds" not in serialized
    assert "Explain gradient descent" not in serialized
    assert "completion:" in serialized  # identities are retained, content is not.
    assert payload["result_identity"].startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        completed.checkpoint_step = 12  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tasks", "max_problems", "kind"),
    [
        (("ARC-Easy", "MMLU"), None, "partial"),
        (
            ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"),
            10,
            "bounded",
        ),
        (("ARC-Easy", "MMLU"), 10, "bounded_partial"),
    ],
)
def test_non_full_scope_is_explicit_and_never_publishes_full_aggregates(
    tasks: tuple[str, ...],
    max_problems: int | None,
    kind: str,
) -> None:
    raw = []
    for task_name in tasks:
        result = (
            _categorical(task_name)
            if task_name in {"ARC-Easy", "ARC-Challenge", "MMLU"}
            else _generative(task_name, passed=True)
        )
        if max_problems is not None:
            result = replace(result, run_kind="bounded", max_problems=max_problems)
        raw.append(result)
    completed = CompletedChatEvaluation(
        config_identity="config:v1",
        checkpoint_identity=_CHECKPOINT,
        checkpoint_step=11,
        tokenizer_identity=_TOKENIZER,
        settings=_settings(*tasks, max_problems=max_problems),
        task_results=tuple(raw),
        diagnostics=_diagnostics(),
    )

    payload = completed.to_dict()

    assert completed.chatcore_metric is None
    assert completed.chatcore_cat is None
    assert payload["scope"]["kind"] == kind
    assert payload["scope"]["full"] is False
    assert payload["chatcore"]["chatcore_metric"] is None
    assert payload["chatcore"]["chatcore_cat"] is None


def test_completed_result_rejects_mismatched_context_scope_and_code_consent() -> None:
    completed = _completed()

    with pytest.raises(ChatEvaluationError, match="checkpoint identity"):
        replace(
            completed,
            task_results=(
                replace(completed.task_results[0], checkpoint_identity="other"),
                *completed.task_results[1:],
            ),
        )
    with pytest.raises(ChatEvaluationError, match="requested task order"):
        replace(completed, task_results=tuple(reversed(completed.task_results)))
    with pytest.raises(ChatEvaluationError, match="categorical evaluation"):
        replace(
            completed,
            task_results=(
                _generative("ARC-Easy", passed=False),
                *completed.task_results[1:],
            ),
        )
    with pytest.raises(ChatEvaluationError, match="generation settings"):
        replace(
            completed,
            task_results=(
                *completed.task_results[:3],
                replace(
                    completed.task_results[3],
                    config=GenerativeEvaluationConfig(seed=99),
                ),
                completed.task_results[4],
            ),
        )
    with pytest.raises(ChatEvaluationError, match="canonical fixed SFT prompt set"):
        replace(
            completed,
            diagnostics=replace(
                completed.diagnostics,
                prompt_set_identity="different-prompt-set",
            ),
        )
    with pytest.raises(ChatEvaluationError, match="explicit consent"):
        replace(
            completed,
            settings=replace(
                completed.settings,
                allow_generated_code_execution=False,
                executor_identity=None,
            ),
        )
    with pytest.raises(ChatEvaluationError, match="require HumanEval"):
        replace(
            completed.settings,
            task_names=("ARC-Easy",),
        )


def test_report_write_is_idempotent_and_never_overwrites_a_conflict(
    tmp_path: Path,
) -> None:
    first = _completed(categorical_elapsed=1.0)
    repeated = _completed(categorical_elapsed=99.0)

    path = write_chat_evaluation_report(first, run_dir=tmp_path)
    original = path.read_bytes()
    assert write_chat_evaluation_report(repeated, run_dir=tmp_path) == path
    assert path.read_bytes() == original

    conflicting = replace(first, checkpoint_step=12)
    with pytest.raises(ChatEvaluationReportConflictError, match="different"):
        write_chat_evaluation_report(conflicting, run_dir=tmp_path)
    assert path.read_bytes() == original

    path.write_text("not JSON\n", encoding="utf-8")
    with pytest.raises(ChatEvaluationReportConflictError, match="valid JSON"):
        write_chat_evaluation_report(first, run_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "not JSON\n"
