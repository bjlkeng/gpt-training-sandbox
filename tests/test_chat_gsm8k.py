"""Tests for the offline GSM8K generative chat-evaluation adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from torch import nn

from scratch_llm.chat.conversation import UserMessage
from scratch_llm.data.hub import publish_local_parquet_cache
from scratch_llm.data.sft_sources import get_sft_dataset_spec
from scratch_llm.evaluation.chat.generative import (
    GenerativeEvaluationConfig,
    evaluate_generative_task,
)
from scratch_llm.evaluation.chat.gsm8k import (
    GSM8K_SHUFFLE_SEED,
    GSM8KDatasetRowError,
    GSM8KProblem,
    build_gsm8k_task,
    extract_gsm8k_answer,
    get_gsm8k_dataset_spec,
    load_gsm8k_task,
    normalize_gsm8k_eval_row,
    score_gsm8k_completion,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer


class _AnswerModel(nn.Module):
    max_seq_len = 512

    def __init__(self, tokenizer: ByteTokenizer) -> None:
        super().__init__()
        self.assistant_start = tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = tokenizer.encode_special("<|assistant_end|>")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*token_ids.shape, 265),
            -torch.inf,
            device=token_ids.device,
        )
        sequence = (*b"#### 2", self.assistant_end)
        for row, values in enumerate(token_ids.detach().cpu().tolist()):
            start = len(values) - 1 - values[::-1].index(self.assistant_start)
            generated_count = len(values) - start - 1
            logits[row, -1, sequence[generated_count]] = 0
        return logits


def _row(index: int, *, final_answer: str | None = None) -> dict[str, str]:
    answer = final_answer or str(index)
    return {
        "answer": f"Reasoning with <<1+1=2>>.\n#### {answer}",
        "question": f"What is problem {index}?",
    }


def _cache(tmp_path: Path, rows: list[dict[str, str]]):
    parquet_path = tmp_path / "gsm8k.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return publish_local_parquet_cache(
        get_gsm8k_dataset_spec(),
        tmp_path / "cache",
        (parquet_path,),
    )


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("work\n#### 1,234.50", "1234.50"),
        ("work\n#### -2.75", "-2.75"),
        ("#### 7 and later #### 8", "7"),
        ("no final marker", None),
        ("####    2", None),
    ],
)
def test_extract_gsm8k_answer_matches_the_pinned_marker_rule(
    completion: str,
    expected: str | None,
) -> None:
    assert extract_gsm8k_answer(completion) == expected


def test_normalize_gsm8k_eval_row_builds_user_only_problem_and_reference() -> None:
    problem = normalize_gsm8k_eval_row(
        _row(2, final_answer="-1,234.50"),
        source_row=4,
        source_identity="source",
        context="fixture row 4",
    )

    assert isinstance(problem, GSM8KProblem)
    assert problem.reference_answer == "-1234.50"
    assert problem.source_row == 4
    assert len(problem.conversation.messages) == 1
    message = problem.conversation.messages[0]
    assert isinstance(message, UserMessage)
    assert message.content == "What is problem 2?"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"question": "", "answer": "#### 2"}, "question must be non-empty"),
        ({"question": "valid", "answer": 2}, "answer must be a string"),
        (
            {"question": "valid", "answer": "no marker"},
            "answer must contain a valid #### numeric marker",
        ),
    ],
)
def test_normalize_gsm8k_eval_row_rejects_invalid_rows(
    row: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(GSM8KDatasetRowError, match=message):
        normalize_gsm8k_eval_row(
            row,
            source_row=0,
            source_identity="source",
            context="fixture row 0",
        )


def test_gsm8k_uses_the_existing_test_cache_and_seed_42_order(tmp_path: Path) -> None:
    assert get_gsm8k_dataset_spec() == get_sft_dataset_spec("gsm8k", "test")
    rows = [_row(index) for index in range(12)]
    cache = _cache(tmp_path, rows)

    first = build_gsm8k_task(cache)
    second = load_gsm8k_task(tmp_path / "cache")

    expected_order = tuple(
        int(index)
        for index in np.random.default_rng(GSM8K_SHUFFLE_SEED).permutation(12)
    )
    assert tuple(problem.source_row for problem in first.problems) == expected_order
    assert second == first
    assert first.name == "GSM8K"
    assert first.source_identity == cache.spec.source_identity
    assert first.dataset_identity == cache.source_identity


def test_gsm8k_scorer_distinguishes_correct_incorrect_and_missing_answers() -> None:
    problem = normalize_gsm8k_eval_row(
        _row(2, final_answer="1,234.5"),
        source_row=0,
        source_identity="source",
        context="fixture row 0",
    )

    assert score_gsm8k_completion(problem, "work\n#### 1,234.5")
    assert not score_gsm8k_completion(problem, "work\n#### 1234.50")
    assert not score_gsm8k_completion(problem, "no marker")


def test_gsm8k_cached_fixture_runs_through_shared_generation(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    task = build_gsm8k_task(
        _cache(
            tmp_path,
            [_row(0, final_answer="2"), _row(1, final_answer="3")],
        )
    )

    result = evaluate_generative_task(
        _AnswerModel(tokenizer),
        tokenizer,
        task,
        score_gsm8k_completion,
        checkpoint_identity="checkpoint",
        config=GenerativeEvaluationConfig(
            num_samples=2,
            max_new_tokens=7,
            temperature=0,
            seed=9,
        ),
        max_problems=None,
        device="cpu",
    )

    assert result.passed_count == 1
    assert result.evaluated_count == 2
    assert result.total_sample_count == 4
    assert result.stop_counts["assistant_end"] == 4
