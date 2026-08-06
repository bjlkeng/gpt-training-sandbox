"""Tests for the offline MMLU categorical chat-evaluation adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from torch import nn

from scratch_llm.chat.conversation import AssistantMessage, UserMessage
from scratch_llm.data.hub import publish_local_parquet_cache
from scratch_llm.data.sft_sources import (
    get_sft_dataset_spec,
    normalize_mmlu_row,
)
from scratch_llm.evaluation.chat.categorical import evaluate_categorical_task
from scratch_llm.evaluation.chat.mmlu import (
    MMLU_SHUFFLE_SEED,
    MMLUDatasetRowError,
    build_mmlu_task,
    get_mmlu_dataset_spec,
    load_mmlu_task,
    normalize_mmlu_eval_row,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer


class _ConstrainedLetterModel(nn.Module):
    max_seq_len = 512

    def __init__(self, predictions: tuple[str, ...]) -> None:
        super().__init__()
        self.predictions = predictions

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert len(token_ids) == len(self.predictions)
        logits = torch.zeros((*token_ids.shape, 265), device=token_ids.device)
        logits[..., ord("Z")] = 100.0
        for row, prediction in enumerate(self.predictions):
            logits[row, :, ord(prediction)] = 10.0
        return logits


def _row(
    index: int,
    *,
    answer: int | None = None,
    subject: str | None = None,
) -> dict[str, object]:
    return {
        "answer": index % 4 if answer is None else answer,
        "choices": [f"choice {index}-{letter}" for letter in "ABCD"],
        "question": f"question {index}?",
        "subject": subject or ("subject_alpha" if index % 2 == 0 else "subject_beta"),
    }


def _cache(tmp_path: Path, rows: list[dict[str, object]]):
    parquet_path = tmp_path / "mmlu.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return publish_local_parquet_cache(
        get_mmlu_dataset_spec(),
        tmp_path / "cache",
        (parquet_path,),
    )


def test_mmlu_reuses_the_existing_pinned_test_cache_contract() -> None:
    spec = get_mmlu_dataset_spec()

    assert spec == get_sft_dataset_spec("mmlu", "test")
    assert spec.repository == "cais/mmlu"
    assert spec.subset == "all"
    assert spec.split == "test"
    assert spec.reference_commit == ("92d63d4e8bb4df75c3b71618f31ddde2378b2bcd")


def test_normalize_mmlu_eval_row_reuses_the_sft_prompt_without_gold_message() -> None:
    row = _row(2, answer=2, subject="college_biology")

    example = normalize_mmlu_eval_row(
        row,
        source_row=7,
        source_identity="source",
        context="fixture row 7",
    )
    sft_conversation = normalize_mmlu_row(row, context="fixture row 7")

    assert example.labels == ("A", "B", "C", "D")
    assert example.answer == "C"
    assert example.group == "college_biology"
    assert example.source_row == 7
    assert example.conversation.messages == (sft_conversation.messages[0],)
    assert isinstance(example.conversation.messages[0], UserMessage)
    assert isinstance(sft_conversation.messages[1], AssistantMessage)
    assert sft_conversation.messages[1].content == "C"
    assert "- choice 2-A=A\n" in example.conversation.messages[0].content
    assert "= A" not in example.conversation.messages[0].content


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({**_row(0), "question": " \t"}, "question must be non-empty"),
        ({**_row(0), "choices": ["a", "b", "c"]}, "exactly four choices"),
        (
            {**_row(0), "choices": ["a", 2, "c", "d"]},
            r"choices\[1\] must be a string",
        ),
        ({**_row(0), "answer": True}, "answer must be an integer in"),
        ({**_row(0), "answer": 4}, "answer must be an integer in"),
        ({**_row(0), "subject": ""}, "subject must be non-empty"),
        ({**_row(0), "subject": 3}, "subject must be a string"),
    ],
)
def test_normalize_mmlu_eval_row_rejects_malformed_rows(
    row: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(MMLUDatasetRowError, match=message):
        normalize_mmlu_eval_row(
            row,
            source_row=0,
            source_identity="source",
            context="fixture row 0",
        )


def test_mmlu_task_uses_seed_42_order_and_preserves_subjects(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(12)]
    cache = _cache(tmp_path, rows)

    first = build_mmlu_task(cache)
    repeated = load_mmlu_task(tmp_path / "cache")

    expected_order = tuple(
        int(index) for index in np.random.default_rng(MMLU_SHUFFLE_SEED).permutation(12)
    )
    assert tuple(example.source_row for example in first.examples) == expected_order
    assert tuple(example.group for example in first.examples) == tuple(
        rows[index]["subject"] for index in expected_order
    )
    assert repeated == first
    assert first.name == "MMLU"
    assert first.source_identity == cache.spec.source_identity
    assert first.dataset_identity == cache.source_identity
    assert first.order_identity.startswith("sha256:")


def test_mmlu_evaluation_constrains_letters_and_reconciles_subject_counts(
    tmp_path: Path,
) -> None:
    task = build_mmlu_task(_cache(tmp_path, [_row(index) for index in range(8)]))
    evaluated = task.examples[:6]
    predictions = [example.answer for example in evaluated]
    predictions[0] = "A" if predictions[0] != "A" else "B"

    result = evaluate_categorical_task(
        _ConstrainedLetterModel(tuple(predictions)),
        ByteTokenizer(),
        task,
        checkpoint_identity="checkpoint",
        batch_size=6,
        max_problems=6,
        device="cpu",
        clock=iter((4.0, 6.0)).__next__,
    )

    assert {example.answer for example in evaluated} == {"A", "B", "C", "D"}
    assert result.passed_count == 5
    assert result.evaluated_count == 6
    assert result.available_count == 8
    assert result.accuracy == 5 / 6
    assert result.run_kind == "bounded"
    assert [group.to_dict() for group in result.groups] == [
        {
            "accuracy": 1.0,
            "evaluated": 3,
            "name": "subject_alpha",
            "passed": 3,
        },
        {
            "accuracy": 2 / 3,
            "evaluated": 3,
            "name": "subject_beta",
            "passed": 2,
        },
    ]
    assert sum(group.passed_count for group in result.groups) == result.passed_count
    assert (
        sum(group.evaluated_count for group in result.groups) == result.evaluated_count
    )
    assert result.to_dict()["groups"] == [group.to_dict() for group in result.groups]
