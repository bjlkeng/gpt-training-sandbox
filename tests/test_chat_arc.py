"""Tests for strict, offline ARC chat-evaluation adapters."""

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
from scratch_llm.evaluation.chat.arc import (
    ARC_SHUFFLE_SEED,
    ArcDatasetRowError,
    build_arc_task,
    get_arc_dataset_spec,
    load_arc_task,
    normalize_arc_row,
)
from scratch_llm.evaluation.chat.categorical import evaluate_categorical_task
from scratch_llm.tokenization.tokenizer import ByteTokenizer


class _PreferAModel(nn.Module):
    max_seq_len = 512

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*token_ids.shape, 265), device=token_ids.device)
        logits[..., ord("A")] = 1.0
        return logits


def _row(
    index: int,
    *,
    labels: list[str] | None = None,
    answer: str | None = None,
) -> dict[str, object]:
    active_labels = labels or ["A", "B", "C", "D"]
    return {
        "answerKey": answer or active_labels[index % len(active_labels)],
        "choices": {
            "label": active_labels,
            "text": [f"choice {index}-{label}" for label in active_labels],
        },
        "question": f"question {index}?",
    }


def _cache(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    task_name: str = "ARC-Easy",
):
    parquet_path = tmp_path / "arc.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return publish_local_parquet_cache(
        get_arc_dataset_spec(task_name),
        tmp_path / "cache",
        (parquet_path,),
    )


def test_arc_specs_pin_named_test_subsets_and_required_columns() -> None:
    easy = get_arc_dataset_spec("ARC-Easy")
    challenge = get_arc_dataset_spec("ARC-Challenge")

    assert easy.repository == challenge.repository == "allenai/ai2_arc"
    assert easy.subset == "ARC-Easy"
    assert challenge.subset == "ARC-Challenge"
    assert easy.split == challenge.split == "test"
    assert easy.reference_commit == ("92d63d4e8bb4df75c3b71618f31ddde2378b2bcd")
    assert easy.required_columns == ("question", "choices", "answerKey")


def test_normalize_arc_row_builds_user_only_non_four_choice_prompt() -> None:
    example = normalize_arc_row(
        _row(1, labels=["A", "B", "C"], answer="B"),
        source_row=8,
        source_identity="source",
        context="fixture row 8",
    )

    assert example.labels == ("A", "B", "C")
    assert example.answer == "B"
    assert example.source_row == 8
    assert len(example.conversation.messages) == 1
    message = example.conversation.messages[0]
    assert isinstance(message, UserMessage)
    assert message.content == (
        "Multiple Choice question: question 1?\n"
        "- choice 1-A=A\n"
        "- choice 1-B=B\n"
        "- choice 1-C=C\n"
        "\nRespond only with the letter of the correct answer."
    )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({**_row(0), "question": "  "}, "question must be non-empty"),
        ({**_row(0), "choices": []}, "choices must be an object"),
        (
            {
                **_row(0),
                "choices": {"label": ["A", "B"], "text": ["only one"]},
            },
            "same non-zero length",
        ),
        (
            {
                **_row(0),
                "choices": {"label": ["A", "A"], "text": ["one", "two"]},
            },
            "labels must be unique",
        ),
        (
            {
                **_row(0),
                "choices": {"label": ["A", 2], "text": ["one", "two"]},
            },
            "choices.label\\[1\\] must be a string",
        ),
        (
            {
                **_row(0),
                "choices": {"label": ["A", "B"], "text": ["one", 2]},
            },
            "choices.text\\[1\\] must be a string",
        ),
        ({**_row(0), "answerKey": "Z"}, "answerKey must be present"),
    ],
)
def test_normalize_arc_row_rejects_invalid_source_rows(
    row: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ArcDatasetRowError, match=message):
        normalize_arc_row(
            row,
            source_row=0,
            source_identity="source",
            context="fixture row 0",
        )


def test_arc_task_matches_seed_42_numpy_order_and_loads_only_from_cache(
    tmp_path: Path,
) -> None:
    rows = [_row(index) for index in range(12)]
    cache = _cache(tmp_path, rows)

    first = build_arc_task(cache)
    second = load_arc_task(tmp_path / "cache", "ARC-Easy")

    expected_order = tuple(
        int(index) for index in np.random.default_rng(ARC_SHUFFLE_SEED).permutation(12)
    )
    assert tuple(example.source_row for example in first.examples) == expected_order
    assert second == first
    assert first.name == "ARC-Easy"
    assert first.source_identity == cache.spec.source_identity
    assert first.dataset_identity == cache.source_identity
    assert first.order_identity.startswith("sha256:")


def test_arc_cached_fixture_runs_through_the_shared_categorical_evaluator(
    tmp_path: Path,
) -> None:
    task = build_arc_task(
        _cache(tmp_path, [_row(index, answer="A") for index in range(4)])
    )

    result = evaluate_categorical_task(
        _PreferAModel(),
        ByteTokenizer(),
        task,
        checkpoint_identity="checkpoint",
        batch_size=2,
        max_problems=3,
        device="cpu",
        clock=iter((5.0, 7.0)).__next__,
    )

    assert result.passed_count == result.evaluated_count == 3
    assert result.available_count == 4
    assert result.elapsed_seconds == 2.0
    assert result.dataset_identity == task.dataset_identity
    assert result.order_identity == task.order_identity
