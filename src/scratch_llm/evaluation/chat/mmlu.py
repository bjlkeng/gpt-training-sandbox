"""Strict offline MMLU adapter for categorical chat evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np

from scratch_llm._validation import (
    require_non_empty_string,
    require_non_negative_integer,
)
from scratch_llm.chat.conversation import Conversation, UserMessage
from scratch_llm.chat.rendering import render_multiple_choice_prompt
from scratch_llm.data.hub import (
    CachedHubParquetDataset,
    HubDatasetSpec,
    load_hub_parquet_cache,
)
from scratch_llm.data.sft_sources import get_sft_dataset_spec
from scratch_llm.evaluation.chat.cache import read_cached_parquet_rows
from scratch_llm.evaluation.chat.categorical import (
    CategoricalExample,
    CategoricalTask,
)
from scratch_llm.identity import canonical_json_identity


MMLU_TASK_NAME: Final = "MMLU"
MMLU_SHUFFLE_SEED: Final = 42
MMLU_LETTERS: Final = ("A", "B", "C", "D")
MMLU_REFERENCE_FILE_SHA256: Final = (
    "0b00ada4ad4bc7675f6e36a1e8770b19ae17c12daedbf7c88965228902421a66"
)


class MMLUDatasetError(ValueError):
    """The MMLU task source or cached view is invalid."""


class MMLUDatasetRowError(MMLUDatasetError):
    """One MMLU row cannot be normalized for categorical evaluation."""


def get_mmlu_dataset_spec() -> HubDatasetSpec:
    """Reuse the pinned cais/mmlu all/test cache contract."""

    return get_sft_dataset_spec("mmlu", "test")


def normalize_mmlu_eval_row(
    row: Mapping[str, object],
    *,
    source_row: int,
    source_identity: str,
    context: str,
) -> CategoricalExample:
    """Normalize one strict MMLU row without its gold assistant message."""

    try:
        if not isinstance(row, Mapping):
            raise MMLUDatasetRowError("row must be an object")
        question = _non_empty_string(row.get("question"), label="question")
        raw_choices = row.get("choices")
        if not isinstance(raw_choices, (list, tuple)) or len(raw_choices) != 4:
            raise MMLUDatasetRowError("choices must contain exactly four choices")
        choices = tuple(
            _string(choice, label=f"choices[{index}]")
            for index, choice in enumerate(raw_choices)
        )
        raw_answer = row.get("answer")
        if (
            not isinstance(raw_answer, int)
            or isinstance(raw_answer, bool)
            or not 0 <= raw_answer < len(MMLU_LETTERS)
        ):
            raise MMLUDatasetRowError("answer must be an integer in [0, 3]")
        subject = _non_empty_string(row.get("subject"), label="subject")
        try:
            source_row = require_non_negative_integer(
                source_row,
                name="source_row",
            )
            source_identity = require_non_empty_string(
                source_identity,
                name="source_identity",
            )
        except (TypeError, ValueError) as error:
            raise MMLUDatasetRowError(str(error)) from error

        answer = MMLU_LETTERS[raw_answer]
        prompt = render_multiple_choice_prompt(question, MMLU_LETTERS, choices)
        identity = canonical_json_identity(
            {
                "answer": answer,
                "labels": list(MMLU_LETTERS),
                "prompt": prompt,
                "source_identity": source_identity,
                "source_row": source_row,
                "subject": subject,
            }
        )
        return CategoricalExample(
            conversation=Conversation(messages=(UserMessage(prompt),)),
            labels=MMLU_LETTERS,
            answer=answer,
            source_row=source_row,
            identity=identity,
            group=subject,
        )
    except MMLUDatasetRowError as error:
        raise _with_context(context, error) from error


def build_mmlu_task(cache: CachedHubParquetDataset) -> CategoricalTask:
    """Materialize deterministic seed-42 MMLU examples from a verified cache."""

    if not isinstance(cache, CachedHubParquetDataset):
        raise TypeError("cache must be a CachedHubParquetDataset")
    expected_spec = get_mmlu_dataset_spec()
    if cache.spec != expected_spec:
        raise MMLUDatasetError(
            "cache spec does not match the pinned MMLU all/test contract"
        )
    rows = read_cached_parquet_rows(cache, additional_columns=("subject",))
    permutation = tuple(
        int(index)
        for index in np.random.default_rng(MMLU_SHUFFLE_SEED).permutation(len(rows))
    )
    examples = tuple(
        normalize_mmlu_eval_row(
            rows[source_row],
            source_row=source_row,
            source_identity=cache.source_identity,
            context=(
                f"{cache.spec.repository}/{cache.spec.subset}/"
                f"{cache.spec.split} row {source_row}"
            ),
        )
        for source_row in permutation
    )
    order_identity = canonical_json_identity(
        {
            "dataset_identity": cache.source_identity,
            "example_identities": [example.identity for example in examples],
            "reference_file_sha256": MMLU_REFERENCE_FILE_SHA256,
            "seed": MMLU_SHUFFLE_SEED,
        }
    )
    return CategoricalTask(
        name=MMLU_TASK_NAME,
        examples=examples,
        source_identity=cache.spec.source_identity,
        dataset_identity=cache.source_identity,
        order_identity=order_identity,
    )


def load_mmlu_task(cache_root: str | Path) -> CategoricalTask:
    """Load the prepared MMLU test cache without a network fallback."""

    spec = get_mmlu_dataset_spec()
    return build_mmlu_task(load_hub_parquet_cache(spec, cache_root))


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise MMLUDatasetRowError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    return value


def _non_empty_string(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if not text.strip():
        raise MMLUDatasetRowError(f"{label} must be non-empty")
    return text


def _with_context(context: str, error: Exception) -> MMLUDatasetRowError:
    try:
        context = require_non_empty_string(context, name="context")
    except (TypeError, ValueError) as context_error:
        raise ValueError(str(context_error)) from context_error
    return MMLUDatasetRowError(f"{context}: {error}")


__all__ = [
    "MMLU_LETTERS",
    "MMLU_REFERENCE_FILE_SHA256",
    "MMLU_SHUFFLE_SEED",
    "MMLU_TASK_NAME",
    "MMLUDatasetError",
    "MMLUDatasetRowError",
    "build_mmlu_task",
    "get_mmlu_dataset_spec",
    "load_mmlu_task",
    "normalize_mmlu_eval_row",
]
