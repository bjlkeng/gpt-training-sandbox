"""Strict cached-parquet adapters for ARC-Easy and ARC-Challenge."""

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
from scratch_llm.data.hub import (
    CachedHubParquetDataset,
    HubDatasetSpec,
    load_hub_parquet_cache,
)
from scratch_llm.evaluation.chat.cache import read_cached_parquet_rows
from scratch_llm.evaluation.chat.categorical import (
    CategoricalExample,
    CategoricalTask,
    render_multiple_choice_prompt,
)
from scratch_llm.evaluation.chat.protocol import CHAT_EVAL_REFERENCE_COMMIT
from scratch_llm.identity import canonical_json_identity


ARC_SHUFFLE_SEED: Final = 42
ARC_TASK_NAMES: Final = ("ARC-Easy", "ARC-Challenge")
ARC_REFERENCE_FILE_SHA256: Final = (
    "f00642888b36ddff8524603f63d417cd39fc36c3c206922cd676df51c10f1b19"
)
_ARC_SPEC_FIELDS: Final = {
    "ARC-Easy": {
        "adapter_version": "arc_easy_chat_v1",
        "dataset": "arc_easy",
        "subset": "ARC-Easy",
    },
    "ARC-Challenge": {
        "adapter_version": "arc_challenge_chat_v1",
        "dataset": "arc_challenge",
        "subset": "ARC-Challenge",
    },
}


class ArcDatasetError(ValueError):
    """An ARC task source or cached view is invalid."""


class ArcDatasetRowError(ArcDatasetError):
    """One ARC row cannot be normalized into a categorical prompt."""


def get_arc_dataset_spec(task_name: str) -> HubDatasetSpec:
    """Return one pinned ARC test-split cache contract."""

    if not isinstance(task_name, str) or task_name not in _ARC_SPEC_FIELDS:
        supported = ", ".join(ARC_TASK_NAMES)
        raise ArcDatasetError(f"supported ARC tasks are: {supported}")
    fields = _ARC_SPEC_FIELDS[task_name]
    return HubDatasetSpec(
        dataset=fields["dataset"],
        repository="allenai/ai2_arc",
        subset=fields["subset"],
        split="test",
        adapter_version=fields["adapter_version"],
        reference_commit=CHAT_EVAL_REFERENCE_COMMIT,
        required_columns=("question", "choices", "answerKey"),
    )


def normalize_arc_row(
    row: Mapping[str, object],
    *,
    source_row: int,
    source_identity: str,
    context: str,
) -> CategoricalExample:
    """Validate and normalize one ARC row without an assistant-answer leak."""

    try:
        if not isinstance(row, Mapping):
            raise ArcDatasetRowError("row must be an object")
        question = _non_empty_string(row.get("question"), label="question")
        raw_choices = row.get("choices")
        if not isinstance(raw_choices, Mapping):
            raise ArcDatasetRowError("choices must be an object")
        raw_labels = raw_choices.get("label")
        raw_texts = raw_choices.get("text")
        if not isinstance(raw_labels, (list, tuple)) or not isinstance(
            raw_texts,
            (list, tuple),
        ):
            raise ArcDatasetRowError("choice labels and text must be lists")
        if not raw_labels or len(raw_labels) != len(raw_texts):
            raise ArcDatasetRowError(
                "choice labels and text must have the same non-zero length"
            )
        labels = tuple(
            _non_empty_string(label, label=f"choices.label[{index}]")
            for index, label in enumerate(raw_labels)
        )
        choices = tuple(
            _non_empty_string(text, label=f"choices.text[{index}]")
            for index, text in enumerate(raw_texts)
        )
        if len(set(labels)) != len(labels):
            raise ArcDatasetRowError("choice labels must be unique")
        answer = _non_empty_string(row.get("answerKey"), label="answerKey")
        if answer not in labels:
            raise ArcDatasetRowError("answerKey must be present in choice labels")
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
            raise ArcDatasetRowError(str(error)) from error
        prompt = render_multiple_choice_prompt(question, labels, choices)
        identity = canonical_json_identity(
            {
                "answer": answer,
                "labels": list(labels),
                "prompt": prompt,
                "source_identity": source_identity,
                "source_row": source_row,
            }
        )
        return CategoricalExample(
            conversation=Conversation(messages=(UserMessage(prompt),)),
            labels=labels,
            answer=answer,
            source_row=source_row,
            identity=identity,
        )
    except ArcDatasetRowError as error:
        raise _with_context(context, error) from error


def build_arc_task(cache: CachedHubParquetDataset) -> CategoricalTask:
    """Materialize one deterministic seed-42 ARC task from a verified cache."""

    if not isinstance(cache, CachedHubParquetDataset):
        raise TypeError("cache must be a CachedHubParquetDataset")
    task_name = _task_name_for_subset(cache.spec.subset)
    expected_spec = get_arc_dataset_spec(task_name)
    if cache.spec != expected_spec:
        raise ArcDatasetError(
            f"cache spec does not match the pinned {task_name} source contract"
        )
    rows = read_cached_parquet_rows(cache)
    permutation = tuple(
        int(index)
        for index in np.random.default_rng(ARC_SHUFFLE_SEED).permutation(len(rows))
    )
    examples = tuple(
        normalize_arc_row(
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
            "seed": ARC_SHUFFLE_SEED,
        }
    )
    return CategoricalTask(
        name=task_name,
        examples=examples,
        source_identity=cache.spec.source_identity,
        dataset_identity=cache.source_identity,
        order_identity=order_identity,
    )


def load_arc_task(
    cache_root: str | Path,
    task_name: str,
) -> CategoricalTask:
    """Load one already-prepared ARC cache without any network fallback."""

    spec = get_arc_dataset_spec(task_name)
    return build_arc_task(load_hub_parquet_cache(spec, cache_root))


def _task_name_for_subset(subset: str) -> str:
    if subset in ARC_TASK_NAMES:
        return subset
    raise ArcDatasetError(f"unsupported ARC cache subset {subset!r}")


def _non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ArcDatasetRowError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise ArcDatasetRowError(f"{label} must be non-empty")
    return value


def _with_context(context: str, error: Exception) -> ArcDatasetRowError:
    try:
        context = require_non_empty_string(context, name="context")
    except (TypeError, ValueError) as context_error:
        raise ValueError(str(context_error)) from context_error
    return ArcDatasetRowError(f"{context}: {error}")


__all__ = [
    "ARC_REFERENCE_FILE_SHA256",
    "ARC_SHUFFLE_SEED",
    "ARC_TASK_NAMES",
    "ArcDatasetError",
    "ArcDatasetRowError",
    "build_arc_task",
    "get_arc_dataset_spec",
    "load_arc_task",
    "normalize_arc_row",
]
