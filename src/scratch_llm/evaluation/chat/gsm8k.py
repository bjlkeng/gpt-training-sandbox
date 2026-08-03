"""Strict offline GSM8K adapter and pinned final-answer scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
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
from scratch_llm.data.sft_sources import get_sft_dataset_spec
from scratch_llm.evaluation.chat.cache import read_cached_parquet_rows
from scratch_llm.evaluation.chat.generative import (
    GenerativeProblem,
    GenerativeTask,
)
from scratch_llm.identity import canonical_json_identity


GSM8K_TASK_NAME: Final = "GSM8K"
GSM8K_SHUFFLE_SEED: Final = 42
GSM8K_REFERENCE_FILE_SHA256: Final = (
    "b22e38691173c8265c1bd24f693b6eb31572fc466f5235352b0213a4d7f26fd1"
)
_GSM8K_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")


class GSM8KDatasetError(ValueError):
    """A GSM8K task source or cached view is invalid."""


class GSM8KDatasetRowError(GSM8KDatasetError):
    """One GSM8K row cannot be normalized for evaluation."""


@dataclass(frozen=True, slots=True)
class GSM8KProblem(GenerativeProblem):
    """One GSM8K prompt with its normalized final numeric answer."""

    reference_answer: str

    def __post_init__(self) -> None:
        GenerativeProblem.__post_init__(self)
        try:
            require_non_empty_string(
                self.reference_answer,
                name="reference_answer",
            )
        except (TypeError, ValueError) as error:
            raise GSM8KDatasetRowError(str(error)) from error


def get_gsm8k_dataset_spec() -> HubDatasetSpec:
    """Reuse the pinned openai/gsm8k main/test cache contract."""

    return get_sft_dataset_spec("gsm8k", "test")


def extract_gsm8k_answer(completion: str) -> str | None:
    """Extract the first pinned ``#### number`` answer, removing commas."""

    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    match = _GSM8K_ANSWER.search(completion)
    if match is None:
        return None
    return match.group(1).strip().replace(",", "")


def normalize_gsm8k_eval_row(
    row: Mapping[str, object],
    *,
    source_row: int,
    source_identity: str,
    context: str,
) -> GSM8KProblem:
    """Normalize one strict row into a user-only prompt and numeric reference."""

    try:
        if not isinstance(row, Mapping):
            raise GSM8KDatasetRowError("row must be an object")
        question = _non_empty_string(row.get("question"), label="question")
        answer = _non_empty_string(row.get("answer"), label="answer")
        reference_answer = extract_gsm8k_answer(answer)
        if reference_answer is None:
            raise GSM8KDatasetRowError(
                "answer must contain a valid #### numeric marker"
            )
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
            raise GSM8KDatasetRowError(str(error)) from error
        identity = canonical_json_identity(
            {
                "question": question,
                "reference_answer": reference_answer,
                "source_identity": source_identity,
                "source_row": source_row,
            }
        )
        return GSM8KProblem(
            conversation=Conversation(messages=(UserMessage(question),)),
            source_row=source_row,
            identity=identity,
            reference_answer=reference_answer,
        )
    except GSM8KDatasetRowError as error:
        raise _with_context(context, error) from error


def score_gsm8k_completion(
    problem: GenerativeProblem,
    completion: str,
) -> bool:
    """Compare one completion with the problem's normalized final answer."""

    if not isinstance(problem, GSM8KProblem):
        raise TypeError("problem must be a GSM8KProblem")
    predicted_answer = extract_gsm8k_answer(completion)
    return predicted_answer is not None and predicted_answer == problem.reference_answer


def build_gsm8k_task(cache: CachedHubParquetDataset) -> GenerativeTask:
    """Materialize deterministic seed-42 GSM8K problems from a verified cache."""

    if not isinstance(cache, CachedHubParquetDataset):
        raise TypeError("cache must be a CachedHubParquetDataset")
    expected_spec = get_gsm8k_dataset_spec()
    if cache.spec != expected_spec:
        raise GSM8KDatasetError(
            "cache spec does not match the pinned GSM8K main/test contract"
        )
    rows = read_cached_parquet_rows(cache)
    permutation = tuple(
        int(index)
        for index in np.random.default_rng(GSM8K_SHUFFLE_SEED).permutation(len(rows))
    )
    problems = tuple(
        normalize_gsm8k_eval_row(
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
            "problem_identities": [problem.identity for problem in problems],
            "reference_file_sha256": GSM8K_REFERENCE_FILE_SHA256,
            "seed": GSM8K_SHUFFLE_SEED,
        }
    )
    return GenerativeTask(
        name=GSM8K_TASK_NAME,
        problems=problems,
        source_identity=cache.spec.source_identity,
        dataset_identity=cache.source_identity,
        order_identity=order_identity,
    )


def load_gsm8k_task(cache_root: str | Path) -> GenerativeTask:
    """Load the already-prepared GSM8K test cache without a network fallback."""

    spec = get_gsm8k_dataset_spec()
    return build_gsm8k_task(load_hub_parquet_cache(spec, cache_root))


def _non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise GSM8KDatasetRowError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise GSM8KDatasetRowError(f"{label} must be non-empty")
    return value


def _with_context(context: str, error: Exception) -> GSM8KDatasetRowError:
    try:
        context = require_non_empty_string(context, name="context")
    except (TypeError, ValueError) as context_error:
        raise ValueError(str(context_error)) from context_error
    return GSM8KDatasetRowError(f"{context}: {error}")


__all__ = [
    "GSM8K_REFERENCE_FILE_SHA256",
    "GSM8K_SHUFFLE_SEED",
    "GSM8K_TASK_NAME",
    "GSM8KDatasetError",
    "GSM8KDatasetRowError",
    "GSM8KProblem",
    "build_gsm8k_task",
    "extract_gsm8k_answer",
    "get_gsm8k_dataset_spec",
    "load_gsm8k_task",
    "normalize_gsm8k_eval_row",
    "score_gsm8k_completion",
]
