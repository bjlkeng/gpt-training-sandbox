"""Prompt rendering and token-span construction for nanochat CORE tasks."""

from __future__ import annotations

from dataclasses import dataclass

from scratch_llm._validation import require_positive_integer
from scratch_llm.core_bundle import CoreTask
from scratch_llm.core_evaluation import CoreEvaluationError, CoreTaskType
from scratch_llm.core_examples import (
    CoreExample,
    LanguageModelingExample,
    MultipleChoiceExample,
    SchemaExample,
)
from scratch_llm.tokenizer import Tokenizer


class CorePromptingError(CoreEvaluationError):
    """A task example cannot be rendered into a valid scoring span."""


@dataclass(frozen=True)
class CoreTokenBatch:
    """Variable-length token rows and the target span in each row."""

    task_type: CoreTaskType
    token_ids: tuple[tuple[int, ...], ...]
    start_indices: tuple[int, ...]
    end_indices: tuple[int, ...]
    gold_index: int | None

    def __post_init__(self) -> None:
        row_count = len(self.token_ids)
        if row_count == 0:
            raise CorePromptingError("CORE token batch must contain at least one row")
        if len(self.start_indices) != row_count or len(self.end_indices) != row_count:
            raise CorePromptingError("CORE token rows and spans must have equal length")
        for row, start, end in zip(
            self.token_ids,
            self.start_indices,
            self.end_indices,
            strict=True,
        ):
            if not 1 <= start < end <= len(row):
                raise CorePromptingError(
                    "CORE scoring spans must be non-empty and follow one context token"
                )
        if self.task_type == "language_modeling":
            if row_count != 1 or self.gold_index is not None:
                raise CorePromptingError(
                    "language-modeling batches require one row and no gold index"
                )
        elif self.gold_index is None or not 0 <= self.gold_index < row_count:
            raise CorePromptingError(
                "categorical CORE batches require a valid gold index"
            )


def render_core_prompts(
    task: CoreTask,
    example: CoreExample,
    fewshot_examples: tuple[CoreExample, ...],
) -> tuple[str, ...]:
    """Render the pinned prompt strings for one task example."""

    if not isinstance(task, CoreTask):
        raise TypeError("task must be a CoreTask")
    if not isinstance(fewshot_examples, tuple):
        raise TypeError("fewshot_examples must be a tuple")
    if task.task_type == "multiple_choice":
        current = _require_example_type(
            example,
            MultipleChoiceExample,
            task_type=task.task_type,
        )
        fewshot = tuple(
            _require_example_type(item, MultipleChoiceExample, task_type=task.task_type)
            for item in fewshot_examples
        )
        return _render_multiple_choice(task, current, fewshot)
    if task.task_type == "schema":
        current = _require_example_type(
            example,
            SchemaExample,
            task_type=task.task_type,
        )
        fewshot = tuple(
            _require_example_type(item, SchemaExample, task_type=task.task_type)
            for item in fewshot_examples
        )
        return _render_schema(task, current, fewshot)
    current = _require_example_type(
        example,
        LanguageModelingExample,
        task_type=task.task_type,
    )
    fewshot = tuple(
        _require_example_type(
            item,
            LanguageModelingExample,
            task_type=task.task_type,
        )
        for item in fewshot_examples
    )
    return _render_language_modeling(task, current, fewshot)


def build_core_token_batch(
    task: CoreTask,
    example: CoreExample,
    fewshot_examples: tuple[CoreExample, ...],
    *,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> CoreTokenBatch:
    """Tokenize prompts, locate answer spans, and apply left truncation."""

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement Tokenizer")
    max_seq_len = require_positive_integer(max_seq_len, name="max_seq_len")
    prompts = render_core_prompts(task, example, fewshot_examples)
    sequences = tuple(
        tuple(
            tokenizer.encode(
                prompt,
                prepend=tokenizer.get_bos_token_id(),
            )
        )
        for prompt in prompts
    )
    starts, ends, gold_index = _scoring_spans(
        task,
        example,
        sequences=sequences,
    )
    if task.task_type == "language_modeling":
        sequences = (sequences[1],)
    truncated, starts, ends = _left_truncate(
        sequences,
        starts,
        ends,
        max_seq_len=max_seq_len,
    )
    return CoreTokenBatch(
        task_type=task.task_type,
        token_ids=truncated,
        start_indices=starts,
        end_indices=ends,
        gold_index=gold_index,
    )


def _render_multiple_choice(
    task: CoreTask,
    current: MultipleChoiceExample,
    fewshot: tuple[MultipleChoiceExample, ...],
) -> tuple[str, ...]:
    prefix = [
        item.query + task.continuation_delimiter + item.choices[item.gold]
        for item in fewshot
    ]
    return tuple(
        "\n\n".join((*prefix, current.query + task.continuation_delimiter + choice))
        for choice in current.choices
    )


def _render_schema(
    task: CoreTask,
    current: SchemaExample,
    fewshot: tuple[SchemaExample, ...],
) -> tuple[str, ...]:
    prefix = [
        item.context_options[item.gold]
        + task.continuation_delimiter
        + item.continuation
        for item in fewshot
    ]
    return tuple(
        "\n\n".join(
            (*prefix, context + task.continuation_delimiter + current.continuation)
        )
        for context in current.context_options
    )


def _render_language_modeling(
    task: CoreTask,
    current: LanguageModelingExample,
    fewshot: tuple[LanguageModelingExample, ...],
) -> tuple[str, str]:
    prefix = [
        item.context.strip() + task.continuation_delimiter + item.continuation
        for item in fewshot
    ]
    current_prefix = current.context.strip() + task.continuation_delimiter
    prompt_without = "\n\n".join((*prefix, current_prefix)).strip()
    prompt_with = "\n\n".join((*prefix, current_prefix + current.continuation))
    return prompt_without, prompt_with


def _scoring_spans(
    task: CoreTask,
    example: CoreExample,
    *,
    sequences: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
    if task.task_type == "multiple_choice":
        if not isinstance(example, MultipleChoiceExample):  # pragma: no cover.
            raise TypeError("multiple-choice task has the wrong example type")
        start = _common_length(sequences, from_right=False)
        return (
            (start,) * len(sequences),
            tuple(len(sequence) for sequence in sequences),
            example.gold,
        )
    if task.task_type == "schema":
        if not isinstance(example, SchemaExample):  # pragma: no cover.
            raise TypeError("schema task has the wrong example type")
        suffix_length = _common_length(sequences, from_right=True)
        ends = tuple(len(sequence) for sequence in sequences)
        starts = tuple(end - suffix_length for end in ends)
        return starts, ends, example.gold
    tokens_without, tokens_with = sequences
    start = len(tokens_without)
    if start >= len(tokens_with) or tokens_without != tokens_with[:start]:
        raise CorePromptingError(
            "language-modeling prompt without continuation must be a token prefix"
        )
    return (start,), (len(tokens_with),), None


def _left_truncate(
    sequences: tuple[tuple[int, ...], ...],
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    *,
    max_seq_len: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    adjusted_starts: list[int] = []
    adjusted_ends: list[int] = []
    for sequence, start, end in zip(sequences, starts, ends, strict=True):
        crop = max(0, len(sequence) - max_seq_len)
        adjusted_start = start - crop
        adjusted_end = end - crop
        if adjusted_start < 1 or adjusted_end > max_seq_len:
            raise CorePromptingError(
                "CORE continuation does not fit within max_seq_len after left truncation"
            )
        rows.append(sequence[crop:])
        adjusted_starts.append(adjusted_start)
        adjusted_ends.append(adjusted_end)
    return tuple(rows), tuple(adjusted_starts), tuple(adjusted_ends)


def _common_length(
    sequences: tuple[tuple[int, ...], ...],
    *,
    from_right: bool,
) -> int:
    minimum = min(len(sequence) for sequence in sequences)
    indices = range(-1, -minimum - 1, -1) if from_right else range(minimum)
    for length, index in enumerate(indices):
        token_id = sequences[0][index]
        if any(sequence[index] != token_id for sequence in sequences[1:]):
            return length
    return minimum


def _require_example_type(
    value: CoreExample,
    expected: type[MultipleChoiceExample]
    | type[SchemaExample]
    | type[LanguageModelingExample],
    *,
    task_type: CoreTaskType,
):
    if not isinstance(value, expected):
        raise CorePromptingError(
            f"{task_type} task requires {expected.__name__} values"
        )
    return value


__all__ = [
    "CorePromptingError",
    "CoreTokenBatch",
    "build_core_token_batch",
    "render_core_prompts",
]
