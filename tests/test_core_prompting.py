"""Tests for nanochat-compatible CORE prompt and token-span construction."""

from __future__ import annotations

import pytest

from scratch_llm.evaluation.core.bundle import CoreTask
from scratch_llm.evaluation.core.examples import (
    LanguageModelingExample,
    MultipleChoiceExample,
    SchemaExample,
)
from scratch_llm.evaluation.core.prompting import (
    CorePromptingError,
    build_core_token_batch,
    render_core_prompts,
)
from scratch_llm.tokenizer import ByteTokenizer


def _task(task_type: str, *, delimiter: str = " ") -> CoreTask:
    return CoreTask(
        label="fixture",
        task_type=task_type,  # type: ignore[arg-type]
        dataset_member="eval_bundle/eval_data/fixture.jsonl",
        num_fewshot=1,
        continuation_delimiter=delimiter,
        random_baseline_percent=25.0,
    )


def test_render_multiple_choice_prompts_include_seeded_examples() -> None:
    task = _task("multiple_choice", delimiter="\nAnswer: ")
    current = MultipleChoiceExample("Current?", ("No", "Yes"), 1)
    fewshot = MultipleChoiceExample("Earlier?", ("Wrong", "Right"), 1)

    assert render_core_prompts(task, current, (fewshot,)) == (
        "Earlier?\nAnswer: Right\n\nCurrent?\nAnswer: No",
        "Earlier?\nAnswer: Right\n\nCurrent?\nAnswer: Yes",
    )


def test_render_schema_and_language_modeling_prompts_match_the_reference() -> None:
    schema_task = _task("schema")
    schema = SchemaExample(("Alice", "Bob"), "won.", 0)
    schema_fewshot = SchemaExample(("Cat", "Dog"), "slept.", 1)
    assert render_core_prompts(schema_task, schema, (schema_fewshot,)) == (
        "Dog slept.\n\nAlice won.",
        "Dog slept.\n\nBob won.",
    )

    lm_task = _task("language_modeling", delimiter="\nAnswer: ")
    language = LanguageModelingExample("  Current question  ", "answer")
    language_fewshot = LanguageModelingExample("  Earlier question  ", "old")
    assert render_core_prompts(lm_task, language, (language_fewshot,)) == (
        "Earlier question\nAnswer: old\n\nCurrent question\nAnswer:",
        "Earlier question\nAnswer: old\n\nCurrent question\nAnswer: answer",
    )


@pytest.mark.parametrize(
    ("task", "example", "expected_batch", "expected_gold"),
    [
        (
            _task("multiple_choice"),
            MultipleChoiceExample("Pick", ("A", "B"), 1),
            2,
            1,
        ),
        (
            _task("schema"),
            SchemaExample(("A", "B"), "suffix", 0),
            2,
            0,
        ),
        (
            _task("language_modeling"),
            LanguageModelingExample("prefix", " suffix"),
            1,
            None,
        ),
    ],
)
def test_build_core_token_batch_marks_non_empty_scoring_spans(
    task: CoreTask,
    example: MultipleChoiceExample | SchemaExample | LanguageModelingExample,
    expected_batch: int,
    expected_gold: int | None,
) -> None:
    batch = build_core_token_batch(
        task,
        example,
        (),
        tokenizer=ByteTokenizer(),
        max_seq_len=64,
    )

    assert len(batch.token_ids) == expected_batch
    assert batch.gold_index == expected_gold
    assert all(
        1 <= start < end <= len(tokens)
        for tokens, start, end in zip(
            batch.token_ids,
            batch.start_indices,
            batch.end_indices,
            strict=True,
        )
    )


def test_multiple_choice_allows_a_prefix_option_with_an_empty_difference_span() -> None:
    batch = build_core_token_batch(
        _task("multiple_choice"),
        MultipleChoiceExample("Pick", ("answer plus", "answer"), 0),
        (),
        tokenizer=ByteTokenizer(),
        max_seq_len=64,
    )

    assert batch.start_indices[0] == batch.start_indices[1]
    assert batch.start_indices[0] < batch.end_indices[0]
    assert batch.start_indices[1] == batch.end_indices[1]


def test_build_core_token_batch_left_truncates_context_but_preserves_answers() -> None:
    batch = build_core_token_batch(
        _task("multiple_choice"),
        MultipleChoiceExample("x" * 30, ("A", "B"), 0),
        (),
        tokenizer=ByteTokenizer(),
        max_seq_len=8,
    )

    assert all(len(tokens) == 8 for tokens in batch.token_ids)
    assert batch.start_indices == (7, 7)
    assert batch.end_indices == (8, 8)


def test_build_core_token_batch_rejects_a_continuation_that_cannot_fit() -> None:
    with pytest.raises(CorePromptingError, match="continuation does not fit"):
        build_core_token_batch(
            _task("language_modeling"),
            LanguageModelingExample("x", "y" * 20),
            (),
            tokenizer=ByteTokenizer(),
            max_seq_len=8,
        )
