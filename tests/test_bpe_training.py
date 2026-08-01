"""Tests for the readable reference regex byte-BPE trainer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import permutations
from typing import Any

import pytest

from scratch_llm.tokenization.bpe import (
    PAIR_TIE_BREAK,
    BPETrainingError,
    apply_merge,
    count_pairs,
    merge_pair,
    select_best_pair,
    train_reference_bpe,
)
from scratch_llm.tokenization.regex_chunking import iter_bpe_training_chunks
from scratch_llm.tokenization.tokenizer import BYTE_VOCAB_SIZE, NANOCHAT_SPECIAL_TOKENS


def test_pair_counts_are_chunk_local() -> None:
    assert count_pairs(((1, 2, 1), (2, 1))) == {
        (1, 2): 1,
        (2, 1): 2,
    }

    chunks = tuple(tuple(chunk) for chunk in iter_bpe_training_chunks(["hello world"]))
    counts = count_pairs(chunks)
    assert (ord("o"), ord(" ")) not in counts
    assert counts[(ord(" "), ord("w"))] == 1


def test_pair_counts_match_hand_worked_multi_chunk_corpus() -> None:
    assert count_pairs(
        (
            (1, 1, 2, 1),
            (),
            (1,),
            (2, 1, 1),
        )
    ) == {
        (1, 1): 2,
        (1, 2): 1,
        (2, 1): 2,
    }


def test_pair_selection_uses_documented_deterministic_tie_break() -> None:
    counts = {(9, 1): 3, (1, 9): 3, (0, 99): 2}

    assert "highest frequency" in PAIR_TIE_BREAK
    assert "lexicographically smallest" in PAIR_TIE_BREAK
    assert select_best_pair(counts) == (1, 9)
    assert select_best_pair(dict(reversed(tuple(counts.items())))) == (1, 9)


def test_merge_is_non_overlapping_left_to_right_and_non_mutating() -> None:
    original = [1, 1, 1, 1, 2]

    assert merge_pair(original, (1, 1), 7) == (7, 7, 2)
    assert merge_pair((1, 1, 1), (1, 1), 7) == (7, 1)
    assert original == [1, 1, 1, 1, 2]


def test_merge_application_keeps_chunks_independent_and_unrelated() -> None:
    chunks: tuple[list[int], ...] = ([1, 2, 1, 2], [9, 9], [], [1])

    assert apply_merge(chunks, (1, 2), 10) == (
        (10, 10),
        (9, 9),
        (),
        (1,),
    )
    assert chunks == ([1, 2, 1, 2], [9, 9], [], [1])


def test_training_follows_a_hand_traced_merge_sequence_and_final_ids() -> None:
    result = train_reference_bpe(
        ["aa aa"],
        vocab_size=BYTE_VOCAB_SIZE + 2 + len(NANOCHAT_SPECIAL_TOKENS),
    )

    assert [(merge.pair, merge.token_id, merge.count) for merge in result.merges] == [
        ((ord("a"), ord("a")), 256, 2),
        ((ord(" "), 256), 257, 1),
    ]
    assert result.vocabulary[256] == b"aa"
    assert result.vocabulary[257] == b" aa"
    assert result.mergeable_vocab_size == 258
    assert result.vocab_size == 267
    assert tuple(result.vocabulary) == tuple(range(267))
    assert result.special_token_ids == {
        token: 258 + offset for offset, token in enumerate(NANOCHAT_SPECIAL_TOKENS)
    }
    assert result.document_count == 1
    assert result.character_count == 5
    assert result.chunk_count == 2


def test_custom_special_tokens_keep_their_configured_final_order() -> None:
    result = train_reference_bpe(
        ["aa"],
        vocab_size=259,
        special_tokens=("<second>", "<first>"),
    )

    assert result.mergeable_vocab_size == 257
    assert result.special_token_ids == {"<second>": 257, "<first>": 258}
    assert result.vocabulary[257] == b"<second>"
    assert result.vocabulary[258] == b"<first>"


def test_repeated_and_shuffled_equivalent_corpora_train_identically() -> None:
    documents = ("banana", " bandana", "banana")
    target_size = BYTE_VOCAB_SIZE + 5 + len(NANOCHAT_SPECIAL_TOKENS)
    baseline = train_reference_bpe(documents, vocab_size=target_size)
    repeated = train_reference_bpe(documents, vocab_size=target_size)

    assert repeated == baseline
    for shuffled in permutations(documents):
        result = train_reference_bpe(shuffled, vocab_size=target_size)
        assert result.merges == baseline.merges
        assert result.vocabulary == baseline.vocabulary
        assert result.special_token_ids == baseline.special_token_ids


class _OnePassTexts:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.iteration_count = 0

    def __iter__(self) -> Any:
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise AssertionError("corpus was iterated more than once")
        return iter(self.texts)


def test_non_reiterable_input_is_consumed_once_and_caps_are_exact() -> None:
    texts = _OnePassTexts(("aaaa", "bbbb", "must-not-be-consumed"))
    result = train_reference_bpe(
        texts,
        vocab_size=BYTE_VOCAB_SIZE + 1 + len(NANOCHAT_SPECIAL_TOKENS),
        max_documents=2,
        max_characters=5,
    )

    assert texts.iteration_count == 1
    assert result.document_count == 2
    assert result.character_count == 5
    assert result.merges[0].pair == (ord("a"), ord("a"))


def test_document_cap_stops_before_pulling_an_extra_item() -> None:
    def guarded_texts() -> Any:
        yield "aaaa"
        raise AssertionError("document cap pulled an extra corpus item")

    result = train_reference_bpe(
        guarded_texts(),
        vocab_size=BYTE_VOCAB_SIZE + 1 + len(NANOCHAT_SPECIAL_TOKENS),
        max_documents=1,
    )

    assert result.document_count == 1
    assert result.character_count == 4


@pytest.mark.parametrize(
    ("texts", "vocab_size", "kwargs", "message"),
    [
        ((), 265, {}, "did not yield any documents"),
        (("",), 265, {}, "produced no regex byte chunks"),
        (("a",), 266, {}, "exhausted all adjacent pairs"),
        (("aaaa",), 266, {"max_documents": 0}, "did not yield any documents"),
        (("aaaa",), 266, {"max_characters": 0}, "did not yield any documents"),
    ],
)
def test_empty_or_exhausted_corpora_fail_explicitly(
    texts: tuple[str, ...],
    vocab_size: int,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(BPETrainingError, match=message):
        train_reference_bpe(texts, vocab_size=vocab_size, **kwargs)


@pytest.mark.parametrize("vocab_size", [264, 0, -1])
def test_invalid_small_vocabulary_target_fails_before_consuming_input(
    vocab_size: int,
) -> None:
    consumed = False

    def texts() -> Any:
        nonlocal consumed
        consumed = True
        yield "aaaa"

    with pytest.raises(ValueError, match="vocab_size must be at least 265"):
        train_reference_bpe(texts(), vocab_size=vocab_size)

    assert not consumed


@pytest.mark.parametrize("vocab_size", [True, 265.0, "265"])
def test_non_integer_vocabulary_target_fails_clearly(vocab_size: object) -> None:
    with pytest.raises(TypeError, match="vocab_size must be an integer"):
        train_reference_bpe(["aaaa"], vocab_size=vocab_size)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"max_documents": -1}, ValueError, "max_documents"),
        ({"max_documents": True}, TypeError, "max_documents"),
        ({"max_characters": -1}, ValueError, "max_characters"),
        ({"max_characters": 1.5}, TypeError, "max_characters"),
        ({"special_tokens": ("",)}, ValueError, "must not be empty"),
        ({"special_tokens": ("<s>", "<s>")}, ValueError, "duplicates"),
        ({"special_tokens": "<s>"}, TypeError, "ordered iterable"),
    ],
)
def test_invalid_limits_and_special_tokens_fail_clearly(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        train_reference_bpe(
            ["aaaa"],
            vocab_size=266,
            **kwargs,  # type: ignore[arg-type]
        )


def test_training_result_is_immutable() -> None:
    result = train_reference_bpe(
        ["aaaa"],
        vocab_size=BYTE_VOCAB_SIZE + 1 + len(NANOCHAT_SPECIAL_TOKENS),
    )

    with pytest.raises(TypeError):
        result.vocabulary[0] = b"changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.special_token_ids["<|bos|>"] = 0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.vocab_size = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("call", "error_type", "message"),
    [
        (
            lambda: count_pairs([1, 2]),  # type: ignore[list-item]
            TypeError,
            "token IDs",
        ),
        (lambda: count_pairs(((True, 1),)), TypeError, "must be an integer"),
        (lambda: merge_pair((1, 2), (1,), 3), ValueError, "exactly two"),
        (lambda: merge_pair((1, 2), (1, 2), -1), ValueError, "non-negative"),
        (lambda: select_best_pair({}), BPETrainingError, "no adjacent"),
        (lambda: select_best_pair({(1, 2): 0}), ValueError, "positive"),
    ],
)
def test_reference_primitives_reject_invalid_inputs(
    call: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        call()
