"""Tests for regex-local BPE encoding and raw-byte decoding."""

from __future__ import annotations

from collections.abc import Callable
import random
from types import MappingProxyType
from typing import Any

import pytest

import scratch_llm.tokenization.bpe as bpe
from scratch_llm.tokenization.bpe import (
    BPEMerge,
    ReferenceBPETrainingResult,
    RegexBPETokenizer,
    train_reference_bpe,
)
from scratch_llm.tokenization.regex_chunking import bpe_encoding_chunks
from scratch_llm.tokenization.tokenizer import (
    BYTE_VOCAB_SIZE,
    NANOCHAT_SPECIAL_TOKENS,
    ByteTokenizer,
    Tokenizer,
)


ROUND_TRIP_TEXTS = (
    "plain ASCII",
    "naïve café — 東京",
    " \tleading\n\ntrailing\r\n ",
    "def square(x: int) -> int:\n    return x ** 2\n",
    r"$e^{i\pi} + 1 = 0$ and \frac{a}{b}",
    "안녕하세요, 세계!",
    "emoji: 🚀🧪✨",
    "<|bos|> is ordinary text unless explicitly requested",
    "",
)


def _train(
    texts: tuple[str, ...],
    *,
    merge_count: int,
) -> RegexBPETokenizer:
    result = train_reference_bpe(
        texts,
        vocab_size=BYTE_VOCAB_SIZE + merge_count + len(NANOCHAT_SPECIAL_TOKENS),
    )
    return RegexBPETokenizer(result)


def _synthetic_boundary_tokenizer() -> RegexBPETokenizer:
    """Build a valid rank whose pair can only occur across the test chunks."""

    mergeable_vocab_size = BYTE_VOCAB_SIZE + 1
    special_token_ids = {
        token: mergeable_vocab_size + offset
        for offset, token in enumerate(NANOCHAT_SPECIAL_TOKENS)
    }
    vocabulary = {token_id: bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)}
    vocabulary[BYTE_VOCAB_SIZE] = b"a "
    vocabulary.update(
        {
            token_id: token.encode("utf-8")
            for token, token_id in special_token_ids.items()
        }
    )
    result = ReferenceBPETrainingResult(
        vocab_size=mergeable_vocab_size + len(NANOCHAT_SPECIAL_TOKENS),
        mergeable_vocab_size=mergeable_vocab_size,
        merges=(
            BPEMerge(
                pair=(ord("a"), ord(" ")),
                token_id=BYTE_VOCAB_SIZE,
                count=1,
            ),
        ),
        vocabulary=MappingProxyType(vocabulary),
        special_token_ids=MappingProxyType(special_token_ids),
        document_count=1,
        character_count=2,
        chunk_count=1,
    )
    return RegexBPETokenizer(result)


def _synthetic_rank_chain_tokenizer(merge_count: int) -> RegexBPETokenizer:
    """Build many valid ranks when only the first can match ``b"ab"``."""

    mergeable_vocab_size = BYTE_VOCAB_SIZE + merge_count
    special_token_ids = {
        token: mergeable_vocab_size + offset
        for offset, token in enumerate(NANOCHAT_SPECIAL_TOKENS)
    }
    vocabulary = {token_id: bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)}
    merges: list[BPEMerge] = []
    previous_id = ord("a")
    for offset in range(merge_count):
        right_id = ord("b") if offset == 0 else ord("a")
        token_id = BYTE_VOCAB_SIZE + offset
        merges.append(
            BPEMerge(pair=(previous_id, right_id), token_id=token_id, count=1)
        )
        vocabulary[token_id] = vocabulary[previous_id] + vocabulary[right_id]
        previous_id = token_id
    vocabulary.update(
        {
            token_id: token.encode("utf-8")
            for token, token_id in special_token_ids.items()
        }
    )
    return RegexBPETokenizer(
        ReferenceBPETrainingResult(
            vocab_size=mergeable_vocab_size + len(NANOCHAT_SPECIAL_TOKENS),
            mergeable_vocab_size=mergeable_vocab_size,
            merges=tuple(merges),
            vocabulary=MappingProxyType(vocabulary),
            special_token_ids=MappingProxyType(special_token_ids),
            document_count=1,
            character_count=2,
            chunk_count=1,
        )
    )


def _reference_rank_sweep(tokenizer: RegexBPETokenizer, text: str) -> list[int]:
    """Apply the original obviously-correct merge-table sweep."""

    token_ids: list[int] = []
    for byte_chunk in bpe_encoding_chunks(text):
        encoded_chunk = tuple(byte_chunk)
        for merge in tokenizer._training_result.merges:
            encoded_chunk = bpe.merge_pair(
                encoded_chunk,
                merge.pair,
                merge.token_id,
            )
        token_ids.extend(encoded_chunk)
    return token_ids


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_unmerged_bpe_matches_byte_tokenizer_and_round_trips(text: str) -> None:
    byte_tokenizer = ByteTokenizer()
    tokenizer: Tokenizer = _train(("training fixture",), merge_count=0)

    assert tokenizer.encode(text) == byte_tokenizer.encode(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.get_vocab_size() == byte_tokenizer.get_vocab_size()
    assert tokenizer.get_bos_token_id() == byte_tokenizer.get_bos_token_id()
    assert tokenizer.get_special_tokens() == byte_tokenizer.get_special_tokens()


def test_encoder_applies_learned_ranks_inside_each_regex_chunk() -> None:
    tokenizer = _train(("aa aa",), merge_count=2)

    assert tokenizer.encode("aa aa") == [256, 257]
    assert tokenizer.decode([256, 257]) == "aa aa"
    assert tokenizer.decode_single_token_bytes(256) == b"aa"
    assert tokenizer.decode_single_token_bytes(257) == b" aa"
    assert all(
        0 <= token_id < tokenizer.get_vocab_size()
        for token_id in tokenizer.encode("aa aa")
    )


def test_encoder_does_not_sweep_irrelevant_learned_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _synthetic_rank_chain_tokenizer(1_024)
    merge_pair_calls = 0
    reference_merge_pair = bpe.merge_pair

    def record_merge_pair(
        chunk: tuple[int, ...],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> tuple[int, ...]:
        nonlocal merge_pair_calls
        merge_pair_calls += 1
        return reference_merge_pair(chunk, pair, new_token_id)

    monkeypatch.setattr(bpe, "merge_pair", record_merge_pair)

    assert tokenizer.encode("ab") == [BYTE_VOCAB_SIZE]
    assert merge_pair_calls <= 1


def test_ranked_encoder_resolves_overlaps_left_to_right() -> None:
    tokenizer = _train(("aaaa",), merge_count=1)

    assert tokenizer.encode("aaa") == [BYTE_VOCAB_SIZE, ord("a")]
    assert tokenizer.encode("aaaa") == [BYTE_VOCAB_SIZE, BYTE_VOCAB_SIZE]
    assert tokenizer.encode("aaaaa") == [
        BYTE_VOCAB_SIZE,
        BYTE_VOCAB_SIZE,
        ord("a"),
    ]


def test_ranked_encoder_matches_reference_sweep_on_randomized_unicode() -> None:
    random_generator = random.Random(20260729)
    alphabet = "aaabbbccc  \n\t0123_+-=é한🚀"
    training_texts = tuple(
        "".join(
            random_generator.choice(alphabet)
            for _ in range(random_generator.randint(16, 64))
        )
        for _ in range(48)
    )
    tokenizer = _train(training_texts, merge_count=64)
    evaluation_texts = (
        *ROUND_TRIP_TEXTS,
        "aaa",
        "aaaa",
        "abababa",
        *(
            "".join(
                random_generator.choice(alphabet)
                for _ in range(random_generator.randint(0, 96))
            )
            for _ in range(128)
        ),
    )

    for text in evaluation_texts:
        expected = _reference_rank_sweep(tokenizer, text)
        assert tokenizer.encode(text) == expected
        assert tokenizer.decode(expected) == text


def test_encoder_never_applies_a_learned_pair_across_regex_chunks() -> None:
    tokenizer = _synthetic_boundary_tokenizer()

    assert tokenizer.encode("a a") == [ord("a"), ord(" "), ord("a")]
    assert BYTE_VOCAB_SIZE not in tokenizer.encode("a a")


def test_decoder_concatenates_raw_bytes_before_one_utf8_decode() -> None:
    tokenizer = _train(("🚀",), merge_count=1)
    encoded = tokenizer.encode("🚀")

    assert encoded == [0xF0, 0x9F, 256]
    assert tokenizer.decode_single_token_bytes(256) == b"\x9a\x80"
    assert tokenizer.decode([encoded[0]]) == "\ufffd"
    assert tokenizer.decode([encoded[1]]) == "\ufffd"
    assert tokenizer.decode([encoded[2]]) == "\ufffd\ufffd"
    assert tokenizer.decode(encoded) == "🚀"


def test_special_tokens_are_explicit_stable_and_mix_with_ordinary_text() -> None:
    tokenizer = _train(("ordinary ordinary",), merge_count=1)
    bos_id = tokenizer.encode_special("<|bos|>")
    assistant_end_id = tokenizer.encode_special("<|assistant_end|>")
    ordinary_special_text = tokenizer.encode("<|bos|>")

    assert bos_id == tokenizer.get_vocab_size() - len(NANOCHAT_SPECIAL_TOKENS)
    assert ordinary_special_text != [bos_id]
    assert all(token_id < bos_id for token_id in ordinary_special_text)
    assert tokenizer.encode(
        "hello",
        prepend="<|bos|>",
        append=assistant_end_id,
    ) == [bos_id, *tokenizer.encode("hello"), assistant_end_id]
    assert (
        tokenizer.decode(
            [
                *tokenizer.encode("left"),
                bos_id,
                *tokenizer.encode("right"),
                assistant_end_id,
            ]
        )
        == "left<|bos|>right<|assistant_end|>"
    )

    for offset, token in enumerate(NANOCHAT_SPECIAL_TOKENS):
        token_id = bos_id + offset
        assert tokenizer.encode_special(token) == token_id
        assert tokenizer.decode([token_id]) == token
        assert tokenizer.decode_single_token_bytes(token_id) == token.encode("utf-8")


@pytest.mark.parametrize(
    "token_ids",
    (
        [0xFF, ord("a"), 0xC3],
        [0, 255],
        [BYTE_VOCAB_SIZE],
    ),
)
def test_unmerged_decode_matches_byte_tokenizer_replacement_policy(
    token_ids: list[int],
) -> None:
    tokenizer = _train(("fixture",), merge_count=0)

    assert tokenizer.decode(token_ids) == ByteTokenizer().decode(token_ids)


@pytest.mark.parametrize(
    "call",
    (
        lambda tokenizer: tokenizer.encode(b"bytes"),
        lambda tokenizer: tokenizer.decode(1),
        lambda tokenizer: tokenizer.decode([True]),
        lambda tokenizer: tokenizer.decode([-1]),
        lambda tokenizer: tokenizer.decode([tokenizer.get_vocab_size()]),
        lambda tokenizer: tokenizer.decode_single_token_bytes(1.5),
        lambda tokenizer: tokenizer.encode_special("<|pad|>"),
        lambda tokenizer: tokenizer.encode_special(1),
        lambda tokenizer: tokenizer.encode("text", prepend=0),
        lambda tokenizer: tokenizer.encode("text", append=True),
    ),
)
def test_unmerged_validation_matches_byte_tokenizer(
    call: Callable[[Tokenizer], Any],
) -> None:
    byte_tokenizer = ByteTokenizer()
    tokenizer = _train(("fixture",), merge_count=0)

    with pytest.raises(Exception) as byte_error:
        call(byte_tokenizer)
    with pytest.raises(type(byte_error.value), match=".*") as bpe_error:
        call(tokenizer)

    assert str(bpe_error.value) == str(byte_error.value)


def test_bpe_identity_is_stable_and_depends_on_the_learned_vocabulary() -> None:
    tokenizer = _train(("aa aa",), merge_count=1)
    same = _train(("aa aa",), merge_count=1)
    different = _train(("bb bb",), merge_count=1)

    assert tokenizer.get_identity() == same.get_identity()
    assert tokenizer.get_identity() != different.get_identity()
    assert tokenizer.get_identity().startswith("sha256:")


def test_constructor_rejects_non_training_results() -> None:
    with pytest.raises(
        TypeError,
        match="training_result must be a ReferenceBPETrainingResult",
    ):
        RegexBPETokenizer(object())  # type: ignore[arg-type]
