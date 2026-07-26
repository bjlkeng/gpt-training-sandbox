"""Tests for the deterministic UTF-8 byte tokenizer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_type_hints

import pytest

import scratch_llm.checkpoint as checkpoint
import scratch_llm.pretraining as pretraining
from scratch_llm.config import DEFAULT_SPECIAL_TOKENS
from scratch_llm.tokenizer import (
    BYTE_VOCAB_SIZE,
    NANOCHAT_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    ByteTokenizer,
    Tokenizer,
    UnsupportedTokenizerOperationError,
)
from scratch_llm.training import TinyTextTrainingResult


EXPECTED_SPECIAL_TOKENS = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)


@pytest.mark.parametrize(
    "text",
    [
        "plain ASCII",
        "na\u00efve caf\u00e9 \u2014 \u6771\u4eac",
        "emoji: \U0001f680\U0001f9ea\u2728",
        "\uc548\ub155\ud558\uc138\uc694, \uc138\uacc4!",
        "def square(x: int) -> int:\n    return x ** 2\n",
        r"$e^{i\pi} + 1 = 0$ and \\frac{a}{b}",
        " \tleading\n\ntrailing\r\n ",
        "",
    ],
)
def test_ordinary_text_round_trips_through_utf8_bytes(text: str) -> None:
    tokenizer = ByteTokenizer()

    token_ids = tokenizer.encode(text)

    assert token_ids == list(text.encode("utf-8"))
    assert tokenizer.decode(token_ids) == text


def test_special_tokens_have_locked_ids_after_the_byte_vocabulary() -> None:
    tokenizer = ByteTokenizer()

    assert BYTE_VOCAB_SIZE == 256
    assert NANOCHAT_SPECIAL_TOKENS is SPECIAL_TOKENS
    assert DEFAULT_SPECIAL_TOKENS is NANOCHAT_SPECIAL_TOKENS
    assert SPECIAL_TOKENS == EXPECTED_SPECIAL_TOKENS
    assert VOCAB_SIZE == 265
    assert tokenizer.get_vocab_size() == VOCAB_SIZE
    assert tokenizer.get_special_tokens() == set(EXPECTED_SPECIAL_TOKENS)
    assert "<|pad|>" not in tokenizer.get_special_tokens()

    for offset, token in enumerate(EXPECTED_SPECIAL_TOKENS):
        token_id = BYTE_VOCAB_SIZE + offset
        assert tokenizer.encode_special(token) == token_id
        assert tokenizer.decode([token_id]) == token
        assert tokenizer.decode_single_token_bytes(token_id) == token.encode("utf-8")

    assert tokenizer.get_bos_token_id() == BYTE_VOCAB_SIZE


def test_byte_tokenizer_implements_the_shared_contract() -> None:
    tokenizer: Tokenizer = ByteTokenizer()

    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.get_identity().startswith("sha256:")
    assert len(tokenizer.get_identity()) == len("sha256:") + 64
    assert tokenizer.get_identity() == ByteTokenizer().get_identity()
    assert tokenizer("hello") == tokenizer.encode("hello")


def test_project_consumers_depend_on_the_shared_tokenizer_contract() -> None:
    assert get_type_hints(checkpoint.ModelCheckpoint)["tokenizer"] is Tokenizer
    assert get_type_hints(checkpoint.save_checkpoint)["tokenizer"] is Tokenizer
    assert get_type_hints(pretraining._build_batches)["tokenizer"] is Tokenizer
    assert get_type_hints(TinyTextTrainingResult)["tokenizer"] is Tokenizer


def test_byte_tokenizer_reports_unsupported_lifecycle_operations(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()

    with pytest.raises(
        UnsupportedTokenizerOperationError,
        match=r"^ByteTokenizer does not support training$",
    ):
        tokenizer.train(["text"])
    with pytest.raises(
        UnsupportedTokenizerOperationError,
        match=r"^ByteTokenizer does not support saving tokenizer artifacts$",
    ):
        tokenizer.save(tmp_path)
    with pytest.raises(
        UnsupportedTokenizerOperationError,
        match=r"^ByteTokenizer does not support loading tokenizer artifacts$",
    ):
        ByteTokenizer.load(tmp_path)


def test_special_tokens_are_explicit_and_can_wrap_ordinary_text() -> None:
    tokenizer = ByteTokenizer()
    text = "hello, \uc138\uacc4"
    bos_id = tokenizer.encode_special("<|bos|>")
    assistant_end_id = tokenizer.encode_special("<|assistant_end|>")

    token_ids = tokenizer.encode(
        text,
        prepend="<|bos|>",
        append=assistant_end_id,
    )

    assert token_ids == [bos_id, *text.encode("utf-8"), assistant_end_id]
    assert tokenizer.decode(token_ids) == f"<|bos|>{text}<|assistant_end|>"
    assert all(0 <= token_id < tokenizer.get_vocab_size() for token_id in token_ids)
    assert tokenizer("<|bos|>") == list(b"<|bos|>")


def test_valid_byte_ids_always_decode_for_generated_output() -> None:
    tokenizer = ByteTokenizer()

    assert tokenizer.decode([0xFF, ord("a"), 0xC3]) == "\ufffda\ufffd"
    assert [
        tokenizer.decode_single_token_bytes(token_id)
        for token_id in range(BYTE_VOCAB_SIZE)
    ] == [bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)]


@pytest.mark.parametrize("token_ids", [[-1], [VOCAB_SIZE], [0, VOCAB_SIZE + 10]])
def test_decode_rejects_out_of_range_ids(token_ids: list[int]) -> None:
    tokenizer = ByteTokenizer()

    with pytest.raises(ValueError, match=r"token ID.*range \[0, 265\)"):
        tokenizer.decode(token_ids)


@pytest.mark.parametrize("token_ids", [[True], [1.5], ["1"]])
def test_decode_rejects_non_integer_ids(token_ids: list[object]) -> None:
    tokenizer = ByteTokenizer()

    with pytest.raises(TypeError, match=r"token ID.*must be an integer"):
        tokenizer.decode(token_ids)  # type: ignore[arg-type]


@pytest.mark.parametrize("token_ids", [1, None])
def test_decode_rejects_non_iterable_token_collections(token_ids: object) -> None:
    tokenizer = ByteTokenizer()

    with pytest.raises(
        TypeError,
        match=rf"^token IDs must be an iterable of integers, got "
        rf"{type(token_ids).__name__}$",
    ):
        tokenizer.decode(token_ids)  # type: ignore[arg-type]


@pytest.mark.parametrize("token_id", [-1, VOCAB_SIZE, True, 1.5])
def test_single_token_decode_validates_the_id(token_id: object) -> None:
    tokenizer = ByteTokenizer()
    error_type = TypeError if isinstance(token_id, (bool, float)) else ValueError

    with pytest.raises(error_type, match="token ID"):
        tokenizer.decode_single_token_bytes(token_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("token", ["<|pad|>", "<|unknown|>", "bos"])
def test_unknown_special_tokens_fail_clearly(token: str) -> None:
    tokenizer = ByteTokenizer()

    expected_message = re.escape(f"unsupported special token {token!r}")
    with pytest.raises(ValueError, match=rf"^{expected_message}$"):
        tokenizer.encode_special(token)


@pytest.mark.parametrize("special", [0, 255, VOCAB_SIZE, True, 1.5])
def test_prepend_and_append_accept_only_supported_specials(special: object) -> None:
    tokenizer = ByteTokenizer()
    error_type = TypeError if isinstance(special, (bool, float)) else ValueError

    with pytest.raises(error_type, match="special token"):
        tokenizer.encode("text", prepend=special)  # type: ignore[arg-type]

    with pytest.raises(error_type, match="special token"):
        tokenizer.encode("text", append=special)  # type: ignore[arg-type]


def test_encode_rejects_non_text_input_clearly() -> None:
    tokenizer = ByteTokenizer()

    with pytest.raises(TypeError, match="text must be a string"):
        tokenizer.encode(b"bytes")  # type: ignore[arg-type]
