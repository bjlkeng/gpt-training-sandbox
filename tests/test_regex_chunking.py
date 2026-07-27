"""Tests for nanochat-compatible Unicode regex chunk boundaries."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any

import pytest
import regex  # type: ignore[import-untyped]

from scratch_llm import regex_chunking
from scratch_llm.regex_chunking import (
    SPLIT_PATTERN,
    RegexChunkingDependencyError,
    bpe_encoding_chunks,
    iter_bpe_training_chunks,
    split_regex_byte_chunks,
    split_regex_chunks,
)
from scratch_llm.tokenizer import SPLIT_PATTERN as TOKENIZER_SPLIT_PATTERN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "data" / "fixtures" / "tokenizer" / "regex_chunks.json"
FIXTURE_README = FIXTURE_PATH.with_name("README.md")
UPSTREAM_COMMIT = "41865401f73ff1c5321ae53297bceb2b78d4c8b4"
EXPECTED_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"
    r"\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|"
    r"\s+(?!\S)|\s+"
)


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_locked_pattern_is_exact_and_requires_third_party_regex() -> None:
    assert SPLIT_PATTERN == EXPECTED_PATTERN
    assert TOKENIZER_SPLIT_PATTERN is SPLIT_PATTERN
    assert regex.compile(SPLIT_PATTERN).pattern == EXPECTED_PATTERN
    with pytest.raises(re.error):
        re.compile(SPLIT_PATTERN)

    distribution = importlib.metadata.distribution("scratch-llm")
    requirements = distribution.metadata.get_all("Requires-Dist") or []
    assert any(
        requirement.split(";", 1)[0].strip() == "regex" for requirement in requirements
    )


def test_pattern_is_compiled_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_compile = regex.compile
    compiled_patterns: list[str] = []

    def record_compile(pattern: str) -> regex.Pattern[str]:
        compiled_patterns.append(pattern)
        return real_compile(pattern)

    regex_chunking._compiled_split_pattern.cache_clear()
    monkeypatch.setattr(regex, "compile", record_compile)

    assert split_regex_chunks("first call") == ("first", " call")
    assert split_regex_chunks("second call") == ("second", " call")
    assert compiled_patterns == [SPLIT_PATTERN]


def test_fixed_upstream_parity_cases_and_utf8_reconstruction() -> None:
    fixture = _load_fixture()

    assert fixture["format"] == "scratch_llm_regex_chunk_parity"
    assert fixture["format_version"] == 1
    assert fixture["pattern"] == SPLIT_PATTERN
    assert fixture["upstream"] == {
        "commit": UPSTREAM_COMMIT,
        "path": "nanochat/tokenizer.py",
        "repository": "https://github.com/karpathy/nanochat",
    }
    assert {case["name"] for case in fixture["cases"]} == {
        "ambiguous_trailing_whitespace",
        "ascii_words",
        "code",
        "contraction_case",
        "emoji",
        "empty",
        "korean",
        "math",
        "one_two_digit_groups",
        "punctuation_plus_newline",
        "spaces_only",
        "tabs_crlf_newlines",
        "unicode_letters_numbers",
    }

    for case in fixture["cases"]:
        text = case["text"]
        expected = tuple(case["chunks"])
        chunks = split_regex_chunks(text)
        byte_chunks = split_regex_byte_chunks(text)

        assert chunks == expected, case["name"]
        assert byte_chunks == tuple(chunk.encode("utf-8") for chunk in expected)
        assert "".join(chunks) == text
        assert b"".join(byte_chunks) == text.encode("utf-8")


def test_chunks_are_immutable_and_callers_receive_fresh_tuples() -> None:
    first = split_regex_chunks("hello world")
    second = split_regex_chunks("hello world")
    byte_chunks = split_regex_byte_chunks("hello world")

    assert first == second == ("hello", " world")
    assert first is not second
    assert isinstance(first, tuple)
    assert isinstance(byte_chunks, tuple)
    assert all(isinstance(chunk, str) for chunk in first)
    assert all(isinstance(chunk, bytes) for chunk in byte_chunks)
    with pytest.raises(TypeError):
        first[0] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        byte_chunks[0] = b"changed"  # type: ignore[index]


def test_training_and_encoding_entry_points_share_the_byte_chunk_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_split(text: str) -> tuple[bytes, ...]:
        calls.append(text)
        return (text.encode("utf-8"),)

    monkeypatch.setattr(regex_chunking, "split_regex_byte_chunks", fake_split)

    assert tuple(iter_bpe_training_chunks(["train one", "train two"])) == (
        b"train one",
        b"train two",
    )
    assert bpe_encoding_chunks("encode") == (b"encode",)
    assert calls == ["train one", "train two", "encode"]


def test_pair_candidates_are_strictly_chunk_local_for_both_entry_points() -> None:
    text = "hello world"
    training_chunks = tuple(iter_bpe_training_chunks([text]))
    encoding_chunks = bpe_encoding_chunks(text)

    assert training_chunks == encoding_chunks == (b"hello", b" world")
    candidate_pairs = {
        pair
        for chunk in training_chunks
        for pair in zip(chunk, chunk[1:], strict=False)
    }
    assert (ord("o"), ord(" ")) not in candidate_pairs
    assert (ord(" "), ord("w")) in candidate_pairs


@pytest.mark.parametrize("value", [None, b"bytes", 123, ["text"]])
def test_splitter_rejects_non_string_input(value: object) -> None:
    with pytest.raises(
        TypeError,
        match=rf"text must be a string, got {type(value).__name__}",
    ):
        split_regex_chunks(value)  # type: ignore[arg-type]


def test_training_entry_point_rejects_a_single_string_or_non_text_item() -> None:
    with pytest.raises(TypeError, match="texts must be an iterable of strings"):
        tuple(iter_bpe_training_chunks("not a corpus"))
    with pytest.raises(TypeError, match=r"text at position 1.*got bytes"):
        tuple(iter_bpe_training_chunks(["valid", b"invalid"]))  # type: ignore[list-item]


def test_missing_regex_dependency_fails_with_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_regex(name: str) -> object:
        assert name == "regex"
        raise ModuleNotFoundError("No module named 'regex'", name="regex")

    regex_chunking._compiled_split_pattern.cache_clear()
    monkeypatch.setattr(regex_chunking, "import_module", missing_regex)

    with pytest.raises(
        RegexChunkingDependencyError,
        match=r"requires the third-party 'regex' package.*base project",
    ):
        split_regex_chunks("text")


def test_fixture_documents_pinned_offline_refresh_procedure() -> None:
    documentation = FIXTURE_README.read_text(encoding="utf-8")

    assert UPSTREAM_COMMIT in documentation
    assert "do not silently regenerate" in documentation
    assert "tests do not" in documentation.lower()
    assert "network" in documentation.lower()
