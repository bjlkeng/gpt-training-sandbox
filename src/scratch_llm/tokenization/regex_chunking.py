"""Locked nanochat regex boundaries shared by byte-BPE training and encoding."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from importlib import import_module
from typing import Final, Protocol, cast


SPLIT_PATTERN: Final = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"
    r"\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|"
    r"\s+(?!\S)|\s+"
)


class RegexChunkingDependencyError(RuntimeError):
    """Unicode regex chunking cannot load its required regex engine."""


class _RegexMatch(Protocol):
    def group(self, group: int = 0) -> str:
        """Return the matched text."""


class _CompiledRegex(Protocol):
    def finditer(self, text: str) -> Iterator[_RegexMatch]:
        """Yield non-overlapping matches in order."""


@lru_cache(maxsize=1)
def _compiled_split_pattern() -> _CompiledRegex:
    """Import and compile the third-party regex pattern exactly once."""

    try:
        regex_module = import_module("regex")
    except ModuleNotFoundError as error:
        if error.name != "regex":
            raise
        raise RegexChunkingDependencyError(
            "nanochat regex chunking requires the third-party 'regex' package; "
            "install the base project dependencies"
        ) from error

    compile_pattern = getattr(regex_module, "compile", None)
    if not callable(compile_pattern):
        raise RegexChunkingDependencyError(
            "the installed 'regex' package does not expose regex.compile"
        )
    return cast(_CompiledRegex, compile_pattern(SPLIT_PATTERN))


def split_regex_chunks(text: str) -> tuple[str, ...]:
    """Return immutable nanochat-compatible chunks that exactly reconstruct text."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    chunks = tuple(match.group(0) for match in _compiled_split_pattern().finditer(text))
    if "".join(chunks) != text:
        raise RuntimeError(
            "locked regex split failed to preserve every input character"
        )
    return chunks


def split_regex_byte_chunks(text: str) -> tuple[bytes, ...]:
    """Return independently encoded immutable byte chunks for byte-BPE."""

    return tuple(chunk.encode("utf-8") for chunk in split_regex_chunks(text))


def iter_bpe_training_chunks(texts: Iterable[str]) -> Iterator[bytes]:
    """Yield chunk-local byte sequences for the reference BPE trainer."""

    if isinstance(texts, (str, bytes)):
        raise TypeError(
            "texts must be an iterable of strings, not a single string or bytes value"
        )
    try:
        iterator = iter(texts)
    except TypeError as error:
        raise TypeError(
            f"texts must be an iterable of strings, got {type(texts).__name__}"
        ) from error

    for position, text in enumerate(iterator):
        if not isinstance(text, str):
            raise TypeError(
                f"text at position {position} must be a string, "
                f"got {type(text).__name__}"
            )
        yield from split_regex_byte_chunks(text)


def bpe_encoding_chunks(text: str) -> tuple[bytes, ...]:
    """Return the shared chunk-local byte input for future BPE encoding."""

    return split_regex_byte_chunks(text)


__all__ = [
    "SPLIT_PATTERN",
    "RegexChunkingDependencyError",
    "bpe_encoding_chunks",
    "iter_bpe_training_chunks",
    "split_regex_byte_chunks",
    "split_regex_chunks",
]
