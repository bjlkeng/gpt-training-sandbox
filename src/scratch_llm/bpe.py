"""Obviously-correct reference training primitives for regex byte-BPE."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from scratch_llm.regex_chunking import iter_bpe_training_chunks
from scratch_llm.tokenizer import BYTE_VOCAB_SIZE, NANOCHAT_SPECIAL_TOKENS


PAIR_TIE_BREAK: Final = (
    "highest frequency, then lexicographically smallest (left_id, right_id)"
)
TokenPair = tuple[int, int]
TokenChunk = tuple[int, ...]


class BPETrainingError(ValueError):
    """The requested vocabulary cannot be learned from the bounded corpus."""


@dataclass(frozen=True)
class BPEMerge:
    """One learned pair replacement in rank and token-ID order."""

    pair: TokenPair
    token_id: int
    count: int

    @property
    def left_id(self) -> int:
        return self.pair[0]

    @property
    def right_id(self) -> int:
        return self.pair[1]


@dataclass(frozen=True)
class ReferenceBPETrainingResult:
    """Immutable learned vocabulary plus bounded-corpus accounting."""

    vocab_size: int
    mergeable_vocab_size: int
    merges: tuple[BPEMerge, ...]
    vocabulary: Mapping[int, bytes]
    special_token_ids: Mapping[str, int]
    document_count: int
    character_count: int
    chunk_count: int


def count_pairs(chunks: Iterable[Sequence[int]]) -> dict[TokenPair, int]:
    """Count adjacent pairs independently inside each token chunk.

    The sorted result order is only for readability; training selection uses
    the explicit frequency/lexicographic rule in :data:`PAIR_TIE_BREAK`.
    """

    counts: Counter[TokenPair] = Counter()
    for chunk in _normalized_chunks(chunks):
        counts.update(zip(chunk, chunk[1:], strict=False))
    return dict(sorted(counts.items()))


def merge_pair(
    chunk: Sequence[int],
    pair: Sequence[int],
    new_token_id: int,
) -> TokenChunk:
    """Replace non-overlapping ``pair`` occurrences from left to right."""

    normalized_chunk = _normalize_chunk(chunk, label="chunk")
    normalized_pair = _normalize_pair(pair)
    new_token_id = _require_token_id(new_token_id, label="new_token_id")

    merged: list[int] = []
    index = 0
    while index < len(normalized_chunk):
        if (
            index + 1 < len(normalized_chunk)
            and normalized_chunk[index] == normalized_pair[0]
            and normalized_chunk[index + 1] == normalized_pair[1]
        ):
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(normalized_chunk[index])
            index += 1
    return tuple(merged)


def apply_merge(
    chunks: Iterable[Sequence[int]],
    pair: Sequence[int],
    new_token_id: int,
) -> tuple[TokenChunk, ...]:
    """Apply one merge independently to every chunk without mutating inputs."""

    normalized_pair = _normalize_pair(pair)
    new_token_id = _require_token_id(new_token_id, label="new_token_id")
    return tuple(
        merge_pair(chunk, normalized_pair, new_token_id)
        for chunk in _normalized_chunks(chunks)
    )


def select_best_pair(pair_counts: Mapping[TokenPair, int]) -> TokenPair:
    """Select the most frequent pair under the documented stable tie-break."""

    if not isinstance(pair_counts, Mapping):
        raise TypeError(
            "pair_counts must be a mapping from token pairs to positive counts, "
            f"got {type(pair_counts).__name__}"
        )
    if not pair_counts:
        raise BPETrainingError("no adjacent token pairs remain to merge")

    normalized: dict[TokenPair, int] = {}
    for raw_pair, raw_count in pair_counts.items():
        pair = _normalize_pair(raw_pair)
        if not isinstance(raw_count, int) or isinstance(raw_count, bool):
            raise TypeError(f"pair count for {pair} must be an integer")
        if raw_count <= 0:
            raise ValueError(f"pair count for {pair} must be positive")
        normalized[pair] = raw_count
    return min(normalized, key=lambda pair: (-normalized[pair], pair))


def train_reference_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int,
    special_tokens: Iterable[str] = NANOCHAT_SPECIAL_TOKENS,
    max_documents: int | None = None,
    max_characters: int | None = None,
) -> ReferenceBPETrainingResult:
    """Train the deliberately slow reference regex byte-BPE vocabulary.

    ``vocab_size`` includes the ordered special tokens. Corpus inputs are
    consumed exactly once. ``max_documents`` counts empty documents, while
    ``max_characters`` is an exact Unicode-character budget that may retain a
    final document prefix. Exhaustion before the requested mergeable vocabulary
    is reached is an error rather than a silently undersized result.
    """

    ordered_special_tokens = _validate_special_tokens(special_tokens)
    vocab_size = _validate_vocab_size(
        vocab_size,
        special_token_count=len(ordered_special_tokens),
    )
    max_documents = _optional_non_negative_integer(
        max_documents,
        name="max_documents",
    )
    max_characters = _optional_non_negative_integer(
        max_characters,
        name="max_characters",
    )
    chunks, document_count, character_count = _collect_training_chunks(
        texts,
        max_documents=max_documents,
        max_characters=max_characters,
    )
    if document_count == 0:
        raise BPETrainingError(
            "training corpus did not yield any documents under the configured limits"
        )
    if not chunks:
        raise BPETrainingError(
            "training corpus documents produced no regex byte chunks"
        )

    mergeable_vocab_size = vocab_size - len(ordered_special_tokens)
    vocabulary: dict[int, bytes] = {
        token_id: bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)
    }
    merges: list[BPEMerge] = []
    working_chunks = chunks
    for new_token_id in range(BYTE_VOCAB_SIZE, mergeable_vocab_size):
        pair_counts = count_pairs(working_chunks)
        if not pair_counts:
            raise BPETrainingError(
                "training corpus exhausted all adjacent pairs after "
                f"{len(merges)} merges; requested {mergeable_vocab_size - BYTE_VOCAB_SIZE}"
            )
        pair = select_best_pair(pair_counts)
        vocabulary[new_token_id] = vocabulary[pair[0]] + vocabulary[pair[1]]
        merges.append(
            BPEMerge(
                pair=pair,
                token_id=new_token_id,
                count=pair_counts[pair],
            )
        )
        working_chunks = apply_merge(working_chunks, pair, new_token_id)

    special_token_ids = {
        token: mergeable_vocab_size + offset
        for offset, token in enumerate(ordered_special_tokens)
    }
    vocabulary.update(
        {
            token_id: token.encode("utf-8")
            for token, token_id in special_token_ids.items()
        }
    )
    if tuple(vocabulary) != tuple(range(vocab_size)):
        raise RuntimeError("reference BPE trainer produced a non-contiguous vocabulary")

    return ReferenceBPETrainingResult(
        vocab_size=vocab_size,
        mergeable_vocab_size=mergeable_vocab_size,
        merges=tuple(merges),
        vocabulary=MappingProxyType(vocabulary),
        special_token_ids=MappingProxyType(special_token_ids),
        document_count=document_count,
        character_count=character_count,
        chunk_count=len(chunks),
    )


def _collect_training_chunks(
    texts: Iterable[str],
    *,
    max_documents: int | None,
    max_characters: int | None,
) -> tuple[tuple[TokenChunk, ...], int, int]:
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

    chunks: list[TokenChunk] = []
    document_count = 0
    character_count = 0
    while max_documents is None or document_count < max_documents:
        if max_characters is not None and character_count >= max_characters:
            break
        try:
            text = next(iterator)
        except StopIteration:
            break
        if not isinstance(text, str):
            raise TypeError(
                f"text at position {document_count} must be a string, "
                f"got {type(text).__name__}"
            )

        bounded_text = text
        if max_characters is not None:
            remaining = max_characters - character_count
            bounded_text = text[:remaining]
        document_count += 1
        character_count += len(bounded_text)
        chunks.extend(
            tuple(byte_chunk)
            for byte_chunk in iter_bpe_training_chunks((bounded_text,))
        )
        if len(bounded_text) < len(text):
            break
    return tuple(chunks), document_count, character_count


def _normalized_chunks(
    chunks: Iterable[Sequence[int]],
) -> Iterable[TokenChunk]:
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be an iterable of token sequences")
    try:
        iterator = iter(chunks)
    except TypeError as error:
        raise TypeError(
            f"chunks must be iterable, got {type(chunks).__name__}"
        ) from error
    for index, chunk in enumerate(iterator):
        yield _normalize_chunk(chunk, label=f"chunk at position {index}")


def _normalize_chunk(chunk: Sequence[int], *, label: str) -> TokenChunk:
    if isinstance(chunk, str):
        raise TypeError(f"{label} must be a sequence of integer token IDs")
    try:
        values = tuple(chunk)
    except TypeError as error:
        raise TypeError(
            f"{label} must be a sequence of integer token IDs, "
            f"got {type(chunk).__name__}"
        ) from error
    return tuple(
        _require_token_id(token_id, label=f"{label} token {index}")
        for index, token_id in enumerate(values)
    )


def _normalize_pair(pair: Sequence[int]) -> TokenPair:
    normalized = _normalize_chunk(pair, label="pair")
    if len(normalized) != 2:
        raise ValueError(
            f"pair must contain exactly two token IDs, got {len(normalized)}"
        )
    return normalized[0], normalized[1]


def _require_token_id(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


def _validate_special_tokens(special_tokens: Iterable[str]) -> tuple[str, ...]:
    if isinstance(special_tokens, (str, bytes)):
        raise TypeError("special_tokens must be an ordered iterable of strings")
    try:
        ordered = tuple(special_tokens)
    except TypeError as error:
        raise TypeError(
            "special_tokens must be an ordered iterable of strings"
        ) from error
    for index, token in enumerate(ordered):
        if not isinstance(token, str):
            raise TypeError(
                f"special token at position {index} must be a string, "
                f"got {type(token).__name__}"
            )
        if not token:
            raise ValueError(f"special token at position {index} must not be empty")
    if len(set(ordered)) != len(ordered):
        raise ValueError("special_tokens must not contain duplicates")
    return ordered


def _validate_vocab_size(vocab_size: object, *, special_token_count: int) -> int:
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        raise TypeError(
            f"vocab_size must be an integer, got {type(vocab_size).__name__}"
        )
    minimum = BYTE_VOCAB_SIZE + special_token_count
    if vocab_size < minimum:
        raise ValueError(
            f"vocab_size must be at least {minimum} for 256 bytes and "
            f"{special_token_count} special tokens, got {vocab_size}"
        )
    return vocab_size


def _optional_non_negative_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


__all__ = [
    "PAIR_TIE_BREAK",
    "BPEMerge",
    "BPETrainingError",
    "ReferenceBPETrainingResult",
    "TokenChunk",
    "TokenPair",
    "apply_merge",
    "count_pairs",
    "merge_pair",
    "select_best_pair",
    "train_reference_bpe",
]
