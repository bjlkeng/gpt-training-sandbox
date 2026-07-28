"""Incremental pair accounting for scalable, exact regex byte-BPE training."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
import gc
import heapq
from pathlib import Path
import time
import tracemalloc
from types import MappingProxyType
from typing import Any, Final

from scratch_llm.bpe import (
    BPEMerge,
    BPETrainingError,
    ReferenceBPETrainingResult,
    TokenChunk,
    TokenPair,
    _collect_training_chunks,
    _normalized_chunks,
    _normalize_pair,
    _optional_non_negative_integer,
    _require_token_id,
    _validate_special_tokens,
    _validate_vocab_size,
)
from scratch_llm.tokenizer import BYTE_VOCAB_SIZE, NANOCHAT_SPECIAL_TOKENS
from scratch_llm.tokenizer_artifacts import regex_bpe_identity
from scratch_llm.utils import save_json


_MISSING_NODE: Final = -1
_MIN_HEAP_COMPACTION_SIZE: Final = 1024
BPE_TRAINING_BENCHMARK_FORMAT: Final = "scratch_llm_bpe_training_benchmark"
BPE_TRAINING_BENCHMARK_FORMAT_VERSION: Final = 1

TrainingThunk = Callable[[], ReferenceBPETrainingResult]
TrainingMeasure = Callable[
    [str, TrainingThunk],
    tuple[ReferenceBPETrainingResult, float, int],
]


@dataclass(frozen=True)
class BPETrainingBenchmarkMeasurement:
    """One trainer's bounded timing, memory, and semantic result."""

    algorithm: str
    elapsed_seconds: float
    peak_memory_bytes: int
    document_count: int
    character_count: int
    merge_count: int
    tokenizer_identity: str

    def to_dict(self) -> dict[str, int | float | str]:
        """Return deterministic JSON-compatible measurement fields."""

        return {
            "algorithm": self.algorithm,
            "character_count": self.character_count,
            "document_count": self.document_count,
            "elapsed_seconds": self.elapsed_seconds,
            "merge_count": self.merge_count,
            "peak_memory_bytes": self.peak_memory_bytes,
            "tokenizer_identity": self.tokenizer_identity,
        }


@dataclass(frozen=True)
class BPETrainingBenchmark:
    """Immutable reference-versus-optimized bounded benchmark."""

    vocab_size: int
    input_document_count: int
    configured_max_documents: int | None
    configured_max_characters: int | None
    measurements: tuple[BPETrainingBenchmarkMeasurement, ...]
    equivalent: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the benchmark's canonical JSON-compatible contract."""

        return {
            "equivalent": self.equivalent,
            "format": BPE_TRAINING_BENCHMARK_FORMAT,
            "format_version": BPE_TRAINING_BENCHMARK_FORMAT_VERSION,
            "input": {
                "configured_max_characters": self.configured_max_characters,
                "configured_max_documents": self.configured_max_documents,
                "input_document_count": self.input_document_count,
                "vocab_size": self.vocab_size,
            },
            "measurements": [
                measurement.to_dict() for measurement in self.measurements
            ],
            "protocol": {
                "clock": "monotonic",
                "memory": (
                    "Python peak allocations measured independently with tracemalloc"
                ),
                "performance_assertion": (
                    "record values without a wall-clock ratio threshold"
                ),
            },
        }


class IncrementalPairIndex:
    """Track all within-chunk adjacencies and update only merge neighborhoods.

    Token occurrences live in compact linked arrays. Each active adjacency is
    represented exactly once in ``_occurrences`` by its left node index.
    Pair selection uses a lazy max-count/min-pair heap; stale entries are
    discarded and the heap is periodically rebuilt to bound retained history.
    """

    def __init__(self, chunks: Iterable[Sequence[int]]) -> None:
        normalized_chunks = tuple(_normalized_chunks(chunks))
        self._token_ids: list[int] = []
        self._previous = array("q")
        self._next = array("q")
        self._active = bytearray()
        self._chunk_heads: list[int] = []
        self._occurrences: dict[TokenPair, set[int]] = {}
        self._pair_heap: list[tuple[int, int, int]] = []

        for chunk in normalized_chunks:
            head = len(self._token_ids) if chunk else _MISSING_NODE
            self._chunk_heads.append(head)
            for offset, token_id in enumerate(chunk):
                node = len(self._token_ids)
                self._token_ids.append(token_id)
                self._previous.append(node - 1 if offset else _MISSING_NODE)
                self._next.append(
                    node + 1 if offset + 1 < len(chunk) else _MISSING_NODE
                )
                self._active.append(1)

        for left_node in range(len(self._next)):
            self._add_occurrence_at(left_node)
        self._rebuild_heap()

    def pair_counts(self) -> dict[TokenPair, int]:
        """Return exact sorted counts equivalent to a full recount."""

        return dict(
            sorted(
                (pair, len(left_nodes))
                for pair, left_nodes in self._occurrences.items()
                if left_nodes
            )
        )

    def best_pair(self) -> TokenPair:
        """Return the reference trainer's frequency/lexicographic choice."""

        while self._pair_heap:
            negative_count, left_id, right_id = self._pair_heap[0]
            pair = (left_id, right_id)
            current_count = len(self._occurrences.get(pair, ()))
            if current_count and current_count == -negative_count:
                return pair
            heapq.heappop(self._pair_heap)
        raise BPETrainingError("no adjacent token pairs remain to merge")

    def merge_pair(
        self,
        pair: Sequence[int],
        new_token_id: object,
    ) -> int:
        """Merge one selected pair left-to-right and update local adjacencies."""

        normalized_pair = _normalize_pair(pair)
        normalized_new_token_id = _require_token_id(
            new_token_id,
            label="new_token_id",
        )
        selected_left_nodes = self._occurrences.get(normalized_pair)
        if not selected_left_nodes:
            raise BPETrainingError(
                f"pair {normalized_pair} has no active occurrences to merge"
            )

        selected_count = len(selected_left_nodes)
        affected_pairs: set[TokenPair] = set()
        # Node indices retain original corpus order, so this exactly matches the
        # reference implementation's independent left-to-right chunk scans.
        for left_node in sorted(selected_left_nodes):
            if not self._active[left_node]:
                continue
            right_node = self._next[left_node]
            if (
                right_node == _MISSING_NODE
                or not self._active[right_node]
                or (
                    self._token_ids[left_node],
                    self._token_ids[right_node],
                )
                != normalized_pair
            ):
                continue

            previous_node = self._previous[left_node]
            following_node = self._next[right_node]
            self._remove_occurrence_at(previous_node, affected_pairs)
            self._remove_occurrence_at(left_node, affected_pairs)
            self._remove_occurrence_at(right_node, affected_pairs)

            self._token_ids[left_node] = normalized_new_token_id
            self._active[right_node] = 0
            self._previous[right_node] = _MISSING_NODE
            self._next[right_node] = _MISSING_NODE
            self._next[left_node] = following_node
            if following_node != _MISSING_NODE:
                self._previous[following_node] = left_node

            self._add_occurrence_at(previous_node, affected_pairs)
            self._add_occurrence_at(left_node, affected_pairs)

        for affected_pair in affected_pairs:
            count = len(self._occurrences.get(affected_pair, ()))
            if count:
                heapq.heappush(
                    self._pair_heap,
                    (-count, affected_pair[0], affected_pair[1]),
                )
        if len(self._pair_heap) > max(
            _MIN_HEAP_COMPACTION_SIZE,
            4 * len(self._occurrences),
        ):
            self._rebuild_heap()
        return selected_count

    def chunks(self) -> tuple[TokenChunk, ...]:
        """Materialize current chunks for differential checks and fallback."""

        chunks: list[TokenChunk] = []
        for head in self._chunk_heads:
            token_ids: list[int] = []
            node = head
            while node != _MISSING_NODE:
                if not self._active[node]:
                    raise RuntimeError(
                        "inactive node remained linked from a chunk head"
                    )
                token_ids.append(self._token_ids[node])
                node = self._next[node]
            chunks.append(tuple(token_ids))
        return tuple(chunks)

    def _add_occurrence_at(
        self,
        left_node: int,
        affected_pairs: set[TokenPair] | None = None,
    ) -> None:
        if left_node == _MISSING_NODE or not self._active[left_node]:
            return
        right_node = self._next[left_node]
        if right_node == _MISSING_NODE or not self._active[right_node]:
            return
        pair = (
            self._token_ids[left_node],
            self._token_ids[right_node],
        )
        left_nodes = self._occurrences.setdefault(pair, set())
        if left_node in left_nodes:
            return
        left_nodes.add(left_node)
        if affected_pairs is not None:
            affected_pairs.add(pair)

    def _remove_occurrence_at(
        self,
        left_node: int,
        affected_pairs: set[TokenPair],
    ) -> None:
        if left_node == _MISSING_NODE or not self._active[left_node]:
            return
        right_node = self._next[left_node]
        if right_node == _MISSING_NODE or not self._active[right_node]:
            return
        pair = (
            self._token_ids[left_node],
            self._token_ids[right_node],
        )
        left_nodes = self._occurrences.get(pair)
        if left_nodes is None or left_node not in left_nodes:
            raise RuntimeError(
                f"active adjacency at node {left_node} was absent from pair index"
            )
        left_nodes.remove(left_node)
        affected_pairs.add(pair)
        if not left_nodes:
            del self._occurrences[pair]

    def _rebuild_heap(self) -> None:
        self._pair_heap = [
            (-len(left_nodes), pair[0], pair[1])
            for pair, left_nodes in self._occurrences.items()
            if left_nodes
        ]
        heapq.heapify(self._pair_heap)


def train_optimized_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int,
    special_tokens: Iterable[str] = NANOCHAT_SPECIAL_TOKENS,
    max_documents: int | None = None,
    max_characters: int | None = None,
) -> ReferenceBPETrainingResult:
    """Train exact regex byte-BPE using incremental adjacency updates."""

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
    pair_index = IncrementalPairIndex(chunks)
    for new_token_id in range(BYTE_VOCAB_SIZE, mergeable_vocab_size):
        try:
            pair = pair_index.best_pair()
        except BPETrainingError as error:
            raise BPETrainingError(
                "training corpus exhausted all adjacent pairs after "
                f"{len(merges)} merges; requested "
                f"{mergeable_vocab_size - BYTE_VOCAB_SIZE}"
            ) from error
        selected_count = pair_index.merge_pair(pair, new_token_id)
        vocabulary[new_token_id] = vocabulary[pair[0]] + vocabulary[pair[1]]
        merges.append(
            BPEMerge(
                pair=pair,
                token_id=new_token_id,
                count=selected_count,
            )
        )

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
        raise RuntimeError("optimized BPE trainer produced a non-contiguous vocabulary")
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


def benchmark_bpe_trainers(
    texts: Sequence[str],
    *,
    vocab_size: int,
    max_documents: int | None = None,
    max_characters: int | None = None,
    measure: TrainingMeasure | None = None,
) -> BPETrainingBenchmark:
    """Measure both trainers on one bounded, reiterable corpus."""

    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings")
    normalized_texts = tuple(texts)
    if not normalized_texts:
        raise ValueError("texts must contain at least one document")
    max_documents = _optional_non_negative_integer(
        max_documents,
        name="max_documents",
    )
    max_characters = _optional_non_negative_integer(
        max_characters,
        name="max_characters",
    )
    active_measure = _measure_training if measure is None else measure
    if not callable(active_measure):
        raise TypeError(f"measure must be callable, got {type(measure).__name__}")

    measured_results: list[tuple[str, ReferenceBPETrainingResult, float, int]] = []
    for algorithm in ("reference", "optimized"):
        train_candidate: TrainingThunk = partial(
            _train_for_benchmark,
            normalized_texts,
            vocab_size=vocab_size,
            algorithm=algorithm,
            max_documents=max_documents,
            max_characters=max_characters,
        )

        result, elapsed_seconds, peak_memory_bytes = active_measure(
            algorithm,
            train_candidate,
        )
        if not isinstance(result, ReferenceBPETrainingResult):
            raise TypeError("benchmark measure must return ReferenceBPETrainingResult")
        if (
            not isinstance(elapsed_seconds, (int, float))
            or isinstance(elapsed_seconds, bool)
            or elapsed_seconds < 0
        ):
            raise ValueError("benchmark elapsed seconds must be a non-negative number")
        if (
            not isinstance(peak_memory_bytes, int)
            or isinstance(peak_memory_bytes, bool)
            or peak_memory_bytes < 0
        ):
            raise ValueError(
                "benchmark peak memory bytes must be a non-negative integer"
            )
        measured_results.append(
            (
                algorithm,
                result,
                float(elapsed_seconds),
                peak_memory_bytes,
            )
        )

    measurements = tuple(
        BPETrainingBenchmarkMeasurement(
            algorithm=algorithm,
            elapsed_seconds=elapsed_seconds,
            peak_memory_bytes=peak_memory_bytes,
            document_count=result.document_count,
            character_count=result.character_count,
            merge_count=len(result.merges),
            tokenizer_identity=regex_bpe_identity(result),
        )
        for algorithm, result, elapsed_seconds, peak_memory_bytes in measured_results
    )
    return BPETrainingBenchmark(
        vocab_size=vocab_size,
        input_document_count=len(normalized_texts),
        configured_max_documents=max_documents,
        configured_max_characters=max_characters,
        measurements=measurements,
        equivalent=measured_results[0][1] == measured_results[1][1],
    )


def write_bpe_training_benchmark(
    benchmark: BPETrainingBenchmark,
    path: str | Path,
) -> Path:
    """Atomically persist one deterministic benchmark report."""

    if not isinstance(benchmark, BPETrainingBenchmark):
        raise TypeError(
            f"benchmark must be BPETrainingBenchmark, got {type(benchmark).__name__}"
        )
    return save_json(benchmark.to_dict(), path)


def _train_for_benchmark(
    texts: Sequence[str],
    *,
    vocab_size: int,
    algorithm: str,
    max_documents: int | None,
    max_characters: int | None,
) -> ReferenceBPETrainingResult:
    if algorithm == "reference":
        from scratch_llm.bpe import train_reference_bpe

        return train_reference_bpe(
            texts,
            vocab_size=vocab_size,
            max_documents=max_documents,
            max_characters=max_characters,
        )
    return train_optimized_bpe(
        texts,
        vocab_size=vocab_size,
        max_documents=max_documents,
        max_characters=max_characters,
    )


def _measure_training(
    algorithm: str,
    train: TrainingThunk,
) -> tuple[ReferenceBPETrainingResult, float, int]:
    del algorithm
    gc.collect()
    tracemalloc.start()
    started_at = time.monotonic()
    try:
        result = train()
        elapsed_seconds = time.monotonic() - started_at
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed_seconds, peak_memory_bytes


__all__ = [
    "BPE_TRAINING_BENCHMARK_FORMAT",
    "BPE_TRAINING_BENCHMARK_FORMAT_VERSION",
    "BPETrainingBenchmark",
    "BPETrainingBenchmarkMeasurement",
    "IncrementalPairIndex",
    "benchmark_bpe_trainers",
    "train_optimized_bpe",
    "write_bpe_training_benchmark",
]
