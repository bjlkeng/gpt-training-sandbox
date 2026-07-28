"""Bounded parquet-to-tokenizer training orchestration and local reports."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import time
import tracemalloc
from typing import Any, Final

from scratch_llm._validation import require_positive_integer
from scratch_llm.bpe import (
    ReferenceBPETrainingResult,
    RegexBPETokenizer,
    train_bpe,
)
from scratch_llm.config import ProjectConfig
from scratch_llm.data import (
    list_parquet_files,
    parquets_iter_batched,
    select_parquet_files,
)
from scratch_llm.run import RunPaths
from scratch_llm.tokenizer_artifacts import TOKENIZER_ARTIFACT_FILENAMES
from scratch_llm.utils import save_json


TOKENIZER_TRAINING_REPORT_FORMAT: Final = "scratch_llm_tokenizer_training"
TOKENIZER_TRAINING_REPORT_FORMAT_VERSION: Final = 1
TOKENIZER_TRAINING_REPORT_FILENAME: Final = "tokenizer_training.json"


@dataclass(frozen=True)
class TokenizerTrainingRunResult:
    """Durable tokenizer artifacts plus training accounting."""

    algorithm: str
    training_result: ReferenceBPETrainingResult
    tokenizer: RegexBPETokenizer
    selected_shards: tuple[str, ...]
    configured_max_documents: int
    configured_max_characters: int
    document_char_cap: int
    elapsed_seconds: float
    peak_memory_bytes: int
    artifact_dir: Path
    report_path: Path
    run_dir: Path

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible training report."""

        return {
            "algorithm": self.algorithm,
            "artifacts": {
                "directory": str(self.artifact_dir.relative_to(self.run_dir)),
                "files": list(TOKENIZER_ARTIFACT_FILENAMES),
            },
            "corpus": {
                "character_count": self.training_result.character_count,
                "chunk_count": self.training_result.chunk_count,
                "configured_max_characters": self.configured_max_characters,
                "configured_max_documents": self.configured_max_documents,
                "document_char_cap": self.document_char_cap,
                "document_count": self.training_result.document_count,
                "selected_shard_count": len(self.selected_shards),
                "selected_shards": list(self.selected_shards),
            },
            "format": TOKENIZER_TRAINING_REPORT_FORMAT,
            "format_version": TOKENIZER_TRAINING_REPORT_FORMAT_VERSION,
            "merge_count": len(self.training_result.merges),
            "performance": {
                "elapsed_seconds": self.elapsed_seconds,
                "peak_memory_bytes": self.peak_memory_bytes,
            },
            "tokenizer_identity": self.tokenizer.get_identity(),
            "vocab_size": self.training_result.vocab_size,
        }


def train_tokenizer_from_parquet(
    config: ProjectConfig,
    paths: RunPaths,
    *,
    algorithm: str = "optimized",
    clock: Callable[[], float] = time.monotonic,
) -> TokenizerTrainingRunResult:
    """Train and atomically publish one configured regex byte-BPE tokenizer."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be ProjectConfig, got {type(config).__name__}")
    if not isinstance(paths, RunPaths):
        raise TypeError(f"paths must be RunPaths, got {type(paths).__name__}")
    if config.tokenizer.type != "regex_byte_bpe":
        raise ValueError("train_tokenizer requires tokenizer.type='regex_byte_bpe'")
    if not callable(clock):
        raise TypeError(f"clock must be callable, got {type(clock).__name__}")
    config.validate()

    data_dir = Path(config.data.parquet_dir)
    selected_files = select_parquet_files(
        list_parquet_files(data_dir),
        "train",
        num_train_shards=config.data.num_tokenizer_train_shards,
        validation_shard_index=config.data.max_shard,
    )
    artifact_dir = paths.run_dir / "artifacts" / "tokenizer"
    report_path = paths.metrics_dir / TOKENIZER_TRAINING_REPORT_FILENAME

    tracemalloc.start()
    started_at = clock()
    try:
        training_result = train_bpe(
            iter_parquet_training_texts(config),
            vocab_size=config.tokenizer.vocab_size,
            algorithm=algorithm,
            special_tokens=config.tokenizer.special_tokens,
            max_documents=config.tokenizer.doc_cap,
            max_characters=config.tokenizer.max_chars,
        )
        elapsed_seconds = clock() - started_at
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if elapsed_seconds < 0:
        raise ValueError("monotonic clock moved backwards during tokenizer training")

    tokenizer = RegexBPETokenizer(training_result)
    tokenizer.save(artifact_dir)
    result = TokenizerTrainingRunResult(
        algorithm=algorithm,
        training_result=training_result,
        tokenizer=tokenizer,
        selected_shards=tuple(path.name for path in selected_files),
        configured_max_documents=config.tokenizer.doc_cap,
        configured_max_characters=config.tokenizer.max_chars,
        document_char_cap=config.data.doc_cap_chars,
        elapsed_seconds=elapsed_seconds,
        peak_memory_bytes=peak_memory_bytes,
        artifact_dir=artifact_dir,
        report_path=report_path,
        run_dir=paths.run_dir,
    )
    save_json(result.to_dict(), report_path)
    return result


def iter_parquet_training_texts(config: ProjectConfig) -> Iterator[str]:
    """Yield canonical training documents with the configured per-doc cap."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be ProjectConfig, got {type(config).__name__}")
    for batch in parquets_iter_batched(
        "train",
        data_dir=config.data.parquet_dir,
        num_train_shards=config.data.num_tokenizer_train_shards,
        validation_shard_index=config.data.max_shard,
        text_column=config.data.text_column,
    ):
        for text in batch:
            yield text[: config.data.doc_cap_chars]


def collect_bounded_parquet_training_texts(
    config: ProjectConfig,
    *,
    max_documents: int,
    max_characters: int,
) -> tuple[str, ...]:
    """Materialize an explicitly bounded prefix for differential benchmarks."""

    max_documents = require_positive_integer(
        max_documents,
        name="max_documents",
    )
    max_characters = require_positive_integer(
        max_characters,
        name="max_characters",
    )
    documents: list[str] = []
    character_count = 0
    for text in iter_parquet_training_texts(config):
        if len(documents) >= max_documents or character_count >= max_characters:
            break
        bounded = text[: max_characters - character_count]
        documents.append(bounded)
        character_count += len(bounded)
    if not documents:
        raise ValueError(
            "bounded tokenizer benchmark did not yield any training documents"
        )
    return tuple(documents)


__all__ = [
    "TOKENIZER_TRAINING_REPORT_FILENAME",
    "TOKENIZER_TRAINING_REPORT_FORMAT",
    "TOKENIZER_TRAINING_REPORT_FORMAT_VERSION",
    "TokenizerTrainingRunResult",
    "collect_bounded_parquet_training_texts",
    "iter_parquet_training_texts",
    "train_tokenizer_from_parquet",
]
