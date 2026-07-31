"""Deterministic, JSON-compatible statistics for raw ClimbMix parquet data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scratch_llm._validation import (
    require_optional_positive_integer,
    require_positive_integer,
)
from scratch_llm.climbmix import (
    CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    DEFAULT_CLIMBMIX_DATA_DIR,
)
from scratch_llm.data import (
    list_parquet_files,
    parquets_iter_batched,
    select_parquet_files,
)
from scratch_llm.utils import save_json


RAW_DATA_STATISTICS_FORMAT = "scratch_llm_raw_data_statistics"
RAW_DATA_STATISTICS_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RawDataSplitStatistics:
    """Counts and canonical source identities for one raw-data split."""

    selected_shards: tuple[str, ...]
    documents: int
    characters: int
    utf8_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return the split's deterministic JSON representation."""

        return {
            "characters": self.characters,
            "documents": self.documents,
            "selected_shard_count": len(self.selected_shards),
            "selected_shards": list(self.selected_shards),
            "utf8_bytes": self.utf8_bytes,
        }


@dataclass(frozen=True)
class RawDataStatistics:
    """Immutable result shared by the data-stats CLI and later tracking."""

    data_dir: str
    text_column: str
    num_train_shards: int | None
    include_validation: bool
    max_documents: int | None
    max_characters: int | None
    document_char_cap: int | None
    train: RawDataSplitStatistics
    validation: RawDataSplitStatistics

    @property
    def bounded(self) -> bool:
        """Whether explicit document or character limits bound each split."""

        return any(
            limit is not None
            for limit in (
                self.max_documents,
                self.max_characters,
                self.document_char_cap,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible result contract."""

        return {
            "bounded": self.bounded,
            "format": RAW_DATA_STATISTICS_FORMAT,
            "format_version": RAW_DATA_STATISTICS_FORMAT_VERSION,
            "limits": {
                "document_char_cap": self.document_char_cap,
                "max_characters": self.max_characters,
                "max_documents": self.max_documents,
            },
            "selection": {
                "data_dir": self.data_dir,
                "include_validation": self.include_validation,
                "num_train_shards": self.num_train_shards,
                "text_column": self.text_column,
            },
            "splits": {
                "train": self.train.to_dict(),
                "validation": self.validation.to_dict(),
            },
            "total": {
                "characters": self.train.characters + self.validation.characters,
                "documents": self.train.documents + self.validation.documents,
                "selected_shard_count": (
                    len(self.train.selected_shards)
                    + len(self.validation.selected_shards)
                ),
                "utf8_bytes": self.train.utf8_bytes + self.validation.utf8_bytes,
            },
        }


def compute_raw_data_statistics(
    data_dir: str | Path = DEFAULT_CLIMBMIX_DATA_DIR,
    *,
    num_train_shards: int | None = None,
    include_validation: bool = True,
    max_documents: int | None = None,
    max_characters: int | None = None,
    document_char_cap: int | None = None,
    validation_shard_index: int = CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    batch_size: int = 1024,
    text_column: str = "text",
) -> RawDataStatistics:
    """Count selected raw documents after applying explicit input limits.

    Limits apply independently to each selected split. ``document_char_cap`` is
    applied to each document first, ``max_documents`` bounds the number of
    documents consumed (including empty strings), and ``max_characters`` may
    retain a final document prefix to reach its exact character budget.
    """

    if not isinstance(include_validation, bool):
        raise TypeError(
            "include_validation must be a boolean, "
            f"got {type(include_validation).__name__}"
        )
    max_documents = require_optional_positive_integer(
        max_documents,
        name="max_documents",
    )
    max_characters = require_optional_positive_integer(
        max_characters,
        name="max_characters",
    )
    document_char_cap = require_optional_positive_integer(
        document_char_cap,
        name="document_char_cap",
    )
    batch_size = require_positive_integer(batch_size, name="batch_size")

    directory = Path(data_dir)
    discovered = list_parquet_files(directory)
    train_files = select_parquet_files(
        discovered,
        "train",
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
    )
    validation_files = (
        select_parquet_files(
            discovered,
            "val",
            validation_shard_index=validation_shard_index,
        )
        if include_validation
        else []
    )

    train = _count_split(
        "train",
        selected_files=train_files,
        data_dir=directory,
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
        batch_size=batch_size,
        text_column=text_column,
        max_documents=max_documents,
        max_characters=max_characters,
        document_char_cap=document_char_cap,
    )
    validation = (
        _count_split(
            "val",
            selected_files=validation_files,
            data_dir=directory,
            num_train_shards=None,
            validation_shard_index=validation_shard_index,
            batch_size=batch_size,
            text_column=text_column,
            max_documents=max_documents,
            max_characters=max_characters,
            document_char_cap=document_char_cap,
        )
        if include_validation
        else RawDataSplitStatistics(
            selected_shards=(),
            documents=0,
            characters=0,
            utf8_bytes=0,
        )
    )
    return RawDataStatistics(
        data_dir=str(directory),
        text_column=text_column,
        num_train_shards=num_train_shards,
        include_validation=include_validation,
        max_documents=max_documents,
        max_characters=max_characters,
        document_char_cap=document_char_cap,
        train=train,
        validation=validation,
    )


def write_raw_data_statistics(
    result: RawDataStatistics,
    path: str | Path,
) -> Path:
    """Atomically write one canonical raw-data statistics report."""

    if not isinstance(result, RawDataStatistics):
        raise TypeError(
            f"result must be RawDataStatistics, got {type(result).__name__}"
        )
    return save_json(result.to_dict(), path)


def _count_split(
    split: str,
    *,
    selected_files: list[Path],
    data_dir: Path,
    num_train_shards: int | None,
    validation_shard_index: int,
    batch_size: int,
    text_column: str,
    max_documents: int | None,
    max_characters: int | None,
    document_char_cap: int | None,
) -> RawDataSplitStatistics:
    documents = 0
    characters = 0
    utf8_bytes = 0
    stop = False

    for batch in parquets_iter_batched(
        split,
        data_dir=data_dir,
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
        batch_size=batch_size,
        text_column=text_column,
    ):
        for text in batch:
            if max_documents is not None and documents >= max_documents:
                stop = True
                break
            if max_characters is not None and characters >= max_characters:
                stop = True
                break

            bounded_text = (
                text[:document_char_cap] if document_char_cap is not None else text
            )
            if max_characters is not None:
                bounded_text = bounded_text[: max_characters - characters]

            documents += 1
            characters += len(bounded_text)
            utf8_bytes += len(bounded_text.encode("utf-8"))
        if stop:
            break

    return RawDataSplitStatistics(
        selected_shards=tuple(path.name for path in selected_files),
        documents=documents,
        characters=characters,
        utf8_bytes=utf8_bytes,
    )


__all__ = [
    "RAW_DATA_STATISTICS_FORMAT",
    "RAW_DATA_STATISTICS_FORMAT_VERSION",
    "RawDataSplitStatistics",
    "RawDataStatistics",
    "compute_raw_data_statistics",
    "write_raw_data_statistics",
]
