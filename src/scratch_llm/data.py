"""Small deterministic datasets for the educational training pipeline."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from operator import index as integer_index
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch import Tensor
from torch.utils.data import Dataset

from scratch_llm._validation import require_positive_integer
from scratch_llm.climbmix import (
    CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    DEFAULT_CLIMBMIX_DATA_DIR,
)
from scratch_llm.tokenizer import VOCAB_SIZE


_PARQUET_SHARD_NAME = re.compile(r"^shard_([0-9]+)\.parquet$")


def list_parquet_files(data_dir: str | Path) -> list[Path]:
    """Return canonical parquet shards ordered by their numeric shard index."""

    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"parquet data directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"parquet data path is not a directory: {directory}")

    canonical_paths: list[Path] = []
    for path in directory.iterdir():
        match = _PARQUET_SHARD_NAME.fullmatch(path.name)
        if match is not None and path.is_file():
            canonical_paths.append(path)

    return [path for _, path in _index_parquet_files(canonical_paths)]


def select_parquet_files(
    files: Sequence[str | Path],
    split: str,
    *,
    num_train_shards: int | None = None,
    validation_shard_index: int = CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
) -> list[Path]:
    """Select a train prefix or the fixed validation shard from local files."""

    _validate_parquet_split(split)
    if num_train_shards is not None:
        num_train_shards = _require_non_negative_integer(
            num_train_shards,
            name="num_train_shards",
        )
    validation_shard_index = _require_non_negative_integer(
        validation_shard_index,
        name="validation_shard_index",
    )

    indexed_paths = _index_parquet_files(files)
    for index, path in indexed_paths:
        if index > validation_shard_index:
            raise ValueError(
                f"parquet shard index {index} exceeds configured final validation "
                f"shard index {validation_shard_index}: {path.name}"
            )

    if split == "val":
        for index, path in indexed_paths:
            if index == validation_shard_index:
                return [path]
        raise FileNotFoundError(
            "fixed validation parquet shard with index "
            f"{validation_shard_index} was not found"
        )

    train_paths = [
        path for index, path in indexed_paths if index != validation_shard_index
    ]
    if num_train_shards is None:
        return train_paths
    return train_paths[:num_train_shards]


def parquets_iter_batched(
    split: str,
    start: int = 0,
    step: int = 1,
    *,
    data_dir: str | Path = DEFAULT_CLIMBMIX_DATA_DIR,
    num_train_shards: int | None = None,
    validation_shard_index: int = CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    batch_size: int = 1024,
    text_column: str = "text",
) -> Iterator[list[str]]:
    """Stream bounded text batches from a deterministic stride of split shards."""

    _validate_parquet_split(split)
    start = _require_non_negative_integer(start, name="start")
    step = require_positive_integer(step, name="step")
    if start >= step:
        raise ValueError(
            f"start must be less than step; got start={start}, step={step}"
        )
    batch_size = require_positive_integer(batch_size, name="batch_size")
    if not isinstance(text_column, str):
        raise TypeError(
            f"text_column must be a non-empty string, got {type(text_column).__name__}"
        )
    if not text_column.strip():
        raise ValueError("text_column must be a non-empty string")

    files = select_parquet_files(
        list_parquet_files(data_dir),
        split,
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
    )
    for path in files[start::step]:
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        column_indices = schema.get_all_field_indices(text_column)
        if not column_indices:
            raise ValueError(
                f"text column {text_column!r} was not found in parquet shard "
                f"{path.name}"
            )
        if len(column_indices) > 1:
            raise ValueError(
                f"text column {text_column!r} is ambiguous in parquet shard {path.name}"
            )

        column_type = schema.field(column_indices[0]).type
        if not (
            pa.types.is_string(column_type) or pa.types.is_large_string(column_type)
        ):
            raise TypeError(
                f"text column {text_column!r} must have a string Arrow type; "
                f"got {column_type} in parquet shard {path.name}"
            )

        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[text_column],
            use_threads=False,
        ):
            text_array = record_batch.column(0)
            if text_array.null_count:
                raise ValueError(
                    f"text column {text_column!r} contains null values in "
                    f"parquet shard {path.name}"
                )
            texts = text_array.to_pylist()
            if texts:
                yield texts


def _index_parquet_files(
    files: Sequence[str | Path],
) -> list[tuple[int, Path]]:
    indexed_paths: list[tuple[int, Path]] = []
    for raw_path in files:
        path = Path(raw_path)
        match = _PARQUET_SHARD_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(
                "selected parquet path does not use the canonical "
                f"'shard_<index>.parquet' name: {path}"
            )
        indexed_paths.append((int(match.group(1)), path))

    indexed_paths.sort(key=lambda item: (item[0], str(item[1])))
    for previous, current in zip(indexed_paths, indexed_paths[1:], strict=False):
        if previous[0] == current[0]:
            raise ValueError(
                f"duplicate parquet shard index {current[0]}: "
                f"{previous[1]} and {current[1]}"
            )

    return indexed_paths


def _validate_parquet_split(split: object) -> None:
    if not isinstance(split, str) or split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")


def _require_non_negative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


class NextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose every contiguous fixed-length next-token window in a token stream."""

    def __init__(
        self,
        token_ids: Sequence[int],
        seq_len: int,
        *,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self.seq_len = require_positive_integer(seq_len, name="seq_len")
        self.vocab_size = require_positive_integer(vocab_size, name="vocab_size")
        validated_ids = [
            _validate_token_id(
                token_id,
                position=position,
                vocab_size=self.vocab_size,
            )
            for position, token_id in enumerate(token_ids)
        ]
        self._token_ids = torch.tensor(validated_ids, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of complete ``seq_len + 1`` windows."""

        return max(self._token_ids.numel() - self.seq_len, 0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return input IDs and their one-token-shifted targets."""

        if isinstance(index, bool):
            raise TypeError("dataset index must be an integer, got bool")
        try:
            index = integer_index(index)
        except TypeError as error:
            raise TypeError(
                f"dataset index must be an integer, got {type(index).__name__}"
            ) from error

        dataset_length = len(self)
        if index < 0:
            index += dataset_length
        if index < 0 or index >= dataset_length:
            raise IndexError("dataset index out of range")

        stop = index + self.seq_len
        inputs = self._token_ids[index:stop]
        targets = self._token_ids[index + 1 : stop + 1]
        return inputs, targets


def _validate_token_id(token_id: object, *, position: int, vocab_size: int) -> int:
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise TypeError(
            f"token ID at position {position} must be an integer, "
            f"got {type(token_id).__name__}"
        )
    if not 0 <= token_id < vocab_size:
        raise ValueError(
            f"token ID at position {position} must be in range "
            f"[0, {vocab_size}); got {token_id}"
        )
    return token_id
