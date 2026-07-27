"""Small deterministic datasets for the educational training pipeline."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
import re
from operator import index as integer_index
from pathlib import Path
from typing import Literal

import numpy as np
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
from scratch_llm.tokenized_data import (
    TOKENIZED_MANIFEST_NAME,
    TOKENIZED_SHARD_FORMAT,
    TOKENIZED_SHARD_FORMAT_VERSION,
    TokenizedDataError,
    TokenizedDatasetManifest,
    TokenizedShardManifest,
    TokenizedShardReader,
    TokenizedShardSource,
    TokenizedSplitManifest,
    write_tokenized_shards,
)
from scratch_llm.tokenizer import VOCAB_SIZE, Tokenizer


_PARQUET_SHARD_NAME = re.compile(r"^shard_([0-9]+)\.parquet$")
RANDOM_OFFSET_LOADER_STATE_FORMAT = "scratch_llm_random_offset_loader_state"
RANDOM_OFFSET_LOADER_STATE_FORMAT_VERSION = 1
_RANDOM_OFFSET_LOADER_STATE_KEYS = frozenset(
    {
        "batch_size",
        "format",
        "format_version",
        "manifest_identity",
        "position",
        "rng_state",
        "seq_len",
        "split",
    }
)
_MAX_TORCH_SEED = 2**63 - 1


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
    _validate_text_column(text_column)

    files = select_parquet_files(
        list_parquet_files(data_dir),
        split,
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
    )
    for path in files[start::step]:
        yield from _parquet_file_batches(
            path,
            batch_size=batch_size,
            text_column=text_column,
        )


def write_tokenized_parquet_shards(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Tokenizer,
    num_train_shards: int | None = None,
    validation_shard_index: int = CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    batch_size: int = 1024,
    text_column: str = "text",
    overwrite: bool = False,
) -> TokenizedDatasetManifest:
    """Tokenize selected parquet shards into a validated local dataset."""

    batch_size = require_positive_integer(batch_size, name="batch_size")
    _validate_text_column(text_column)
    files = list_parquet_files(data_dir)
    train_files = select_parquet_files(
        files,
        "train",
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
    )
    val_files = select_parquet_files(
        files,
        "val",
        validation_shard_index=validation_shard_index,
    )
    train_sources = tuple(
        TokenizedShardSource(
            identity=path.name,
            documents=_parquet_file_documents(
                path,
                batch_size=batch_size,
                text_column=text_column,
            ),
        )
        for path in train_files
    )
    val_sources = tuple(
        TokenizedShardSource(
            identity=path.name,
            documents=_parquet_file_documents(
                path,
                batch_size=batch_size,
                text_column=text_column,
            ),
        )
        for path in val_files
    )
    return write_tokenized_shards(
        output_dir,
        tokenizer=tokenizer,
        train_sources=train_sources,
        val_sources=val_sources,
        overwrite=overwrite,
    )


def _parquet_file_documents(
    path: Path,
    *,
    batch_size: int,
    text_column: str,
) -> Iterator[str]:
    for batch in _parquet_file_batches(
        path,
        batch_size=batch_size,
        text_column=text_column,
    ):
        yield from batch


def _parquet_file_batches(
    path: Path,
    *,
    batch_size: int,
    text_column: str,
) -> Iterator[list[str]]:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    column_indices = schema.get_all_field_indices(text_column)
    if not column_indices:
        raise ValueError(
            f"text column {text_column!r} was not found in parquet shard {path.name}"
        )
    if len(column_indices) > 1:
        raise ValueError(
            f"text column {text_column!r} is ambiguous in parquet shard {path.name}"
        )

    column_type = schema.field(column_indices[0]).type
    if not (pa.types.is_string(column_type) or pa.types.is_large_string(column_type)):
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


def _validate_text_column(text_column: object) -> None:
    if not isinstance(text_column, str):
        raise TypeError(
            f"text_column must be a non-empty string, got {type(text_column).__name__}"
        )
    if not text_column.strip():
        raise ValueError("text_column must be a non-empty string")


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


class RandomOffsetTokenLoaderStateError(ValueError):
    """A saved random-offset loader state is malformed or incompatible."""


class RandomOffsetTokenLoader(
    Iterator[tuple[Tensor, Tensor]],
):
    """Yield restartable random contiguous batches from validated token shards.

    Sampling is uniform over the union of every shard-local start with
    ``seq_len + 1`` available tokens. Shards stay memory-mapped; only the
    requested windows are copied into a CPU ``torch.long`` batch.
    """

    def __init__(
        self,
        reader: TokenizedShardReader,
        *,
        split: Literal["train", "val"],
        batch_size: int,
        seq_len: int,
        seed: int,
    ) -> None:
        if not isinstance(reader, TokenizedShardReader):
            raise TypeError(
                f"reader must be a TokenizedShardReader, got {type(reader).__name__}"
            )
        if not isinstance(split, str) or split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.batch_size = require_positive_integer(batch_size, name="batch_size")
        self.seq_len = require_positive_integer(seq_len, name="seq_len")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
        if not 0 <= seed <= _MAX_TORCH_SEED:
            raise ValueError(
                f"seed must be in range [0, {_MAX_TORCH_SEED}], got {seed}"
            )

        mapped_shards = reader.shards(split)
        cumulative_starts: list[int] = []
        valid_start_count = 0
        for shard in mapped_shards:
            valid_start_count += max(len(shard) - self.seq_len, 0)
            cumulative_starts.append(valid_start_count)
        if valid_start_count == 0:
            required_tokens = self.seq_len + 1
            lengths = [len(shard) for shard in mapped_shards]
            raise ValueError(
                f"{split} split has no complete windows requiring "
                f"{required_tokens} tokens; shard lengths={lengths}"
            )
        if valid_start_count > torch.iinfo(torch.int64).max:
            raise ValueError(
                "valid random-offset start count exceeds torch.int64 capacity: "
                f"{valid_start_count}"
            )

        self.reader = reader
        self.split = split
        self.valid_start_count = valid_start_count
        self._mapped_shards = mapped_shards
        self._cumulative_starts = tuple(cumulative_starts)
        self._manifest_identity = _tokenized_manifest_identity(reader.manifest)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self.position = 0

    def __iter__(self) -> RandomOffsetTokenLoader:
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        return self.next_batch()

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Sample and materialize the next CPU batch."""

        # This also turns a reader closed after loader construction into an
        # actionable error before touching a closed memmap.
        self.reader.shards(self.split)
        offsets = torch.randint(
            0,
            self.valid_start_count,
            (self.batch_size,),
            generator=self._generator,
            dtype=torch.int64,
            device="cpu",
        )
        windows = torch.empty(
            (self.batch_size, self.seq_len + 1),
            dtype=torch.long,
            device="cpu",
        )
        for row, global_offset in enumerate(offsets.tolist()):
            shard_index = bisect_right(
                self._cumulative_starts,
                global_offset,
            )
            previous_end = (
                self._cumulative_starts[shard_index - 1] if shard_index > 0 else 0
            )
            local_offset = global_offset - previous_end
            mapped_window = self._mapped_shards[shard_index][
                local_offset : local_offset + self.seq_len + 1
            ]
            copied_window = np.array(mapped_window, dtype=np.int64, copy=True)
            windows[row].copy_(torch.from_numpy(copied_window))

        self.position += self.batch_size
        return windows[:, :-1], windows[:, 1:]

    def state_dict(self) -> dict[str, object]:
        """Return a small backend-neutral state for exact next-batch resume."""

        return {
            "batch_size": self.batch_size,
            "format": RANDOM_OFFSET_LOADER_STATE_FORMAT,
            "format_version": RANDOM_OFFSET_LOADER_STATE_FORMAT_VERSION,
            "manifest_identity": self._manifest_identity,
            "position": self.position,
            "rng_state": self._generator.get_state().tolist(),
            "seq_len": self.seq_len,
            "split": self.split,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate and restore state without partially mutating this loader."""

        if not isinstance(state, Mapping):
            raise RandomOffsetTokenLoaderStateError(
                f"loader state must be a mapping, got {type(state).__name__}"
            )
        state_keys = set(state)
        if state_keys != _RANDOM_OFFSET_LOADER_STATE_KEYS:
            missing = sorted(_RANDOM_OFFSET_LOADER_STATE_KEYS - state_keys)
            unexpected = sorted(
                state_keys - _RANDOM_OFFSET_LOADER_STATE_KEYS,
                key=str,
            )
            raise RandomOffsetTokenLoaderStateError(
                "loader state fields do not match format version "
                f"{RANDOM_OFFSET_LOADER_STATE_FORMAT_VERSION}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if state["format"] != RANDOM_OFFSET_LOADER_STATE_FORMAT:
            raise RandomOffsetTokenLoaderStateError(
                f"unknown loader state format {state['format']!r}"
            )
        format_version = _loader_state_integer(
            state["format_version"],
            name="format version",
        )
        if format_version != RANDOM_OFFSET_LOADER_STATE_FORMAT_VERSION:
            raise RandomOffsetTokenLoaderStateError(
                f"unknown loader state format version {format_version}; "
                f"expected {RANDOM_OFFSET_LOADER_STATE_FORMAT_VERSION}"
            )
        if state["manifest_identity"] != self._manifest_identity:
            raise RandomOffsetTokenLoaderStateError(
                "loader state manifest identity does not match the mapped dataset"
            )
        _require_loader_setting(state, "split", self.split)
        _require_loader_setting(state, "batch_size", self.batch_size)
        _require_loader_setting(state, "seq_len", self.seq_len)

        position = _loader_state_integer(state["position"], name="position")
        if position < 0 or position % self.batch_size != 0:
            raise RandomOffsetTokenLoaderStateError(
                "loader state position must be a non-negative multiple of "
                f"batch_size={self.batch_size}, got {position}"
            )
        raw_rng_state = state["rng_state"]
        if not isinstance(raw_rng_state, list) or not raw_rng_state:
            raise RandomOffsetTokenLoaderStateError(
                "loader state rng_state must be a non-empty list of bytes"
            )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 255
            for value in raw_rng_state
        ):
            raise RandomOffsetTokenLoaderStateError(
                "loader state rng_state must contain only integer bytes"
            )
        rng_state = torch.tensor(raw_rng_state, dtype=torch.uint8, device="cpu")
        candidate_generator = torch.Generator(device="cpu")
        try:
            candidate_generator.set_state(rng_state)
        except RuntimeError as error:
            raise RandomOffsetTokenLoaderStateError(
                f"loader state rng_state is invalid: {error}"
            ) from error

        self._generator.set_state(rng_state)
        self.position = position


def _tokenized_manifest_identity(manifest: TokenizedDatasetManifest) -> str:
    payload = json.dumps(
        manifest.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _loader_state_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RandomOffsetTokenLoaderStateError(
            f"loader state {name} must be an integer"
        )
    return value


def _require_loader_setting(
    state: Mapping[str, object],
    name: str,
    expected: object,
) -> None:
    actual = state[name]
    if isinstance(expected, int):
        actual = _loader_state_integer(actual, name=name)
    elif isinstance(expected, str) and not isinstance(actual, str):
        raise RandomOffsetTokenLoaderStateError(f"loader state {name} must be a string")
    if actual != expected:
        raise RandomOffsetTokenLoaderStateError(
            f"loader state {name} does not match this loader: "
            f"state has {actual!r}, loader requires {expected!r}"
        )


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


__all__ = [
    "CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX",
    "DEFAULT_CLIMBMIX_DATA_DIR",
    "NextTokenDataset",
    "RANDOM_OFFSET_LOADER_STATE_FORMAT",
    "RANDOM_OFFSET_LOADER_STATE_FORMAT_VERSION",
    "RandomOffsetTokenLoader",
    "RandomOffsetTokenLoaderStateError",
    "TOKENIZED_MANIFEST_NAME",
    "TOKENIZED_SHARD_FORMAT",
    "TOKENIZED_SHARD_FORMAT_VERSION",
    "TokenizedDataError",
    "TokenizedDatasetManifest",
    "TokenizedShardManifest",
    "TokenizedShardReader",
    "TokenizedShardSource",
    "TokenizedSplitManifest",
    "list_parquet_files",
    "parquets_iter_batched",
    "select_parquet_files",
    "write_tokenized_parquet_shards",
    "write_tokenized_shards",
]
