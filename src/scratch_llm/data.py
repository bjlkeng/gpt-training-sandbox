"""Small deterministic datasets for the educational training pipeline."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import heapq
import hashlib
import json
import logging
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
    TokenizedDocumentSpan,
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
_PACKING_PROGRESS_INTERVAL = 100_000
DOCUMENT_PACKING_LOADER_STATE_FORMAT = "scratch_llm_document_packing_loader_state"
DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION = 2
_DOCUMENT_PACKING_LOADER_STATE_KEYS = frozenset(
    {
        "batch_size",
        "epoch",
        "epoch_seed",
        "format",
        "format_version",
        "manifest_identity",
        "position",
        "rng_state",
        "row_position",
        "seq_len",
        "split",
    }
)
_BOS_TOKEN = "<|bos|>"
_LOGGER = logging.getLogger(__name__)


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


class DocumentPackingTokenLoaderStateError(ValueError):
    """A saved document-packing loader state is malformed or incompatible."""


@dataclass(frozen=True)
class _DocumentPiece:
    shard_index: int
    start: int
    token_count: int
    is_continuation: bool
    is_document_end: bool

    @property
    def packed_token_count(self) -> int:
        """Count the piece's BOS or carried-token prefix plus new tokens."""

        return self.token_count + 1


@dataclass(frozen=True)
class _PackedRow:
    pieces: tuple[_DocumentPiece, ...]
    used_token_count: int


@dataclass(frozen=True)
class DocumentPackingPlanStats:
    """Deterministic work counters for one capacity-indexed epoch plan."""

    document_count: int
    piece_count: int
    row_count: int
    capacity_searches: int
    row_candidate_checks: int
    max_capacity_bucket_size: int


class _AvailableCapacities:
    """Ordered set of non-empty capacity buckets backed by an integer bitset."""

    def __init__(self) -> None:
        self._bits = 0

    def add(self, capacity: int) -> None:
        """Mark one exact residual-capacity bucket as non-empty."""

        self._bits |= 1 << capacity

    def discard(self, capacity: int) -> None:
        """Mark one exact residual-capacity bucket as empty."""

        self._bits &= ~(1 << capacity)

    def smallest_at_least(self, required_capacity: int) -> int | None:
        """Return the smallest available capacity that meets the requirement.

        Bit position ``capacity`` records whether that exact bucket is non-empty.
        Shifting by the requirement discards undersized buckets and moves an exact
        fit to bit zero. The lowest remaining set bit is therefore the smallest
        available capacity offset above the requirement.
        """

        eligible_capacity_bits = self._bits >> required_capacity
        if eligible_capacity_bits == 0:
            return None
        smallest_offset_bit = eligible_capacity_bits & -eligible_capacity_bits
        smallest_capacity_offset = smallest_offset_bit.bit_length() - 1
        return required_capacity + smallest_capacity_offset


class _CapacityIndexedRows:
    """Mutable best-fit rows indexed by their exact residual capacity."""

    def __init__(self, row_token_count: int) -> None:
        self.row_token_count = row_token_count
        self.rows: list[list[_DocumentPiece]] = []
        self.used_token_counts: list[int] = []
        self.capacity_searches = 0
        self.row_candidate_checks = 0
        self.max_capacity_bucket_size = 0
        self._capacity_heaps: list[list[int]] = [[] for _ in range(row_token_count + 1)]
        self._available_capacities = _AvailableCapacities()

    def add_row(self, piece: _DocumentPiece) -> None:
        """Append a row and make any residual capacity available for reuse."""

        packed_token_count = piece.packed_token_count
        if packed_token_count > self.row_token_count:
            raise ValueError(
                "document piece exceeds packed row capacity: "
                f"{packed_token_count} > {self.row_token_count}"
            )
        row_index = len(self.rows)
        self.rows.append([piece])
        self.used_token_counts.append(packed_token_count)
        self._register_capacity(
            self.row_token_count - packed_token_count,
            row_index,
        )

    def place_best_fit(self, piece: _DocumentPiece) -> None:
        """Place one piece with bounded lookup and earliest-row tie behavior."""

        packed_token_count = piece.packed_token_count
        if packed_token_count > self.row_token_count:
            raise ValueError(
                "document piece exceeds packed row capacity: "
                f"{packed_token_count} > {self.row_token_count}"
            )
        self.capacity_searches += 1
        capacity = self._available_capacities.smallest_at_least(packed_token_count)
        if capacity is None:
            self.add_row(piece)
            return

        capacity_heap = self._capacity_heaps[capacity]
        row_index = heapq.heappop(capacity_heap)
        self.row_candidate_checks += 1
        if not capacity_heap:
            self._available_capacities.discard(capacity)

        self.rows[row_index].append(piece)
        self.used_token_counts[row_index] += packed_token_count
        self._register_capacity(capacity - packed_token_count, row_index)

    def freeze(self) -> tuple[_PackedRow, ...]:
        """Return the immutable row plan."""

        return tuple(
            _PackedRow(pieces=tuple(pieces), used_token_count=used)
            for pieces, used in zip(
                self.rows,
                self.used_token_counts,
                strict=True,
            )
        )

    def _register_capacity(self, capacity: int, row_index: int) -> None:
        if capacity == 0:
            return
        capacity_heap = self._capacity_heaps[capacity]
        heapq.heappush(capacity_heap, row_index)
        self._available_capacities.add(capacity)
        self.max_capacity_bucket_size = max(
            self.max_capacity_bucket_size,
            len(capacity_heap),
        )


class DocumentPackingTokenLoader(
    Iterator[tuple[Tensor, Tensor, Tensor]],
):
    """Yield deterministic boundary-aware batches without corpus concatenation.

    A document starts with BOS. Continuation windows instead carry the previous
    real token as context, so an artificial chunk boundary is never presented
    as a document opening. Best-fit placement combines complete documents that
    fit in one ``seq_len + 1`` row. Residual positions and residual batch rows
    use BOS, with an explicit boolean loss mask that selects ordinary document
    tokens and the first BOS after a real document end.
    """

    def __init__(
        self,
        reader: TokenizedShardReader,
        *,
        split: Literal["train", "val"],
        batch_size: int,
        seq_len: int,
        seed: int,
        planning_progress: Callable[[str], None] | None = None,
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
        if planning_progress is not None and not callable(planning_progress):
            raise TypeError("planning_progress must be callable or None")

        mapped_shards = reader.shards(split)
        document_spans = reader.document_spans(split)
        if not document_spans:
            raise ValueError(f"{split} split has no documents to pack")
        try:
            bos_token_id = reader.manifest.special_token_ids[_BOS_TOKEN]
        except KeyError as error:
            raise TokenizedDataError(
                f"tokenized manifest does not define required {_BOS_TOKEN!r}"
            ) from error

        self.reader = reader
        self.split = split
        self._mapped_shards = mapped_shards
        self._document_spans = document_spans
        self._bos_token_id = bos_token_id
        self._manifest_identity = _tokenized_manifest_identity(reader.manifest)
        self._planning_progress = planning_progress
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)
        self.position = 0
        self.epoch = -1
        self.epoch_seed = 0
        self.row_position = 0
        self._rows: tuple[_PackedRow, ...] = ()
        self.packed_example_count = 0
        self.plan_stats = DocumentPackingPlanStats(0, 0, 0, 0, 0, 0)
        self._start_next_epoch()

    def __iter__(self) -> DocumentPackingTokenLoader:
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.next_batch()

    def next_batch(self) -> tuple[Tensor, Tensor, Tensor]:
        """Materialize the next packed CPU batch and its explicit loss mask."""

        self.reader.shards(self.split)
        if self.row_position == len(self._rows):
            self._start_next_epoch()

        batch_rows = self._rows[self.row_position : self.row_position + self.batch_size]
        if len(batch_rows) != self.batch_size:
            raise RuntimeError("document-packing plan ended with an incomplete batch")

        windows = torch.full(
            (self.batch_size, self.seq_len + 1),
            self._bos_token_id,
            dtype=torch.long,
            device="cpu",
        )
        loss_mask = torch.zeros(
            (self.batch_size, self.seq_len),
            dtype=torch.bool,
            device="cpu",
        )
        for row_index, row in enumerate(batch_rows):
            packed_offset = 0
            for piece in row.pieces:
                if piece.is_continuation:
                    previous_token = int(
                        self._mapped_shards[piece.shard_index][piece.start - 1]
                    )
                    windows[row_index, packed_offset] = previous_token
                content_start = packed_offset + 1
                content_stop = content_start + piece.token_count
                if piece.token_count:
                    mapped_tokens = self._mapped_shards[piece.shard_index][
                        piece.start : piece.start + piece.token_count
                    ]
                    copied_tokens = np.array(
                        mapped_tokens,
                        dtype=np.int64,
                        copy=True,
                    )
                    windows[row_index, content_start:content_stop].copy_(
                        torch.from_numpy(copied_tokens)
                    )
                    loss_mask[
                        row_index,
                        packed_offset : packed_offset + piece.token_count,
                    ] = True
                packed_offset += piece.packed_token_count
                if piece.is_document_end and packed_offset < self.seq_len + 1:
                    loss_mask[row_index, packed_offset - 1] = True

        self.row_position += self.batch_size
        self.position += self.batch_size
        return windows[:, :-1], windows[:, 1:], loss_mask

    def state_dict(self) -> dict[str, object]:
        """Return JSON-compatible state for exact next-packed-batch resume."""

        return {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "epoch_seed": self.epoch_seed,
            "format": DOCUMENT_PACKING_LOADER_STATE_FORMAT,
            "format_version": DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION,
            "manifest_identity": self._manifest_identity,
            "position": self.position,
            "rng_state": self._generator.get_state().tolist(),
            "row_position": self.row_position,
            "seq_len": self.seq_len,
            "split": self.split,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate and restore state without partially mutating this loader."""

        if not isinstance(state, Mapping):
            raise DocumentPackingTokenLoaderStateError(
                f"loader state must be a mapping, got {type(state).__name__}"
            )
        state_keys = set(state)
        if state_keys != _DOCUMENT_PACKING_LOADER_STATE_KEYS:
            missing = sorted(_DOCUMENT_PACKING_LOADER_STATE_KEYS - state_keys)
            unexpected = sorted(
                state_keys - _DOCUMENT_PACKING_LOADER_STATE_KEYS,
                key=str,
            )
            raise DocumentPackingTokenLoaderStateError(
                "loader state fields do not match format version "
                f"{DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if state["format"] != DOCUMENT_PACKING_LOADER_STATE_FORMAT:
            raise DocumentPackingTokenLoaderStateError(
                f"unknown loader state format {state['format']!r}"
            )
        format_version = _packing_state_integer(
            state["format_version"],
            name="format version",
        )
        if format_version != DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION:
            raise DocumentPackingTokenLoaderStateError(
                f"unknown loader state format version {format_version}; "
                f"expected {DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION}"
            )
        if state["manifest_identity"] != self._manifest_identity:
            raise DocumentPackingTokenLoaderStateError(
                "loader state manifest identity does not match the mapped dataset"
            )
        _require_packing_loader_setting(state, "split", self.split)
        _require_packing_loader_setting(state, "batch_size", self.batch_size)
        _require_packing_loader_setting(state, "seq_len", self.seq_len)

        epoch = _packing_state_integer(state["epoch"], name="epoch")
        epoch_seed = _packing_state_integer(state["epoch_seed"], name="epoch_seed")
        position = _packing_state_integer(state["position"], name="position")
        row_position = _packing_state_integer(
            state["row_position"],
            name="row_position",
        )
        if epoch < 0:
            raise DocumentPackingTokenLoaderStateError(
                f"loader state epoch must be non-negative, got {epoch}"
            )
        if not 0 <= epoch_seed <= _MAX_TORCH_SEED:
            raise DocumentPackingTokenLoaderStateError(
                "loader state epoch_seed must be in range "
                f"[0, {_MAX_TORCH_SEED}], got {epoch_seed}"
            )
        if position < 0 or position % self.batch_size != 0:
            raise DocumentPackingTokenLoaderStateError(
                "loader state position must be a non-negative multiple of "
                f"batch_size={self.batch_size}, got {position}"
            )

        rows, packed_example_count, plan_stats = self._build_epoch(epoch_seed)
        if (
            row_position < 0
            or row_position > len(rows)
            or row_position % self.batch_size != 0
        ):
            raise DocumentPackingTokenLoaderStateError(
                "loader state row_position must be a batch-aligned offset in "
                f"[0, {len(rows)}], got {row_position}"
            )
        rng_state = _packing_rng_state(state["rng_state"])
        candidate_generator = torch.Generator(device="cpu")
        try:
            candidate_generator.set_state(rng_state)
        except RuntimeError as error:
            raise DocumentPackingTokenLoaderStateError(
                f"loader state rng_state is invalid: {error}"
            ) from error

        self._generator.set_state(rng_state)
        self._rows = rows
        self.packed_example_count = packed_example_count
        self.plan_stats = plan_stats
        self.epoch = epoch
        self.epoch_seed = epoch_seed
        self.position = position
        self.row_position = row_position

    def _start_next_epoch(self) -> None:
        epoch_seed = int(
            torch.randint(
                0,
                _MAX_TORCH_SEED,
                (1,),
                generator=self._generator,
                dtype=torch.int64,
                device="cpu",
            ).item()
        )
        rows, packed_example_count, plan_stats = self._build_epoch(epoch_seed)
        self.epoch += 1
        self.epoch_seed = epoch_seed
        self.row_position = 0
        self._rows = rows
        self.packed_example_count = packed_example_count
        self.plan_stats = plan_stats

    def _build_epoch(
        self,
        epoch_seed: int,
    ) -> tuple[tuple[_PackedRow, ...], int, DocumentPackingPlanStats]:
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(epoch_seed)
        order = torch.randperm(
            len(self._document_spans),
            generator=order_generator,
            dtype=torch.int64,
            device="cpu",
        ).tolist()
        rows, plan_stats = _plan_best_fit_document_rows(
            self._document_spans,
            order=order,
            seq_len=self.seq_len,
            progress=self._planning_progress,
        )
        packed_example_count = len(rows)
        padding_row_count = (-packed_example_count) % self.batch_size
        if padding_row_count:
            rows = (
                *rows,
                *(
                    _PackedRow(pieces=(), used_token_count=0)
                    for _ in range(padding_row_count)
                ),
            )
        return tuple(rows), packed_example_count, plan_stats


def create_token_loader(
    reader: TokenizedShardReader,
    *,
    strategy: Literal["flat", "packed"],
    split: Literal["train", "val"],
    batch_size: int,
    seq_len: int,
    seed: int,
    planning_progress: Callable[[str], None] | None = None,
) -> RandomOffsetTokenLoader | DocumentPackingTokenLoader:
    """Select the explicit flat baseline or BOS-aware packing strategy."""

    if strategy == "flat":
        return RandomOffsetTokenLoader(
            reader,
            split=split,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed,
        )
    if strategy == "packed":
        return DocumentPackingTokenLoader(
            reader,
            split=split,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed,
            planning_progress=planning_progress,
        )
    raise ValueError(f"strategy must be 'flat' or 'packed', got {strategy!r}")


def _plan_best_fit_document_rows(
    spans: Sequence[TokenizedDocumentSpan],
    *,
    order: Sequence[int],
    seq_len: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[tuple[_PackedRow, ...], DocumentPackingPlanStats]:
    row_token_count = seq_len + 1
    planner = _CapacityIndexedRows(row_token_count)
    document_count = len(order)
    piece_count = 0
    _report_packing_progress(
        progress,
        "packing planner started: documents=%d row_tokens=%d",
        document_count,
        row_token_count,
    )

    for processed_documents, document_index in enumerate(order, start=1):
        span = spans[document_index]
        remaining = span.token_count
        document_offset = 0
        while remaining > 0:
            piece_token_count = min(remaining, seq_len)
            is_continuation = document_offset > 0
            piece = _DocumentPiece(
                shard_index=span.shard_index,
                start=span.start + document_offset,
                token_count=piece_token_count,
                is_continuation=is_continuation,
                is_document_end=piece_token_count == remaining,
            )
            piece_count += 1
            if is_continuation:
                # The carried prefix must be the preceding source token, never
                # an unrelated document that happened to leave residual room.
                planner.add_row(piece)
            else:
                planner.place_best_fit(piece)
            remaining -= piece_token_count
            document_offset += piece_token_count
        if span.token_count == 0:
            piece_count += 1
            planner.place_best_fit(
                _DocumentPiece(
                    shard_index=span.shard_index,
                    start=span.start,
                    token_count=0,
                    is_continuation=False,
                    is_document_end=True,
                )
            )
        if (
            processed_documents % _PACKING_PROGRESS_INTERVAL == 0
            and processed_documents < document_count
        ):
            _report_packing_progress(
                progress,
                "packing planner progress: documents=%d/%d rows=%d pieces=%d",
                processed_documents,
                document_count,
                len(planner.rows),
                piece_count,
            )

    rows = planner.freeze()
    stats = DocumentPackingPlanStats(
        document_count=document_count,
        piece_count=piece_count,
        row_count=len(rows),
        capacity_searches=planner.capacity_searches,
        row_candidate_checks=planner.row_candidate_checks,
        max_capacity_bucket_size=planner.max_capacity_bucket_size,
    )
    _report_packing_progress(
        progress,
        "packing planner completed: documents=%d rows=%d pieces=%d "
        "capacity_searches=%d row_candidate_checks=%d",
        stats.document_count,
        stats.row_count,
        stats.piece_count,
        stats.capacity_searches,
        stats.row_candidate_checks,
    )
    return rows, stats


def _report_packing_progress(
    progress: Callable[[str], None] | None,
    message: str,
    *arguments: object,
) -> None:
    rendered = message % arguments
    _LOGGER.info("%s", rendered)
    if progress is not None:
        progress(rendered)


def _place_best_fit_piece(
    rows: list[list[_DocumentPiece]],
    used_token_counts: list[int],
    piece: _DocumentPiece,
    *,
    row_token_count: int,
) -> None:
    best_index: int | None = None
    best_remaining: int | None = None
    for row_index, used in enumerate(used_token_counts):
        remaining = row_token_count - used - piece.packed_token_count
        if remaining >= 0 and (best_remaining is None or remaining < best_remaining):
            best_index = row_index
            best_remaining = remaining
    if best_index is None:
        rows.append([piece])
        used_token_counts.append(piece.packed_token_count)
        return
    rows[best_index].append(piece)
    used_token_counts[best_index] += piece.packed_token_count


def _packing_state_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DocumentPackingTokenLoaderStateError(
            f"loader state {name} must be an integer"
        )
    return value


def _require_packing_loader_setting(
    state: Mapping[str, object],
    name: str,
    expected: object,
) -> None:
    actual = state[name]
    if isinstance(expected, int):
        actual = _packing_state_integer(actual, name=name)
    elif isinstance(expected, str) and not isinstance(actual, str):
        raise DocumentPackingTokenLoaderStateError(
            f"loader state {name} must be a string"
        )
    if actual != expected:
        raise DocumentPackingTokenLoaderStateError(
            f"loader state {name} does not match this loader: "
            f"state has {actual!r}, loader requires {expected!r}"
        )


def _packing_rng_state(value: object) -> Tensor:
    if not isinstance(value, list) or not value:
        raise DocumentPackingTokenLoaderStateError(
            "loader state rng_state must be a non-empty list of bytes"
        )
    if any(
        not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 255
        for item in value
    ):
        raise DocumentPackingTokenLoaderStateError(
            "loader state rng_state must contain only integer bytes"
        )
    return torch.tensor(value, dtype=torch.uint8, device="cpu")


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
    "DOCUMENT_PACKING_LOADER_STATE_FORMAT",
    "DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION",
    "DocumentPackingTokenLoader",
    "DocumentPackingTokenLoaderStateError",
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
    "TokenizedDocumentSpan",
    "TokenizedShardManifest",
    "TokenizedShardReader",
    "TokenizedShardSource",
    "TokenizedSplitManifest",
    "create_token_loader",
    "list_parquet_files",
    "parquets_iter_batched",
    "select_parquet_files",
    "write_tokenized_parquet_shards",
    "write_tokenized_shards",
]
