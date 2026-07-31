"""Pinned nanochat-compatible validation packing and BPB evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch import Tensor, nn

from scratch_llm.bpb import (
    BaseValidationResult,
    evaluate_bpb_batches,
)
from scratch_llm.climbmix import CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX
from scratch_llm.data import list_parquet_files, select_parquet_files
from scratch_llm.tokenized_data import (
    TokenizedDataError,
    TokenizedDocumentSpan,
    TokenizedShardReader,
    tokenized_manifest_identity,
)
from scratch_llm.tokenizer import Tokenizer


NANOCHAT_COMPAT_PROTOCOL_ID: Final = "nanochat_compat_v1"
NANOCHAT_COMPAT_PROTOCOL_VERSION: Final = 1
NANOCHAT_COMPAT_TRAIN_METRIC: Final = "val_bpb"
NANOCHAT_COMPAT_EVAL_METRIC: Final = "eval/val_bpb"
NANOCHAT_REFERENCE_COMMIT: Final = "41865401f73ff1c5321ae53297bceb2b78d4c8b4"
NANOCHAT_REFERENCE_FILE_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "nanochat/dataloader.py": (
            "5cc72d7207931f112d685ba8e04c112e1a4ab7756dbbb29b95bdb4908a21864d"
        ),
        "nanochat/loss_eval.py": (
            "00faad1e0ae8912022f79ee4bf583c4f9b4c058e4523c5674144648c49229fd6"
        ),
        "scripts/base_train.py": (
            "d806cfa36d51f246186bd24e8693cc09ddcf96545ab5c5355a3450d1eddfd8ac"
        ),
    }
)
_NANOCHAT_BUFFER_SIZE = 1000
_NANOCHAT_TOKENIZER_BATCH_SIZE = 128
_NANOCHAT_TOKENIZER_THREADS = 4
_NANOCHAT_TEXT_COLUMN = "text"
_METRIC_KEYS = frozenset({NANOCHAT_COMPAT_TRAIN_METRIC, NANOCHAT_COMPAT_EVAL_METRIC})
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


@dataclass(frozen=True)
class NanochatCompatibilityConfig:
    """Resolved single-process settings from the pinned base evaluator."""

    device_batch_size: int
    context_length: int
    eval_tokens: int
    buffer_size: int = field(default=_NANOCHAT_BUFFER_SIZE, init=False)
    tokenizer_batch_size: int = field(
        default=_NANOCHAT_TOKENIZER_BATCH_SIZE,
        init=False,
    )
    tokenizer_threads: int = field(
        default=_NANOCHAT_TOKENIZER_THREADS,
        init=False,
    )
    world_size: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        _positive_integer(self.device_batch_size, name="device_batch_size")
        _positive_integer(self.context_length, name="context_length")
        _positive_integer(self.eval_tokens, name="eval_tokens")
        if self.eval_steps == 0:
            raise ValueError(
                "eval_tokens must cover at least one complete evaluation step "
                f"of {self.device_batch_size * self.context_length} tokens"
            )

    @property
    def row_capacity(self) -> int:
        """Return nanochat's pre-shift row width, ``T + 1``."""

        return self.context_length + 1

    @property
    def eval_steps(self) -> int:
        """Return nanochat's floor-divided number of evaluation batches."""

        return self.eval_tokens // (
            self.device_batch_size * self.context_length * self.world_size
        )

    @property
    def processed_eval_tokens(self) -> int:
        """Return the resolved model-token count after floor division."""

        return (
            self.eval_steps
            * self.device_batch_size
            * self.context_length
            * self.world_size
        )

    def to_dict(self) -> dict[str, object]:
        """Return immutable protocol provenance plus resolved evaluator values."""

        return {
            "buffer_size": self.buffer_size,
            "device_batch_size": self.device_batch_size,
            "document_order": "final_validation_shard_row_group_document",
            "eval_steps": self.eval_steps,
            "eval_tokens": self.eval_tokens,
            "max_seq_len": self.context_length,
            "packing": {
                "crop": "first_shortest_prefix_discard_suffix",
                "largest_fit_tie": "first",
                "prepend_bos": True,
                "refill": "whole_tokenizer_batch_below_buffer_size",
                "selection": "first_largest_document_that_fits",
                "shortest_tie": "first",
            },
            "processed_eval_tokens": self.processed_eval_tokens,
            "reference_files": dict(NANOCHAT_REFERENCE_FILE_SHA256),
            "repeated_cycles": "restart_final_validation_shard_in_order",
            "row_capacity": self.row_capacity,
            "split": "val",
            "tokenizer_batch_size": self.tokenizer_batch_size,
            "tokenizer_threads": self.tokenizer_threads,
            "world_size": self.world_size,
        }


@dataclass(frozen=True)
class NanochatDocument:
    """One tokenized source document without its protocol-owned BOS."""

    source_document_index: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _non_negative_integer(
            self.source_document_index,
            name="source_document_index",
        )
        if not isinstance(self.token_ids, tuple):
            raise TypeError("token_ids must be a tuple of integers")
        for position, token_id in enumerate(self.token_ids):
            _non_negative_integer(token_id, name=f"token_ids[{position}]")


class NanochatCompatiblePacker(Iterator[tuple[Tensor, Tensor]]):
    """Reproduce nanochat's buffered BOS best-fit rows and destructive crops."""

    def __init__(
        self,
        document_batches: Iterator[Sequence[NanochatDocument]],
        *,
        batch_size: int,
        context_length: int,
        bos_token_id: int,
        token_bytes: Tensor,
        buffer_size: int = _NANOCHAT_BUFFER_SIZE,
    ) -> None:
        try:
            self._document_batches = iter(document_batches)
        except TypeError as error:
            raise TypeError("document_batches must be iterable") from error
        self.batch_size = _positive_integer(batch_size, name="batch_size")
        self.context_length = _positive_integer(
            context_length,
            name="context_length",
        )
        self.row_capacity = self.context_length + 1
        self.buffer_size = _positive_integer(buffer_size, name="buffer_size")
        self._token_bytes = _normalized_token_bytes(token_bytes)
        self.bos_token_id = _token_id(
            bos_token_id,
            vocab_size=self._token_bytes.numel(),
            name="bos_token_id",
        )
        if int(self._token_bytes[self.bos_token_id].item()) != 0:
            raise ValueError("BOS token byte length must be zero")

        self._document_buffer: list[NanochatDocument] = []
        # This protocol can retain only prefixes, never interior ranges or
        # suffix continuations. One maximum prefix length per source document
        # therefore represents exact unique coverage without a per-token set.
        self._retained_source_prefixes: dict[int, int] = {}
        self._counted_source_tokens = 0
        self._counted_source_bytes = 0
        self._unique_source_tokens = 0
        self._unique_source_bytes = 0

    @property
    def counted_source_tokens(self) -> int:
        """Return ordinary target occurrences, including repeated cycles."""

        return self._counted_source_tokens

    @property
    def counted_source_bytes(self) -> int:
        """Return ordinary target bytes, including repeated cycles."""

        return self._counted_source_bytes

    @property
    def unique_source_tokens(self) -> int:
        """Return distinct source token positions retained at least once."""

        return self._unique_source_tokens

    @property
    def unique_source_bytes(self) -> int:
        """Return bytes at distinct retained source token positions."""

        return self._unique_source_bytes

    def __iter__(self) -> NanochatCompatiblePacker:
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        rows: list[list[int]] = []
        for _ in range(self.batch_size):
            row: list[int] = []
            while len(row) < self.row_capacity:
                self._refill_buffer()
                remaining = self.row_capacity - len(row)
                selected_index = self._largest_fitting_document(remaining)
                if selected_index is None:
                    selected_index = min(
                        range(len(self._document_buffer)),
                        key=lambda index: len(self._document_buffer[index].token_ids),
                    )
                    document = self._document_buffer.pop(selected_index)
                    copied_count = remaining
                else:
                    document = self._document_buffer.pop(selected_index)
                    copied_count = len(document.token_ids) + 1

                prefixed_tokens = (self.bos_token_id, *document.token_ids)
                row.extend(prefixed_tokens[:copied_count])
                self._record_source_prefix(
                    document,
                    ordinary_token_count=max(copied_count - 1, 0),
                )
            rows.append(row)

        windows = torch.tensor(rows, dtype=torch.long, device="cpu")
        return windows[:, :-1], windows[:, 1:]

    def _refill_buffer(self) -> None:
        while len(self._document_buffer) < self.buffer_size:
            try:
                raw_batch = next(self._document_batches)
            except StopIteration as error:
                raise RuntimeError(
                    "nanochat document batch source ended; it must repeat "
                    "the validation shard indefinitely"
                ) from error
            if isinstance(raw_batch, (str, bytes)) or not isinstance(
                raw_batch,
                Sequence,
            ):
                raise TypeError(
                    "each nanochat document batch must be a sequence of documents"
                )
            batch = tuple(raw_batch)
            if not batch:
                raise ValueError("nanochat document batches must not be empty")
            for position, document in enumerate(batch):
                if not isinstance(document, NanochatDocument):
                    raise TypeError(
                        f"document batch item {position} must be a NanochatDocument"
                    )
                self._validate_document_tokens(document)
            self._document_buffer.extend(batch)

    def _largest_fitting_document(self, remaining: int) -> int | None:
        best_index: int | None = None
        best_length = 0
        for index, document in enumerate(self._document_buffer):
            document_length = len(document.token_ids) + 1
            if document_length <= remaining and document_length > best_length:
                best_index = index
                best_length = document_length
        return best_index

    def _validate_document_tokens(self, document: NanochatDocument) -> None:
        for position, token_id in enumerate(document.token_ids):
            normalized_id = _token_id(
                token_id,
                vocab_size=self._token_bytes.numel(),
                name=f"document token_ids[{position}]",
            )
            if int(self._token_bytes[normalized_id].item()) == 0:
                raise ValueError(
                    "source documents must contain only positive-byte ordinary "
                    f"tokens; found token ID {normalized_id}"
                )

    def _record_source_prefix(
        self,
        document: NanochatDocument,
        *,
        ordinary_token_count: int,
    ) -> None:
        retained_tokens = document.token_ids[:ordinary_token_count]
        retained_bytes = sum(
            int(self._token_bytes[token_id].item()) for token_id in retained_tokens
        )
        self._counted_source_tokens += len(retained_tokens)
        self._counted_source_bytes += retained_bytes

        previous_prefix = self._retained_source_prefixes.get(
            document.source_document_index,
            0,
        )
        if ordinary_token_count <= previous_prefix:
            return
        new_unique_tokens = document.token_ids[previous_prefix:ordinary_token_count]
        self._retained_source_prefixes[document.source_document_index] = (
            ordinary_token_count
        )
        self._unique_source_tokens += len(new_unique_tokens)
        self._unique_source_bytes += sum(
            int(self._token_bytes[token_id].item()) for token_id in new_unique_tokens
        )


class _NanochatParquetDocumentBatches(
    Iterator[tuple[NanochatDocument, ...]],
):
    """Cycle one final parquet shard in upstream row-group/document order."""

    def __init__(
        self,
        path: Path,
        *,
        reader: TokenizedShardReader,
        tokenizer: Tokenizer,
    ) -> None:
        self._path = path
        self._reader = reader
        self._tokenizer = tokenizer
        self._spans = reader.document_spans("val")
        self._mapped_shards = reader.shards("val")
        self._text_batches: Iterator[tuple[str, ...]] = iter(())
        self._document_index = 0
        self._cycle_started = False
        self._validate_source_contract()

    def __iter__(self) -> _NanochatParquetDocumentBatches:
        return self

    def __next__(self) -> tuple[NanochatDocument, ...]:
        texts = self._next_text_batch()
        documents: list[NanochatDocument] = []
        for text in texts:
            if self._document_index >= len(self._spans):
                raise TokenizedDataError(
                    f"validation parquet {self._path.name} contains more "
                    "documents than the tokenized manifest"
                )
            span = self._spans[self._document_index]
            token_ids = tuple(self._tokenizer.encode(text))
            expected = self._mapped_document_tokens(span)
            if token_ids != expected:
                raise TokenizedDataError(
                    "validation parquet tokenization does not match the "
                    f"tokenized manifest at document {self._document_index}"
                )
            documents.append(
                NanochatDocument(
                    source_document_index=self._document_index,
                    token_ids=token_ids,
                )
            )
            self._document_index += 1
        return tuple(documents)

    def _next_text_batch(self) -> tuple[str, ...]:
        try:
            return next(self._text_batches)
        except StopIteration:
            if self._cycle_started and self._document_index != len(self._spans):
                raise TokenizedDataError(
                    f"validation parquet {self._path.name} contains "
                    f"{self._document_index} documents but the tokenized "
                    f"manifest records {len(self._spans)}"
                )
            self._text_batches = _parquet_row_group_batches(self._path)
            self._document_index = 0
            self._cycle_started = True
            try:
                return next(self._text_batches)
            except StopIteration as error:
                raise TokenizedDataError(
                    f"validation parquet {self._path.name} contains no documents"
                ) from error

    def _validate_source_contract(self) -> None:
        validation_manifest = self._reader.manifest.splits["val"]
        if len(validation_manifest.shards) != 1:
            raise TokenizedDataError(
                "nanochat_compat_v1 requires exactly one final validation shard"
            )
        shard = validation_manifest.shards[0]
        if shard.source_shards != (self._path.name,):
            raise TokenizedDataError(
                "tokenized validation manifest source does not match the "
                f"selected final parquet shard {self._path.name}"
            )
        if len(self._spans) != validation_manifest.document_count:
            raise TokenizedDataError(
                "tokenized validation document spans do not match the manifest"
            )

    def _mapped_document_tokens(
        self,
        span: TokenizedDocumentSpan,
    ) -> tuple[int, ...]:
        mapped = self._mapped_shards[span.shard_index][span.start : span.stop]
        return tuple(int(token_id) for token_id in mapped.tolist())


def evaluate_nanochat_compatible_bpb(
    model: nn.Module,
    tokenizer: Tokenizer,
    reader: TokenizedShardReader,
    token_bytes: Tensor,
    *,
    parquet_dir: str | Path,
    checkpoint_identity: str,
    config: NanochatCompatibilityConfig,
    device: str | torch.device,
) -> BaseValidationResult:
    """Evaluate one checkpoint under the pinned nanochat compatibility domain."""

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    if not isinstance(reader, TokenizedShardReader):
        raise TypeError(
            f"reader must be a TokenizedShardReader, got {type(reader).__name__}"
        )
    if not isinstance(config, NanochatCompatibilityConfig):
        raise TypeError(
            f"config must be a NanochatCompatibilityConfig, got {type(config).__name__}"
        )
    if not isinstance(checkpoint_identity, str) or not checkpoint_identity.strip():
        raise ValueError("checkpoint_identity must be a non-empty string")
    normalized_token_bytes = _normalized_token_bytes(token_bytes)
    if normalized_token_bytes.numel() != tokenizer.get_vocab_size():
        raise ValueError(
            "token_bytes size must match the tokenizer vocabulary: "
            f"{normalized_token_bytes.numel()} != {tokenizer.get_vocab_size()}"
        )
    if reader.manifest.tokenizer_identity != tokenizer.get_identity():
        raise TokenizedDataError(
            "tokenized manifest identity does not match the evaluator tokenizer"
        )

    validation_paths = select_parquet_files(
        list_parquet_files(parquet_dir),
        "val",
        validation_shard_index=CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    )
    if len(validation_paths) != 1:
        raise TokenizedDataError(
            "nanochat_compat_v1 requires exactly one final validation parquet"
        )
    document_batches = _NanochatParquetDocumentBatches(
        validation_paths[0],
        reader=reader,
        tokenizer=tokenizer,
    )
    source_documents, source_tokens, source_bytes = _source_coverage(
        reader,
        normalized_token_bytes,
    )
    packer = NanochatCompatiblePacker(
        document_batches,
        batch_size=config.device_batch_size,
        context_length=config.context_length,
        bos_token_id=tokenizer.get_bos_token_id(),
        token_bytes=normalized_token_bytes,
        buffer_size=config.buffer_size,
    )
    accumulation = evaluate_bpb_batches(
        model,
        islice(packer, config.eval_steps),
        normalized_token_bytes,
        device=device,
    )
    if accumulation.processed_model_tokens != config.processed_eval_tokens:
        raise RuntimeError(
            "nanochat-compatible evaluation processed an unexpected token count"
        )
    if (
        accumulation.counted_target_tokens != packer.counted_source_tokens
        or accumulation.counted_target_bytes != packer.counted_source_bytes
    ):
        raise RuntimeError(
            "nanochat-compatible source coverage does not match BPB accumulation"
        )

    return BaseValidationResult.from_accumulation(
        accumulation,
        protocol_id=NANOCHAT_COMPAT_PROTOCOL_ID,
        protocol_version=NANOCHAT_COMPAT_PROTOCOL_VERSION,
        reference_commit=NANOCHAT_REFERENCE_COMMIT,
        reference_config=config.to_dict(),
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer.get_identity(),
        validation_manifest_identity=tokenized_manifest_identity(reader.manifest),
        source_documents=source_documents,
        source_tokens=source_tokens,
        source_bytes=source_bytes,
        unique_source_tokens=packer.unique_source_tokens,
        unique_source_bytes=packer.unique_source_bytes,
    )


def nanochat_compatible_metric_value(
    result: BaseValidationResult,
    *,
    key: str,
) -> float:
    """Return BPB only for the two permanently reserved compatibility keys."""

    if not isinstance(result, BaseValidationResult):
        raise TypeError(
            f"result must be a BaseValidationResult, got {type(result).__name__}"
        )
    if result.protocol_id != NANOCHAT_COMPAT_PROTOCOL_ID:
        raise ValueError(f"result protocol must be {NANOCHAT_COMPAT_PROTOCOL_ID!r}")
    if key not in _METRIC_KEYS:
        raise ValueError(
            "reserved nanochat-compatible metric key must be exactly "
            f"{NANOCHAT_COMPAT_TRAIN_METRIC!r} or "
            f"{NANOCHAT_COMPAT_EVAL_METRIC!r}; got {key!r}"
        )
    return result.bpb


def _parquet_row_group_batches(path: Path) -> Iterator[tuple[str, ...]]:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    column_indices = schema.get_all_field_indices(_NANOCHAT_TEXT_COLUMN)
    if not column_indices:
        raise TokenizedDataError(
            f"text column {_NANOCHAT_TEXT_COLUMN!r} was not found in {path.name}"
        )
    if len(column_indices) != 1:
        raise TokenizedDataError(
            f"text column {_NANOCHAT_TEXT_COLUMN!r} is ambiguous in {path.name}"
        )
    column_type = schema.field(column_indices[0]).type
    if not (pa.types.is_string(column_type) or pa.types.is_large_string(column_type)):
        raise TokenizedDataError(
            f"text column {_NANOCHAT_TEXT_COLUMN!r} must be a string in "
            f"{path.name}; got {column_type}"
        )

    for row_group_index in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_index,
            columns=[_NANOCHAT_TEXT_COLUMN],
            use_threads=False,
        )
        text_column = table.column(0)
        if text_column.null_count:
            raise TokenizedDataError(
                f"text column {_NANOCHAT_TEXT_COLUMN!r} contains null values "
                f"in {path.name} row group {row_group_index}"
            )
        texts = text_column.to_pylist()
        for start in range(0, len(texts), _NANOCHAT_TOKENIZER_BATCH_SIZE):
            batch = tuple(texts[start : start + _NANOCHAT_TOKENIZER_BATCH_SIZE])
            if batch:
                yield batch


def _source_coverage(
    reader: TokenizedShardReader,
    token_bytes: Tensor,
) -> tuple[int, int, int]:
    validation_manifest = reader.manifest.splits["val"]
    source_documents = validation_manifest.document_count
    source_tokens = validation_manifest.token_count
    if source_documents <= 0:
        raise TokenizedDataError("validation manifest must contain documents")
    if source_tokens <= 0:
        raise TokenizedDataError(
            "validation manifest must contain ordinary source tokens"
        )

    source_bytes = 0
    for shard in reader.shards("val"):
        for start in range(0, len(shard), 1_000_000):
            token_ids = np.array(
                shard[start : start + 1_000_000],
                dtype=np.int64,
                copy=True,
            )
            ids = torch.from_numpy(token_ids)
            if ids.numel() == 0:
                continue
            if (
                int(ids.min().item()) < 0
                or int(ids.max().item()) >= token_bytes.numel()
            ):
                raise TokenizedDataError(
                    "validation token ID is outside the token_bytes table"
                )
            byte_lengths = token_bytes[ids]
            if bool(byte_lengths.eq(0).any().item()):
                raise TokenizedDataError(
                    "validation source contains a zero-byte special token"
                )
            source_bytes += int(byte_lengths.sum().item())
    if source_bytes <= 0:
        raise TokenizedDataError(
            "validation manifest must contain positive-byte source tokens"
        )
    return source_documents, source_tokens, source_bytes


def _normalized_token_bytes(token_bytes: Tensor) -> Tensor:
    if not isinstance(token_bytes, Tensor):
        raise TypeError(
            f"token_bytes must be a Tensor, got {type(token_bytes).__name__}"
        )
    if token_bytes.ndim != 1 or token_bytes.numel() == 0:
        raise ValueError("token_bytes must be a non-empty one-dimensional tensor")
    if token_bytes.dtype not in _INTEGER_DTYPES:
        raise TypeError("token_bytes must use an integer dtype")
    normalized = token_bytes.detach().to(device="cpu", dtype=torch.int64).clone()
    if bool(normalized.lt(0).any().item()):
        raise ValueError("token_bytes values must be non-negative")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _non_negative_integer(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _non_negative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _token_id(value: object, *, vocab_size: int, name: str) -> int:
    normalized = _non_negative_integer(value, name=name)
    if normalized >= vocab_size:
        raise ValueError(
            f"{name} must be less than token_bytes size {vocab_size}; got {normalized}"
        )
    return normalized


__all__ = [
    "NANOCHAT_COMPAT_EVAL_METRIC",
    "NANOCHAT_COMPAT_PROTOCOL_ID",
    "NANOCHAT_COMPAT_PROTOCOL_VERSION",
    "NANOCHAT_COMPAT_TRAIN_METRIC",
    "NANOCHAT_REFERENCE_COMMIT",
    "NANOCHAT_REFERENCE_FILE_SHA256",
    "NanochatCompatibilityConfig",
    "NanochatCompatiblePacker",
    "NanochatDocument",
    "evaluate_nanochat_compatible_bpb",
    "nanochat_compatible_metric_value",
]
