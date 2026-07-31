"""Finite continuation-aware full-document BPB evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import heapq
import math
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn

from scratch_llm._validation import (
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.bpb import BaseValidationResult, evaluate_bpb_batches
from scratch_llm.tokenized_data import (
    TokenizedDataError,
    TokenizedDocumentSpan,
    TokenizedShardReader,
    tokenized_manifest_identity,
)
from scratch_llm.tokenizer import Tokenizer


FULL_DOCUMENT_PROTOCOL_ID: Final = "full_documents_v1"
FULL_DOCUMENT_PROTOCOL_VERSION: Final = 1
FULL_DOCUMENT_TRAIN_METRIC: Final = "val_bpb_full_documents"
FULL_DOCUMENT_EVAL_METRIC: Final = "eval/val_bpb_full_documents"
_METRIC_KEYS = frozenset(
    {
        FULL_DOCUMENT_TRAIN_METRIC,
        FULL_DOCUMENT_EVAL_METRIC,
    }
)
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
class FullDocumentProtocolConfig:
    """Resolved settings for one finite full-document validation pass."""

    device_batch_size: int
    context_length: int

    def __post_init__(self) -> None:
        require_positive_integer(self.device_batch_size, name="device_batch_size")
        require_positive_integer(self.context_length, name="context_length")

    @property
    def row_capacity(self) -> int:
        """Return the pre-shift row width, ``T + 1``."""

        return self.context_length + 1

    def to_dict(self) -> dict[str, object]:
        """Return the frozen protocol choices and resolved tensor shape."""

        return {
            "batch_padding": "bos_with_no_supervision",
            "device_batch_size": self.device_batch_size,
            "document_order": "manifest_shard_document",
            "max_seq_len": self.context_length,
            "packing": {
                "continuation_context": "previous_ordinary_token",
                "first_window_context": "bos",
                "largest_fit_tie": "earliest_row",
                "ordinary_target_coverage": "exactly_once",
                "selection": "best_fit_complete_first_piece",
            },
            "row_capacity": self.row_capacity,
            "seed": None,
            "termination": "one_validation_manifest_pass",
        }


@dataclass(frozen=True)
class _DocumentPiece:
    shard_index: int
    start: int
    token_count: int
    is_continuation: bool
    is_document_end: bool

    @property
    def packed_token_count(self) -> int:
        """Count the BOS or carried context plus newly supervised tokens."""

        return self.token_count + 1

    @property
    def signature(self) -> tuple[int, int, int, bool, bool]:
        return (
            self.shard_index,
            self.start,
            self.token_count,
            self.is_continuation,
            self.is_document_end,
        )


@dataclass(frozen=True)
class _PackedRow:
    pieces: tuple[_DocumentPiece, ...]
    used_token_count: int


class _CapacityIndexedRows:
    """Place first pieces by best residual fit with earliest-row ties."""

    def __init__(self, row_capacity: int) -> None:
        self._row_capacity = row_capacity
        self._rows: list[list[_DocumentPiece]] = []
        self._used_token_counts: list[int] = []
        self._capacity_heaps: list[list[int]] = [[] for _ in range(row_capacity + 1)]
        self._nonempty_capacities = 0

    def add_row(self, piece: _DocumentPiece) -> None:
        packed_token_count = piece.packed_token_count
        if packed_token_count > self._row_capacity:
            raise ValueError(
                "document piece exceeds full-document row capacity: "
                f"{packed_token_count} > {self._row_capacity}"
            )
        row_index = len(self._rows)
        self._rows.append([piece])
        self._used_token_counts.append(packed_token_count)
        self._register_capacity(
            self._row_capacity - packed_token_count,
            row_index,
        )

    def place_best_fit(self, piece: _DocumentPiece) -> None:
        packed_token_count = piece.packed_token_count
        if packed_token_count > self._row_capacity:
            raise ValueError(
                "document piece exceeds full-document row capacity: "
                f"{packed_token_count} > {self._row_capacity}"
            )
        eligible_capacities = self._nonempty_capacities >> packed_token_count
        if eligible_capacities == 0:
            self.add_row(piece)
            return

        capacity_offset = (eligible_capacities & -eligible_capacities).bit_length() - 1
        capacity = packed_token_count + capacity_offset
        candidates = self._capacity_heaps[capacity]
        row_index = heapq.heappop(candidates)
        if not candidates:
            self._nonempty_capacities &= ~(1 << capacity)
        self._rows[row_index].append(piece)
        self._used_token_counts[row_index] += packed_token_count
        self._register_capacity(capacity - packed_token_count, row_index)

    def freeze(self) -> tuple[_PackedRow, ...]:
        return tuple(
            _PackedRow(tuple(pieces), used_token_count)
            for pieces, used_token_count in zip(
                self._rows,
                self._used_token_counts,
                strict=True,
            )
        )

    def _register_capacity(self, capacity: int, row_index: int) -> None:
        if capacity == 0:
            return
        heapq.heappush(self._capacity_heaps[capacity], row_index)
        self._nonempty_capacities |= 1 << capacity


class FullDocumentValidationBatches(
    Iterator[tuple[Tensor, Tensor, Tensor]],
):
    """Yield one deterministic, finite, full-document validation pass.

    Every ordinary validation token appears as a supervised target exactly
    once. A later piece of an oversized document carries the immediately
    preceding ordinary token as context, rather than pretending to start a new
    document.
    """

    def __init__(
        self,
        reader: TokenizedShardReader,
        *,
        config: FullDocumentProtocolConfig,
        bos_token_id: int,
    ) -> None:
        if not isinstance(reader, TokenizedShardReader):
            raise TypeError(
                f"reader must be a TokenizedShardReader, got {type(reader).__name__}"
            )
        if not isinstance(config, FullDocumentProtocolConfig):
            raise TypeError(
                "config must be a FullDocumentProtocolConfig, "
                f"got {type(config).__name__}"
            )
        normalized_bos = require_non_negative_integer(
            bos_token_id,
            name="bos_token_id",
        )
        if normalized_bos >= reader.manifest.vocab_size:
            raise ValueError(
                "bos_token_id must be less than the manifest vocabulary size "
                f"{reader.manifest.vocab_size}; got {normalized_bos}"
            )
        manifest_bos = reader.manifest.special_token_ids.get("<|bos|>")
        if manifest_bos is None:
            raise TokenizedDataError(
                "tokenized manifest does not define required '<|bos|>'"
            )
        if normalized_bos != manifest_bos:
            raise TokenizedDataError(
                "bos_token_id does not match the tokenized manifest"
            )

        spans = reader.document_spans("val")
        if not spans:
            raise TokenizedDataError("validation manifest must contain documents")
        self._reader = reader
        self._mapped_shards = reader.shards("val")
        self.config = config
        self.bos_token_id = normalized_bos
        self._rows = _plan_rows(
            spans,
            context_length=config.context_length,
        )
        self._batch_index = 0

    @property
    def plan_signature(
        self,
    ) -> tuple[tuple[tuple[int, int, int, bool, bool], ...], ...]:
        """Return stable source-coordinate metadata for conformance tests."""

        return tuple(
            tuple(piece.signature for piece in row.pieces) for row in self._rows
        )

    def __len__(self) -> int:
        """Return the finite number of padded model batches."""

        return math.ceil(len(self._rows) / self.config.device_batch_size)

    def __iter__(self) -> FullDocumentValidationBatches:
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._batch_index >= len(self):
            raise StopIteration

        batch_start = self._batch_index * self.config.device_batch_size
        batch_rows = self._rows[
            batch_start : batch_start + self.config.device_batch_size
        ]
        windows = torch.full(
            (
                self.config.device_batch_size,
                self.config.row_capacity,
            ),
            self.bos_token_id,
            dtype=torch.long,
            device="cpu",
        )
        loss_mask = torch.zeros(
            (
                self.config.device_batch_size,
                self.config.context_length,
            ),
            dtype=torch.bool,
            device="cpu",
        )
        for row_index, row in enumerate(batch_rows):
            packed_offset = 0
            for piece in row.pieces:
                if piece.is_continuation:
                    if piece.start == 0:
                        raise RuntimeError(
                            "continuation piece has no preceding source token"
                        )
                    windows[row_index, packed_offset] = int(
                        self._mapped_shards[piece.shard_index][piece.start - 1]
                    )
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
                    windows[
                        row_index,
                        content_start:content_stop,
                    ].copy_(torch.from_numpy(copied_tokens))
                    loss_mask[
                        row_index,
                        packed_offset : packed_offset + piece.token_count,
                    ] = True
                packed_offset += piece.packed_token_count
                if piece.is_document_end and packed_offset < self.config.row_capacity:
                    loss_mask[row_index, packed_offset - 1] = True

        self._batch_index += 1
        return windows[:, :-1], windows[:, 1:], loss_mask


def evaluate_full_document_bpb(
    model: nn.Module,
    tokenizer: Tokenizer,
    reader: TokenizedShardReader,
    token_bytes: Tensor,
    *,
    checkpoint_identity: str,
    config: FullDocumentProtocolConfig,
    device: str | torch.device,
) -> BaseValidationResult:
    """Evaluate exactly one finite validation-manifest pass."""

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    if not isinstance(reader, TokenizedShardReader):
        raise TypeError(
            f"reader must be a TokenizedShardReader, got {type(reader).__name__}"
        )
    if not isinstance(config, FullDocumentProtocolConfig):
        raise TypeError(
            f"config must be a FullDocumentProtocolConfig, got {type(config).__name__}"
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
    bos_token_id = tokenizer.get_bos_token_id()
    if int(normalized_token_bytes[bos_token_id].item()) != 0:
        raise ValueError("BOS token byte length must be zero")

    source_documents, source_tokens, source_bytes = _source_coverage(
        reader,
        normalized_token_bytes,
    )
    batches = FullDocumentValidationBatches(
        reader,
        config=config,
        bos_token_id=bos_token_id,
    )
    accumulation = evaluate_bpb_batches(
        model,
        batches,
        normalized_token_bytes,
        device=device,
    )
    if accumulation.counted_target_tokens != source_tokens:
        raise RuntimeError(
            "full-document evaluation did not count every source token exactly once"
        )
    if accumulation.counted_target_bytes != source_bytes:
        raise RuntimeError(
            "full-document evaluation did not count every source byte exactly once"
        )

    return BaseValidationResult.from_accumulation(
        accumulation,
        protocol_id=FULL_DOCUMENT_PROTOCOL_ID,
        protocol_version=FULL_DOCUMENT_PROTOCOL_VERSION,
        reference_commit=None,
        reference_config=config.to_dict(),
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer.get_identity(),
        validation_manifest_identity=tokenized_manifest_identity(reader.manifest),
        source_documents=source_documents,
        source_tokens=source_tokens,
        source_bytes=source_bytes,
        unique_source_tokens=source_tokens,
        unique_source_bytes=source_bytes,
    )


def full_document_metric_value(
    result: BaseValidationResult,
    *,
    key: str,
) -> float:
    """Return BPB only for the full-document protocol's reserved keys."""

    if not isinstance(result, BaseValidationResult):
        raise TypeError(
            f"result must be a BaseValidationResult, got {type(result).__name__}"
        )
    if result.protocol_id != FULL_DOCUMENT_PROTOCOL_ID:
        raise ValueError(f"result protocol must be {FULL_DOCUMENT_PROTOCOL_ID!r}")
    if key not in _METRIC_KEYS:
        raise ValueError(
            "reserved full-document metric key must be exactly "
            f"{FULL_DOCUMENT_TRAIN_METRIC!r} or "
            f"{FULL_DOCUMENT_EVAL_METRIC!r}; got {key!r}"
        )
    return result.bpb


def _plan_rows(
    spans: tuple[TokenizedDocumentSpan, ...],
    *,
    context_length: int,
) -> tuple[_PackedRow, ...]:
    planner = _CapacityIndexedRows(context_length + 1)
    for span in spans:
        remaining = span.token_count
        document_offset = 0
        while remaining > 0:
            piece_token_count = min(remaining, context_length)
            is_continuation = document_offset > 0
            piece = _DocumentPiece(
                shard_index=span.shard_index,
                start=span.start + document_offset,
                token_count=piece_token_count,
                is_continuation=is_continuation,
                is_document_end=piece_token_count == remaining,
            )
            if is_continuation:
                planner.add_row(piece)
            else:
                planner.place_best_fit(piece)
            remaining -= piece_token_count
            document_offset += piece_token_count
        if span.token_count == 0:
            planner.place_best_fit(
                _DocumentPiece(
                    shard_index=span.shard_index,
                    start=span.start,
                    token_count=0,
                    is_continuation=False,
                    is_document_end=True,
                )
            )
    return planner.freeze()


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


__all__ = [
    "FULL_DOCUMENT_EVAL_METRIC",
    "FULL_DOCUMENT_PROTOCOL_ID",
    "FULL_DOCUMENT_PROTOCOL_VERSION",
    "FULL_DOCUMENT_TRAIN_METRIC",
    "FullDocumentProtocolConfig",
    "FullDocumentValidationBatches",
    "evaluate_full_document_bpb",
    "full_document_metric_value",
]
