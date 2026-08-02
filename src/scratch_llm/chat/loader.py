"""Resumable bounded best-fit packing for assistant-masked SFT batches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from os import PathLike
from pathlib import Path
import random
from typing import Final, Protocol

import torch
from torch import Tensor

from scratch_llm._validation import (
    require_integer,
    require_positive_integer,
)
from scratch_llm.chat.conversation import (
    AssistantMessage,
    Conversation,
    PythonOutputPart,
    PythonPart,
    SystemMessage,
    TextPart,
    UserMessage,
    parse_conversation,
    read_conversations,
)
from scratch_llm.chat.rendering import (
    IGNORE_INDEX,
    ChatRenderingError,
    RenderedConversation,
    render_conversation,
    shift_sft_targets,
)
from scratch_llm.data.sft_sources import SFTConversationExample
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.identity import file_identity


SFT_LOADER_STATE_FORMAT: Final = "scratch_llm_sft_conversation_loader_state"
SFT_LOADER_STATE_VERSION: Final = 1
SFT_CROP_POLICY: Final = "bounded_prefix_v1"
_MAX_SEED: Final = 2**63 - 1
_SOURCE_SEED_MAX: Final = 2**32 - 1
_STATE_FIELDS: Final = frozenset(
    {
        "base_seed",
        "batch_size",
        "buffer",
        "crop_policy",
        "epoch",
        "epoch_exhausted",
        "epoch_packed_conversations",
        "epoch_seed",
        "epoch_step",
        "format",
        "format_version",
        "global_step",
        "max_seq_len",
        "mixture_cursor",
        "packing_buffer_size",
        "repeat",
        "repeat_weights",
        "rng_state",
        "source_order",
        "stats",
        "tokenizer_identity",
        "vocab_size",
    }
)
_BUFFER_FIELDS: Final = frozenset(
    {
        "item_identity",
        "source_index",
        "source_offset",
    }
)
_STATS_FIELDS: Final = frozenset(
    {
        "cropped_conversations",
        "emitted_batches",
        "emitted_rows",
        "packed_conversations",
        "padding_rows",
        "padding_tokens",
        "seen_conversations",
        "skipped_zero_supervision",
    }
)


class SFTLoaderError(ValueError):
    """An SFT source or packing request cannot produce a safe batch."""


class SFTLoaderStateError(SFTLoaderError):
    """A serialized loader state is malformed or incompatible."""


class FiniteConversationSource(Protocol):
    """Fresh finite iteration contract shared by local and parquet sources."""

    source_identity: str

    def __len__(self) -> int:
        """Return the exact number of examples in one source cycle."""

    def iter_examples(self, *, seed: int) -> Iterator[SFTConversationExample]:
        """Return a fresh deterministic iterator for ``seed``."""


@dataclass(frozen=True, slots=True)
class WeightedConversationSource:
    """One finite source and its explicit number of repeats per epoch."""

    source: FiniteConversationSource
    repeat_weight: int = 1

    def __post_init__(self) -> None:
        if not hasattr(self.source, "source_identity") or not callable(
            getattr(self.source, "iter_examples", None)
        ):
            raise TypeError("source must implement the finite conversation contract")
        try:
            require_positive_integer(self.repeat_weight, name="repeat_weight")
        except (TypeError, ValueError) as error:
            raise type(error)(str(error)) from error


class InMemoryConversationSource:
    """Immutable tiny/local conversation source with optional seeded shuffling."""

    def __init__(
        self,
        conversations: Sequence[Conversation | Mapping[str, object]],
        *,
        source_identity: str | None = None,
        shuffle: bool = True,
    ) -> None:
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean")
        self._conversations = tuple(
            parse_conversation(conversation) for conversation in conversations
        )
        if source_identity is None:
            source_identity = _canonical_identity(
                {
                    "conversations": [
                        _conversation_payload(conversation)
                        for conversation in self._conversations
                    ],
                    "format": "scratch_llm_in_memory_conversations_v1",
                    "shuffle": shuffle,
                }
            )
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise ValueError("source_identity must be a non-empty string")
        self.source_identity = source_identity
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self._conversations)

    def iter_examples(self, *, seed: int) -> Iterator[SFTConversationExample]:
        seed = _validate_source_seed(seed)
        order = list(range(len(self)))
        if self.shuffle:
            random.Random(seed).shuffle(order)
        for source_row in order:
            conversation = self._conversations[source_row]
            yield SFTConversationExample(
                conversation=conversation,
                source_identity=self.source_identity,
                source_row=source_row,
                identity=_canonical_identity(
                    {
                        "conversation": _conversation_payload(conversation),
                        "source_identity": self.source_identity,
                        "source_row": source_row,
                    }
                ),
            )


@dataclass(frozen=True, slots=True)
class SFTLoaderStats:
    """Cumulative observable packing, crop, skip, and fill counters."""

    seen_conversations: int = 0
    cropped_conversations: int = 0
    skipped_zero_supervision: int = 0
    packed_conversations: int = 0
    emitted_rows: int = 0
    emitted_batches: int = 0
    padding_tokens: int = 0
    padding_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in sorted(_STATS_FIELDS)}


@dataclass(frozen=True, slots=True)
class SFTBatchInfo:
    """Non-tensor provenance for the most recently emitted batch."""

    epoch: int
    epoch_step: int
    row_item_identities: tuple[tuple[str, ...], ...]
    content_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BufferedConversation:
    source_index: int
    source_offset: int
    item_identity: str
    rendered: RenderedConversation

    def state_dict(self) -> dict[str, object]:
        return {
            "item_identity": self.item_identity,
            "source_index": self.source_index,
            "source_offset": self.source_offset,
        }


@dataclass(frozen=True, slots=True)
class _PackedRow:
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    item_identities: tuple[str, ...]
    content_length: int


@dataclass(frozen=True, slots=True)
class _LocatedConversation:
    source_index: int
    source_offset: int
    example: SFTConversationExample


@dataclass(frozen=True, slots=True)
class _LoaderProgress:
    rng: random.Random
    epoch: int
    epoch_seed: int
    epoch_step: int
    global_step: int
    epoch_exhausted: bool
    epoch_packed_conversations: int
    mixture_cursor: int
    stats: SFTLoaderStats


@dataclass(frozen=True, slots=True)
class _RestoredLoaderState:
    progress: _LoaderProgress
    mixture: _ConversationMixture
    buffer: list[_BufferedConversation]


class _ConversationMixture:
    """Deterministic weighted source schedule and its live source iterators."""

    def __init__(
        self,
        sources: tuple[WeightedConversationSource, ...],
        *,
        source_lengths: tuple[int, ...],
        source_identities: tuple[str, ...],
        repeat_weights: tuple[int, ...],
        epoch_seed: int,
        cursor: int = 0,
    ) -> None:
        self._sources = sources
        self._source_lengths = source_lengths
        self._source_identities = source_identities
        self._repeat_weights = repeat_weights
        self.epoch_seed = epoch_seed
        self._order = self._build_order()
        if not 0 <= cursor <= len(self._order):
            raise SFTLoaderStateError("loader state mixture cursor is out of range")
        self.cursor = cursor
        consumed_counts = Counter(self._order[:cursor])
        self._source_cursors = [
            consumed_counts[source_index] for source_index in range(len(sources))
        ]
        self._source_iterators = self._build_source_iterators()

    @property
    def exhausted(self) -> bool:
        return self.cursor == len(self._order)

    def source_cursor(self, source_index: int) -> int:
        return self._source_cursors[source_index]

    def pull(self) -> _LocatedConversation | None:
        if self.exhausted:
            return None
        source_index = self._order[self.cursor]
        source_offset = self._source_cursors[source_index]
        source_length = self._source_lengths[source_index]
        iterator = self._source_iterators[source_index]
        if iterator is None:
            raise SFTLoaderError(
                f"source {source_index} iterator ended before mixture order"
            )
        try:
            example = next(iterator)
        except StopIteration as error:
            raise SFTLoaderError(
                f"source {source_index} yielded fewer than {source_length} "
                "examples in one cycle"
            ) from error
        example = self._validate_example(example, source_index)

        self.cursor += 1
        self._source_cursors[source_index] += 1
        self._advance_source_cycle_if_needed(source_index, iterator)
        return _LocatedConversation(source_index, source_offset, example)

    def fetch(self, source_index: int, source_offset: int) -> SFTConversationExample:
        source_length = self._source_lengths[source_index]
        cycle, source_position = divmod(source_offset, source_length)
        iterator = self._sources[source_index].source.iter_examples(
            seed=_source_cycle_seed(self.epoch_seed, source_index, cycle)
        )
        for position in range(source_position + 1):
            try:
                example = next(iterator)
            except StopIteration as error:
                raise SFTLoaderStateError(
                    f"source {source_index} ended before buffered position "
                    f"{source_position}"
                ) from error
            example = self._validate_example(example, source_index)
            if position == source_position:
                return example
        raise AssertionError("source position loop must return")

    def _build_order(self) -> list[int]:
        order: list[int] = []
        for source_index, (source_length, repeat_weight) in enumerate(
            zip(self._source_lengths, self._repeat_weights, strict=True)
        ):
            order.extend([source_index] * (source_length * repeat_weight))
        random.Random(self.epoch_seed).shuffle(order)
        return order

    def _build_source_iterators(
        self,
    ) -> list[Iterator[SFTConversationExample] | None]:
        iterators: list[Iterator[SFTConversationExample] | None] = []
        for source_index, cursor in enumerate(self._source_cursors):
            source_length = self._source_lengths[source_index]
            source_total = source_length * self._repeat_weights[source_index]
            if cursor == source_total:
                iterators.append(None)
                continue
            cycle, position = divmod(cursor, source_length)
            iterator = self._sources[source_index].source.iter_examples(
                seed=_source_cycle_seed(self.epoch_seed, source_index, cycle)
            )
            for skipped_position in range(position):
                try:
                    skipped = next(iterator)
                except StopIteration as error:
                    raise SFTLoaderError(
                        f"source {source_index} ended before declared length while "
                        f"restoring position {skipped_position}"
                    ) from error
                self._validate_example(skipped, source_index)
            iterators.append(iterator)
        return iterators

    def _advance_source_cycle_if_needed(
        self,
        source_index: int,
        iterator: Iterator[SFTConversationExample],
    ) -> None:
        cursor = self._source_cursors[source_index]
        source_length = self._source_lengths[source_index]
        if cursor % source_length:
            return
        try:
            extra = next(iterator)
        except StopIteration:
            extra = None
        if extra is not None:
            raise SFTLoaderError(
                f"source {source_index} yielded more than its declared "
                f"length {source_length}"
            )
        source_total = source_length * self._repeat_weights[source_index]
        if cursor == source_total:
            self._source_iterators[source_index] = None
            return
        cycle = cursor // source_length
        self._source_iterators[source_index] = self._sources[
            source_index
        ].source.iter_examples(
            seed=_source_cycle_seed(self.epoch_seed, source_index, cycle)
        )

    def _validate_example(
        self,
        example: object,
        source_index: int,
    ) -> SFTConversationExample:
        if not isinstance(example, SFTConversationExample):
            raise SFTLoaderError(
                f"source {source_index} must yield SFTConversationExample values"
            )
        if example.source_identity != self._source_identities[source_index]:
            raise SFTLoaderError(
                f"source {source_index} yielded a conflicting source identity"
            )
        if (
            not isinstance(example.identity, str)
            or not example.identity.strip()
            or not isinstance(example.source_row, int)
            or isinstance(example.source_row, bool)
            or example.source_row < 0
        ):
            raise SFTLoaderError(
                f"source {source_index} yielded invalid example metadata"
            )
        if not isinstance(example.conversation, Conversation):
            raise SFTLoaderError(
                f"source {source_index} yielded an invalid conversation"
            )
        return example


class SFTConversationLoader(Iterator[tuple[Tensor, Tensor]]):
    """Pack complete bounded conversations into exact assistant-label batches."""

    def __init__(
        self,
        sources: Sequence[WeightedConversationSource],
        *,
        tokenizer: Tokenizer,
        batch_size: int,
        max_seq_len: int,
        packing_buffer_size: int = 100,
        seed: int = 42,
        repeat: bool = True,
    ) -> None:
        normalized_sources, source_lengths, source_identities = (
            _validate_source_contract(sources)
        )
        batch_size, max_seq_len, packing_buffer_size, seed, repeat = (
            _validate_loader_settings(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                packing_buffer_size=packing_buffer_size,
                seed=seed,
                repeat=repeat,
            )
        )
        vocab_size, bos_token_id, tokenizer_identity = _validate_tokenizer_contract(
            tokenizer
        )

        self.sources = normalized_sources
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.row_capacity = max_seq_len + 1
        self.packing_buffer_size = packing_buffer_size
        self.base_seed = seed
        self.repeat = repeat
        self._source_lengths = source_lengths
        self._source_identities = source_identities
        self._repeat_weights = tuple(
            source.repeat_weight for source in normalized_sources
        )
        self._vocab_size = vocab_size
        self._bos_token_id = bos_token_id
        self._tokenizer_identity = tokenizer_identity
        self._rng = random.Random(seed)
        self._stats_values = {field: 0 for field in _STATS_FIELDS}
        self.global_step = 0
        self.epoch = -1
        self.epoch_seed = 0
        self.epoch_step = 0
        self.epoch_exhausted = False
        self._epoch_packed_conversations = 0
        self._buffer: list[_BufferedConversation] = []
        self._last_batch_info: SFTBatchInfo | None = None
        self._start_next_epoch()

    def __iter__(self) -> SFTConversationLoader:
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        return self.next_batch()

    @property
    def stats(self) -> SFTLoaderStats:
        """Return a frozen snapshot of cumulative loader counters."""

        return SFTLoaderStats(**self._stats_values)

    @property
    def last_batch_info(self) -> SFTBatchInfo:
        """Return provenance for the last successful batch."""

        if self._last_batch_info is None:
            raise SFTLoaderError("no SFT batch has been emitted yet")
        return self._last_batch_info

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return the next contiguous CPU ``long`` input/label batch."""

        packed_rows = self._collect_batch_rows()
        inputs, labels = self._materialize_batch(packed_rows)
        self._record_batch(packed_rows)
        return inputs, labels

    def _collect_batch_rows(self) -> tuple[_PackedRow, ...]:
        while True:
            if self.epoch_exhausted:
                if not self.repeat:
                    raise StopIteration
                self._start_next_epoch()

            packed_rows: list[_PackedRow] = []
            while len(packed_rows) < self.batch_size:
                row = self._build_row()
                if row is None:
                    break
                packed_rows.append(row)
            if packed_rows:
                break

            self.epoch_exhausted = True
            if self._epoch_packed_conversations == 0:
                raise SFTLoaderError(
                    "SFT epoch has no supervised assistant targets after "
                    "bounded-prefix cropping"
                )

        while len(packed_rows) < self.batch_size:
            packed_rows.append(self._padding_row())
        if self._mixture.exhausted and not self._buffer:
            self.epoch_exhausted = True
        return tuple(packed_rows)

    def _materialize_batch(
        self,
        packed_rows: Sequence[_PackedRow],
    ) -> tuple[Tensor, Tensor]:
        input_rows: list[tuple[int, ...]] = []
        label_rows: list[tuple[int, ...]] = []
        for row in packed_rows:
            if row.content_length == 0:
                input_rows.append(row.token_ids[:-1])
                label_rows.append((IGNORE_INDEX,) * self.max_seq_len)
            else:
                shifted = shift_sft_targets(row.token_ids, row.loss_mask)
                input_rows.append(shifted.input_ids)
                label_rows.append(shifted.labels)

        inputs = torch.tensor(input_rows, dtype=torch.long, device="cpu").contiguous()
        labels = torch.tensor(label_rows, dtype=torch.long, device="cpu").contiguous()
        if inputs.shape != (self.batch_size, self.max_seq_len) or labels.shape != (
            self.batch_size,
            self.max_seq_len,
        ):
            raise AssertionError("SFT batch materialization produced an invalid shape")
        if not bool((labels != IGNORE_INDEX).any()):
            raise SFTLoaderError("refusing to emit an all-ignored SFT batch")
        return inputs, labels

    def _record_batch(self, packed_rows: Sequence[_PackedRow]) -> None:
        self.global_step += 1
        self.epoch_step += 1
        self._stats_values["emitted_batches"] += 1
        self._stats_values["emitted_rows"] += self.batch_size
        self._last_batch_info = SFTBatchInfo(
            epoch=self.epoch,
            epoch_step=self.epoch_step,
            row_item_identities=tuple(row.item_identities for row in packed_rows),
            content_lengths=tuple(row.content_length for row in packed_rows),
        )

    def iter_epoch(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield exactly the current (or next) finite epoch and then stop."""

        if self.epoch_exhausted:
            if not self.repeat:
                return
            self._start_next_epoch()
        active_epoch = self.epoch
        while self.epoch == active_epoch:
            try:
                batch = self.next_batch()
            except StopIteration:
                return
            yield batch
            if self.epoch_exhausted:
                return

    def state_dict(self) -> dict[str, object]:
        """Return JSON-safe exact continuation state without serialized tensors."""

        return {
            "base_seed": self.base_seed,
            "batch_size": self.batch_size,
            "buffer": [item.state_dict() for item in self._buffer],
            "crop_policy": SFT_CROP_POLICY,
            "epoch": self.epoch,
            "epoch_exhausted": self.epoch_exhausted,
            "epoch_packed_conversations": self._epoch_packed_conversations,
            "epoch_seed": self.epoch_seed,
            "epoch_step": self.epoch_step,
            "format": SFT_LOADER_STATE_FORMAT,
            "format_version": SFT_LOADER_STATE_VERSION,
            "global_step": self.global_step,
            "max_seq_len": self.max_seq_len,
            "mixture_cursor": self._mixture.cursor,
            "packing_buffer_size": self.packing_buffer_size,
            "repeat": self.repeat,
            "repeat_weights": list(self._repeat_weights),
            "rng_state": _rng_state_to_json(self._rng.getstate()),
            "source_order": list(self._source_identities),
            "stats": self.stats.to_dict(),
            "tokenizer_identity": self._tokenizer_identity,
            "vocab_size": self._vocab_size,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate and rehydrate exact continuation state transactionally."""

        restored = self._decode_state(state)
        progress = restored.progress
        self._rng = progress.rng
        self.epoch = progress.epoch
        self.epoch_seed = progress.epoch_seed
        self.epoch_step = progress.epoch_step
        self.global_step = progress.global_step
        self.epoch_exhausted = progress.epoch_exhausted
        self._epoch_packed_conversations = progress.epoch_packed_conversations
        self._mixture = restored.mixture
        self._buffer = restored.buffer
        self._stats_values = progress.stats.to_dict()
        self._last_batch_info = None

    def _decode_state(self, state: object) -> _RestoredLoaderState:
        if not isinstance(state, Mapping):
            raise SFTLoaderStateError(
                f"loader state must be a mapping, got {type(state).__name__}"
            )
        self._validate_state_contract(state)
        progress = _parse_loader_progress(state)
        try:
            mixture = self._new_mixture(
                progress.epoch_seed,
                cursor=progress.mixture_cursor,
            )
            buffer = self._rehydrate_buffer(state["buffer"], mixture=mixture)
        except (SFTLoaderError, StopIteration) as error:
            raise SFTLoaderStateError(
                f"loader state cannot rehydrate sources or buffer: {error}"
            ) from error
        self._validate_resume_position(progress, mixture, buffer)
        return _RestoredLoaderState(progress, mixture, buffer)

    def _validate_state_contract(self, state: Mapping[str, object]) -> None:
        keys = set(state)
        if keys != _STATE_FIELDS:
            missing = sorted(_STATE_FIELDS - keys)
            unexpected = sorted(keys - _STATE_FIELDS, key=str)
            raise SFTLoaderStateError(
                "loader state fields do not match format version "
                f"{SFT_LOADER_STATE_VERSION}; missing={missing}, "
                f"unexpected={unexpected}"
            )
        if state["format"] != SFT_LOADER_STATE_FORMAT:
            raise SFTLoaderStateError(
                f"unknown loader state format {state['format']!r}"
            )
        version = _state_integer(state["format_version"], label="format version")
        if version != SFT_LOADER_STATE_VERSION:
            raise SFTLoaderStateError(
                f"unknown loader state format version {version}; "
                f"expected {SFT_LOADER_STATE_VERSION}"
            )
        _require_state_setting(state, "tokenizer_identity", self._tokenizer_identity)
        _require_state_setting(state, "vocab_size", self._vocab_size)
        _require_state_setting(state, "source_order", list(self._source_identities))
        _require_state_setting(state, "repeat_weights", list(self._repeat_weights))
        _require_state_setting(state, "batch_size", self.batch_size)
        _require_state_setting(state, "max_seq_len", self.max_seq_len)
        _require_state_setting(
            state,
            "packing_buffer_size",
            self.packing_buffer_size,
        )
        _require_state_setting(state, "base_seed", self.base_seed)
        _require_state_setting(state, "repeat", self.repeat)
        _require_state_setting(state, "crop_policy", SFT_CROP_POLICY)

    def _validate_resume_position(
        self,
        progress: _LoaderProgress,
        mixture: _ConversationMixture,
        buffer: Sequence[_BufferedConversation],
    ) -> None:
        if progress.epoch_exhausted and (not mixture.exhausted or buffer):
            raise SFTLoaderStateError(
                "loader state marks an epoch exhausted before mixture and buffer end"
            )
        if (
            not progress.epoch_exhausted
            and mixture.exhausted
            and not buffer
            and progress.epoch_packed_conversations > 0
        ):
            raise SFTLoaderStateError(
                "loader state must mark a drained epoch as exhausted"
            )

    def _start_next_epoch(self) -> None:
        self.epoch += 1
        self.epoch_seed = self._rng.randrange(_MAX_SEED + 1)
        self.epoch_step = 0
        self.epoch_exhausted = False
        self._epoch_packed_conversations = 0
        self._mixture = self._new_mixture(self.epoch_seed)
        self._buffer = []
        self._last_batch_info = None

    def _new_mixture(
        self,
        epoch_seed: int,
        *,
        cursor: int = 0,
    ) -> _ConversationMixture:
        return _ConversationMixture(
            self.sources,
            source_lengths=self._source_lengths,
            source_identities=self._source_identities,
            repeat_weights=self._repeat_weights,
            epoch_seed=epoch_seed,
            cursor=cursor,
        )

    def _fill_buffer(self) -> None:
        while len(self._buffer) < self.packing_buffer_size:
            item = self._pull_next_item()
            if item is None:
                return
            self._buffer.append(item)

    def _pull_next_item(self) -> _BufferedConversation | None:
        while located := self._mixture.pull():
            item, cropped = self._render_bounded_item(
                located.example,
                source_index=located.source_index,
                source_offset=located.source_offset,
            )

            self._stats_values["seen_conversations"] += 1
            if cropped:
                self._stats_values["cropped_conversations"] += 1
            if item is None:
                self._stats_values["skipped_zero_supervision"] += 1
                continue
            return item
        return None

    def _render_bounded_item(
        self,
        example: SFTConversationExample,
        *,
        source_index: int,
        source_offset: int,
    ) -> tuple[_BufferedConversation | None, bool]:
        try:
            rendered = render_conversation(example.conversation, self.tokenizer)
        except (ChatRenderingError, TypeError, ValueError) as error:
            raise SFTLoaderError(
                f"could not render source {source_index} example "
                f"{example.identity}: {error}"
            ) from error
        cropped = len(rendered.token_ids) > self.row_capacity
        if cropped:
            rendered = RenderedConversation(
                token_ids=rendered.token_ids[: self.row_capacity],
                loss_mask=rendered.loss_mask[: self.row_capacity],
            )
        if any(token_id >= self._vocab_size for token_id in rendered.token_ids):
            raise SFTLoaderError(
                f"source {source_index} example contains a token outside vocabulary"
            )
        if not any(rendered.loss_mask[1:]):
            return None, cropped
        item_identity = _canonical_identity(
            {
                "example_identity": example.identity,
                "source_index": source_index,
                "source_offset": source_offset,
            }
        )
        return (
            _BufferedConversation(
                source_index=source_index,
                source_offset=source_offset,
                item_identity=item_identity,
                rendered=rendered,
            ),
            cropped,
        )

    def _build_row(self) -> _PackedRow | None:
        row_ids: list[int] = []
        row_mask: list[bool] = []
        item_identities: list[str] = []
        while len(row_ids) < self.row_capacity:
            self._fill_buffer()
            if not self._buffer:
                if not row_ids:
                    return None
                break
            remaining = self.row_capacity - len(row_ids)
            best_index = _largest_fitting_index(self._buffer, remaining)
            if best_index is None:
                if not row_ids:
                    raise AssertionError(
                        "bounded conversation cannot fit into an empty row"
                    )
                break
            item = self._buffer.pop(best_index)
            row_ids.extend(item.rendered.token_ids)
            row_mask.extend(item.rendered.loss_mask)
            item_identities.append(item.item_identity)
            self._stats_values["packed_conversations"] += 1
            self._epoch_packed_conversations += 1

        content_length = len(row_ids)
        padding = self.row_capacity - content_length
        if padding:
            row_ids.extend([self._bos_token_id] * padding)
            row_mask.extend([False] * padding)
            self._stats_values["padding_tokens"] += padding
        return _PackedRow(
            token_ids=tuple(row_ids),
            loss_mask=tuple(row_mask),
            item_identities=tuple(item_identities),
            content_length=content_length,
        )

    def _padding_row(self) -> _PackedRow:
        self._stats_values["padding_rows"] += 1
        self._stats_values["padding_tokens"] += self.row_capacity
        return _PackedRow(
            token_ids=(self._bos_token_id,) * self.row_capacity,
            loss_mask=(False,) * self.row_capacity,
            item_identities=(),
            content_length=0,
        )

    def _rehydrate_buffer(
        self,
        raw_buffer: object,
        *,
        mixture: _ConversationMixture,
    ) -> list[_BufferedConversation]:
        if not isinstance(raw_buffer, list):
            raise SFTLoaderStateError("loader state buffer must be a list")
        if len(raw_buffer) > self.packing_buffer_size:
            raise SFTLoaderStateError("loader state buffer exceeds packing_buffer_size")
        result: list[_BufferedConversation] = []
        identities: set[str] = set()
        for buffer_index, raw_item in enumerate(raw_buffer):
            item = self._rehydrate_buffer_item(raw_item, buffer_index, mixture)
            if item.item_identity in identities:
                raise SFTLoaderStateError(
                    "loader state buffer contains duplicate item identities"
                )
            identities.add(item.item_identity)
            result.append(item)
        return result

    def _rehydrate_buffer_item(
        self,
        raw_item: object,
        buffer_index: int,
        mixture: _ConversationMixture,
    ) -> _BufferedConversation:
        if not isinstance(raw_item, Mapping) or set(raw_item) != _BUFFER_FIELDS:
            raise SFTLoaderStateError(
                f"loader state buffer[{buffer_index}] fields are invalid"
            )
        source_index = _state_non_negative_integer(
            raw_item["source_index"],
            label=f"buffer[{buffer_index}].source_index",
        )
        if source_index >= len(self.sources):
            raise SFTLoaderStateError(
                f"loader state buffer[{buffer_index}] source is out of range"
            )
        source_offset = _state_non_negative_integer(
            raw_item["source_offset"],
            label=f"buffer[{buffer_index}].source_offset",
        )
        source_total = (
            self._source_lengths[source_index] * self._repeat_weights[source_index]
        )
        if source_offset >= source_total:
            raise SFTLoaderStateError(
                f"loader state buffer[{buffer_index}] locator is out of range"
            )
        if source_offset >= mixture.source_cursor(source_index):
            raise SFTLoaderStateError(
                f"loader state buffer[{buffer_index}] was not yet consumed"
            )
        example = mixture.fetch(source_index, source_offset)
        item, _ = self._render_bounded_item(
            example,
            source_index=source_index,
            source_offset=source_offset,
        )
        if item is None:
            raise SFTLoaderStateError(
                f"loader state buffer[{buffer_index}] has no supervision"
            )
        if raw_item["item_identity"] != item.item_identity:
            raise SFTLoaderStateError(
                f"loader state buffer[{buffer_index}] item identity is invalid"
            )
        return item


def build_fresh_sft_validation_loader(
    sources: Sequence[FiniteConversationSource],
    *,
    tokenizer: Tokenizer,
    batch_size: int,
    max_seq_len: int,
    packing_buffer_size: int = 100,
    seed: int = 42,
) -> SFTConversationLoader:
    """Build a new finite validation view independent of every train cursor."""

    return SFTConversationLoader(
        tuple(WeightedConversationSource(source) for source in sources),
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        packing_buffer_size=packing_buffer_size,
        seed=seed,
        repeat=False,
    )


def load_jsonl_conversation_source(
    path: str | PathLike[str],
    *,
    shuffle: bool,
) -> InMemoryConversationSource:
    """Load one strict tracked JSONL file as a stable finite source."""

    conversations = read_conversations(path)
    return InMemoryConversationSource(
        conversations,
        source_identity=_canonical_identity(
            {
                "file_identity": file_identity(Path(path)),
                "format": "scratch_llm_jsonl_conversation_source_v1",
            }
        ),
        shuffle=shuffle,
    )


def _validate_source_contract(
    sources: Sequence[WeightedConversationSource],
) -> tuple[
    tuple[WeightedConversationSource, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    normalized = tuple(sources)
    if not normalized or not all(
        isinstance(source, WeightedConversationSource) for source in normalized
    ):
        raise SFTLoaderError(
            "sources must be a non-empty sequence of WeightedConversationSource"
        )

    lengths: list[int] = []
    identities: list[str] = []
    for index, weighted in enumerate(normalized):
        identity = getattr(weighted.source, "source_identity", None)
        if not isinstance(identity, str) or not identity.strip():
            raise SFTLoaderError(f"source {index} has an invalid source_identity")
        try:
            source_length = len(weighted.source)
        except (TypeError, ValueError) as error:
            raise SFTLoaderError(
                f"source {index} did not return a valid length: {error}"
            ) from error
        if (
            not isinstance(source_length, int)
            or isinstance(source_length, bool)
            or source_length <= 0
        ):
            raise SFTLoaderError(
                f"source {index} is empty or has an invalid length: {source_length!r}"
            )
        lengths.append(source_length)
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise SFTLoaderError("weighted sources require unique source identities")
    return normalized, tuple(lengths), tuple(identities)


def _validate_loader_settings(
    *,
    batch_size: object,
    max_seq_len: object,
    packing_buffer_size: object,
    seed: object,
    repeat: object,
) -> tuple[int, int, int, int, bool]:
    try:
        normalized_batch_size = require_positive_integer(
            batch_size,
            name="batch_size",
        )
        normalized_max_seq_len = require_positive_integer(
            max_seq_len,
            name="max_seq_len",
        )
        normalized_buffer_size = require_positive_integer(
            packing_buffer_size,
            name="packing_buffer_size",
        )
        normalized_seed = require_integer(seed, name="seed")
    except (TypeError, ValueError) as error:
        raise SFTLoaderError(str(error)) from error
    if not 0 <= normalized_seed <= _MAX_SEED:
        raise SFTLoaderError(
            f"seed must be in range [0, {_MAX_SEED}], got {normalized_seed}"
        )
    if not isinstance(repeat, bool):
        raise TypeError("repeat must be a boolean")
    return (
        normalized_batch_size,
        normalized_max_seq_len,
        normalized_buffer_size,
        normalized_seed,
        repeat,
    )


def _validate_tokenizer_contract(tokenizer: Tokenizer) -> tuple[int, int, str]:
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement Tokenizer")
    vocab_size = tokenizer.get_vocab_size()
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size <= 0
    ):
        raise SFTLoaderError("tokenizer vocabulary size must be positive")
    bos_token_id = tokenizer.get_bos_token_id()
    if (
        not isinstance(bos_token_id, int)
        or isinstance(bos_token_id, bool)
        or not 0 <= bos_token_id < vocab_size
    ):
        raise SFTLoaderError("tokenizer BOS token ID is invalid")
    tokenizer_identity = tokenizer.get_identity()
    if not isinstance(tokenizer_identity, str) or not tokenizer_identity.strip():
        raise SFTLoaderError("tokenizer identity must be a non-empty string")
    return vocab_size, bos_token_id, tokenizer_identity


def _conversation_payload(conversation: Conversation) -> dict[str, object]:
    messages: list[dict[str, object]] = []
    for message in conversation.messages:
        if isinstance(message, (SystemMessage, UserMessage)):
            content: object = message.content
        elif isinstance(message, AssistantMessage) and isinstance(message.content, str):
            content = message.content
        elif isinstance(message, AssistantMessage):
            content = [
                {"text": part.text, "type": part.type}
                for part in message.content
                if isinstance(part, (TextPart, PythonPart, PythonOutputPart))
            ]
        else:
            raise AssertionError("validated conversation contains an unknown message")
        messages.append({"content": content, "role": message.role})
    return {
        "messages": messages,
        "schema_version": conversation.schema_version,
    }


def _canonical_identity(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _largest_fitting_index(
    buffer: Sequence[_BufferedConversation],
    capacity: int,
) -> int | None:
    fitting_indices = (
        index
        for index, item in enumerate(buffer)
        if len(item.rendered.token_ids) <= capacity
    )
    return max(
        fitting_indices,
        key=lambda index: len(buffer[index].rendered.token_ids),
        default=None,
    )


def _source_cycle_seed(epoch_seed: int, source_index: int, cycle: int) -> int:
    encoded = f"{epoch_seed}:{source_index}:{cycle}".encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _validate_source_seed(seed: object) -> int:
    try:
        normalized = require_integer(seed, name="seed")
    except TypeError as error:
        raise SFTLoaderError(str(error)) from error
    if not 0 <= normalized <= _SOURCE_SEED_MAX:
        raise SFTLoaderError(f"source seed must be in range [0, {_SOURCE_SEED_MAX}]")
    return normalized


def _rng_state_to_json(state: tuple[object, ...]) -> list[object]:
    if len(state) != 3 or not isinstance(state[1], tuple):
        raise AssertionError("Python random returned an unexpected state")
    return [state[0], list(state[1]), state[2]]


def _rng_state_from_json(value: object) -> tuple[object, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise SFTLoaderStateError(
            "loader state rng_state must be a three-item JSON list"
        )
    version, raw_internal, gaussian = value
    if not isinstance(version, int) or isinstance(version, bool):
        raise SFTLoaderStateError("loader state rng_state version is invalid")
    if (
        not isinstance(raw_internal, list)
        or not raw_internal
        or any(
            not isinstance(item, int) or isinstance(item, bool) for item in raw_internal
        )
    ):
        raise SFTLoaderStateError("loader state rng_state internal values are invalid")
    if gaussian is not None and (
        not isinstance(gaussian, (int, float))
        or isinstance(gaussian, bool)
        or not math.isfinite(float(gaussian))
    ):
        raise SFTLoaderStateError("loader state rng_state gaussian value is invalid")
    state = (version, tuple(raw_internal), gaussian)
    candidate = random.Random()
    try:
        candidate.setstate(state)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SFTLoaderStateError(
            f"loader state rng_state is invalid: {error}"
        ) from error
    return state


def _state_integer(value: object, *, label: str) -> int:
    try:
        return require_integer(value, name=label)
    except TypeError as error:
        raise SFTLoaderStateError(f"loader state {label} must be an integer") from error


def _state_non_negative_integer(value: object, *, label: str) -> int:
    normalized = _state_integer(value, label=label)
    if normalized < 0:
        raise SFTLoaderStateError(f"loader state {label} must be non-negative")
    return normalized


def _require_state_setting(
    state: Mapping[str, object],
    key: str,
    expected: object,
) -> None:
    if state[key] != expected:
        label = "source identities" if key == "source_order" else key
        label = "tokenizer identity" if key == "tokenizer_identity" else label
        raise SFTLoaderStateError(
            f"loader state {label} does not match configured loader"
        )


def _parse_loader_progress(state: Mapping[str, object]) -> _LoaderProgress:
    epoch = _state_non_negative_integer(state["epoch"], label="epoch")
    epoch_seed = _state_integer(state["epoch_seed"], label="epoch_seed")
    if not 0 <= epoch_seed <= _MAX_SEED:
        raise SFTLoaderStateError("loader state epoch_seed is out of range")
    epoch_step = _state_non_negative_integer(state["epoch_step"], label="epoch_step")
    global_step = _state_non_negative_integer(
        state["global_step"],
        label="global_step",
    )
    if epoch_step > global_step:
        raise SFTLoaderStateError("loader state epoch_step cannot exceed global_step")
    epoch_packed = _state_non_negative_integer(
        state["epoch_packed_conversations"],
        label="epoch_packed_conversations",
    )
    epoch_exhausted = state["epoch_exhausted"]
    if not isinstance(epoch_exhausted, bool):
        raise SFTLoaderStateError("loader state epoch_exhausted must be a boolean")
    mixture_cursor = _state_non_negative_integer(
        state["mixture_cursor"],
        label="mixture_cursor",
    )

    rng = random.Random()
    rng.setstate(_rng_state_from_json(state["rng_state"]))
    stats = _parse_stats(state["stats"])
    if stats.emitted_batches != global_step:
        raise SFTLoaderStateError(
            "loader state stats emitted_batches must equal global_step"
        )
    if epoch_packed > stats.packed_conversations:
        raise SFTLoaderStateError(
            "loader state epoch packed count exceeds cumulative packed count"
        )
    return _LoaderProgress(
        rng=rng,
        epoch=epoch,
        epoch_seed=epoch_seed,
        epoch_step=epoch_step,
        global_step=global_step,
        epoch_exhausted=epoch_exhausted,
        epoch_packed_conversations=epoch_packed,
        mixture_cursor=mixture_cursor,
        stats=stats,
    )


def _parse_stats(value: object) -> SFTLoaderStats:
    if not isinstance(value, Mapping) or set(value) != _STATS_FIELDS:
        raise SFTLoaderStateError("loader state stats fields are invalid")
    normalized = {
        field: _state_non_negative_integer(value[field], label=f"stats.{field}")
        for field in _STATS_FIELDS
    }
    return SFTLoaderStats(**normalized)


__all__ = [
    "SFT_CROP_POLICY",
    "SFT_LOADER_STATE_FORMAT",
    "SFT_LOADER_STATE_VERSION",
    "FiniteConversationSource",
    "InMemoryConversationSource",
    "SFTBatchInfo",
    "SFTConversationLoader",
    "SFTLoaderError",
    "SFTLoaderStateError",
    "SFTLoaderStats",
    "WeightedConversationSource",
    "build_fresh_sft_validation_loader",
    "load_jsonl_conversation_source",
]
