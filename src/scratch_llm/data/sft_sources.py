"""Deterministic normalization and bounded parquet views for SFT sources."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import random
from typing import Final

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scratch_llm._validation import (
    require_integer,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.chat.conversation import (
    CHAT_SCHEMA_VERSION,
    AssistantMessage,
    Conversation,
    ConversationValidationError,
    PythonOutputPart,
    PythonPart,
    SystemMessage,
    TextPart,
    UserMessage,
    parse_conversation,
)
from scratch_llm.data.hub import CachedHubParquetDataset, HubDatasetSpec


_SHARED_SEED_MAX: Final = 2**32 - 1
_MMLU_LETTERS: Final = ("A", "B", "C", "D")
NANOCHAT_SFT_REFERENCE_COMMIT: Final = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
_SUPPORTED_SPLITS: Final = {
    "gsm8k": ("train", "test"),
    "mmlu": ("auxiliary_train", "test"),
    "smoltalk": ("train", "test"),
}
_SPEC_FACTORIES: Final = {
    "gsm8k": {
        "repository": "openai/gsm8k",
        "subset": "main",
        "adapter_version": "gsm8k_v1",
        "required_columns": ("question", "answer"),
    },
    "mmlu": {
        "repository": "cais/mmlu",
        "subset": "all",
        "adapter_version": "mmlu_v1",
        "required_columns": ("question", "choices", "answer"),
    },
    "smoltalk": {
        "repository": "HuggingFaceTB/smol-smoltalk",
        "subset": "default",
        "adapter_version": "smoltalk_v1",
        "required_columns": ("messages",),
    },
}


class SFTDatasetError(ValueError):
    """A requested SFT source or deterministic view is invalid."""


class SFTDatasetRowError(SFTDatasetError):
    """One source row cannot be normalized into the conversation schema."""


@dataclass(frozen=True, slots=True)
class SFTConversationExample:
    """A normalized conversation with stable source and row identities."""

    conversation: Conversation
    source_identity: str
    source_row: int
    identity: str


def get_sft_dataset_spec(dataset: str, split: str) -> HubDatasetSpec:
    """Return one supported nanochat-aligned Hub source contract."""

    if not isinstance(dataset, str):
        raise SFTDatasetError("dataset must be a string")
    if dataset not in _SUPPORTED_SPLITS:
        supported = ", ".join(sorted(_SUPPORTED_SPLITS))
        raise SFTDatasetError(f"supported datasets are: {supported}")
    if not isinstance(split, str) or split not in _SUPPORTED_SPLITS[dataset]:
        supported = ", ".join(_SUPPORTED_SPLITS[dataset])
        raise SFTDatasetError(f"supported splits for {dataset} are: {supported}")
    fields = _SPEC_FACTORIES[dataset]
    return HubDatasetSpec(
        dataset=dataset,
        repository=fields["repository"],  # type: ignore[arg-type]
        subset=fields["subset"],  # type: ignore[arg-type]
        split=split,
        adapter_version=fields["adapter_version"],  # type: ignore[arg-type]
        reference_commit=NANOCHAT_SFT_REFERENCE_COMMIT,
        required_columns=fields["required_columns"],  # type: ignore[arg-type]
    )


def normalize_smoltalk_row(
    row: Mapping[str, object],
    *,
    context: str,
) -> Conversation:
    """Preserve one strict SmolTalk string conversation exactly."""

    try:
        source = _require_row(row)
        raw_messages = source.get("messages")
        if not isinstance(raw_messages, list):
            raise SFTDatasetRowError("messages must be a list")
        if not raw_messages:
            raise SFTDatasetRowError("messages must be non-empty")
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise SFTDatasetRowError(f"messages[{index}] must be an object")
            content = raw_message.get("content")
            if not isinstance(content, str):
                raise SFTDatasetRowError(f"messages[{index}].content must be a string")
        conversation = parse_conversation(
            {
                "messages": raw_messages,
                "schema_version": CHAT_SCHEMA_VERSION,
            }
        )
        non_system_messages = (
            conversation.messages[1:]
            if isinstance(conversation.messages[0], SystemMessage)
            else conversation.messages
        )
        if len(non_system_messages) < 2 or not isinstance(
            non_system_messages[-1], AssistantMessage
        ):
            raise SFTDatasetRowError(
                "SmolTalk conversations must contain at least one complete "
                "user/assistant turn"
            )
        return conversation
    except (ConversationValidationError, SFTDatasetRowError) as error:
        raise _with_context(context, error) from error


def normalize_mmlu_row(
    row: Mapping[str, object],
    *,
    context: str,
) -> Conversation:
    """Render one four-choice MMLU row with letter-after-choice binding."""

    try:
        source = _require_row(row)
        question = _require_non_empty_string(source.get("question"), label="question")
        raw_choices = source.get("choices")
        if not isinstance(raw_choices, (list, tuple)) or len(raw_choices) != 4:
            raise SFTDatasetRowError("choices must contain exactly four choices")
        choices = tuple(
            _require_string(choice, label=f"choices[{index}]")
            for index, choice in enumerate(raw_choices)
        )
        raw_answer = source.get("answer")
        if (
            not isinstance(raw_answer, int)
            or isinstance(raw_answer, bool)
            or not 0 <= raw_answer < len(_MMLU_LETTERS)
        ):
            raise SFTDatasetRowError("answer must be an integer in [0, 3]")

        prompt = f"Multiple Choice question: {question}\n"
        prompt += "".join(
            f"- {choice}={letter}\n"
            for letter, choice in zip(_MMLU_LETTERS, choices, strict=True)
        )
        prompt += "\nRespond only with the letter of the correct answer."
        return Conversation(
            messages=(
                UserMessage(content=prompt),
                AssistantMessage(content=_MMLU_LETTERS[raw_answer]),
            )
        )
    except SFTDatasetRowError as error:
        raise _with_context(context, error) from error


def normalize_gsm8k_row(
    row: Mapping[str, object],
    *,
    context: str,
) -> Conversation:
    """Normalize one GSM8K solution and preserve calculator calls as tool parts."""

    try:
        source = _require_row(row)
        question = _require_non_empty_string(source.get("question"), label="question")
        answer = _require_non_empty_string(source.get("answer"), label="answer")
        parts = parse_gsm8k_answer_parts(answer, context=context)
        return Conversation(
            messages=(
                UserMessage(content=question),
                AssistantMessage(content=parts),
            )
        )
    except SFTDatasetRowError as error:
        if str(error).startswith(f"{context}:"):
            raise
        raise _with_context(context, error) from error


def parse_gsm8k_answer_parts(
    answer: str,
    *,
    context: str,
) -> tuple[TextPart | PythonPart | PythonOutputPart, ...]:
    """Split strict ``<<expression=result>>`` markers without losing text."""

    try:
        _require_string(answer, label="answer")
        parts: list[TextPart | PythonPart | PythonOutputPart] = []
        cursor = 0
        while True:
            start = answer.find("<<", cursor)
            stray_end = answer.find(">>", cursor)
            if start < 0:
                if stray_end >= 0:
                    raise SFTDatasetRowError(
                        "malformed calculator marker has a closing delimiter "
                        "without an opening delimiter"
                    )
                parts.append(TextPart(text=answer[cursor:]))
                break
            if 0 <= stray_end < start:
                raise SFTDatasetRowError(
                    "malformed calculator marker has a closing delimiter "
                    "before its opening delimiter"
                )
            parts.append(TextPart(text=answer[cursor:start]))
            end = answer.find(">>", start + 2)
            if end < 0:
                raise SFTDatasetRowError(
                    "malformed calculator marker is missing a closing delimiter"
                )
            inner = answer[start + 2 : end]
            if "<" in inner or ">" in inner or "=" not in inner:
                raise SFTDatasetRowError(
                    "malformed calculator marker must be <<expression=result>>"
                )
            expression, result = inner.rsplit("=", 1)
            if not expression.strip() or not result.strip():
                raise SFTDatasetRowError(
                    "malformed calculator marker requires non-empty expression "
                    "and result"
                )
            parts.append(PythonPart(text=expression))
            parts.append(PythonOutputPart(text=result))
            cursor = end + 2
        return tuple(parts)
    except SFTDatasetRowError as error:
        raise _with_context(context, error) from error


class SFTConversationDataset:
    """Finite, repeatable, bounded-memory conversation view over cached parquet."""

    def __init__(
        self,
        cache: CachedHubParquetDataset,
        *,
        shuffle_buffer_size: int = 1024,
        row_batch_size: int = 1024,
        shuffle: bool = True,
    ) -> None:
        if not isinstance(cache, CachedHubParquetDataset):
            raise TypeError("cache must be a CachedHubParquetDataset")
        try:
            self.shuffle_buffer_size = require_positive_integer(
                shuffle_buffer_size,
                name="shuffle_buffer_size",
            )
            self.row_batch_size = require_positive_integer(
                row_batch_size,
                name="row_batch_size",
            )
        except (TypeError, ValueError) as error:
            raise SFTDatasetError(str(error)) from error
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean")
        self.cache = cache
        self.shuffle = shuffle
        self.source_identity = _canonical_identity(
            {
                "cache_source_identity": cache.source_identity,
                "format": "scratch_llm_sft_parquet_view_v1",
                "row_batch_size": self.row_batch_size,
                "shuffle": self.shuffle,
                "shuffle_buffer_size": self.shuffle_buffer_size,
            }
        )

    def __len__(self) -> int:
        return self.cache.row_count

    def iter_examples(
        self,
        *,
        seed: int = 42,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[SFTConversationExample]:
        """Yield a deterministic seeded logical slice without eager row loading."""

        seed = _shared_seed(seed)
        try:
            start = require_non_negative_integer(start, name="start")
            step = require_positive_integer(step, name="step")
            if stop is not None:
                stop = require_non_negative_integer(stop, name="stop")
        except (TypeError, ValueError) as error:
            raise SFTDatasetError(str(error)) from error
        if stop is not None and stop < start:
            raise SFTDatasetError(
                f"stop must be greater than or equal to start, got {stop} < {start}"
            )

        ordered_rows = self._iter_seeded_rows(seed)
        bounded_rows = ordered_rows if stop is None else islice(ordered_rows, stop)
        for logical_index, (source_row, row) in enumerate(bounded_rows):
            if logical_index < start or (logical_index - start) % step:
                continue
            context = (
                f"{self.cache.spec.repository}/{self.cache.spec.subset}/"
                f"{self.cache.spec.split} row {source_row}"
            )
            conversation = _normalize_row(
                self.cache.spec.dataset,
                row,
                context=context,
            )
            yield SFTConversationExample(
                conversation=conversation,
                source_identity=self.source_identity,
                source_row=source_row,
                identity=_example_identity(
                    self.source_identity,
                    source_row,
                    conversation,
                ),
            )

    def iter_conversations(
        self,
        *,
        seed: int = 42,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[Conversation]:
        """Yield just conversations for consumers that do not need row metadata."""

        for example in self.iter_examples(
            seed=seed,
            start=start,
            stop=stop,
            step=step,
        ):
            yield example.conversation

    def _iter_seeded_rows(
        self,
        seed: int,
    ) -> Iterator[tuple[int, Mapping[str, object]]]:
        if not self.shuffle:
            yield from self._iter_indexed_rows()
            return
        rows = iter(self._iter_indexed_rows())
        buffer: list[tuple[int, Mapping[str, object]]] = []
        for _ in range(min(self.shuffle_buffer_size, len(self))):
            try:
                buffer.append(next(rows))
            except StopIteration:
                break
        rng = random.Random(seed)
        while buffer:
            try:
                incoming = next(rows)
            except StopIteration:
                rng.shuffle(buffer)
                yield from buffer
                return
            selected = rng.randrange(len(buffer))
            outgoing = buffer[selected]
            buffer[selected] = incoming
            yield outgoing

    def _iter_indexed_rows(
        self,
    ) -> Iterator[tuple[int, Mapping[str, object]]]:
        row_index = 0
        for shard_path in self.cache.shard_paths:
            try:
                parquet = pq.ParquetFile(shard_path)
                batches = parquet.iter_batches(
                    batch_size=self.row_batch_size,
                    columns=list(self.cache.spec.required_columns),
                )
                for batch in batches:
                    for row in batch.to_pylist():
                        if not isinstance(row, Mapping):
                            raise SFTDatasetError(
                                f"parquet row {row_index} is not an object"
                            )
                        yield row_index, row
                        row_index += 1
            except pa.ArrowException as error:
                raise SFTDatasetError(
                    f"could not stream cached parquet {shard_path}: {error}"
                ) from error
        if row_index != len(self):
            raise SFTDatasetError(
                f"streamed {row_index} rows but manifest records {len(self)}"
            )


def preview_examples_identity(examples: tuple[SFTConversationExample, ...]) -> str:
    """Return one canonical identity for a deterministic bounded preview."""

    if not isinstance(examples, tuple) or not all(
        isinstance(example, SFTConversationExample) for example in examples
    ):
        raise TypeError("examples must be a tuple of SFTConversationExample values")
    encoded = json.dumps(
        [example.identity for example in examples],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _canonical_identity(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalize_row(
    dataset: str,
    row: Mapping[str, object],
    *,
    context: str,
) -> Conversation:
    if dataset == "smoltalk":
        return normalize_smoltalk_row(row, context=context)
    if dataset == "mmlu":
        return normalize_mmlu_row(row, context=context)
    if dataset == "gsm8k":
        return normalize_gsm8k_row(row, context=context)
    raise AssertionError(f"unsupported validated dataset {dataset!r}")


def _example_identity(
    source_identity: str,
    source_row: int,
    conversation: Conversation,
) -> str:
    payload = {
        "conversation": _conversation_record(conversation),
        "source_identity": source_identity,
        "source_row": source_row,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _conversation_record(conversation: Conversation) -> dict[str, object]:
    messages: list[dict[str, object]] = []
    for message in conversation.messages:
        if isinstance(message, (SystemMessage, UserMessage)):
            content: object = message.content
        elif isinstance(message.content, str):
            content = message.content
        else:
            content = [
                {"text": part.text, "type": part.type} for part in message.content
            ]
        messages.append({"content": content, "role": message.role})
    return {
        "messages": messages,
        "schema_version": conversation.schema_version,
    }


def _require_row(row: object) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        raise SFTDatasetRowError(f"row must be an object, got {type(row).__name__}")
    return row


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise SFTDatasetRowError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    return value


def _require_non_empty_string(value: object, *, label: str) -> str:
    text = _require_string(value, label=label)
    if not text.strip():
        raise SFTDatasetRowError(f"{label} must be non-empty")
    return text


def _with_context(context: str, error: Exception) -> SFTDatasetRowError:
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    return SFTDatasetRowError(f"{context}: {error}")


def _shared_seed(value: object) -> int:
    try:
        seed = require_integer(value, name="seed")
    except TypeError as error:
        raise SFTDatasetError(str(error)) from error
    if not 0 <= seed <= _SHARED_SEED_MAX:
        raise SFTDatasetError(
            f"seed must be in range [0, {_SHARED_SEED_MAX}], got {seed}"
        )
    return seed


__all__ = [
    "NANOCHAT_SFT_REFERENCE_COMMIT",
    "SFTConversationDataset",
    "SFTConversationExample",
    "SFTDatasetError",
    "SFTDatasetRowError",
    "get_sft_dataset_spec",
    "normalize_gsm8k_row",
    "normalize_mmlu_row",
    "normalize_smoltalk_row",
    "parse_gsm8k_answer_parts",
    "preview_examples_identity",
]
