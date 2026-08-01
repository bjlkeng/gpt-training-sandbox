"""Immutable, versioned conversation schema and strict JSONL loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from os import PathLike
from pathlib import Path
from typing import ClassVar, Final, Literal, TypeAlias


CHAT_SCHEMA_VERSION: Final = 1
_CONVERSATION_FIELDS: Final = frozenset({"schema_version", "messages"})
_MESSAGE_FIELDS: Final = frozenset({"role", "content"})
_PART_FIELDS: Final = frozenset({"type", "text"})
_PART_TYPES: Final = frozenset({"text", "python", "python_output"})


class ConversationValidationError(ValueError):
    """A conversation record or JSONL line violates the chat schema."""


def _require_text(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise ConversationValidationError(
            f"{path} must be a string, got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True, slots=True)
class TextPart:
    """Assistant-authored ordinary text."""

    type: ClassVar[Literal["text"]] = "text"
    text: str

    def __post_init__(self) -> None:
        _require_text(self.text, path="text part text")


@dataclass(frozen=True, slots=True)
class PythonPart:
    """Assistant-authored Python tool-call text."""

    type: ClassVar[Literal["python"]] = "python"
    text: str

    def __post_init__(self) -> None:
        _require_text(self.text, path="python part text")


@dataclass(frozen=True, slots=True)
class PythonOutputPart:
    """Environment-authored output returned from the Python tool."""

    type: ClassVar[Literal["python_output"]] = "python_output"
    text: str

    def __post_init__(self) -> None:
        _require_text(self.text, path="python_output part text")


AssistantPart: TypeAlias = TextPart | PythonPart | PythonOutputPart
AssistantContent: TypeAlias = str | tuple[AssistantPart, ...]


@dataclass(frozen=True, slots=True)
class SystemMessage:
    """Optional leading system instructions."""

    role: ClassVar[Literal["system"]] = "system"
    content: str

    def __post_init__(self) -> None:
        _require_text(self.content, path="system message content")


@dataclass(frozen=True, slots=True)
class UserMessage:
    """One user-authored string message."""

    role: ClassVar[Literal["user"]] = "user"
    content: str

    def __post_init__(self) -> None:
        _require_text(self.content, path="user message content")


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """One assistant message, optionally split around Python tool I/O."""

    role: ClassVar[Literal["assistant"]] = "assistant"
    content: AssistantContent

    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            return
        if not isinstance(self.content, tuple):
            raise ConversationValidationError(
                "assistant message content must be a string or tuple of parts"
            )
        if not self.content:
            raise ConversationValidationError(
                "assistant message content parts must be non-empty"
            )
        if not all(
            isinstance(part, (TextPart, PythonPart, PythonOutputPart))
            for part in self.content
        ):
            raise ConversationValidationError(
                "assistant message content contains an unsupported part"
            )


Message: TypeAlias = SystemMessage | UserMessage | AssistantMessage


@dataclass(frozen=True, slots=True)
class Conversation:
    """A validated chat conversation with one stable schema version."""

    messages: tuple[Message, ...]
    schema_version: int = CHAT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != CHAT_SCHEMA_VERSION
        ):
            raise ConversationValidationError(
                f"schema_version must equal {CHAT_SCHEMA_VERSION}, "
                f"got {self.schema_version!r}"
            )
        if not isinstance(self.messages, tuple):
            raise ConversationValidationError("messages must be an immutable tuple")
        _validate_message_sequence(self.messages)


def _validate_message_sequence(messages: tuple[Message, ...]) -> None:
    if not messages:
        raise ConversationValidationError("messages must be non-empty")
    if not all(
        isinstance(message, (SystemMessage, UserMessage, AssistantMessage))
        for message in messages
    ):
        raise ConversationValidationError("messages contain an unsupported value")

    offset = 0
    if isinstance(messages[0], SystemMessage):
        if len(messages) == 1 or not isinstance(messages[1], UserMessage):
            raise ConversationValidationError(
                "system message must be followed by a user message"
            )
        offset = 1

    alternating = messages[offset:]
    for logical_index, message in enumerate(alternating):
        expected_role = "user" if logical_index % 2 == 0 else "assistant"
        if message.role != expected_role:
            absolute_index = logical_index + offset
            raise ConversationValidationError(
                f"messages[{absolute_index}].role must be {expected_role!r}, "
                f"got {message.role!r}"
            )


def parse_conversation(record: object) -> Conversation:
    """Validate a JSON-shaped record and copy it into immutable domain values."""

    if isinstance(record, Conversation):
        return record
    obj = _require_mapping(record, path="record")
    _require_exact_fields(
        obj,
        _CONVERSATION_FIELDS,
        path="record",
        schema_label=f"conversation schema version {CHAT_SCHEMA_VERSION}",
    )

    version = obj["schema_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CHAT_SCHEMA_VERSION
    ):
        raise ConversationValidationError(
            f"schema_version must equal {CHAT_SCHEMA_VERSION}, got {version!r}"
        )

    raw_messages = obj["messages"]
    if not isinstance(raw_messages, list):
        raise ConversationValidationError(
            f"messages must be a list, got {type(raw_messages).__name__}"
        )
    if not raw_messages:
        raise ConversationValidationError("messages must be non-empty")

    messages = tuple(
        _parse_message(raw_message, index=index)
        for index, raw_message in enumerate(raw_messages)
    )
    return Conversation(messages=messages, schema_version=version)


def _parse_message(raw_message: object, *, index: int) -> Message:
    path = f"messages[{index}]"
    message = _require_mapping(raw_message, path=path)
    _require_exact_fields(message, _MESSAGE_FIELDS, path=path, schema_label="message")
    role = message["role"]
    if not isinstance(role, str):
        raise ConversationValidationError(
            f"{path}.role must be a string, got {type(role).__name__}"
        )

    content = message["content"]
    if role == "system":
        return SystemMessage(content=_require_text(content, path=f"{path}.content"))
    if role == "user":
        return UserMessage(content=_require_text(content, path=f"{path}.content"))
    if role == "assistant":
        return AssistantMessage(content=_parse_assistant_content(content, path=path))
    raise ConversationValidationError(
        f"{path}.role must be 'system', 'user', or 'assistant', got {role!r}"
    )


def _parse_assistant_content(content: object, *, path: str) -> AssistantContent:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ConversationValidationError(
            f"{path}.content must be a string or list of parts, "
            f"got {type(content).__name__}"
        )
    if not content:
        raise ConversationValidationError(f"{path}.content must be a non-empty list")
    return tuple(
        _parse_assistant_part(raw_part, path=f"{path}.content[{part_index}]")
        for part_index, raw_part in enumerate(content)
    )


def _parse_assistant_part(raw_part: object, *, path: str) -> AssistantPart:
    part = _require_mapping(raw_part, path=path)
    _require_exact_fields(part, _PART_FIELDS, path=path, schema_label="assistant part")
    part_type = part["type"]
    if not isinstance(part_type, str) or part_type not in _PART_TYPES:
        raise ConversationValidationError(
            f"{path}.type must be 'text', 'python', or 'python_output', "
            f"got {part_type!r}"
        )
    text = _require_text(part["text"], path=f"{path}.text")
    if part_type == "text":
        return TextPart(text=text)
    if part_type == "python":
        return PythonPart(text=text)
    return PythonOutputPart(text=text)


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConversationValidationError(
            f"{path} must be an object, got {type(value).__name__}"
        )
    if not all(isinstance(key, str) for key in value):
        raise ConversationValidationError(f"{path} keys must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    path: str,
    schema_label: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise ConversationValidationError(
        f"{path} fields do not match {schema_label}; "
        f"missing={missing}, unexpected={unexpected}"
    )


def read_conversations(path: str | PathLike[str]) -> tuple[Conversation, ...]:
    """Read strict UTF-8 JSONL and return an immutable conversation collection."""

    source = Path(path)
    conversations: list[Conversation] = []
    with source.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ConversationValidationError(
                    f"{source}:{line_number}: input is not valid UTF-8"
                ) from error
            if not line.strip():
                raise ConversationValidationError(
                    f"{source}:{line_number}: line must contain one JSON object"
                )
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_non_standard_number,
                )
                conversations.append(parse_conversation(record))
            except json.JSONDecodeError as error:
                raise ConversationValidationError(
                    f"{source}:{line_number}: invalid JSON at column {error.colno}: "
                    f"{error.msg}"
                ) from error
            except ConversationValidationError as error:
                raise ConversationValidationError(
                    f"{source}:{line_number}: {error}"
                ) from error

    if not conversations:
        raise ConversationValidationError(f"{source}: file contains no conversations")
    return tuple(conversations)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConversationValidationError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_non_standard_number(value: str) -> object:
    raise ConversationValidationError(f"JSON contains non-standard number {value!r}")


__all__ = [
    "CHAT_SCHEMA_VERSION",
    "AssistantContent",
    "AssistantMessage",
    "AssistantPart",
    "Conversation",
    "ConversationValidationError",
    "Message",
    "PythonOutputPart",
    "PythonPart",
    "SystemMessage",
    "TextPart",
    "UserMessage",
    "parse_conversation",
    "read_conversations",
]
