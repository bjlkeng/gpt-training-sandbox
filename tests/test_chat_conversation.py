"""Conversation schema and JSONL reader contracts for supervised finetuning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scratch_llm.chat.conversation import (
    CHAT_SCHEMA_VERSION,
    AssistantMessage,
    Conversation,
    ConversationValidationError,
    PythonOutputPart,
    PythonPart,
    TextPart,
    UserMessage,
    parse_conversation,
    read_conversations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "chat" / "train.jsonl"
VALIDATION_FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "chat" / "validation.jsonl"


def _record(messages: list[dict[str, object]]) -> dict[str, object]:
    return {"schema_version": CHAT_SCHEMA_VERSION, "messages": messages}


def test_tracked_chat_fixtures_cover_supported_shapes() -> None:
    train = read_conversations(TRAIN_FIXTURE)
    validation = read_conversations(VALIDATION_FIXTURE)

    assert len(train) == 5
    assert len(validation) == 2
    assert all(isinstance(conversation, Conversation) for conversation in train)
    assert all(
        conversation.schema_version == CHAT_SCHEMA_VERSION for conversation in train
    )

    all_conversations = train + validation
    assert any(len(conversation.messages) > 2 for conversation in all_conversations)
    assert any(
        conversation.messages[0].role == "system" for conversation in all_conversations
    )
    assert any(
        any(
            isinstance(message, AssistantMessage)
            and isinstance(message.content, tuple)
            and any(isinstance(part, PythonPart) for part in message.content)
            for message in conversation.messages
        )
        for conversation in all_conversations
    )
    assert any("☕" in str(conversation) for conversation in all_conversations)


def test_parse_conversation_builds_immutable_typed_values_without_mutating_input() -> (
    None
):
    raw = _record(
        [
            {"role": "user", "content": "Calculate."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Working: "},
                    {"type": "python", "text": "2 + 3"},
                    {"type": "python_output", "text": "5"},
                    {"type": "text", "text": " done."},
                ],
            },
        ]
    )
    before = json.loads(json.dumps(raw))

    conversation = parse_conversation(raw)

    assert raw == before
    assert conversation == Conversation(
        messages=(
            UserMessage(content="Calculate."),
            AssistantMessage(
                content=(
                    TextPart(text="Working: "),
                    PythonPart(text="2 + 3"),
                    PythonOutputPart(text="5"),
                    TextPart(text=" done."),
                )
            ),
        )
    )
    with pytest.raises(AttributeError):
        conversation.messages = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"schema_version": 999, "messages": []}, "schema_version must equal 1"),
        ({"schema_version": 1, "messages": []}, "messages must be non-empty"),
        (
            _record([{"role": "assistant", "content": "No user."}]),
            r"messages\[0\].role must be 'user'",
        ),
        (
            _record([{"role": "system", "content": "Rules."}]),
            "system message must be followed by a user message",
        ),
        (
            _record(
                [
                    {"role": "user", "content": "One"},
                    {"role": "user", "content": "Two"},
                ]
            ),
            r"messages\[1\].role must be 'assistant'",
        ),
        (
            _record(
                [
                    {"role": "user", "content": []},
                    {"role": "assistant", "content": "No."},
                ]
            ),
            r"messages\[0\].content must be a string",
        ),
        (
            _record(
                [
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": [{"type": "shell", "text": "pwd"}],
                    },
                ]
            ),
            r"messages\[1\].content\[0\].type",
        ),
        (
            _record(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": []},
                ]
            ),
            r"messages\[1\].content must be a non-empty list",
        ),
        (
            {
                "schema_version": 1,
                "messages": [{"role": "user", "content": "Hi"}],
                "extra": True,
            },
            "record fields do not match conversation schema version 1",
        ),
    ],
)
def test_parse_conversation_rejects_invalid_shapes_before_tokenization(
    record: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ConversationValidationError, match=match):
        parse_conversation(record)


def test_reader_reports_line_number_for_json_schema_and_utf8_errors(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "invalid-json.jsonl"
    invalid_json.write_text(
        json.dumps(_record([{"role": "user", "content": "ok"}]))
        + "\n"
        + '{"schema_version":1,"messages":',
        encoding="utf-8",
    )
    with pytest.raises(ConversationValidationError, match=r"invalid-json\.jsonl:2"):
        read_conversations(invalid_json)

    invalid_schema = tmp_path / "invalid-schema.jsonl"
    invalid_schema.write_text(
        json.dumps(_record([{"role": "assistant", "content": "bad"}])),
        encoding="utf-8",
    )
    with pytest.raises(
        ConversationValidationError,
        match=r"invalid-schema\.jsonl:1: messages\[0\]",
    ):
        read_conversations(invalid_schema)

    duplicate_key = tmp_path / "duplicate.jsonl"
    duplicate_key.write_text(
        '{"schema_version":1,"schema_version":1,"messages":[]}',
        encoding="utf-8",
    )
    with pytest.raises(
        ConversationValidationError,
        match=r"duplicate\.jsonl:1:.*duplicate key 'schema_version'",
    ):
        read_conversations(duplicate_key)

    invalid_utf8 = tmp_path / "invalid-utf8.jsonl"
    invalid_utf8.write_bytes(
        b'{"schema_version":1,"messages":[{"role":"user","content":"ok"}]}\n\xff'
    )
    with pytest.raises(
        ConversationValidationError,
        match=r"invalid-utf8\.jsonl:2: input is not valid UTF-8",
    ):
        read_conversations(invalid_utf8)


def test_reader_rejects_blank_lines_and_empty_files(tmp_path: Path) -> None:
    blank_line = tmp_path / "blank.jsonl"
    blank_line.write_text("\n", encoding="utf-8")
    with pytest.raises(
        ConversationValidationError,
        match=r"blank\.jsonl:1: line must contain one JSON object",
    ):
        read_conversations(blank_line)

    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(
        ConversationValidationError,
        match=r"empty\.jsonl: file contains no conversations",
    ):
        read_conversations(empty)
