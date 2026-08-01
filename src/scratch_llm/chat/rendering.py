"""Exact chat-template rendering and assistant-only target construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from scratch_llm.chat.conversation import (
    AssistantMessage,
    Conversation,
    Message,
    PythonOutputPart,
    PythonPart,
    SystemMessage,
    TextPart,
    UserMessage,
    parse_conversation,
)
from scratch_llm.tokenization.tokenizer import Tokenizer


CHAT_RENDERER_ID: Final = "scratch_llm_chat_renderer_v1"
IGNORE_INDEX: Final = -1
_SPECIAL_TOKENS: Final = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)
ConversationInput: TypeAlias = Conversation | Mapping[str, object]


class ChatRenderingError(ValueError):
    """A valid conversation cannot be used at the requested render boundary."""


@dataclass(frozen=True, slots=True)
class RenderedConversation:
    """Immutable token IDs aligned one-for-one with assistant loss flags."""

    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    renderer_id: str = CHAT_RENDERER_ID

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.loss_mask):
            raise ChatRenderingError("token and mask lengths must match")
        if not self.token_ids:
            raise ChatRenderingError("rendered conversation must not be empty")
        if not all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token_id in self.token_ids
        ):
            raise ChatRenderingError("rendered token IDs must be non-negative integers")
        if not all(isinstance(value, bool) for value in self.loss_mask):
            raise ChatRenderingError("loss mask values must be booleans")
        if self.renderer_id != CHAT_RENDERER_ID:
            raise ChatRenderingError(f"renderer_id must equal {CHAT_RENDERER_ID!r}")


@dataclass(frozen=True, slots=True)
class CompletionPrompt:
    """Immutable prompt ending immediately after ``assistant_start``."""

    token_ids: tuple[int, ...]
    renderer_id: str = CHAT_RENDERER_ID

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ChatRenderingError("completion prompt must not be empty")
        if not all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token_id in self.token_ids
        ):
            raise ChatRenderingError(
                "completion prompt IDs must be non-negative integers"
            )
        if self.renderer_id != CHAT_RENDERER_ID:
            raise ChatRenderingError(f"renderer_id must equal {CHAT_RENDERER_ID!r}")


@dataclass(frozen=True, slots=True)
class ShiftedSFTSequence:
    """One immutable next-token input/label sequence."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ChatRenderingError("input and label lengths must match")
        if not self.input_ids:
            raise ChatRenderingError("shifted SFT sequence must not be empty")


def render_conversation(
    conversation: ConversationInput,
    tokenizer: Tokenizer,
) -> RenderedConversation:
    """Render a complete training conversation with exact supervision flags."""

    normalized = parse_conversation(conversation)
    messages = _merge_leading_system(normalized)
    if not isinstance(messages[-1], AssistantMessage):
        raise ChatRenderingError(
            "an SFT conversation must end with an assistant message"
        )
    special = _special_token_ids(tokenizer)
    token_ids, loss_mask = _render_messages(messages, tokenizer, special)
    rendered = RenderedConversation(tuple(token_ids), tuple(loss_mask))
    if not any(rendered.loss_mask[1:]):
        raise ChatRenderingError(
            "rendered conversation has no supervised next-token target"
        )
    return rendered


def render_completion_prompt(
    conversation: ConversationInput,
    tokenizer: Tokenizer,
) -> CompletionPrompt:
    """Render history ending in a user message and append one assistant start."""

    normalized = parse_conversation(conversation)
    messages = _merge_leading_system(normalized)
    if not isinstance(messages[-1], UserMessage):
        raise ChatRenderingError("completion conversation must end with a user message")
    special = _special_token_ids(tokenizer)
    token_ids, _ = _render_messages(messages, tokenizer, special)
    token_ids.append(special["<|assistant_start|>"])
    return CompletionPrompt(tuple(token_ids))


def shift_sft_targets(
    token_ids: Sequence[int],
    loss_mask: Sequence[bool],
) -> ShiftedSFTSequence:
    """Shift a fully assembled row once and mask ignored next-token labels."""

    immutable_ids = tuple(token_ids)
    immutable_mask = tuple(loss_mask)
    if len(immutable_ids) != len(immutable_mask):
        raise ChatRenderingError("token and mask lengths must match")
    if len(immutable_ids) < 2:
        raise ChatRenderingError("SFT shifting requires at least two tokens")
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
        for token_id in immutable_ids
    ):
        raise ChatRenderingError("token IDs must be non-negative integers")
    if not all(isinstance(value, bool) for value in immutable_mask):
        raise ChatRenderingError("loss mask values must be booleans")

    labels = tuple(
        token_id if supervised else IGNORE_INDEX
        for token_id, supervised in zip(
            immutable_ids[1:], immutable_mask[1:], strict=True
        )
    )
    if all(label == IGNORE_INDEX for label in labels):
        raise ChatRenderingError(
            "SFT sequence must retain at least one supervised next-token target"
        )
    return ShiftedSFTSequence(input_ids=immutable_ids[:-1], labels=labels)


def _merge_leading_system(conversation: Conversation) -> tuple[Message, ...]:
    messages = conversation.messages
    if not isinstance(messages[0], SystemMessage):
        return messages
    first_user = messages[1]
    if not isinstance(first_user, UserMessage):
        raise AssertionError("validated system messages are followed by users")
    merged_user = UserMessage(content=f"{messages[0].content}\n\n{first_user.content}")
    return (merged_user, *messages[2:])


def _special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    supported = tokenizer.get_special_tokens()
    missing = sorted(set(_SPECIAL_TOKENS) - supported)
    if missing:
        raise ChatRenderingError(
            f"tokenizer is missing required chat special tokens: {missing}"
        )
    return {token: tokenizer.encode_special(token) for token in _SPECIAL_TOKENS}


def _render_messages(
    messages: tuple[Message, ...],
    tokenizer: Tokenizer,
    special: Mapping[str, int],
) -> tuple[list[int], list[bool]]:
    token_ids: list[int] = []
    loss_mask: list[bool] = []

    def add(values: int | Sequence[int], *, supervised: bool) -> None:
        normalized = (values,) if isinstance(values, int) else tuple(values)
        token_ids.extend(normalized)
        loss_mask.extend([supervised] * len(normalized))

    add(special["<|bos|>"], supervised=False)
    for message in messages:
        if isinstance(message, UserMessage):
            add(special["<|user_start|>"], supervised=False)
            add(tokenizer.encode(message.content), supervised=False)
            add(special["<|user_end|>"], supervised=False)
            continue
        if not isinstance(message, AssistantMessage):
            raise AssertionError("system messages must be merged before rendering")
        add(special["<|assistant_start|>"], supervised=False)
        if isinstance(message.content, str):
            add(tokenizer.encode(message.content), supervised=True)
        else:
            for part in message.content:
                if isinstance(part, TextPart):
                    add(tokenizer.encode(part.text), supervised=True)
                elif isinstance(part, PythonPart):
                    add(special["<|python_start|>"], supervised=True)
                    add(tokenizer.encode(part.text), supervised=True)
                    add(special["<|python_end|>"], supervised=True)
                elif isinstance(part, PythonOutputPart):
                    add(special["<|output_start|>"], supervised=False)
                    add(tokenizer.encode(part.text), supervised=False)
                    add(special["<|output_end|>"], supervised=False)
                else:
                    raise AssertionError("validated assistant part is unsupported")
        add(special["<|assistant_end|>"], supervised=True)
    return token_ids, loss_mask


__all__ = [
    "CHAT_RENDERER_ID",
    "IGNORE_INDEX",
    "ChatRenderingError",
    "CompletionPrompt",
    "RenderedConversation",
    "ShiftedSFTSequence",
    "render_completion_prompt",
    "render_conversation",
    "shift_sft_targets",
]
