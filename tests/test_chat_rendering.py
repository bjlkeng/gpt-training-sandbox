"""Exact chat rendering and assistant-only supervision tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scratch_llm.chat.conversation import (
    AssistantMessage,
    Conversation,
    PythonOutputPart,
    PythonPart,
    SystemMessage,
    TextPart,
    UserMessage,
)
from scratch_llm.chat.rendering import (
    CHAT_RENDERER_ID,
    IGNORE_INDEX,
    ChatRenderingError,
    render_completion_prompt,
    render_conversation,
    shift_sft_targets,
)
from scratch_llm.tokenization.bpe import RegexBPETokenizer, train_reference_bpe
from scratch_llm.tokenization.tokenizer import ByteTokenizer, Tokenizer


class _FailIfEncodedTokenizer(ByteTokenizer):
    def encode(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]:
        raise AssertionError("invalid conversations must fail before tokenization")


def _conversation() -> Conversation:
    return Conversation(
        messages=(
            UserMessage(content="Hi"),
            AssistantMessage(
                content=(
                    TextPart(text="A"),
                    PythonPart(text="1+1"),
                    PythonOutputPart(text="2"),
                    TextPart(text="B"),
                )
            ),
        )
    )


def _expected_ids(tokenizer: Tokenizer) -> tuple[int, ...]:
    special = tokenizer.encode_special
    return (
        special("<|bos|>"),
        special("<|user_start|>"),
        *tokenizer.encode("Hi"),
        special("<|user_end|>"),
        special("<|assistant_start|>"),
        *tokenizer.encode("A"),
        special("<|python_start|>"),
        *tokenizer.encode("1+1"),
        special("<|python_end|>"),
        special("<|output_start|>"),
        *tokenizer.encode("2"),
        special("<|output_end|>"),
        *tokenizer.encode("B"),
        special("<|assistant_end|>"),
    )


def _expected_mask(tokenizer: Tokenizer) -> tuple[bool, ...]:
    return (
        False,
        False,
        *(False for _ in tokenizer.encode("Hi")),
        False,
        False,
        *(True for _ in tokenizer.encode("A")),
        True,
        *(True for _ in tokenizer.encode("1+1")),
        True,
        False,
        *(False for _ in tokenizer.encode("2")),
        False,
        *(True for _ in tokenizer.encode("B")),
        True,
    )


def test_byte_renderer_places_every_delimiter_once_with_exact_mask() -> None:
    tokenizer = ByteTokenizer()

    rendered = render_conversation(_conversation(), tokenizer)

    assert rendered.renderer_id == CHAT_RENDERER_ID
    assert rendered.token_ids == (
        256,
        257,
        72,
        105,
        258,
        259,
        65,
        261,
        49,
        43,
        49,
        262,
        263,
        50,
        264,
        66,
        260,
    )
    assert rendered.loss_mask == (
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    )
    assert len(rendered.token_ids) == len(rendered.loss_mask)


def test_saved_regex_bpe_renderer_matches_exact_fixture(tmp_path: Path) -> None:
    trained = RegexBPETokenizer(train_reference_bpe(("chat fixture",), vocab_size=265))
    artifact_dir = tmp_path / "tokenizer"
    trained.save(artifact_dir)
    tokenizer = RegexBPETokenizer.load(artifact_dir)

    rendered = render_conversation(_conversation(), tokenizer)

    assert rendered.token_ids == _expected_ids(tokenizer)
    assert rendered.loss_mask == _expected_mask(tokenizer)
    assert rendered.token_ids == (
        256,
        257,
        72,
        105,
        258,
        259,
        65,
        261,
        49,
        43,
        49,
        262,
        263,
        50,
        264,
        66,
        260,
    )


def test_system_message_is_merged_with_exact_separator_without_mutation() -> None:
    conversation = Conversation(
        messages=(
            SystemMessage(content="Be terse."),
            UserMessage(content="Why?"),
            AssistantMessage(content="Because."),
        )
    )
    tokenizer = ByteTokenizer()

    rendered = render_conversation(conversation, tokenizer)

    assert conversation.messages[0] == SystemMessage(content="Be terse.")
    assert conversation.messages[1] == UserMessage(content="Why?")
    expected_user = tokenizer.encode(
        "Be terse.\n\nWhy?",
        prepend="<|user_start|>",
        append="<|user_end|>",
    )
    assert rendered.token_ids[1 : 1 + len(expected_user)] == tuple(expected_user)
    assert tokenizer.decode(rendered.token_ids).count("\n\n") == 1


def test_multi_turn_rendering_preserves_order_and_masks_each_assistant() -> None:
    tokenizer = ByteTokenizer()
    conversation = Conversation(
        messages=(
            UserMessage(content="u1"),
            AssistantMessage(content="a1"),
            UserMessage(content="u2"),
            AssistantMessage(content="a2"),
        )
    )

    rendered = render_conversation(conversation, tokenizer)

    assert tokenizer.decode(rendered.token_ids) == (
        "<|bos|><|user_start|>u1<|user_end|>"
        "<|assistant_start|>a1<|assistant_end|>"
        "<|user_start|>u2<|user_end|>"
        "<|assistant_start|>a2<|assistant_end|>"
    )
    for response in (b"a1", b"a2"):
        start = rendered.token_ids.index(response[0])
        assert rendered.loss_mask[start : start + len(response)] == (True, True)


def test_training_render_preserves_a_trailing_user_turn_as_masked_context() -> None:
    tokenizer = ByteTokenizer()
    conversation = Conversation(
        messages=(
            UserMessage(content="u1"),
            AssistantMessage(content="a1"),
            UserMessage(content="u2"),
        )
    )

    rendered = render_conversation(conversation, tokenizer)

    assert tokenizer.decode(rendered.token_ids) == (
        "<|bos|><|user_start|>u1<|user_end|>"
        "<|assistant_start|>a1<|assistant_end|>"
        "<|user_start|>u2<|user_end|>"
    )
    trailing_start = rendered.token_ids.index(
        tokenizer.encode_special("<|user_start|>"),
        rendered.token_ids.index(tokenizer.encode_special("<|assistant_end|>")) + 1,
    )
    assert not any(rendered.loss_mask[trailing_start:])
    assert any(rendered.loss_mask[:trailing_start])


def test_invalid_raw_conversation_fails_before_tokenizer_use() -> None:
    raw = {
        "schema_version": 1,
        "messages": [{"role": "assistant", "content": "wrong"}],
    }
    with pytest.raises(ValueError, match="must be 'user'"):
        render_conversation(raw, _FailIfEncodedTokenizer())


def test_shifted_labels_follow_next_token_mask_without_mutating_rendering() -> None:
    rendered = render_conversation(_conversation(), ByteTokenizer())
    before_ids = rendered.token_ids
    before_mask = rendered.loss_mask

    shifted = shift_sft_targets(rendered.token_ids, rendered.loss_mask)

    assert shifted.input_ids == rendered.token_ids[:-1]
    assert shifted.labels == tuple(
        token_id if supervised else IGNORE_INDEX
        for token_id, supervised in zip(
            rendered.token_ids[1:], rendered.loss_mask[1:], strict=True
        )
    )
    assert set(label for label in shifted.labels if label < 0) == {IGNORE_INDEX}
    assert any(label != IGNORE_INDEX for label in shifted.labels)
    assert rendered.token_ids is before_ids
    assert rendered.loss_mask is before_mask


@pytest.mark.parametrize(
    ("token_ids", "loss_mask", "match"),
    [
        ((1,), (True,), "at least two tokens"),
        ((1, 2), (True,), "lengths must match"),
        ((1, 2), (True, False), "at least one supervised next-token target"),
    ],
)
def test_shifted_labels_reject_invalid_or_all_ignored_sequences(
    token_ids: tuple[int, ...],
    loss_mask: tuple[bool, ...],
    match: str,
) -> None:
    with pytest.raises(ChatRenderingError, match=match):
        shift_sft_targets(token_ids, loss_mask)


def test_completion_prompt_preserves_history_and_ends_at_one_assistant_start() -> None:
    tokenizer = ByteTokenizer()
    conversation = Conversation(
        messages=(
            SystemMessage(content="Be brief."),
            UserMessage(content="First?"),
            AssistantMessage(content="One."),
            UserMessage(content="Second?"),
        )
    )

    prompt = render_completion_prompt(conversation, tokenizer)

    decoded = tokenizer.decode(prompt.token_ids)
    assert prompt.renderer_id == CHAT_RENDERER_ID
    assert decoded == (
        "<|bos|><|user_start|>Be brief.\n\nFirst?<|user_end|>"
        "<|assistant_start|>One.<|assistant_end|>"
        "<|user_start|>Second?<|user_end|><|assistant_start|>"
    )
    assert prompt.token_ids[-1] == tokenizer.encode_special("<|assistant_start|>")
    assert decoded.count("<|assistant_start|>") == 2
    assert decoded.count("<|assistant_end|>") == 1


def test_completion_prompt_requires_a_final_user_message() -> None:
    with pytest.raises(ChatRenderingError, match="must end with a user message"):
        render_completion_prompt(_conversation(), ByteTokenizer())


def test_bounded_completion_prompt_preserves_exact_fit_and_crops_one_token() -> None:
    tokenizer = ByteTokenizer()
    conversation = Conversation(messages=(UserMessage(content="abcd"),))
    full = render_completion_prompt(conversation, tokenizer)

    exact = render_completion_prompt(
        conversation,
        tokenizer,
        max_token_count=len(full.token_ids),
    )
    cropped = render_completion_prompt(
        conversation,
        tokenizer,
        max_token_count=len(full.token_ids) - 1,
    )

    assert exact.token_ids == full.token_ids
    assert exact.original_token_count == len(full.token_ids)
    assert exact.dropped_turn_count == 0
    assert exact.truncated_user_token_count == 0
    assert cropped.original_token_count == len(full.token_ids)
    assert cropped.dropped_turn_count == 0
    assert cropped.truncated_user_token_count == 1
    assert tokenizer.decode(cropped.token_ids) == (
        "<|bos|><|user_start|>bcd<|user_end|><|assistant_start|>"
    )


def test_bounded_completion_prompt_drops_oldest_complete_turns_first() -> None:
    tokenizer = ByteTokenizer()
    conversation = Conversation(
        messages=(
            UserMessage(content="old"),
            AssistantMessage(content="old answer"),
            UserMessage(content="new"),
            AssistantMessage(content="new answer"),
            UserMessage(content="current"),
        )
    )
    newest_two_turns = Conversation(messages=conversation.messages[2:])
    expected = render_completion_prompt(newest_two_turns, tokenizer)

    bounded = render_completion_prompt(
        conversation,
        tokenizer,
        max_token_count=len(expected.token_ids),
    )

    assert bounded.token_ids == expected.token_ids
    assert bounded.dropped_turn_count == 1
    assert bounded.truncated_user_token_count == 0
    assert bounded.original_token_count > len(bounded.token_ids)


def test_bounded_completion_prompt_only_left_crops_current_user_after_turns() -> None:
    tokenizer = ByteTokenizer()
    empty = render_completion_prompt(
        Conversation(messages=(UserMessage(content=""),)),
        tokenizer,
    )
    conversation = Conversation(
        messages=(
            UserMessage(content="first"),
            AssistantMessage(content="A"),
            UserMessage(content="second"),
            AssistantMessage(content="B"),
            UserMessage(content="abcdefghij"),
        )
    )

    bounded = render_completion_prompt(
        conversation,
        tokenizer,
        max_token_count=len(empty.token_ids) + 3,
    )

    assert bounded.dropped_turn_count == 2
    assert bounded.truncated_user_token_count == 7
    assert tokenizer.decode(bounded.token_ids) == (
        "<|bos|><|user_start|>hij<|user_end|><|assistant_start|>"
    )


def test_bounded_completion_prompt_rejects_impossible_control_budget() -> None:
    tokenizer = ByteTokenizer()
    empty = render_completion_prompt(
        Conversation(messages=(UserMessage(content=""),)),
        tokenizer,
    )

    with pytest.raises(ChatRenderingError, match="fixed chat controls"):
        render_completion_prompt(
            Conversation(messages=(UserMessage(content="x"),)),
            tokenizer,
            max_token_count=len(empty.token_ids) - 1,
        )


def test_fixture_json_is_canonical_utf8_jsonl() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "chat"
    for path in (root / "train.jsonl", root / "validation.jsonl"):
        raw_lines = path.read_bytes().splitlines()
        assert raw_lines
        for raw_line in raw_lines:
            decoded = raw_line.decode("utf-8")
            record = json.loads(decoded)
            assert decoded == json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
