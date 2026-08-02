"""Deterministic best-fit SFT packing and exact-resume contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scratch_llm.chat.conversation import (
    AssistantMessage,
    Conversation,
    UserMessage,
)
from scratch_llm.chat.loader import (
    SFT_LOADER_STATE_FORMAT,
    SFT_LOADER_STATE_VERSION,
    InMemoryConversationSource,
    SFTConversationLoader,
    SFTLoaderError,
    SFTLoaderStateError,
    WeightedConversationSource,
    build_fresh_sft_validation_loader,
    load_jsonl_conversation_source,
)
from scratch_llm.chat.rendering import IGNORE_INDEX, render_conversation
from scratch_llm.tokenization.tokenizer import ByteTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DifferentIdentityTokenizer(ByteTokenizer):
    def get_identity(self) -> str:
        return "sha256:" + "f" * 64


def _conversation(label: str, token_count: int) -> Conversation:
    # ByteTokenizer rendering has five controls around empty user content and
    # the assistant string, so this gives an exact requested rendered length.
    assert token_count >= 6
    assistant = label + label.lower() * (token_count - 6)
    return Conversation(
        messages=(
            UserMessage(content=""),
            AssistantMessage(content=assistant),
        )
    )


def _source(
    conversations: tuple[Conversation, ...],
    *,
    identity: str = "source-a",
    shuffle: bool = False,
) -> InMemoryConversationSource:
    return InMemoryConversationSource(
        conversations,
        source_identity=identity,
        shuffle=shuffle,
    )


def _loader(
    source: InMemoryConversationSource,
    *,
    batch_size: int = 1,
    max_seq_len: int = 19,
    buffer_size: int = 3,
    seed: int = 7,
    repeat: bool = False,
    weight: int = 1,
    tokenizer: ByteTokenizer | None = None,
) -> SFTConversationLoader:
    return SFTConversationLoader(
        (WeightedConversationSource(source, repeat_weight=weight),),
        tokenizer=ByteTokenizer() if tokenizer is None else tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        packing_buffer_size=buffer_size,
        seed=seed,
        repeat=repeat,
    )


def _expected_row(
    conversations: tuple[Conversation, ...],
    *,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = ByteTokenizer()
    ids: list[int] = []
    mask: list[bool] = []
    for conversation in conversations:
        rendered = render_conversation(conversation, tokenizer)
        ids.extend(rendered.token_ids)
        mask.extend(rendered.loss_mask)
    ids.extend([tokenizer.get_bos_token_id()] * (capacity - len(ids)))
    mask.extend([False] * (capacity - len(mask)))
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(
        [
            token_id if supervised else IGNORE_INDEX
            for token_id, supervised in zip(ids[1:], mask[1:], strict=True)
        ],
        dtype=torch.long,
    ).unsqueeze(0)
    return x, y


def test_loader_emits_contiguous_shifted_integer_batches_without_mutation() -> None:
    conversations = (
        _conversation("A", 8),
        _conversation("B", 12),
        _conversation("C", 8),
    )
    before = deepcopy(conversations)
    loader = _loader(_source(conversations))

    x, y = loader.next_batch()
    saved_x = x.clone()
    saved_y = y.clone()
    loader.next_batch()

    expected_x, expected_y = _expected_row(
        (conversations[1], conversations[0]),
        capacity=20,
    )
    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)
    assert torch.equal(x, saved_x)
    assert torch.equal(y, saved_y)
    assert conversations == before
    assert x.shape == y.shape == (1, 19)
    assert x.dtype == y.dtype == torch.long
    assert x.is_contiguous() and y.is_contiguous()
    assert int(x.min()) >= 0
    assert int(x.max()) < ByteTokenizer().get_vocab_size()
    assert set(y[y < 0].tolist()) == {IGNORE_INDEX}
    assert bool((y != IGNORE_INDEX).any())


def test_best_fit_uses_stable_equal_length_ties() -> None:
    conversations = (_conversation("A", 8), _conversation("B", 8))
    loader = _loader(
        _source(conversations),
        max_seq_len=7,
        buffer_size=2,
    )

    first_x, first_y = loader.next_batch()
    second_x, second_y = loader.next_batch()

    assert torch.equal(first_x, _expected_row((conversations[0],), capacity=8)[0])
    assert torch.equal(first_y, _expected_row((conversations[0],), capacity=8)[1])
    assert torch.equal(second_x, _expected_row((conversations[1],), capacity=8)[0])
    assert torch.equal(second_y, _expected_row((conversations[1],), capacity=8)[1])


def test_best_fit_refills_buffer_before_each_largest_fitting_choice() -> None:
    conversations = (
        _conversation("A", 12),
        _conversation("B", 6),
        _conversation("C", 8),
    )
    loader = _loader(_source(conversations), buffer_size=2)

    x, y = loader.next_batch()

    expected_x, expected_y = _expected_row(
        (conversations[0], conversations[2]),
        capacity=20,
    )
    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)
    assert loader.last_batch_info.content_lengths == (20,)
    assert len(loader.last_batch_info.row_item_identities[0]) == 2


def test_residual_and_incomplete_batch_rows_use_masked_bos_fill() -> None:
    conversation = _conversation("A", 8)
    loader = _loader(
        _source((conversation,)),
        batch_size=2,
        max_seq_len=11,
        buffer_size=1,
    )

    x, y = loader.next_batch()

    assert x.shape == y.shape == (2, 11)
    expected_x, expected_y = _expected_row((conversation,), capacity=12)
    assert torch.equal(x[0], expected_x[0])
    assert torch.equal(y[0], expected_y[0])
    assert torch.equal(
        x[1],
        torch.full((11,), ByteTokenizer().get_bos_token_id(), dtype=torch.long),
    )
    assert torch.equal(
        y[1],
        torch.full((11,), IGNORE_INDEX, dtype=torch.long),
    )
    assert loader.last_batch_info.content_lengths == (8, 0)
    assert loader.stats.padding_rows == 1
    assert loader.stats.padding_tokens == 16


def test_prefix_crop_and_zero_supervision_skips_are_observable_and_bounded() -> None:
    no_supervision_prefix = Conversation(
        messages=(
            UserMessage(content="u" * 40),
            AssistantMessage(content="answer"),
        )
    )
    supervised_prefix = Conversation(
        messages=(
            UserMessage(content=""),
            AssistantMessage(content="a" * 40),
        )
    )
    loader = _loader(
        _source((no_supervision_prefix, supervised_prefix)),
        max_seq_len=7,
        buffer_size=2,
    )

    x, y = loader.next_batch()

    assert x.shape == y.shape == (1, 7)
    assert bool((y != IGNORE_INDEX).any())
    assert loader.stats.seen_conversations == 2
    assert loader.stats.cropped_conversations == 2
    assert loader.stats.skipped_zero_supervision == 1
    assert loader.stats.packed_conversations == 1
    assert loader.last_batch_info.content_lengths == (8,)


def test_all_unsupervised_dataset_fails_instead_of_emitting_nan_batch() -> None:
    conversation = Conversation(
        messages=(
            UserMessage(content="u" * 40),
            AssistantMessage(content="answer"),
        )
    )
    loader = _loader(
        _source((conversation,)),
        max_seq_len=3,
        buffer_size=1,
    )

    with pytest.raises(SFTLoaderError, match="no supervised assistant targets"):
        loader.next_batch()
    assert loader.stats.cropped_conversations == 1
    assert loader.stats.skipped_zero_supervision == 1


def test_weighted_mixture_order_is_repeatable_and_counts_every_repeat() -> None:
    first_source = _source(
        (_conversation("A", 8), _conversation("B", 8)),
        identity="first",
    )
    second_source = _source(
        (_conversation("C", 8), _conversation("D", 8)),
        identity="second",
    )

    def collect(seed: int) -> tuple[str, ...]:
        loader = SFTConversationLoader(
            (
                WeightedConversationSource(first_source, repeat_weight=2),
                WeightedConversationSource(second_source, repeat_weight=1),
            ),
            tokenizer=ByteTokenizer(),
            batch_size=1,
            max_seq_len=7,
            packing_buffer_size=1,
            seed=seed,
            repeat=False,
        )
        identities: list[str] = []
        for _ in loader.iter_epoch():
            identities.extend(loader.last_batch_info.row_item_identities[0])
        assert loader.stats.packed_conversations == 6
        return tuple(identities)

    assert collect(29) == collect(29)
    assert collect(29) != collect(30)


def test_train_epoch_boundaries_and_finite_exhaustion_are_explicit() -> None:
    source = _source((_conversation("A", 8), _conversation("B", 8)))
    finite = _loader(source, max_seq_len=7, buffer_size=1, repeat=False)
    repeated = _loader(source, max_seq_len=7, buffer_size=1, repeat=True)

    finite_batches = tuple(finite.iter_epoch())
    repeated_batches = tuple(repeated.iter_epoch())

    assert len(finite_batches) == len(repeated_batches) == 2
    assert finite.epoch == repeated.epoch == 0
    assert finite.epoch_exhausted and repeated.epoch_exhausted
    with pytest.raises(StopIteration):
        finite.next_batch()
    repeated.next_batch()
    assert repeated.epoch == 1
    assert repeated.epoch_step == 1


def test_fresh_validation_loaders_ignore_prior_train_consumption() -> None:
    source = _source(
        tuple(_conversation(chr(65 + index), 8) for index in range(5)),
        shuffle=True,
    )
    train = _loader(source, max_seq_len=7, buffer_size=2, repeat=True)
    train.next_batch()
    train.next_batch()

    first = build_fresh_sft_validation_loader(
        (source,),
        tokenizer=ByteTokenizer(),
        batch_size=2,
        max_seq_len=7,
        packing_buffer_size=3,
        seed=41,
    )
    second = build_fresh_sft_validation_loader(
        (source,),
        tokenizer=ByteTokenizer(),
        batch_size=2,
        max_seq_len=7,
        packing_buffer_size=3,
        seed=41,
    )

    first_batches = tuple((x.clone(), y.clone()) for x, y in first.iter_epoch())
    second_batches = tuple((x.clone(), y.clone()) for x, y in second.iter_epoch())
    assert len(first_batches) == len(second_batches)
    assert all(
        torch.equal(first_x, second_x) and torch.equal(first_y, second_y)
        for (first_x, first_y), (second_x, second_y) in zip(
            first_batches,
            second_batches,
            strict=True,
        )
    )


def test_json_safe_state_resumes_exact_batches_and_buffer_identities() -> None:
    conversations = tuple(
        _conversation(chr(65 + index), 8 + index % 3) for index in range(9)
    )
    source = _source(conversations, shuffle=True)
    uninterrupted = _loader(
        source,
        max_seq_len=15,
        buffer_size=4,
        seed=101,
        repeat=True,
    )
    uninterrupted.next_batch()
    serialized_state = json.loads(json.dumps(uninterrupted.state_dict()))
    expected = []
    for _ in range(5):
        x, y = uninterrupted.next_batch()
        expected.append((x.clone(), y.clone(), uninterrupted.last_batch_info))

    resumed = _loader(
        _source(conversations, shuffle=True),
        max_seq_len=15,
        buffer_size=4,
        seed=101,
        repeat=True,
    )
    resumed.load_state_dict(serialized_state)
    actual = []
    for _ in range(5):
        x, y = resumed.next_batch()
        actual.append((x.clone(), y.clone(), resumed.last_batch_info))

    assert serialized_state["format"] == SFT_LOADER_STATE_FORMAT
    assert serialized_state["format_version"] == SFT_LOADER_STATE_VERSION
    assert serialized_state["buffer"]
    assert all("item_identity" in item for item in serialized_state["buffer"])
    assert "token_ids" not in json.dumps(serialized_state)
    assert "loss_mask" not in json.dumps(serialized_state)
    assert len(actual) == len(expected)
    for (actual_x, actual_y, actual_info), (
        expected_x,
        expected_y,
        expected_info,
    ) in zip(actual, expected, strict=True):
        assert torch.equal(actual_x, expected_x)
        assert torch.equal(actual_y, expected_y)
        assert actual_info == expected_info
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_state_keeps_only_independent_mixture_progress_and_buffer_references() -> None:
    source = _source(tuple(_conversation(chr(65 + index), 8) for index in range(6)))
    loader = _loader(
        source,
        max_seq_len=7,
        buffer_size=4,
        seed=11,
        repeat=True,
    )

    loader.next_batch()
    state = loader.state_dict()

    assert "mixture_order" not in state
    assert "source_cursors" not in state
    assert state["buffer"]
    assert all(
        set(item) == {"item_identity", "source_index", "source_offset"}
        for item in state["buffer"]
    )


def test_compact_state_resumes_a_weighted_multi_source_mixture_exactly() -> None:
    first = _source(
        tuple(_conversation(chr(65 + index), 8) for index in range(4)),
        identity="first",
        shuffle=True,
    )
    second = _source(
        tuple(_conversation(chr(75 + index), 8) for index in range(3)),
        identity="second",
        shuffle=True,
    )

    def build() -> SFTConversationLoader:
        return SFTConversationLoader(
            (
                WeightedConversationSource(first, repeat_weight=2),
                WeightedConversationSource(second),
            ),
            tokenizer=ByteTokenizer(),
            batch_size=1,
            max_seq_len=7,
            packing_buffer_size=3,
            seed=37,
            repeat=True,
        )

    uninterrupted = build()
    uninterrupted.next_batch()
    uninterrupted.next_batch()
    state = json.loads(json.dumps(uninterrupted.state_dict()))
    expected = []
    for _ in range(8):
        expected.append(uninterrupted.next_batch())

    resumed = build()
    resumed.load_state_dict(state)
    actual = [resumed.next_batch() for _ in range(8)]

    assert all(
        torch.equal(actual_x, expected_x) and torch.equal(actual_y, expected_y)
        for (actual_x, actual_y), (expected_x, expected_y) in zip(
            actual,
            expected,
            strict=True,
        )
    )
    assert resumed.state_dict() == uninterrupted.state_dict()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda state: state.update(format_version=999), "format version"),
        (lambda state: state.update(unexpected=True), "fields"),
        (lambda state: state.update(mixture_cursor=999), "mixture cursor"),
        (
            lambda state: state["buffer"][0].update(source_offset=999),
            "buffer.*locator",
        ),
        (lambda state: state.update(rng_state=["bad"]), "rng_state"),
        (
            lambda state: state["buffer"][0].update(item_identity="sha256:" + "0" * 64),
            "buffer.*identity",
        ),
    ],
)
def test_state_rejection_is_transactional(mutation, match: str) -> None:
    source = _source(tuple(_conversation(chr(65 + index), 8) for index in range(6)))
    loader = _loader(
        source,
        max_seq_len=7,
        buffer_size=4,
        seed=11,
        repeat=True,
    )
    loader.next_batch()
    before = deepcopy(loader.state_dict())
    invalid = deepcopy(before)
    mutation(invalid)

    with pytest.raises(SFTLoaderStateError, match=match):
        loader.load_state_dict(invalid)
    assert loader.state_dict() == before


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ({"max_seq_len": 8}, "max_seq_len"),
        ({"batch_size": 2}, "batch_size"),
        ({"buffer_size": 2}, "packing_buffer_size"),
        ({"weight": 2}, "repeat_weights"),
        ({"identity": "different"}, "source identities"),
        ({"tokenizer": _DifferentIdentityTokenizer()}, "tokenizer identity"),
    ],
)
def test_state_rejects_changed_continuation_contracts(
    replacement: dict[str, object],
    match: str,
) -> None:
    conversations = tuple(_conversation(chr(65 + index), 8) for index in range(5))
    original = _loader(
        _source(conversations),
        max_seq_len=7,
        buffer_size=3,
        seed=13,
        repeat=True,
    )
    original.next_batch()
    state = original.state_dict()

    source = _source(
        conversations,
        identity=str(replacement.get("identity", "source-a")),
    )
    incompatible = _loader(
        source,
        batch_size=int(replacement.get("batch_size", 1)),
        max_seq_len=int(replacement.get("max_seq_len", 7)),
        buffer_size=int(replacement.get("buffer_size", 3)),
        seed=13,
        repeat=True,
        weight=int(replacement.get("weight", 1)),
        tokenizer=replacement.get("tokenizer"),  # type: ignore[arg-type]
    )
    before = incompatible.state_dict()

    with pytest.raises(SFTLoaderStateError, match=match):
        incompatible.load_state_dict(state)
    assert incompatible.state_dict() == before


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_seq_len": 0}, "max_seq_len"),
        ({"packing_buffer_size": 0}, "packing_buffer_size"),
        ({"seed": -1}, "seed"),
    ],
)
def test_loader_rejects_invalid_settings(kwargs: dict[str, int], match: str) -> None:
    settings = {
        "batch_size": 1,
        "max_seq_len": 7,
        "packing_buffer_size": 2,
        "seed": 1,
    }
    settings.update(kwargs)
    with pytest.raises((SFTLoaderError, TypeError, ValueError), match=match):
        SFTConversationLoader(
            (WeightedConversationSource(_source((_conversation("A", 8),))),),
            tokenizer=ByteTokenizer(),
            repeat=False,
            **settings,
        )


def test_loader_rejects_empty_duplicate_or_zero_weight_sources() -> None:
    empty = _source((), identity="empty")
    source = _source((_conversation("A", 8),), identity="same")
    with pytest.raises(SFTLoaderError, match="empty"):
        _loader(empty)
    with pytest.raises(SFTLoaderError, match="unique source identities"):
        SFTConversationLoader(
            (WeightedConversationSource(source), WeightedConversationSource(source)),
            tokenizer=ByteTokenizer(),
            batch_size=1,
            max_seq_len=7,
            packing_buffer_size=1,
            seed=1,
        )
    with pytest.raises((TypeError, ValueError), match="repeat_weight"):
        WeightedConversationSource(source, repeat_weight=0)


def test_tracked_jsonl_source_has_stable_identity_and_fresh_seeded_iteration() -> None:
    path = PROJECT_ROOT / "data" / "fixtures" / "chat" / "train.jsonl"
    first = load_jsonl_conversation_source(path, shuffle=True)
    second = load_jsonl_conversation_source(path, shuffle=True)

    first_examples = tuple(first.iter_examples(seed=73))
    second_examples = tuple(second.iter_examples(seed=73))

    assert len(first) == len(second) == 5
    assert first.source_identity == second.source_identity
    assert [example.identity for example in first_examples] == [
        example.identity for example in second_examples
    ]
