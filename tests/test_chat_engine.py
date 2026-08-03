"""Transport-neutral chat-engine contracts and lifecycle tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scratch_llm.chat import (
    CHAT_RENDERER_ID,
    AssistantMessage,
    ChatEngine,
    ChatEngineError,
    ChatRenderingError,
    ChatState,
    Conversation,
    TokenEvent,
    UserMessage,
    read_conversations,
    render_completion_prompt,
)
from scratch_llm.config import (
    GPTConfig,
    GenerationConfig,
    ProjectConfig,
    RunConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.model import GPT
from scratch_llm.training.checkpoint import ExactTrainingState, save_checkpoint
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.rng_state import capture_training_rng_state
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _TransitionModel(torch.nn.Module):
    def __init__(
        self,
        transitions: dict[int, int],
        *,
        max_seq_len: int = 128,
        fail_after: int | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.transitions = transitions
        self.fail_after = fail_after
        self.forward_calls = 0
        self.contexts: list[torch.Tensor] = []
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.contexts.append(token_ids.detach().cpu().clone())
        random.random()
        np.random.random()
        if self.fail_after is not None and self.forward_calls > self.fail_after:
            raise RuntimeError("fixture generation failed")
        next_ids = [
            self.transitions[int(token_id)]
            for token_id in token_ids[:, -1].detach().cpu().tolist()
        ]
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], 265),
            -torch.inf,
            device=token_ids.device,
        )
        for row, token_id in enumerate(next_ids):
            logits[row, -1, token_id] = 0
        return logits


def _checkpoint(
    model: torch.nn.Module,
    *,
    stage: str = "sft",
    generation: GenerationConfig | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        tokenizer=ByteTokenizer(),
        config=SimpleNamespace(generation=generation or GenerationConfig()),
        step=17,
        training_stage=stage,
    )


def _engine(
    model: torch.nn.Module,
    *,
    clock: _Clock | None = None,
    stage: str = "sft",
    generation: GenerationConfig | None = None,
) -> tuple[ChatEngine, list[tuple[Path, str]]]:
    calls: list[tuple[Path, str]] = []

    def load(path: Path, *, device: str) -> SimpleNamespace:
        calls.append((path, device))
        return _checkpoint(model, stage=stage, generation=generation)

    return (
        ChatEngine(
            "fixture.pt",
            device="cpu",
            checkpoint_loader=load,
            clock=clock or _Clock(),
        ),
        calls,
    )


def _assistant_start(tokenizer: ByteTokenizer) -> int:
    return tokenizer.encode_special("<|assistant_start|>")


def _assistant_end(tokenizer: ByteTokenizer) -> int:
    return tokenizer.encode_special("<|assistant_end|>")


def _bos(tokenizer: ByteTokenizer) -> int:
    return tokenizer.get_bos_token_id()


def test_engine_loads_sft_checkpoint_and_exposes_frozen_json_state() -> None:
    tokenizer = ByteTokenizer()
    model = _TransitionModel({_assistant_start(tokenizer): ord("A")})
    model.train()

    engine, calls = _engine(model)
    state = engine.get_state()

    assert calls == [(Path("fixture.pt"), "cpu")]
    assert model.training is False
    assert isinstance(state, ChatState)
    assert state.checkpoint_path == "fixture.pt"
    assert state.checkpoint_step == 17
    assert state.training_stage == "sft"
    assert state.device == "cpu"
    assert state.renderer_id == CHAT_RENDERER_ID
    assert state.tokenizer_identity == tokenizer.get_identity()
    assert state.status == "idle"
    assert state.messages == ()
    assert json.loads(json.dumps(state.to_dict()))["messages"] == []
    with pytest.raises(FrozenInstanceError):
        state.status = "failed"  # type: ignore[misc]


def test_engine_owns_tokenizer_utilities_context_limit_and_resource_release() -> None:
    tokenizer = ByteTokenizer()
    model = _TransitionModel({_assistant_start(tokenizer): ord("A")}, max_seq_len=23)
    engine, _ = _engine(model)

    token_ids = engine.tokenize("café 🚀")

    assert token_ids == tuple(tokenizer.encode("café 🚀"))
    assert engine.detokenize(token_ids) == "café 🚀"
    assert engine.max_context_tokens == 23

    engine.close()
    engine.close()

    with pytest.raises(ChatEngineError, match="closed"):
        engine.tokenize("after close")


def test_engine_rejects_non_sft_checkpoint_actionably() -> None:
    tokenizer = ByteTokenizer()
    model = _TransitionModel({_assistant_start(tokenizer): ord("A")})

    with pytest.raises(ChatEngineError, match="SFT checkpoint.*pretrain"):
        _engine(model, stage="pretrain")


def test_default_loader_reconstructs_tokenizer_and_checkpoint_defaults(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=16,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=16,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0,
        ),
        sft=SFTConfig(
            device_batch_size=1,
            total_batch_size_tokens=16,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0,
            eval_every=1,
            eval_batches=1,
            save_every=1,
            log_every=1,
        ),
        generation=GenerationConfig(temperature=0, max_new_tokens=1),
    )
    model = GPT(config.model)
    active_train = config.sft.to_train_config(config.model.seq_len)
    optimizer = build_optimizer(model, active_train)
    scheduler = build_lr_scheduler(optimizer, active_train)
    checkpoint_path = save_checkpoint(
        tmp_path / "sft.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=ByteTokenizer(),
        continuation=ExactTrainingState(
            loader_format="fixture_loader_v1",
            loader_state={"format": "fixture_loader_v1", "position": 0},
            rng_state=capture_training_rng_state("cpu"),
            tracker_step=0,
            total_training_time_seconds=0,
            total_training_flops=0,
        ),
        training_stage="sft",
        base_checkpoint_identity="sha256:" + "a" * 64,
    )

    engine = ChatEngine(checkpoint_path, device="cpu")
    defaults = engine.default_generation_config
    defaults.max_new_tokens = 9
    engine.append_user_message("x")
    events = tuple(engine.generate_stream())

    assert engine.get_state().tokenizer_identity == ByteTokenizer().get_identity()
    assert engine.default_generation_config.max_new_tokens == 1
    assert events[-1].completion_reason == "max_new_tokens"
    assert events[-1].generated_token_count == 1


def test_streaming_is_utf8_lossless_and_commits_one_normalized_turn() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    assistant_end = _assistant_end(tokenizer)
    model = _TransitionModel(
        {
            assistant_start: 0xF0,
            0xF0: 0x9F,
            0x9F: 0x9A,
            0x9A: 0x80,
            0x80: assistant_end,
        }
    )
    clock = _Clock()
    engine, _ = _engine(model, clock=clock)
    engine.append_user_message("Launch?")

    stream = engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=8))
    start = next(stream)
    assert start == TokenEvent(
        type="start",
        token_ids=(),
        text_delta="",
        prompt_token_count=start.prompt_token_count,
        generated_token_count=0,
        sampled_token_count=0,
        elapsed_seconds=0.0,
        completion_reason=None,
        stop_token_id=None,
    )
    clock.now = 2.0
    events = (start, *tuple(stream))

    visible_ids = tuple(token_id for event in events for token_id in event.token_ids)
    text = "".join(event.text_delta for event in events)
    assert visible_ids == (0xF0, 0x9F, 0x9A, 0x80)
    assert text == tokenizer.decode(visible_ids) == "🚀"
    assert [event.type for event in events] == [
        "start",
        "token",
        "token",
        "token",
        "token",
        "complete",
    ]
    assert events[-1].completion_reason == "stop_token"
    assert events[-1].stop_token_id == assistant_end
    assert events[-1].generated_token_count == 4
    assert events[-1].sampled_token_count == 5
    assert events[-1].elapsed_seconds == 2.0
    assert json.loads(json.dumps(events[-1].to_dict()))["type"] == "complete"
    state = engine.get_state()
    assert state.status == "completed"
    assert state.generated_token_count == 4
    assert state.sampled_token_count == 5
    assert state.generation_seconds == 2.0
    assert tuple((message.role, message.content) for message in state.messages) == (
        ("user", "Launch?"),
        ("assistant", "🚀"),
    )


@pytest.mark.parametrize("stop_name", ["assistant_end", "bos"])
def test_chat_stop_tokens_are_control_metadata_and_stop_immediately(
    stop_name: str,
) -> None:
    tokenizer = ByteTokenizer()
    stop_token_id = (
        _assistant_end(tokenizer) if stop_name == "assistant_end" else _bos(tokenizer)
    )
    model = _TransitionModel({_assistant_start(tokenizer): stop_token_id})
    engine, _ = _engine(model)
    engine.append_user_message("stop")

    events = tuple(
        engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=8))
    )

    assert [event.type for event in events] == ["start", "complete"]
    assert events[-1].token_ids == ()
    assert events[-1].text_delta == ""
    assert events[-1].completion_reason == "stop_token"
    assert events[-1].stop_token_id == stop_token_id
    assert events[-1].generated_token_count == 0
    assert events[-1].sampled_token_count == 1
    assert model.forward_calls == 1
    assert engine.get_state().messages[-1] == AssistantMessage("")


def test_chat_generation_passes_exact_stop_set_to_shared_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scratch_llm.chat.engine as engine_module

    tokenizer = ByteTokenizer()
    model = _TransitionModel(
        {_assistant_start(tokenizer): ord("A"), ord("A"): ord("A")}
    )
    engine, _ = _engine(model)
    engine.append_user_message("hello")
    observed: list[frozenset[int]] = []
    real_stream = engine_module.stream_generate_sequence

    def record_stop_set(*args: object, **kwargs: object):
        observed.append(frozenset(kwargs["stop_token_ids"]))  # type: ignore[arg-type]
        return real_stream(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(engine_module, "stream_generate_sequence", record_stop_set)

    events = tuple(
        engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=1))
    )

    assert events[-1].completion_reason == "max_new_tokens"
    assert events[-1].stop_token_id is None
    assert events[-1].generated_token_count == 1
    assert events[-1].sampled_token_count == 1
    assert model.forward_calls == 1
    assert observed == [frozenset({_assistant_end(tokenizer), _bos(tokenizer)})]


def test_engine_drops_turns_then_crops_current_user_without_losing_transcript() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    assistant_end = _assistant_end(tokenizer)
    model = _TransitionModel(
        {assistant_start: ord("A"), ord("A"): assistant_end},
        max_seq_len=13,
    )
    engine, _ = _engine(model)
    for user_text in ("u1", "u2"):
        engine.append_user_message(user_text)
        tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))

    engine.append_user_message("abcdefghij")
    pending = engine.get_state()

    assert pending.prompt_token_count == model.max_seq_len
    assert pending.dropped_turn_count == 2
    assert pending.truncated_user_token_count == 1
    assert [(message.role, message.content) for message in pending.messages] == [
        ("user", "u1"),
        ("assistant", "A"),
        ("user", "u2"),
        ("assistant", "A"),
        ("user", "abcdefghij"),
    ]
    model.contexts.clear()
    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))

    assert all(context.shape[1] <= model.max_seq_len for context in model.contexts)
    assert tokenizer.decode(model.contexts[0][0].tolist()) == (
        "<|bos|><|user_start|>bcdefghij<|user_end|><|assistant_start|>"
    )
    assert engine.get_state().dropped_turn_count == 2
    assert engine.get_state().truncated_user_token_count == 1
    assert engine.get_state().messages[-1] == AssistantMessage("A")


def test_impossible_fixed_control_budget_fails_before_engine_or_model_mutation() -> (
    None
):
    tokenizer = ByteTokenizer()
    model = _TransitionModel(
        {_assistant_start(tokenizer): ord("A")},
        max_seq_len=3,
    )
    engine, _ = _engine(model)
    before = engine.get_state()

    with pytest.raises(ChatRenderingError, match="fixed chat controls"):
        engine.append_user_message("x")

    assert engine.get_state() == before
    assert model.forward_calls == 0


def test_multi_turn_prompt_is_rendered_only_through_canonical_renderer() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    assistant_end = _assistant_end(tokenizer)
    model = _TransitionModel({assistant_start: ord("A"), ord("A"): assistant_end})
    engine, _ = _engine(model)

    engine.append_user_message("first")
    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))
    engine.append_user_message("second")
    expected = render_completion_prompt(
        Conversation(
            messages=(
                UserMessage("first"),
                AssistantMessage("A"),
                UserMessage("second"),
            )
        ),
        tokenizer,
    )
    model.contexts.clear()
    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))

    assert torch.equal(model.contexts[0], torch.tensor([expected.token_ids]))


def test_invalid_transitions_and_top_p_fail_before_state_mutation() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    model = _TransitionModel({assistant_start: ord("A"), ord("A"): ord("A")})
    engine, _ = _engine(model)

    with pytest.raises(ChatEngineError, match="user message"):
        engine.generate_stream()
    engine.append_user_message("hello")
    before = engine.get_state()
    with pytest.raises(ChatEngineError, match="assistant response"):
        engine.append_user_message("too soon")
    with pytest.raises(ChatEngineError, match="top_p"):
        engine.generate_stream(GenerationConfig(top_p=0.9))
    assert engine.get_state() == before

    stream = engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=3))
    assert next(stream).type == "start"
    active = engine.get_state()
    with pytest.raises(ChatEngineError, match="already active"):
        engine.generate_stream()
    with pytest.raises(ChatEngineError, match="already active"):
        engine.reset()
    with pytest.raises(ChatEngineError, match="already active"):
        engine.append_user_message("blocked")
    assert engine.get_state() == active
    stream.close()


def test_lazy_stream_rechecks_transaction_ownership_and_conversation_revision() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    model = _TransitionModel({assistant_start: ord("A"), ord("A"): ord("A")})
    engine, _ = _engine(model)
    engine.append_user_message("first")
    first = engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=1))
    duplicate = engine.generate_stream(
        GenerationConfig(temperature=0, max_new_tokens=1)
    )

    assert next(first).type == "start"
    with pytest.raises(ChatEngineError, match="already active"):
        next(duplicate)
    first.close()

    stale = engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=1))
    engine.reset()
    engine.append_user_message("replacement")
    before = engine.get_state()
    with pytest.raises(ChatEngineError, match="conversation changed"):
        next(stale)
    assert engine.get_state() == before


def test_iterator_close_restores_rng_and_mode_and_rolls_back_assistant() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    model = _TransitionModel({assistant_start: ord("A"), ord("A"): ord("A")})
    engine, _ = _engine(model)
    engine.append_user_message("hello")
    random.seed(51)
    np.random.seed(52)
    torch.manual_seed(53)
    python_state = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    torch_state = torch.get_rng_state().clone()
    stream = engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=3))

    assert next(stream).type == "start"
    assert next(stream).type == "token"
    stream.close()

    assert model.training is False
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state(legacy=True)
    assert restored_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    state = engine.get_state()
    assert state.status == "cancelled"
    assert [(message.role, message.content) for message in state.messages] == [
        ("user", "hello")
    ]

    engine.rollback_last_turn()

    assert engine.get_state().status == "idle"
    assert engine.get_state().messages == ()


def test_pending_turn_rollback_preserves_earlier_completed_history() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    assistant_end = _assistant_end(tokenizer)
    model = _TransitionModel({assistant_start: ord("A"), ord("A"): assistant_end})
    engine, _ = _engine(model)
    engine.append_user_message("first")
    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))
    before = engine.get_state().messages
    engine.append_user_message("cancel me")

    engine.rollback_last_turn(include_completed=True)

    state = engine.get_state()
    assert state.status == "completed"
    assert state.messages == before
    assert state.prompt_token_count == 0

    engine.append_user_message("completed at cancellation boundary")
    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))
    engine.rollback_last_turn(include_completed=True)

    assert engine.get_state().messages == before
    with pytest.raises(ChatEngineError, match="latest chat turn"):
        engine.rollback_last_turn()


def test_generation_failure_restores_state_and_allows_retry() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    model = _TransitionModel(
        {assistant_start: ord("A"), ord("A"): ord("A")},
        fail_after=1,
    )
    engine, _ = _engine(model)
    engine.append_user_message("hello")

    with pytest.raises(RuntimeError, match="fixture generation failed"):
        tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=3)))

    assert engine.get_state().status == "failed"
    assert len(engine.get_state().messages) == 1
    model.fail_after = None
    events = tuple(
        engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=1))
    )
    assert events[-1].type == "complete"
    assert engine.get_state().status == "completed"


def test_reset_clears_completed_history_without_reloading_checkpoint() -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    assistant_end = _assistant_end(tokenizer)
    model = _TransitionModel({assistant_start: ord("A"), ord("A"): assistant_end})
    engine, calls = _engine(model)
    engine.append_user_message("hello")
    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))

    engine.reset()

    state = engine.get_state()
    assert state.status == "idle"
    assert state.messages == ()
    assert state.prompt_token_count == 0
    assert state.generated_token_count == 0
    assert calls == [(Path("fixture.pt"), "cpu")]


def test_transcript_save_is_atomic_completed_only_and_reset_preserves_file(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    assistant_start = _assistant_start(tokenizer)
    assistant_end = _assistant_end(tokenizer)
    model = _TransitionModel({assistant_start: ord("A"), ord("A"): assistant_end})
    engine, _ = _engine(model)
    transcript = tmp_path / "chat.jsonl"
    engine.append_user_message("Café")

    with pytest.raises(ChatEngineError, match="completed conversation"):
        engine.save_transcript(transcript)
    assert not transcript.exists()

    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))
    engine.save_transcript(transcript)

    assert read_conversations(transcript) == (
        Conversation(messages=(UserMessage("Café"), AssistantMessage("A"))),
    )
    engine.append_user_message("第二")
    tuple(engine.generate_stream(GenerationConfig(temperature=0, max_new_tokens=2)))
    engine.save_transcript(transcript)
    saved = transcript.read_bytes()
    assert read_conversations(transcript) == (
        Conversation(
            messages=(
                UserMessage("Café"),
                AssistantMessage("A"),
                UserMessage("第二"),
                AssistantMessage("A"),
            )
        ),
    )
    engine.reset()
    assert transcript.read_bytes() == saved
    engine.append_user_message("unfinished")
    with pytest.raises(ChatEngineError, match="completed conversation"):
        engine.save_transcript(transcript)
    assert transcript.read_bytes() == saved
