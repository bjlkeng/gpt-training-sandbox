"""Cached decoding parity, lifecycle, and shared-consumer integration."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.chat import run_terminal_chat
from scratch_llm.chat import ChatEngine
from scratch_llm.config import GPTConfig, GenerationConfig
from scratch_llm.generation import (
    GeneratedToken,
    generate_sequences,
    stream_generate_sequence,
)
from scratch_llm.model import GPT
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from scratch_llm.web.service import (
    ChatSessionService,
    GenerationOverrides,
    GenerationTerminal,
)


class _RecordingCache:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _CacheAwareLogitsModel(torch.nn.Module):
    def __init__(
        self,
        logits: torch.Tensor,
        *,
        max_seq_len: int = 16,
        use_kv_cache: bool = True,
        fail_after: int | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.config = SimpleNamespace(use_kv_cache=use_kv_cache)
        self.logits: torch.Tensor
        self.register_buffer("logits", logits)
        self.fail_after = fail_after
        self.forward_lengths: list[int] = []
        self.forward_caches: list[object | None] = []
        self.created_caches: list[_RecordingCache] = []
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def create_kv_cache(self, *, batch_size: int, capacity: int) -> _RecordingCache:
        assert batch_size == 1
        assert capacity == self.max_seq_len
        cache = _RecordingCache()
        self.created_caches.append(cache)
        return cache

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        kv_cache: object | None = None,
    ) -> torch.Tensor:
        self.forward_lengths.append(token_ids.shape[1])
        self.forward_caches.append(kv_cache)
        random.random()
        np.random.random()
        if self.fail_after is not None and len(self.forward_lengths) > self.fail_after:
            raise RuntimeError("fixture decode failed")
        return self.logits.reshape(1, 1, -1).expand(
            token_ids.shape[0],
            token_ids.shape[1],
            -1,
        )


class _CacheAwareTransitionModel(_CacheAwareLogitsModel):
    def __init__(
        self,
        transitions: dict[int, int],
        *,
        vocab_size: int = 265,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__(torch.zeros(vocab_size), max_seq_len=max_seq_len)
        self.transitions = transitions

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        kv_cache: object | None = None,
    ) -> torch.Tensor:
        self.forward_lengths.append(token_ids.shape[1])
        self.forward_caches.append(kv_cache)
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.logits.numel()),
            -torch.inf,
            device=token_ids.device,
        )
        for row, last_token in enumerate(token_ids[:, -1].detach().cpu().tolist()):
            logits[row, -1, self.transitions[int(last_token)]] = 0
        return logits


def _rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_equal(
    before: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0]
    np.testing.assert_array_equal(after_numpy[1], before[1][1])
    assert after_numpy[2:] == before[1][2:]
    assert torch.equal(torch.random.get_rng_state(), before[2])


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("max_new_tokens", [1, 5])
def test_cached_greedy_matches_naive_and_prefills_once(
    backend: str,
    max_new_tokens: int,
) -> None:
    torch.manual_seed(401)
    model = GPT(
        GPTConfig(
            vocab_size=32,
            seq_len=12,
            n_layer=2,
            n_head=2,
            n_embd=8,
            mlp_ratio=2,
            dropout=0,
            attention_backend=backend,  # type: ignore[arg-type]
        )
    ).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    naive = generate_sequences(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0,
        mode="naive",
    )
    observed_lengths: list[int] = []
    observed_caches: list[object] = []

    def record_forward(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, object],
    ) -> None:
        observed_lengths.append(args[0].shape[1])
        observed_caches.append(kwargs["kv_cache"])

    handle = model.register_forward_pre_hook(record_forward, with_kwargs=True)
    try:
        cached = generate_sequences(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0,
            mode="cached",
        )
    finally:
        handle.remove()

    assert cached == naive
    assert observed_lengths == [7, *([1] * (max_new_tokens - 1))]
    assert len({id(cache) for cache in observed_caches}) == 1
    assert observed_caches[0].position == 0  # type: ignore[attr-defined]


def test_cached_seeded_sampling_matches_naive_rng_steps() -> None:
    model = _CacheAwareLogitsModel(torch.zeros(32), use_kv_cache=False)
    prompt = torch.tensor([[4, 5, 6]])

    naive = generate_sequences(
        model,
        prompt,
        max_new_tokens=7,
        temperature=1,
        top_k=9,
        seed=409,
        mode="naive",
    )
    model.forward_lengths.clear()
    model.forward_caches.clear()
    cached = generate_sequences(
        model,
        prompt,
        max_new_tokens=7,
        temperature=1,
        top_k=9,
        seed=409,
        mode="cached",
    )

    assert cached == naive
    assert model.forward_lengths == [3, 1, 1, 1, 1, 1, 1]
    assert len({id(cache) for cache in model.forward_caches}) == 1
    assert model.created_caches[-1].reset_calls == 1


@pytest.mark.parametrize("stop_token_id", [9, 11])
def test_cached_immediate_stop_is_omitted_with_exact_metadata(
    stop_token_id: int,
) -> None:
    logits = torch.full((16,), -torch.inf)
    logits[stop_token_id] = 0
    prompt = torch.tensor([[1]])

    outcomes = []
    for mode in ("naive", "cached"):
        model = _CacheAwareLogitsModel(logits)
        outcomes.append(
            generate_sequences(
                model,
                prompt,
                max_new_tokens=4,
                temperature=0,
                stop_token_ids={stop_token_id},
                mode=mode,
            )
        )

    assert outcomes[0] == outcomes[1]
    sequence = outcomes[1].sequences[0]
    assert sequence.generated_token_ids == ()
    assert sequence.completion_reason == "stop_token"
    assert sequence.stop_token_id == stop_token_id
    assert sequence.sampled_token_count == 1


def test_cached_batch_and_overflow_fail_before_runtime_mutation() -> None:
    model = _CacheAwareLogitsModel(torch.zeros(32), max_seq_len=8)
    model.train()
    before = _rng_snapshot()

    with pytest.raises(ValueError, match="exactly one batch row"):
        generate_sequences(
            model,
            torch.tensor([[1], [2]]),
            max_new_tokens=2,
            mode="cached",
        )
    with pytest.raises(ValueError, match="cache capacity"):
        generate_sequences(
            model,
            torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
            max_new_tokens=3,
            mode="cached",
        )

    assert model.training is True
    assert model.created_caches == []
    assert model.forward_lengths == []
    _assert_rng_equal(before)


@pytest.mark.parametrize("exit_kind", ["close", "failure"])
def test_cached_stream_resets_lease_and_restores_modes_and_rng(exit_kind: str) -> None:
    model = _CacheAwareLogitsModel(
        torch.zeros(16),
        fail_after=1 if exit_kind == "failure" else None,
    )
    model.train()
    before = _rng_snapshot()
    stream = stream_generate_sequence(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=4,
        temperature=0,
        mode="cached",
    )

    assert isinstance(next(stream), GeneratedToken)
    if exit_kind == "close":
        stream.close()
    else:
        with pytest.raises(RuntimeError, match="fixture decode failed"):
            next(stream)

    assert model.training is True
    assert model.created_caches[-1].reset_calls == 1
    _assert_rng_equal(before)


def test_model_config_selects_cached_mode_and_explicit_naive_overrides_it() -> None:
    model = _CacheAwareLogitsModel(torch.zeros(8), use_kv_cache=True)
    prompt = torch.tensor([[1]])

    generate_sequences(model, prompt, max_new_tokens=2, temperature=0)
    assert model.forward_lengths == [1, 1]
    assert len(model.created_caches) == 1

    model.forward_lengths.clear()
    generate_sequences(
        model,
        prompt,
        max_new_tokens=2,
        temperature=0,
        mode="naive",
    )
    assert model.forward_lengths == [1, 2]
    assert len(model.created_caches) == 1


def _chat_engine(model: torch.nn.Module) -> ChatEngine:
    tokenizer = ByteTokenizer()

    def load(_path: Path, *, device: str) -> SimpleNamespace:
        assert device == "cpu"
        return SimpleNamespace(
            model=model,
            tokenizer=tokenizer,
            config=SimpleNamespace(
                generation=GenerationConfig(
                    temperature=0,
                    max_new_tokens=4,
                )
            ),
            step=1,
            training_stage="sft",
        )

    return ChatEngine("fixture.pt", device="cpu", checkpoint_loader=load)


def _chat_model() -> _CacheAwareTransitionModel:
    tokenizer = ByteTokenizer()
    return _CacheAwareTransitionModel(
        {
            tokenizer.encode_special("<|assistant_start|>"): ord("A"),
            ord("A"): tokenizer.encode_special("<|assistant_end|>"),
        }
    )


def test_terminal_and_web_share_cached_events_across_repeated_turns(
    tmp_path: Path,
) -> None:
    terminal_model = _chat_model()
    terminal_engine = _chat_engine(terminal_model)
    output = StringIO()
    run_terminal_chat(
        terminal_engine,
        GenerationConfig(temperature=0, max_new_tokens=4),
        prompt="hello",
        output_stream=output,
    )

    web_model = _chat_model()
    web_engine = _chat_engine(web_model)
    checkpoint_root = tmp_path / "catalog"
    checkpoint_root.mkdir()
    (checkpoint_root / "fixture.pt").write_bytes(b"fixture")
    service = ChatSessionService(
        checkpoint_root,
        engine_factory=lambda _path, _device: web_engine,
    )

    async def run_web_turns() -> list[GenerationTerminal]:
        await service.load_checkpoint("fixture.pt")
        terminals = []
        for message in ("hello", "again"):
            lease = await service.start_generation(
                message,
                GenerationOverrides(temperature=0, max_new_tokens=4),
            )
            items = [item async for item in lease]
            assert [item.text_delta for item in items[:-1]] == ["", "A"]
            terminal = items[-1]
            assert isinstance(terminal, GenerationTerminal)
            terminals.append(terminal)
        return terminals

    terminals = asyncio.run(run_web_turns())

    assert output.getvalue() == "A\n"
    assert [terminal.completion_event.stop_token_id for terminal in terminals] == [
        ByteTokenizer().encode_special("<|assistant_end|>"),
        ByteTokenizer().encode_special("<|assistant_end|>"),
    ]
    assert len(terminal_model.forward_lengths) == 2
    assert terminal_model.forward_lengths[0] > 1
    assert terminal_model.forward_lengths[1] == 1
    assert len(web_model.forward_lengths) == 4
    assert web_model.forward_lengths[0] > 1
    assert web_model.forward_lengths[1] == 1
    assert web_model.forward_lengths[2] > web_model.forward_lengths[0]
    assert web_model.forward_lengths[3] == 1
    assert [cache.reset_calls for cache in terminal_model.created_caches] == [1]
    assert [cache.reset_calls for cache in web_model.created_caches] == [1, 1]
