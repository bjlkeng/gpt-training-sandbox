"""Tests for shared no-cache autoregressive generation."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from scratch_llm.config import GPTConfig
from scratch_llm.generation import (
    GeneratedToken,
    GenerationBatchResult,
    GenerationComplete,
    generate,
    generate_sequences,
    stream_generate_sequence,
)
from scratch_llm.model import GPT
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer


def _model_config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "seq_len": 3,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "dropout": 0.0,
        "bias": True,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


class _FixedLogitsModel(torch.nn.Module):
    def __init__(self, next_token_logits: torch.Tensor) -> None:
        super().__init__()
        self.max_seq_len = 3
        self.fixed_logits: torch.Tensor
        self.register_buffer("fixed_logits", next_token_logits)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits.reshape(1, 1, -1).expand(
            token_ids.shape[0],
            token_ids.shape[1],
            -1,
        )


class _TransitionLogitsModel(torch.nn.Module):
    def __init__(self, transitions: dict[int, int], *, vocab_size: int = 32) -> None:
        super().__init__()
        self.max_seq_len = 8
        self.transitions = transitions
        self.vocab_size = vocab_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        next_ids = [
            self.transitions[int(token_id)]
            for token_id in token_ids[:, -1].detach().cpu().tolist()
        ]
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.vocab_size),
            -torch.inf,
            device=token_ids.device,
        )
        for row_index, next_id in enumerate(next_ids):
            logits[row_index, -1, next_id] = 0
        return logits


class _RngConsumingLogitsModel(_FixedLogitsModel):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        random.random()
        np.random.random()
        return super().forward(token_ids)


class _VariableLogitsModel(torch.nn.Module):
    def __init__(self, logits_by_last_token: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.max_seq_len = 8
        self.logits_by_last_token = logits_by_last_token

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        rows = [
            self.logits_by_last_token[int(token_id)]
            for token_id in token_ids[:, -1].detach().cpu().tolist()
        ]
        next_logits = torch.stack(rows).to(token_ids.device)
        return next_logits[:, None, :].expand(
            token_ids.shape[0],
            token_ids.shape[1],
            -1,
        )


def test_generate_crops_each_forward_pass_and_appends_exact_token_count() -> None:
    torch.manual_seed(7)
    model = GPT(_model_config()).eval()
    prompt = torch.tensor([[0, 1, 2, 3, 4]])
    observed_contexts: list[torch.Tensor] = []

    def record_context(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        observed_contexts.append(inputs[0].detach().clone())

    hook = model.register_forward_pre_hook(record_context)
    try:
        generated = generate(
            model,
            prompt,
            max_new_tokens=3,
            temperature=0.0,
        )
    finally:
        hook.remove()

    assert generated.shape == (1, prompt.shape[1] + 3)
    assert torch.equal(generated[:, : prompt.shape[1]], prompt)
    assert len(observed_contexts) == 3
    for step, context in enumerate(observed_contexts):
        current_length = prompt.shape[1] + step
        expected = generated[
            :,
            max(0, current_length - model.max_seq_len) : current_length,
        ]
        assert torch.equal(context, expected)


def test_temperature_zero_is_greedy_and_nonzero_temperature_samples() -> None:
    model = _FixedLogitsModel(torch.linspace(1.0, 0.0, steps=32))
    prompt = torch.tensor([[4]])

    greedy = generate(
        model,
        prompt,
        max_new_tokens=24,
        temperature=0.0,
        seed=11,
    )
    sampled = generate(
        model,
        prompt,
        max_new_tokens=24,
        temperature=10.0,
        seed=11,
    )

    assert torch.equal(greedy[:, 1:], torch.zeros((1, 24), dtype=torch.long))
    assert sampled[:, 1:].ne(0).any()


def test_top_k_limits_sampling_to_the_highest_scoring_tokens() -> None:
    model = _FixedLogitsModel(torch.arange(32.0, 0.0, step=-1.0))

    generated = generate(
        model,
        torch.tensor([[4]]),
        max_new_tokens=24,
        temperature=2.0,
        top_k=2,
        seed=17,
    )

    assert set(generated[0, 1:].tolist()) == {0, 1}


def test_seeded_sampling_is_reproducible_without_using_global_rng_state() -> None:
    model = _FixedLogitsModel(torch.zeros(32))
    prompt = torch.tensor([[4]])

    first = generate(
        model,
        prompt,
        max_new_tokens=16,
        temperature=1.0,
        seed=23,
    )
    torch.rand(50)
    repeated = generate(
        model,
        prompt,
        max_new_tokens=16,
        temperature=1.0,
        seed=23,
    )
    different_seed = generate(
        model,
        prompt,
        max_new_tokens=16,
        temperature=1.0,
        seed=24,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different_seed)


def test_explicit_row_seeds_match_independent_generation_streams() -> None:
    sampling_logits = torch.full((32,), -torch.inf)
    sampling_logits[20:22] = 0
    model = _VariableLogitsModel(
        {
            10: sampling_logits,
            20: sampling_logits,
            21: sampling_logits,
        }
    )

    batched = generate_sequences(
        model,
        torch.tensor([[10], [10]]),
        max_new_tokens=6,
        temperature=1,
        row_seeds=(101, 202),
    )
    independent = tuple(
        generate_sequences(
            model,
            torch.tensor([[10]]),
            max_new_tokens=6,
            temperature=1,
            seed=seed,
        ).sequences[0]
        for seed in (101, 202)
    )

    assert batched.sequences == independent


def test_explicit_row_seeds_require_one_seed_per_row_and_no_shared_seed() -> None:
    model = _FixedLogitsModel(torch.zeros(32))
    prompts = torch.tensor([[10], [10]])

    with pytest.raises(ValueError, match="exactly 2"):
        generate_sequences(
            model,
            prompts,
            max_new_tokens=1,
            row_seeds=(101,),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_sequences(
            model,
            prompts,
            max_new_tokens=1,
            seed=42,
            row_seeds=(101, 202),
        )


def test_generated_ids_stay_in_vocab_and_byte_decode_without_error() -> None:
    tokenizer = ByteTokenizer()
    model = GPT(_model_config(vocab_size=VOCAB_SIZE, seq_len=4))
    prompt = torch.tensor([tokenizer.encode("Hello")])

    generated = generate(
        model,
        prompt,
        max_new_tokens=12,
        temperature=0.8,
        top_k=50,
        seed=29,
    )
    generated_ids = generated[0].tolist()

    assert all(0 <= token_id < tokenizer.get_vocab_size() for token_id in generated_ids)
    assert isinstance(tokenizer.decode(generated_ids), str)


def test_explicit_stop_set_handles_immediate_mid_and_fallback_rows() -> None:
    model = _TransitionLogitsModel(
        {
            10: 31,
            11: 12,
            12: 31,
            20: 21,
            21: 22,
            22: 23,
        }
    )

    generated = generate(
        model,
        torch.tensor([[10], [11], [20]]),
        max_new_tokens=3,
        temperature=0,
        stop_token_ids={31},
    )

    assert isinstance(generated, GenerationBatchResult)
    assert tuple(sequence.token_ids for sequence in generated.sequences) == (
        (10,),
        (11, 12),
        (20, 21, 22, 23),
    )
    assert tuple(sequence.generated_token_ids for sequence in generated.sequences) == (
        (),
        (12,),
        (21, 22, 23),
    )
    assert tuple(sequence.completion_reason for sequence in generated.sequences) == (
        "stop_token",
        "stop_token",
        "max_new_tokens",
    )
    assert tuple(sequence.stop_token_id for sequence in generated.sequences) == (
        31,
        31,
        None,
    )
    assert tuple(sequence.sampled_token_count for sequence in generated.sequences) == (
        1,
        2,
        3,
    )

    fallback_alone = generate(
        model,
        torch.tensor([[20]]),
        max_new_tokens=3,
        temperature=0,
        stop_token_ids={31},
    )
    assert isinstance(fallback_alone, GenerationBatchResult)
    assert (
        fallback_alone.sequences[0].generated_token_ids
        == generated.sequences[2].generated_token_ids
    )


def test_stream_generate_sequence_yields_visible_tokens_then_completion() -> None:
    model = _TransitionLogitsModel({10: 11, 11: 12, 12: 31})

    events = tuple(
        stream_generate_sequence(
            model,
            torch.tensor([[10]]),
            max_new_tokens=4,
            temperature=0,
            stop_token_ids={31},
        )
    )

    assert events[:-1] == (
        GeneratedToken(token_id=11, generated_token_count=1, sampled_token_count=1),
        GeneratedToken(token_id=12, generated_token_count=2, sampled_token_count=2),
    )
    assert isinstance(events[-1], GenerationComplete)
    assert events[-1].sequence.generated_token_ids == (11, 12)
    assert events[-1].sequence.completion_reason == "stop_token"
    assert events[-1].sequence.stop_token_id == 31
    assert events[-1].sequence.sampled_token_count == 3


def test_closing_stream_restores_model_modes_and_caller_rng_state() -> None:
    model = _RngConsumingLogitsModel(torch.zeros(32))
    model.train()
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    python_state = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    torch_state = torch.get_rng_state().clone()
    stream = stream_generate_sequence(
        model,
        torch.tensor([[4]]),
        max_new_tokens=4,
        temperature=1.0,
        seed=44,
    )

    first = next(stream)
    assert isinstance(first, GeneratedToken)
    assert model.training is False
    stream.close()

    assert model.training is True
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state(legacy=True)
    assert restored_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_finished_rows_do_not_change_an_unfinished_rows_sampling_stream() -> None:
    stop_logits = torch.full((32,), -torch.inf)
    stop_logits[31] = 0
    sampling_logits = torch.full((32,), -torch.inf)
    sampling_logits[20:22] = 0
    model = _VariableLogitsModel(
        {
            10: stop_logits,
            20: sampling_logits,
            21: sampling_logits,
        }
    )

    batched = generate(
        model,
        torch.tensor([[10], [20]]),
        max_new_tokens=6,
        temperature=1,
        seed=43,
        stop_token_ids={31},
    )
    alone = generate(
        model,
        torch.tensor([[20]]),
        max_new_tokens=6,
        temperature=1,
        seed=43,
        stop_token_ids={31},
    )

    assert isinstance(batched, GenerationBatchResult)
    assert isinstance(alone, GenerationBatchResult)
    assert (
        batched.sequences[1].generated_token_ids
        == alone.sequences[0].generated_token_ids
    )


def test_stop_sets_are_generic_for_future_chat_termination() -> None:
    model = _TransitionLogitsModel({7: 30})

    generated = generate(
        model,
        torch.tensor([[7]]),
        max_new_tokens=4,
        temperature=0,
        stop_token_ids={30, 31},
    )

    assert isinstance(generated, GenerationBatchResult)
    assert generated.sequences[0].generated_token_ids == ()
    assert generated.sequences[0].stop_token_id == 30


def test_stopped_generation_restores_modes_and_caller_rng_state() -> None:
    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    model = _RngConsumingLogitsModel(torch.zeros(32)).train()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    generated = generate(
        model,
        torch.tensor([[10]]),
        max_new_tokens=3,
        temperature=1,
        stop_token_ids=set(),
    )

    assert isinstance(generated, GenerationBatchResult)
    assert model.training
    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
