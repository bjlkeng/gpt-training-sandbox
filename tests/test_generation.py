"""Tests for shared no-cache autoregressive generation."""

from __future__ import annotations

import torch

from scratch_llm.config import GPTConfig
from scratch_llm.generation import generate
from scratch_llm.model import GPT
from scratch_llm.tokenizer import VOCAB_SIZE, ByteTokenizer


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
