"""Regression tests for the explicit causal self-attention implementation."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from scratch_llm.attention import CausalSelfAttention
from scratch_llm.config import ConfigValidationError, GPTConfig


def _attention_config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "seq_len": 6,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 4,
        "dropout": 0.0,
        "bias": True,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def _reference_attention(module: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Compute attention one query position at a time for an independent oracle."""

    batch_size, sequence_length, channels = x.shape
    q, k, v = F.linear(x, module.qkv.weight, module.qkv.bias).chunk(3, dim=-1)
    q = q.view(batch_size, sequence_length, module.n_head, module.head_dim)
    k = k.view(batch_size, sequence_length, module.n_head, module.head_dim)
    v = v.view(batch_size, sequence_length, module.n_head, module.head_dim)

    context = torch.empty_like(q)
    scale = math.sqrt(module.head_dim)
    for batch_index in range(batch_size):
        for head_index in range(module.n_head):
            for query_index in range(sequence_length):
                allowed_keys = k[batch_index, : query_index + 1, head_index]
                scores = (
                    q[batch_index, query_index, head_index] @ allowed_keys.T
                ) / scale
                weights = torch.softmax(scores, dim=-1)
                context[batch_index, query_index, head_index] = (
                    weights[:, None] * v[batch_index, : query_index + 1, head_index]
                ).sum(dim=0)

    merged = context.reshape(batch_size, sequence_length, channels)
    return F.linear(merged, module.out_proj.weight, module.out_proj.bias)


def test_attention_preserves_shape_and_has_finite_gradients() -> None:
    module = CausalSelfAttention(_attention_config(n_embd=8, n_head=4))
    x = torch.randn(2, 5, 8, requires_grad=True)

    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_attention_matches_a_position_by_position_reference() -> None:
    module = CausalSelfAttention(_attention_config(seq_len=3))
    x = torch.tensor(
        [
            [
                [0.2, -0.1, 0.4, 0.3],
                [0.0, 0.5, -0.2, 0.1],
                [0.7, -0.3, 0.2, -0.4],
            ]
        ]
    )

    with torch.no_grad():
        module.qkv.weight.copy_(
            torch.arange(48, dtype=x.dtype).reshape(12, 4) / 50 - 0.4
        )
        assert module.qkv.bias is not None
        module.qkv.bias.copy_(torch.linspace(-0.2, 0.2, 12))
        module.out_proj.weight.copy_(
            torch.tensor(
                [
                    [0.4, -0.2, 0.1, 0.3],
                    [-0.1, 0.5, 0.2, -0.4],
                    [0.3, 0.1, -0.5, 0.2],
                    [0.2, -0.3, 0.4, 0.1],
                ]
            )
        )
        assert module.out_proj.bias is not None
        module.out_proj.bias.copy_(torch.tensor([0.05, -0.1, 0.15, -0.2]))

    expected = _reference_attention(module, x)

    torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)


def test_future_tokens_cannot_change_earlier_outputs() -> None:
    torch.manual_seed(7)
    module = CausalSelfAttention(_attention_config(n_embd=8, n_head=2))
    original = torch.randn(1, 6, 8)
    changed = original.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100

    original_output = module(original)
    changed_output = module(changed)

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_attention_revalidates_head_dimensions_when_constructed() -> None:
    config = _attention_config(n_embd=8, n_head=2)
    config.n_head = 3

    with pytest.raises(ConfigValidationError, match="model.n_embd:.*divisible"):
        CausalSelfAttention(config)


def test_attention_rejects_sequences_longer_than_its_context() -> None:
    module = CausalSelfAttention(_attention_config(seq_len=4, n_embd=8))

    with pytest.raises(
        ValueError, match="sequence length 5 exceeds configured context length 4"
    ):
        module(torch.randn(2, 5, 8))
