"""Tests for the baseline GPT feed-forward network and transformer block."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from scratch_llm.config import GPTConfig
from scratch_llm.model import Block, MLP


def _model_config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "seq_len": 8,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "dropout": 0.0,
        "bias": True,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "channels"),
    [(1, 1, 4), (2, 5, 8), (3, 2, 12)],
)
def test_mlp_preserves_batch_sequence_and_channel_shape(
    batch_size: int,
    sequence_length: int,
    channels: int,
) -> None:
    module = MLP(_model_config(n_embd=channels))
    x = torch.randn(batch_size, sequence_length, channels)

    assert module(x).shape == x.shape


def test_mlp_uses_configured_expansion_and_gelu() -> None:
    module = MLP(_model_config(n_embd=8, mlp_ratio=3, bias=False))

    assert module.in_proj.in_features == 8
    assert module.in_proj.out_features == 24
    assert isinstance(module.activation, nn.GELU)
    assert module.out_proj.in_features == 24
    assert module.out_proj.out_features == 8
    assert module.in_proj.bias is None
    assert module.out_proj.bias is None


def test_mlp_dropout_is_active_only_during_training() -> None:
    module = MLP(_model_config(dropout=0.5, bias=False))
    with torch.no_grad():
        module.in_proj.weight.fill_(0.125)
        module.out_proj.weight.fill_(0.125)
    x = torch.ones(4, 5, 8)

    module.eval()
    torch.manual_seed(1)
    first_eval = module(x)
    torch.manual_seed(2)
    second_eval = module(x)

    module.train()
    torch.manual_seed(1)
    first_train = module(x)
    torch.manual_seed(2)
    second_train = module(x)

    torch.testing.assert_close(first_eval, second_eval)
    assert not torch.equal(first_train, second_train)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "channels"),
    [(1, 1, 4), (2, 5, 8), (3, 2, 12)],
)
def test_block_preserves_batch_sequence_and_channel_shape(
    batch_size: int,
    sequence_length: int,
    channels: int,
) -> None:
    module = Block(_model_config(n_embd=channels, mlp_ratio=3))
    x = torch.randn(batch_size, sequence_length, channels)

    assert module(x).shape == x.shape


def _gradient_magnitude(module: nn.Module) -> float:
    return sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_both_block_residual_sublayers_participate_in_gradients() -> None:
    torch.manual_seed(7)
    module = Block(_model_config())
    x = torch.randn(2, 6, 8, requires_grad=True)

    module(x).square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert _gradient_magnitude(module.attn) > 0
    assert _gradient_magnitude(module.mlp) > 0


def test_block_dropout_is_active_only_during_training() -> None:
    torch.manual_seed(11)
    module = Block(_model_config(dropout=0.5))
    x = torch.randn(2, 6, 8)

    module.eval()
    torch.manual_seed(1)
    first_eval = module(x)
    torch.manual_seed(2)
    second_eval = module(x)

    module.train()
    torch.manual_seed(1)
    first_train = module(x)
    torch.manual_seed(2)
    second_train = module(x)

    torch.testing.assert_close(first_eval, second_eval)
    assert not torch.equal(first_train, second_train)


class _RecordingTransform(nn.Module):
    def __init__(
        self,
        name: str,
        events: list[tuple[str, torch.Tensor]],
        *,
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.scale = scale
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.events.append((self.name, x.detach().clone()))
        return self.scale * x + self.offset


def test_block_adds_both_residuals_with_pre_layernorm_ordering() -> None:
    block = Block(_model_config())
    assert isinstance(block.ln_1, nn.LayerNorm)
    assert isinstance(block.ln_2, nn.LayerNorm)

    events: list[tuple[str, torch.Tensor]] = []
    setattr(block, "ln_1", _RecordingTransform("ln_1", events, offset=1.0))
    setattr(block, "attn", _RecordingTransform("attn", events, scale=2.0))
    setattr(block, "ln_2", _RecordingTransform("ln_2", events, offset=3.0))
    setattr(block, "mlp", _RecordingTransform("mlp", events, scale=4.0))
    x = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)

    output = block(x)

    first_residual = x + 2.0 * (x + 1.0)
    expected = first_residual + 4.0 * (first_residual + 3.0)
    assert [name for name, _ in events] == ["ln_1", "attn", "ln_2", "mlp"]
    torch.testing.assert_close(events[0][1], x)
    torch.testing.assert_close(events[1][1], x + 1.0)
    torch.testing.assert_close(events[2][1], first_residual)
    torch.testing.assert_close(events[3][1], first_residual + 3.0)
    torch.testing.assert_close(output, expected)
