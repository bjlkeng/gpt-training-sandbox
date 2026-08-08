"""Regression tests for the explicit causal self-attention implementation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scratch_llm.attention import CausalSelfAttention
from scratch_llm.config import ConfigValidationError, GPTConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize("sequence_length", [1, 6])
def test_sdpa_matches_manual_forward_and_backward(
    sequence_length: int,
) -> None:
    torch.manual_seed(19)
    manual = CausalSelfAttention(_attention_config(attention_backend="manual"))
    sdpa = CausalSelfAttention(_attention_config(attention_backend="sdpa"))
    sdpa.load_state_dict(manual.state_dict())
    manual_input = torch.randn(2, sequence_length, 4, requires_grad=True)
    sdpa_input = manual_input.detach().clone().requires_grad_(True)

    manual_output = manual(manual_input)
    sdpa_output = sdpa(sdpa_input)
    manual_output.square().mean().backward()
    sdpa_output.square().mean().backward()

    assert set(manual.state_dict()) == set(sdpa.state_dict())
    torch.testing.assert_close(sdpa_output, manual_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        sdpa_input.grad,
        manual_input.grad,
        rtol=2e-5,
        atol=1e-6,
    )
    for (_, manual_parameter), (_, sdpa_parameter) in zip(
        manual.named_parameters(),
        sdpa.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            sdpa_parameter.grad,
            manual_parameter.grad,
            rtol=2e-5,
            atol=1e-6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_reduced_precision_sdpa_parity(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    torch.manual_seed(41)
    manual = CausalSelfAttention(
        _attention_config(attention_backend="manual", n_embd=8, n_head=2)
    ).to(device="cuda", dtype=dtype)
    sdpa = CausalSelfAttention(
        _attention_config(attention_backend="sdpa", n_embd=8, n_head=2)
    ).to(device="cuda", dtype=dtype)
    sdpa.load_state_dict(manual.state_dict())
    manual_input = torch.randn(
        2,
        6,
        8,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    sdpa_input = manual_input.detach().clone().requires_grad_(True)

    manual_output = manual(manual_input)
    sdpa_output = sdpa(sdpa_input)
    manual_output.float().square().mean().backward()
    sdpa_output.float().square().mean().backward()

    torch.testing.assert_close(sdpa_output, manual_output, rtol=0.05, atol=0.005)
    torch.testing.assert_close(
        sdpa_input.grad,
        manual_input.grad,
        rtol=0.08,
        atol=0.005,
    )
    for (_, manual_parameter), (_, sdpa_parameter) in zip(
        manual.named_parameters(),
        sdpa.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            sdpa_parameter.grad,
            manual_parameter.grad,
            rtol=0.08,
            atol=0.005,
        )


def test_sdpa_path_has_no_materialized_square_causal_mask() -> None:
    manual = CausalSelfAttention(_attention_config(attention_backend="manual"))
    sdpa = CausalSelfAttention(_attention_config(attention_backend="sdpa"))

    assert manual.causal_mask.shape == (6, 6)
    assert sdpa.causal_mask is None


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_each_backend_preserves_causality(backend: str) -> None:
    torch.manual_seed(23)
    module = CausalSelfAttention(_attention_config(attention_backend=backend))
    original = torch.randn(1, 6, 4)
    changed = original.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100

    torch.testing.assert_close(module(original)[:, :3], module(changed)[:, :3])


def test_sdpa_uses_causal_mode_and_training_only_dropout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    real_sdpa = F.scaled_dot_product_attention

    def record_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(dict(kwargs))
        return real_sdpa(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(F, "scaled_dot_product_attention", record_sdpa)
    module = CausalSelfAttention(
        _attention_config(attention_backend="sdpa", dropout=0.25)
    )
    x = torch.randn(2, 4, 4)

    module.train()
    module(x)
    module.eval()
    module(x)

    assert calls == [
        {
            "attn_mask": None,
            "dropout_p": 0.25,
            "is_causal": True,
        },
        {
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": True,
        },
    ]


def test_nonzero_dropout_fixture_preserves_eval_parity_and_seeded_modes() -> None:
    torch.manual_seed(29)
    manual = CausalSelfAttention(
        _attention_config(attention_backend="manual", dropout=0.2)
    )
    sdpa = CausalSelfAttention(_attention_config(attention_backend="sdpa", dropout=0.2))
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(2, 5, 4)

    manual.eval()
    sdpa.eval()
    torch.testing.assert_close(sdpa(x), manual(x), rtol=1e-5, atol=1e-6)
    for module in (manual, sdpa):
        module.train()
        torch.manual_seed(31)
        first = module(x)
        torch.manual_seed(31)
        second = module(x)
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_readme_documents_sdpa_selection_parity_and_benchmark_identity() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "attention_backend: manual  # manual, sdpa, or flash" in readme
    assert "never materializes the manual square causal mask" in readme
    assert "Requested attention backend" in readme
    assert "--override model.attention_backend=sdpa" in readme
