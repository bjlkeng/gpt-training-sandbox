"""Gated alternating token-value embedding coverage."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from scratch_llm.attention import CausalSelfAttention, mix_token_value_embeddings
from scratch_llm.config import (
    ConfigValidationError,
    GPTConfig,
    ProjectConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.resource_estimation import (
    estimate_gpt_model_size,
    estimate_training_resources,
    summarize_module_parameters,
)
from scratch_llm.model import GPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 24,
        "seq_len": 8,
        "n_layer": 3,
        "n_head": 4,
        "n_kv_head": 2,
        "n_embd": 16,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "bias": False,
        "use_gqa": True,
        "use_value_embeddings": True,
        "value_embedding_gate_channels": 4,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (1, (0,)),
        (2, (1,)),
        (3, (0, 2)),
        (4, (1, 3)),
        (5, (0, 2, 4)),
    ],
)
def test_value_embedding_placement_alternates_with_final_layer_parity(
    depth: int,
    expected: tuple[int, ...],
) -> None:
    enabled = _config(n_layer=depth)
    disabled = _config(n_layer=depth, use_value_embeddings=False)

    assert enabled.value_embedding_layer_indices() == expected
    assert disabled.value_embedding_layer_indices() == ()
    assert expected[-1] == depth - 1


def test_value_embedding_config_validates_gate_bounds_and_identity() -> None:
    for overrides, path in [
        ({"use_value_embeddings": "yes"}, "model.use_value_embeddings"),
        ({"value_embedding_gate_channels": 0}, "model.value_embedding_gate_channels"),
        (
            {"value_embedding_gate_channels": True},
            "model.value_embedding_gate_channels",
        ),
        ({"value_embedding_gate_channels": 17}, "model.value_embedding_gate_channels"),
    ]:
        with pytest.raises(ConfigValidationError) as error:
            _config(**overrides)
        assert error.value.path == path

    tiny_disabled = GPTConfig(
        vocab_size=12,
        seq_len=4,
        n_layer=1,
        n_head=2,
        n_embd=8,
    )
    assert tiny_disabled.value_embedding_gate_channels == 12
    assert tiny_disabled.value_embedding_layer_indices() == ()
    assert _config().value_embedding_identity() == {
        "enabled": True,
        "gate_channels": 4,
        "gate_scale": 3.0,
        "kv_width": 8,
        "layer_indices": [0, 2],
        "placement": "alternating_by_final_layer_parity",
    }


def test_hand_computed_value_mix_uses_one_sigmoid_gate_per_kv_head() -> None:
    projected = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    embedded = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    gate_logits = torch.zeros(1, 1, 2)

    actual = mix_token_value_embeddings(projected, embedded, gate_logits)

    torch.testing.assert_close(
        actual,
        torch.tensor([[[[4.0, 8.0]], [[12.0, 16.0]]]]),
    )


def test_disabled_mode_adds_no_modules_parameters_or_state_and_is_exact() -> None:
    disabled = _config(use_value_embeddings=False)
    torch.manual_seed(811)
    first = GPT(disabled)
    torch.manual_seed(811)
    second = GPT(replace(disabled))

    assert all(block.attn.value_embedding is None for block in first.blocks)
    assert all(block.attn.value_gate is None for block in first.blocks)
    assert not any("value_embedding" in key for key in first.state_dict())
    assert not any("value_gate" in key for key in first.state_dict())
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(first(tokens), second(tokens), rtol=0, atol=0)


def test_enabled_layers_own_compact_embedding_and_gate_with_pinned_init() -> None:
    torch.manual_seed(821)
    first = GPT(_config())
    torch.manual_seed(821)
    second = GPT(_config())
    bound = math.sqrt(3.0 / first.config.n_embd)

    for layer_index, block in enumerate(first.blocks):
        if layer_index in (0, 2):
            assert isinstance(block.attn.value_embedding, nn.Embedding)
            assert isinstance(block.attn.value_gate, nn.Linear)
            assert block.attn.value_embedding.weight.shape == (24, 8)
            assert block.attn.value_gate.weight.shape == (2, 4)
            assert block.attn.value_gate.bias is None
            assert torch.all(block.attn.value_embedding.weight >= -bound)
            assert torch.all(block.attn.value_embedding.weight <= bound)
            assert torch.all(block.attn.value_gate.weight >= 0.0)
            assert torch.all(block.attn.value_gate.weight <= 0.02)
        else:
            assert block.attn.value_embedding is None
            assert block.attn.value_gate is None
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_attention_changes_only_projected_values_before_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(823)
    module = CausalSelfAttention(_config(n_layer=1), layer_index=0)
    inputs = torch.randn(2, 3, 16)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    projected = module.qkv_projection(inputs)
    query_width = module.n_head * module.head_dim
    kv_width = module.n_kv_head * module.head_dim
    query, key, original_value = projected.split(
        (query_width, kv_width, kv_width), dim=-1
    )
    query = query.view(2, 3, module.n_head, module.head_dim).transpose(1, 2)
    key = key.view(2, 3, module.n_kv_head, module.head_dim).transpose(1, 2)
    original_value = original_value.view(
        2, 3, module.n_kv_head, module.head_dim
    ).transpose(1, 2)
    captured: dict[str, torch.Tensor] = {}

    def capture(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        captured.update(q=q, k=k, v=v)
        return torch.zeros_like(q)

    monkeypatch.setattr(module, "_manual_attention", capture)
    module(inputs, token_ids=token_ids)

    torch.testing.assert_close(captured["q"], query)
    torch.testing.assert_close(captured["k"], key)
    assert module.value_embedding is not None
    assert module.value_gate is not None
    expected_value = mix_token_value_embeddings(
        original_value,
        module.value_embedding(token_ids).view(2, 3, 2, 4),
        module.value_gate(inputs[..., :4]),
    )
    torch.testing.assert_close(captured["v"], expected_value)


@pytest.mark.parametrize("n_kv_head", [4, 2])
def test_manual_and_sdpa_forward_and_gradients_match_with_value_embeddings(
    n_kv_head: int,
) -> None:
    base = _config(
        n_layer=2,
        n_kv_head=n_kv_head,
        use_gqa=n_kv_head < 4,
        attention_backend="manual",
    )
    torch.manual_seed(827)
    manual = GPT(base)
    sdpa = GPT(replace(base, attention_backend="sdpa"))
    sdpa.load_state_dict(manual.state_dict(), strict=True)
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    targets = torch.tensor([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])

    manual_loss = manual(tokens, targets)
    sdpa_loss = sdpa(tokens, targets)
    manual_loss.backward()
    sdpa_loss.backward()

    torch.testing.assert_close(manual_loss, sdpa_loss, rtol=3e-5, atol=1e-6)
    for (manual_name, manual_parameter), (sdpa_name, sdpa_parameter) in zip(
        manual.named_parameters(), sdpa.named_parameters(), strict=True
    ):
        assert manual_name == sdpa_name
        torch.testing.assert_close(
            manual_parameter.grad,
            sdpa_parameter.grad,
            rtol=5e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("n_kv_head", [4, 2])
def test_full_prefill_and_cached_decode_share_token_indexed_values(
    backend: str,
    n_kv_head: int,
) -> None:
    torch.manual_seed(829)
    config = _config(
        n_layer=3,
        n_kv_head=n_kv_head,
        use_gqa=n_kv_head < 4,
        attention_backend=backend,
        use_kv_cache=True,
        sliding_window_pattern="S",
        sliding_window_size=2,
    )
    model = GPT(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.inference_mode():
        full = model(tokens)
        cache = model.create_kv_cache(batch_size=1, capacity=tokens.shape[1])
        pieces = [model(tokens[:, :3], kv_cache=cache)]
        pieces.extend(
            model(tokens[:, index : index + 1], kv_cache=cache)
            for index in range(3, tokens.shape[1])
        )

    torch.testing.assert_close(
        torch.cat(pieces, dim=1),
        full,
        rtol=4e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    ("depth", "n_kv_head"),
    [(1, 4), (3, 2), (4, 4)],
)
def test_shapes_gradients_and_dtype_cover_depth_parity_mha_and_gqa(
    depth: int,
    n_kv_head: int,
) -> None:
    torch.manual_seed(839)
    config = _config(
        n_layer=depth,
        n_kv_head=n_kv_head,
        use_gqa=n_kv_head < 4,
    )
    model = GPT(config).to(dtype=torch.float64)
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]])

    logits = model(tokens)
    loss = model(tokens, targets)
    loss.backward()

    assert logits.shape == (2, 4, config.vocab_size)
    assert logits.dtype == torch.float64
    for layer_index in config.value_embedding_layer_indices():
        attention = model.blocks[layer_index].attn
        assert attention.value_embedding is not None
        assert attention.value_gate is not None
        assert attention.value_embedding.weight.dtype == torch.float64
        assert attention.value_gate.weight.dtype == torch.float64
        assert attention.value_embedding.weight.grad is not None
        assert attention.value_gate.weight.grad is not None
        assert torch.isfinite(attention.value_embedding.weight.grad).all()
        assert torch.isfinite(attention.value_gate.weight.grad).all()


def test_parameters_optimizer_and_checkpoint_contracts_are_exact() -> None:
    disabled_config = _config(use_value_embeddings=False)
    enabled_config = _config()
    disabled = GPT(disabled_config)
    enabled = GPT(enabled_config)
    expected_embedding_parameters = 2 * 24 * 8
    expected_gate_parameters = 2 * 4 * 2
    expected_delta = expected_embedding_parameters + expected_gate_parameters
    disabled_count = summarize_module_parameters(disabled).unique_parameters
    enabled_count = summarize_module_parameters(enabled).unique_parameters
    estimate = estimate_gpt_model_size(enabled_config)

    assert enabled_count - disabled_count == expected_delta
    assert estimate.value_embedding_parameters == expected_embedding_parameters
    assert estimate.value_embedding_gate_parameters == expected_gate_parameters
    assert estimate.unique_parameters == enabled_count
    optimizer = torch.optim.AdamW(enabled.parameters(), lr=0.001)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids == {
        id(parameter) for parameter in enabled.parameters()
    }

    restored = GPT(enabled_config)
    restored.load_state_dict(enabled.state_dict(), strict=True)
    assert set(restored.state_dict()) == set(enabled.state_dict())
    with pytest.raises(RuntimeError, match="value_(embedding|gate)"):
        disabled.load_state_dict(enabled.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="value_(embedding|gate)"):
        enabled.load_state_dict(disabled.state_dict(), strict=True)


def test_training_resource_identity_includes_exact_value_embedding_placement() -> None:
    config = _config(vocab_size=32_768)
    train = TrainConfig(
        device_batch_size=1,
        total_batch_size_tokens=config.seq_len,
        grad_accum_steps=1,
        max_steps=1,
        warmup_steps=0,
        warmdown_ratio=0.0,
    )
    result = estimate_training_resources(
        ProjectConfig(
            model=config,
            train=train,
        )
    )
    disabled = estimate_training_resources(
        ProjectConfig(model=replace(config, use_value_embeddings=False), train=train)
    )
    payload = result.to_dict()

    assert payload["model"]["value_embeddings"] == config.value_embedding_identity()
    assert result.memory.parameter_bytes == result.model.unique_parameters * 4
    assert result.memory.activation_bytes - disabled.memory.activation_bytes == 768


def test_value_embedding_model_overfits_one_tiny_batch() -> None:
    torch.manual_seed(853)
    config = _config(vocab_size=16, seq_len=6, n_layer=2)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])

    for _ in range(70):
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.inference_mode():
        assert model(inputs, targets).item() < 0.12


def test_value_embedding_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = (
        PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-9-value-embeddings"
    )
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Gated value embeddings" in readme
    assert "experimental" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.use_value_embeddings",
        "run.name",
    ]
    assert payload["runs"]["off"]["layer_indices"] == []
    assert payload["runs"]["on"]["layer_indices"] == [1]
    assert (
        payload["runs"]["on"]["unique_parameters"]
        > (payload["runs"]["off"]["unique_parameters"])
    )
    assert len(offline["runs"]) == 2
    identities = {
        run["run"]: run["identities"]["parameterization"]["value_embeddings"]
        for run in offline["runs"]
    }
    assert identities["m10-value-off"]["layer_indices"] == []
    assert identities["m10-value-on"]["layer_indices"] == [1]
