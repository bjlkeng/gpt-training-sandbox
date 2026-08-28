"""Architecture audit for the existing model-wide bias policy."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scratch_llm.config import ConfigValidationError, GPTConfig
from scratch_llm.diagnostics.resource_estimation import (
    estimate_gpt_model_size,
    summarize_module_parameters,
)
from scratch_llm.model import GPT, RMSNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 24,
        "seq_len": 8,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "mlp_ratio": 3,
        "dropout": 0.0,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def _linear_modules(model: GPT) -> dict[str, nn.Linear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }


def _expected_projection_names(layer_count: int) -> set[str]:
    return {
        *(f"blocks.{index}.attn.qkv_projection" for index in range(layer_count)),
        *(f"blocks.{index}.attn.out_proj" for index in range(layer_count)),
        *(f"blocks.{index}.mlp.in_proj" for index in range(layer_count)),
        *(f"blocks.{index}.mlp.out_proj" for index in range(layer_count)),
    }


def _expected_added_bias_keys(config: GPTConfig) -> set[str]:
    keys = {
        *(f"{name}.bias" for name in _expected_projection_names(config.n_layer)),
    }
    if config.norm == "layernorm":
        keys.update(
            f"blocks.{index}.ln_{position}.bias"
            for index in range(config.n_layer)
            for position in (1, 2)
        )
        keys.add("ln_f.bias")
    return keys


def test_bias_policy_rejects_non_boolean_config_values() -> None:
    with pytest.raises(ConfigValidationError) as error:
        _config(bias=1)
    assert error.value.path == "model.bias"


@pytest.mark.parametrize("bias", [False, True])
def test_every_projection_and_layernorm_obeys_the_documented_policy(
    bias: bool,
) -> None:
    config = _config(bias=bias)
    model = GPT(config)
    linears = _linear_modules(model)
    projection_names = _expected_projection_names(config.n_layer)

    assert set(linears) == projection_names | {"lm_head"}
    assert all((linears[name].bias is not None) is bias for name in projection_names)
    assert linears["lm_head"].bias is None
    norms = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    }
    assert len(norms) == 2 * config.n_layer + 1
    assert all((module.bias is not None) is bias for module in norms.values())
    assert all(module.weight is not None for module in norms.values())
    assert all(type(module) is nn.Linear for module in linears.values())


@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("tie_weights", [False, True])
@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
def test_parameter_delta_equals_full_audited_bias_inventory(
    layers: int,
    tie_weights: bool,
    norm: str,
) -> None:
    common = {
        "n_layer": layers,
        "tie_weights": tie_weights,
        "norm": norm,
        "use_rmsnorm": norm == "rmsnorm",
    }
    bias_free = GPT(_config(**common, bias=False))
    biased = GPT(_config(**common, bias=True))
    expected_keys = _expected_added_bias_keys(biased.config)

    assert set(biased.state_dict()) - set(bias_free.state_dict()) == expected_keys
    assert set(bias_free.state_dict()) - set(biased.state_dict()) == set()
    assert all(key.endswith(".bias") for key in expected_keys)
    expected_delta = sum(biased.state_dict()[key].numel() for key in expected_keys)
    assert (
        summarize_module_parameters(biased).unique_parameters
        - summarize_module_parameters(bias_free).unique_parameters
        == expected_delta
    )
    assert estimate_gpt_model_size(biased.config).unique_parameters == (
        summarize_module_parameters(biased).unique_parameters
    )
    assert estimate_gpt_model_size(bias_free.config).unique_parameters == (
        summarize_module_parameters(bias_free).unique_parameters
    )
    shared_keys = set(bias_free.state_dict())
    assert all(
        bias_free.state_dict()[key].shape == biased.state_dict()[key].shape
        for key in shared_keys
    )


def test_rmsnorm_has_no_hidden_bias_interaction() -> None:
    model = GPT(_config(norm="rmsnorm", use_rmsnorm=True, bias=True))

    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())
    assert all(
        len(list(module.parameters())) == 0
        for module in model.modules()
        if isinstance(module, RMSNorm)
    )
    assert _expected_added_bias_keys(model.config) == {
        *(f"{name}.bias" for name in _expected_projection_names(2)),
    }


@pytest.mark.parametrize("bias", [False, True])
def test_bias_modes_round_trip_deterministically_and_keep_forward_api(
    tmp_path: Path,
    bias: bool,
) -> None:
    torch.manual_seed(401)
    config = _config(bias=bias)
    source = GPT(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = source(tokens)
    path = tmp_path / f"bias-{bias}.pt"
    torch.save(source.state_dict(), path)

    restored = GPT(config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(tokens), expected, rtol=0, atol=0)
    assert inspect.signature(source.forward) == inspect.signature(restored.forward)
    assert config.parameter_compatibility_dict()["bias"] is bias


def test_bias_architecture_mismatch_fails_with_exact_missing_or_unexpected_keys() -> (
    None
):
    bias_free = GPT(_config(bias=False))
    biased = GPT(_config(bias=True))
    expected = _expected_added_bias_keys(biased.config)

    with pytest.raises(RuntimeError, match="Missing key.*attn.qkv_projection.bias"):
        biased.load_state_dict(bias_free.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Unexpected key.*attn.qkv_projection.bias"):
        bias_free.load_state_dict(biased.state_dict(), strict=True)
    assert set(biased.state_dict()) - set(bias_free.state_dict()) == expected


@pytest.mark.parametrize("bias", [False, True])
def test_bias_modes_are_causal_have_finite_gradients_and_overfit(bias: bool) -> None:
    torch.manual_seed(409)
    model = GPT(
        _config(
            vocab_size=16,
            seq_len=6,
            n_layer=1,
            n_embd=16,
            bias=bias,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])
    model.eval()
    with torch.inference_mode():
        original = model(inputs)
        changed = model(inputs.clone().index_fill(1, torch.tensor([4, 5]), 9))
    torch.testing.assert_close(original[:, :4], changed[:, :4])

    model.train()
    for _ in range(60):
        loss = model(inputs, targets)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.inference_mode():
        final_loss = model(inputs, targets)
    assert final_loss.item() < 0.1


def test_bias_audit_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-3-bias"
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Bias policy audit" in readme
    assert "baseline is already bias-free" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.bias",
        "run.name",
    ]
    assert payload["inventory"]["lm_head_bias"] is False
    assert payload["inventory"]["layernorm_bias_changes_with_flag"] is True
    assert payload["deltas"]["unique_parameters"] == 2_944
    assert payload["interpretation"]["all_training_telemetry_finite"] is True
    assert len(offline["runs"]) == 2
