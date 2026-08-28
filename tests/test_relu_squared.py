"""ReLU-squared MLP activation option and bounded evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from scratch_llm.config import ConfigValidationError, GPTConfig
from scratch_llm.diagnostics.resource_estimation import estimate_gpt_model_size
from scratch_llm.model import GPT, MLP, ReLUSquared, build_activation


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
        "bias": False,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def test_relu_squared_matches_hand_computed_forward_and_derivative() -> None:
    inputs = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)

    output = ReLUSquared()(inputs)
    output.sum().backward()

    torch.testing.assert_close(output, torch.tensor([0.0, 0.0, 9.0]))
    torch.testing.assert_close(inputs.grad, torch.tensor([0.0, 0.0, 6.0]))


@pytest.mark.parametrize("shape", [(1, 1, 4), (2, 5, 8), (3, 2, 12)])
def test_relu_squared_preserves_arbitrary_token_tensor_shapes(
    shape: tuple[int, ...],
) -> None:
    inputs = torch.linspace(-2, 2, steps=torch.Size(shape).numel()).reshape(shape)

    actual = ReLUSquared()(inputs)

    assert actual.shape == inputs.shape
    assert actual.dtype == inputs.dtype
    torch.testing.assert_close(actual, torch.relu(inputs).square())


def test_activation_factory_is_explicit_and_config_rejects_unknown_values() -> None:
    assert isinstance(build_activation(_config()), nn.GELU)
    assert isinstance(build_activation(_config(activation="relu_squared")), ReLUSquared)
    with pytest.raises(ConfigValidationError) as error:
        _config(activation="relu2")
    assert error.value.path == "model.activation"


def test_switch_changes_no_parameter_state_width_or_initialization() -> None:
    torch.manual_seed(503)
    gelu = GPT(_config(activation="gelu"))
    torch.manual_seed(503)
    relu_squared = GPT(_config(activation="relu_squared"))

    assert set(gelu.state_dict()) == set(relu_squared.state_dict())
    for key, expected in gelu.state_dict().items():
        torch.testing.assert_close(
            relu_squared.state_dict()[key], expected, rtol=0, atol=0
        )
    assert sum(parameter.numel() for parameter in gelu.parameters()) == sum(
        parameter.numel() for parameter in relu_squared.parameters()
    )
    for first, second in zip(gelu.blocks, relu_squared.blocks, strict=True):
        assert first.mlp.in_proj.out_features == second.mlp.in_proj.out_features
        assert first.mlp.out_proj.in_features == second.mlp.out_proj.in_features
        assert first.mlp.dropout.p == second.mlp.dropout.p


def test_implicit_and_explicit_gelu_preserve_exact_logits() -> None:
    torch.manual_seed(509)
    implicit = GPT(_config()).eval()
    torch.manual_seed(509)
    explicit = GPT(_config(activation="gelu")).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])

    with torch.inference_mode():
        torch.testing.assert_close(implicit(tokens), explicit(tokens), rtol=0, atol=0)


class _RecordingTransform(nn.Module):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__()
        self.name = name
        self.events = events

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.events.append(self.name)
        return x


@pytest.mark.parametrize("activation", ["gelu", "relu_squared"])
def test_activation_stays_between_existing_projections_and_dropout(
    activation: str,
) -> None:
    mlp = MLP(_config(activation=activation))
    events: list[str] = []
    mlp.in_proj = _RecordingTransform("in_proj", events)  # type: ignore[assignment]
    mlp.activation = _RecordingTransform("activation", events)
    mlp.out_proj = _RecordingTransform("out_proj", events)  # type: ignore[assignment]
    mlp.dropout = _RecordingTransform("dropout", events)  # type: ignore[assignment]

    output = mlp(torch.ones(2, 3, mlp.n_embd))

    assert output.shape == (2, 3, mlp.n_embd)
    assert events == ["in_proj", "activation", "out_proj", "dropout"]


@pytest.mark.parametrize("activation", ["gelu", "relu_squared"])
def test_training_eval_dropout_contract_is_unchanged(activation: str) -> None:
    torch.manual_seed(521)
    mlp = MLP(_config(activation=activation, dropout=0.5))
    inputs = torch.ones(4, 5, mlp.n_embd)

    mlp.eval()
    torch.manual_seed(1)
    first_eval = mlp(inputs)
    torch.manual_seed(2)
    second_eval = mlp(inputs)
    mlp.train()
    torch.manual_seed(1)
    first_train = mlp(inputs)
    torch.manual_seed(2)
    second_train = mlp(inputs)

    torch.testing.assert_close(first_eval, second_eval)
    assert not torch.equal(first_train, second_train)


@pytest.mark.parametrize("activation", ["gelu", "relu_squared"])
def test_activation_checkpoint_round_trip_and_identity(
    tmp_path: Path,
    activation: str,
) -> None:
    torch.manual_seed(523)
    config = _config(activation=activation)
    source = GPT(config).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = source(inputs)
    checkpoint = tmp_path / f"{activation}.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = GPT(config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    assert config.parameter_compatibility_dict()["activation"] == activation


@pytest.mark.parametrize("activation", ["gelu", "relu_squared"])
def test_activation_modes_have_finite_gradients_and_tiny_overfit(
    activation: str,
) -> None:
    torch.manual_seed(541)
    model = GPT(
        _config(
            activation=activation,
            vocab_size=16,
            seq_len=6,
            n_layer=1,
            n_embd=16,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])

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


def test_resource_report_records_activation_without_parameter_delta() -> None:
    gelu = estimate_gpt_model_size(_config(activation="gelu")).to_dict()
    relu_squared = estimate_gpt_model_size(_config(activation="relu_squared")).to_dict()

    assert gelu["activation"] == "gelu"
    assert relu_squared["activation"] == "relu_squared"
    assert gelu["unique_parameters"] == relu_squared["unique_parameters"]
    assert gelu["component_parameters"] == relu_squared["component_parameters"]


def test_relu_squared_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-4-relu2"
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### ReLU-squared MLP" in readme
    assert "relu(x).square()" in readme
    assert "experimental" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.activation",
        "run.name",
    ]
    assert payload["deltas"]["unique_parameters"] == 0
    assert payload["runs"]["gelu"]["activation"] == "gelu"
    assert payload["runs"]["relu_squared"]["activation"] == "relu_squared"
    assert payload["interpretation"]["all_training_telemetry_finite"] is True
    assert len(offline["runs"]) == 2
