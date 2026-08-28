"""Per-layer residual-stream and normalized-input scalar coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from scratch_llm.config import (
    ConfigValidationError,
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.resource_estimation import (
    estimate_gpt_model_size,
    estimate_training_resources,
    summarize_module_parameters,
)
from scratch_llm.model import GPT, normalize_initial_token_representation
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.training.checkpoint import load_model_checkpoint, save_checkpoint
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.telemetry import estimate_gpt_training_flops


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
        "use_residual_scalars": True,
        "residual_scalar_init": "neutral",
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def test_named_scalar_initializers_are_validated_and_exact() -> None:
    neutral = _config(n_layer=3)
    pinned_one = _config(n_layer=1, residual_scalar_init="nanochat_depth")
    pinned_many = _config(n_layer=5, residual_scalar_init="nanochat_depth")

    assert neutral.residual_scalar_initial_values() == (
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    )
    assert pinned_one.residual_scalar_initial_values() == ((1.15,), (0.2,))
    assert pinned_many.residual_scalar_initial_values() == (
        (1.15, 1.125, 1.0999999999999999, 1.075, 1.0499999999999998),
        (0.2, 0.1625, 0.125, 0.08750000000000002, 0.05000000000000002),
    )

    with pytest.raises(ConfigValidationError) as error:
        _config(residual_scalar_init="unknown")
    assert error.value.path == "model.residual_scalar_init"

    with pytest.raises(ConfigValidationError) as error:
        _config(use_residual_scalars="yes")
    assert error.value.path == "model.use_residual_scalars"


def test_disabled_mode_adds_no_parameters_or_state_and_preserves_exact_logits() -> None:
    disabled = _config(use_residual_scalars=False)
    torch.manual_seed(907)
    first = GPT(disabled)
    torch.manual_seed(907)
    second = GPT(replace(disabled))

    assert first.residual_scalars is None
    assert first.input_scalars is None
    assert not any("residual_scalars" in key for key in first.state_dict())
    assert not any("input_scalars" in key for key in first.state_dict())
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(first(tokens), second(tokens), rtol=0, atol=0)


def test_enabled_mode_creates_exactly_two_vectors_and_neutral_is_functional_noop() -> (
    None
):
    disabled_config = _config(use_residual_scalars=False)
    enabled_config = _config()
    torch.manual_seed(911)
    disabled = GPT(disabled_config)
    torch.manual_seed(911)
    enabled = GPT(enabled_config)
    missing, unexpected = enabled.load_state_dict(disabled.state_dict(), strict=False)

    assert missing == ["residual_scalars", "input_scalars"]
    assert unexpected == []
    assert isinstance(enabled.residual_scalars, nn.Parameter)
    assert isinstance(enabled.input_scalars, nn.Parameter)
    assert enabled.residual_scalars.shape == enabled.input_scalars.shape == (3,)
    torch.testing.assert_close(enabled.residual_scalars, torch.ones(3))
    torch.testing.assert_close(enabled.input_scalars, torch.zeros(3))
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    torch.testing.assert_close(enabled(tokens), disabled(tokens), rtol=0, atol=0)


class _RecordingBlock(nn.Module):
    def __init__(self, increment: float, records: list[torch.Tensor]) -> None:
        super().__init__()
        self.increment = increment
        self.records = records

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.records.append(x.detach().clone())
        return x + self.increment


def test_hand_computed_recurrence_runs_before_each_block_and_keeps_x0_fixed() -> None:
    config = _config(
        vocab_size=8,
        seq_len=3,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=4,
        use_gqa=False,
        use_value_embeddings=False,
    )
    model = GPT(config)
    assert model.residual_scalars is not None
    assert model.input_scalars is not None
    with torch.no_grad():
        model.token_embedding.weight.zero_()
        model.token_embedding.weight[1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert model.position_embedding is not None
        model.position_embedding.weight.zero_()
        model.residual_scalars.copy_(torch.tensor([2.0, 3.0]))
        model.input_scalars.copy_(torch.tensor([5.0, 7.0]))
    records: list[torch.Tensor] = []
    model.blocks = nn.ModuleList(
        [_RecordingBlock(10.0, records), _RecordingBlock(20.0, records)]
    )
    tokens = torch.tensor([[1, 1]])
    initial = model.token_embedding(tokens)
    x0 = normalize_initial_token_representation(initial)
    first_input = 2.0 * initial + 5.0 * x0
    second_input = 3.0 * (first_input + 10.0) + 7.0 * x0

    model(tokens)

    assert len(records) == 2
    torch.testing.assert_close(records[0], first_input)
    torch.testing.assert_close(records[1], second_input)
    torch.testing.assert_close(x0, normalize_initial_token_representation(initial))


@pytest.mark.parametrize("use_rope", [False, True])
def test_learned_positions_and_rope_match_full_prefill_and_cached_decode(
    use_rope: bool,
) -> None:
    torch.manual_seed(919)
    config = _config(
        use_rope=use_rope,
        use_kv_cache=True,
        use_value_embeddings=True,
        value_embedding_gate_channels=4,
        residual_scalar_init="nanochat_depth",
        attention_backend="sdpa",
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
        rtol=5e-5,
        atol=2e-6,
    )


def test_scalar_gradients_optimizer_counts_and_checkpoint_identity_are_exact() -> None:
    torch.manual_seed(929)
    config = _config(residual_scalar_init="nanochat_depth")
    model = GPT(config).to(dtype=torch.float64)
    assert model.residual_scalars is not None
    assert model.input_scalars is not None
    expected_residual, expected_input = config.residual_scalar_initial_values()
    torch.testing.assert_close(
        model.residual_scalars,
        torch.tensor(expected_residual, dtype=torch.float64),
    )
    torch.testing.assert_close(
        model.input_scalars,
        torch.tensor(expected_input, dtype=torch.float64),
    )
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]])
    model.set_activation_checkpointing(True)
    model(tokens, targets).backward()

    assert model.residual_scalars.grad is not None
    assert model.input_scalars.grad is not None
    assert torch.isfinite(model.residual_scalars.grad).all()
    assert torch.isfinite(model.input_scalars.grad).all()
    assert torch.count_nonzero(model.residual_scalars.grad) == config.n_layer
    assert torch.count_nonzero(model.input_scalars.grad) == config.n_layer
    assert model.residual_scalars.grad.data_ptr() != model.input_scalars.grad.data_ptr()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    occurrences = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter is model.residual_scalars or parameter is model.input_scalars
    ]
    assert occurrences == [model.residual_scalars, model.input_scalars]

    restored = GPT(config).to(dtype=torch.float64)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.config.residual_scalar_identity() == (
        config.residual_scalar_identity()
    )
    disabled = GPT(replace(config, use_residual_scalars=False))
    with pytest.raises(RuntimeError, match="(residual|input)_scalars"):
        disabled.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="(residual|input)_scalars"):
        model.load_state_dict(disabled.state_dict(), strict=True)


def test_full_checkpoint_round_trip_preserves_scalar_initializer_and_logits(
    tmp_path: Path,
) -> None:
    model_config = _config(
        vocab_size=VOCAB_SIZE,
        residual_scalar_init="nanochat_depth",
    )
    project = ProjectConfig(
        run=RunConfig(device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=model_config,
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=model_config.seq_len,
            grad_accum_steps=1,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
    )
    torch.manual_seed(933)
    model = GPT(project.model).eval()
    optimizer = build_optimizer(model, project.train)
    scheduler = build_lr_scheduler(optimizer, project.train)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected_logits = model(tokens)
    checkpoint_path = save_checkpoint(
        tmp_path / "residual-scalars.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=project,
        step=0,
        tokenizer=ByteTokenizer(),
    )

    loaded = load_model_checkpoint(checkpoint_path)

    assert loaded.config.model.residual_scalar_identity() == (
        project.model.residual_scalar_identity()
    )
    assert loaded.model.residual_scalars is not None
    assert loaded.model.input_scalars is not None
    torch.testing.assert_close(
        loaded.model.residual_scalars,
        model.residual_scalars,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        loaded.model.input_scalars,
        model.input_scalars,
        rtol=0,
        atol=0,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded.model(tokens),
            expected_logits,
            rtol=0,
            atol=0,
        )


def test_parameter_resource_flop_and_memory_identities_are_exact() -> None:
    enabled_config = _config(vocab_size=32_768, residual_scalar_init="nanochat_depth")
    disabled_config = replace(enabled_config, use_residual_scalars=False)
    enabled_model = GPT(_config(residual_scalar_init="nanochat_depth"))
    disabled_model = GPT(_config(use_residual_scalars=False))
    assert (
        summarize_module_parameters(enabled_model).unique_parameters
        - summarize_module_parameters(disabled_model).unique_parameters
        == 2 * enabled_config.n_layer
    )
    estimate = estimate_gpt_model_size(enabled_config)
    assert estimate.residual_scalar_parameters == 2 * enabled_config.n_layer
    assert estimate.unique_parameters - estimate_gpt_model_size(
        disabled_config
    ).unique_parameters == 2 * enabled_config.n_layer

    train = TrainConfig(
        device_batch_size=1,
        total_batch_size_tokens=enabled_config.seq_len,
        grad_accum_steps=1,
        max_steps=1,
        warmup_steps=0,
        warmdown_ratio=0.0,
    )
    enabled_resources = estimate_training_resources(
        ProjectConfig(model=enabled_config, train=train)
    )
    disabled_resources = estimate_training_resources(
        ProjectConfig(model=disabled_config, train=train)
    )
    assert enabled_resources.model.to_dict()["residual_scalars"] == (
        enabled_config.residual_scalar_identity()
    )
    assert (
        enabled_resources.memory.activation_bytes
        - disabled_resources.memory.activation_bytes
        == enabled_config.seq_len * enabled_config.n_embd * 4
    )
    enabled_flops = estimate_gpt_training_flops(enabled_config)
    disabled_flops = estimate_gpt_training_flops(disabled_config)
    assert enabled_flops.flops_per_token == disabled_flops.flops_per_token
    assert enabled_flops.to_dict()["residual_scalars"] == (
        enabled_config.residual_scalar_identity()
    )


@pytest.mark.parametrize("initializer", ["neutral", "nanochat_depth"])
def test_residual_scalar_model_overfits_one_tiny_batch(initializer: str) -> None:
    torch.manual_seed(937)
    config = _config(
        vocab_size=16,
        seq_len=6,
        n_layer=2,
        residual_scalar_init=initializer,
    )
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


def test_residual_scalar_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = (
        PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-10-residual-scalars"
    )
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Residual and input scalars" in readme
    assert "experimental" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.residual_scalar_init",
        "model.use_residual_scalars",
        "run.name",
    ]
    assert payload["runs"]["off"]["parameter_delta"] == 0
    assert payload["runs"]["neutral"]["initial_residual_scalars"] == [1.0, 1.0]
    assert payload["runs"]["pinned"]["initial_input_scalars"] == [0.2, 0.05]
    assert len(payload["runs"]["pinned"]["learned_residual_scalars"]) == 2
    assert len(payload["runs"]["pinned"]["learned_input_scalars"]) == 2
    assert len(offline["runs"]) == 3
