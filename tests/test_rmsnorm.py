"""Parameter-free RMSNorm architecture option and comparison evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import scratch_llm.model as model_module
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.data.loaders import NextTokenDataset
from scratch_llm.diagnostics.resource_estimation import (
    estimate_gpt_model_size,
    summarize_module_parameters,
)
from scratch_llm.model import RMS_NORM_EPSILON, Block, GPT, RMSNorm
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.training.checkpoint import load_model_checkpoint, save_checkpoint
from scratch_llm.training.loop import run_training_steps
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("shape", "dtype", "rtol", "atol"),
    [
        ((2, 3, 4), torch.float64, 1e-12, 1e-12),
        ((2, 3, 4), torch.float32, 1e-6, 1e-7),
        ((3, 8), torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_rmsnorm_matches_parameter_free_reference_and_gradient(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    values = torch.linspace(-2.0, 3.0, steps=torch.Size(shape).numel(), dtype=dtype)
    inputs = values.reshape(shape).requires_grad_(True)
    reference_dtype = (
        torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    )
    reference_inputs = inputs.detach().to(reference_dtype).clone().requires_grad_(True)
    norm = RMSNorm(shape[-1])

    actual = norm(inputs)
    expected = reference_inputs * torch.rsqrt(
        reference_inputs.square().mean(dim=-1, keepdim=True) + RMS_NORM_EPSILON
    )
    actual.square().sum().backward()
    expected.square().sum().backward()

    assert actual.shape == inputs.shape
    assert actual.dtype == inputs.dtype
    assert actual.device == inputs.device
    torch.testing.assert_close(
        actual,
        expected.to(dtype),
        rtol=rtol,
        atol=atol,
    )
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert reference_inputs.grad is not None
    torch.testing.assert_close(
        inputs.grad,
        reference_inputs.grad.to(dtype),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("channels", [0, -1, 1.5, True])
def test_rmsnorm_rejects_invalid_channel_counts(channels: object) -> None:
    with pytest.raises((TypeError, ValueError), match="channels.*positive integer"):
        RMSNorm(channels)  # type: ignore[arg-type]


def test_rmsnorm_rejects_inputs_without_the_configured_channel_axis() -> None:
    norm = RMSNorm(4)

    with pytest.raises(ValueError, match="at least one dimension"):
        norm(torch.tensor(1.0))
    with pytest.raises(ValueError, match="channel dimension 3.*configured.*4"):
        norm(torch.ones(2, 3))


def test_rmsnorm_has_no_parameters_or_persistent_state() -> None:
    norm = RMSNorm(8)

    assert list(norm.parameters()) == []
    assert list(norm.buffers()) == []
    assert norm.state_dict() == {}


def test_rmsnorm_delegates_to_native_parameter_free_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, tuple[int, ...], object, object]] = []

    def native_rms_norm(
        inputs: torch.Tensor,
        normalized_shape: tuple[int, ...],
        weight: torch.Tensor | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        calls.append((inputs, normalized_shape, weight, eps))
        return torch.full_like(inputs, 7)

    monkeypatch.setattr(model_module.F, "rms_norm", native_rms_norm)
    inputs = torch.randn(2, 3, 4)

    output = RMSNorm(4)(inputs)

    torch.testing.assert_close(output, torch.full_like(inputs, 7))
    assert calls == [(inputs, (4,), None, RMS_NORM_EPSILON)]


def test_rmsnorm_factory_covers_every_block_and_final_norm() -> None:
    config = GPTConfig(
        vocab_size=32,
        seq_len=8,
        n_layer=3,
        n_head=2,
        n_embd=8,
        norm="rmsnorm",
        use_rmsnorm=True,
    )
    model = GPT(config)

    assert isinstance(model.ln_f, RMSNorm)
    assert all(
        isinstance(block, Block)
        and isinstance(block.ln_1, RMSNorm)
        and isinstance(block.ln_2, RMSNorm)
        for block in model.blocks
    )
    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())


@pytest.mark.parametrize("bias", [False, True])
def test_rmsnorm_removes_only_normalization_state_and_parameters(bias: bool) -> None:
    common = {
        "vocab_size": 32,
        "seq_len": 8,
        "n_layer": 3,
        "n_head": 2,
        "n_embd": 8,
        "mlp_ratio": 2,
        "bias": bias,
    }
    layernorm = GPT(GPTConfig(**common))  # type: ignore[arg-type]
    rmsnorm = GPT(
        GPTConfig(**common, norm="rmsnorm", use_rmsnorm=True)  # type: ignore[arg-type]
    )
    layernorm_state = set(layernorm.state_dict())
    rmsnorm_state = set(rmsnorm.state_dict())
    expected_removed = {
        *(
            f"blocks.{index}.ln_{position}.weight"
            for index in range(3)
            for position in (1, 2)
        ),
        "ln_f.weight",
    }
    if bias:
        expected_removed.update(
            f"blocks.{index}.ln_{position}.bias"
            for index in range(3)
            for position in (1, 2)
        )
        expected_removed.add("ln_f.bias")

    assert layernorm_state - rmsnorm_state == expected_removed
    assert rmsnorm_state - layernorm_state == set()
    expected_delta = (2 * common["n_layer"] + 1) * common["n_embd"] * (2 if bias else 1)
    assert (
        summarize_module_parameters(layernorm).unique_parameters
        - summarize_module_parameters(rmsnorm).unique_parameters
        == expected_delta
    )
    assert estimate_gpt_model_size(rmsnorm.config).unique_parameters == (
        summarize_module_parameters(rmsnorm).unique_parameters
    )
    with pytest.raises(RuntimeError, match="Missing key.*ln_1.weight"):
        layernorm.load_state_dict(rmsnorm.state_dict(), strict=True)


def _checkpoint_config(tmp_path: Path, *, norm: str) -> ProjectConfig:
    use_rmsnorm = norm == "rmsnorm"
    return ProjectConfig(
        run=RunConfig(
            name=f"{norm}-checkpoint", device="cpu", output_dir=str(tmp_path)
        ),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
            norm=norm,  # type: ignore[arg-type]
            use_rmsnorm=use_rmsnorm,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=8,
            grad_accum_steps=1,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
    )


def test_rmsnorm_checkpoint_round_trip_and_architecture_identity(
    tmp_path: Path,
) -> None:
    config = _checkpoint_config(tmp_path, norm="rmsnorm")
    torch.manual_seed(31)
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    checkpoint_path = save_checkpoint(
        tmp_path / "rmsnorm.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=ByteTokenizer(),
    )

    loaded = load_model_checkpoint(checkpoint_path, device="cpu")
    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    model.eval()
    loaded.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), loaded.model(inputs), rtol=0, atol=0)
    layernorm_identity = _checkpoint_config(tmp_path, norm="layernorm").model
    rmsnorm_identity = config.model
    differences = {
        key: (layernorm_identity.parameter_compatibility_dict()[key], value)
        for key, value in rmsnorm_identity.parameter_compatibility_dict().items()
        if layernorm_identity.parameter_compatibility_dict()[key] != value
    }
    assert differences == {
        "norm": ("layernorm", "rmsnorm"),
        "use_rmsnorm": (False, True),
    }


def test_tiny_rmsnorm_model_overfits_one_fixed_batch() -> None:
    torch.manual_seed(1234)
    tokenizer = ByteTokenizer()
    dataset = NextTokenDataset(
        tokenizer.encode("abcd efgh abcd efgh abcd efgh"),
        seq_len=8,
        vocab_size=tokenizer.get_vocab_size(),
    )
    first_inputs, first_targets = dataset[0]
    second_inputs, second_targets = dataset[1]
    batch_inputs = torch.stack((first_inputs, second_inputs))
    batch_targets = torch.stack((first_targets, second_targets))
    model = GPT(
        GPTConfig(
            vocab_size=tokenizer.get_vocab_size(),
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_ratio=2,
            norm="rmsnorm",
            use_rmsnorm=True,
        )
    )
    train = TrainConfig(
        device_batch_size=2,
        total_batch_size_tokens=16,
        grad_accum_steps=1,
        max_steps=60,
        learning_rate=0.03,
        weight_decay=0.0,
        warmup_steps=0,
        warmdown_ratio=0.0,
    )
    optimizer = build_optimizer(model, train)
    scheduler = build_lr_scheduler(optimizer, train)

    run_training_steps(
        model,
        [(batch_inputs, batch_targets)],
        optimizer,
        scheduler,
        max_steps=train.max_steps,
        grad_accum_steps=1,
        grad_clip=train.grad_clip,
        device="cpu",
    )

    with torch.inference_mode():
        final_loss = model(batch_inputs, batch_targets)
    assert torch.isfinite(final_loss)
    assert final_loss.item() < 0.1


def test_rmsnorm_readme_and_bounded_comparison_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    comparison_dir = PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-1-rmsnorm"
    report = (comparison_dir / "README.md").read_text(encoding="utf-8")
    payload = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (comparison_dir / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Parameter-free RMSNorm" in readme
    assert "sqrt(mean(x^2) + 1e-5)" in readme
    assert "comparisons/gpt-training-sandbox-as7-1-rmsnorm" in readme
    assert payload["format"] == "scratch_llm_bounded_architecture_flag_comparison"
    assert payload["format_version"] == 1
    assert payload["controls"]["changed_config_fields"] == [
        "model.norm",
        "model.use_rmsnorm",
        "run.name",
    ]
    assert payload["controls"]["rmsnorm_operation"] == ("torch.nn.functional.rms_norm")
    assert payload["controls"]["rmsnorm_weight"] is None
    assert payload["runs"]["layernorm"]["unique_parameters"] == 444_160
    assert payload["runs"]["rmsnorm"]["unique_parameters"] == 443_520
    assert payload["deltas"]["unique_parameters"] == -640
    assert payload["interpretation"]["all_training_telemetry_finite"] is True
    assert offline["run_count"] == 2
    assert [run["training"]["unique_parameters"] for run in offline["runs"]] == [
        444_160,
        443_520,
    ]
    assert all(not run["rankable"] for run in offline["runs"])
    assert "do not establish" in report
    assert "quality or performance trend" in report
