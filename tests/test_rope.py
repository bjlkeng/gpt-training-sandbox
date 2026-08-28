"""Rotary position embedding architecture option and parity coverage."""

from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import torch

from scratch_llm.config import ConfigValidationError, GPTConfig
from scratch_llm.diagnostics.resource_estimation import (
    estimate_gpt_model_size,
    summarize_module_parameters,
)
from scratch_llm.attention import RotaryEmbedding, apply_rotary_emb
from scratch_llm.model import GPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "seq_len": 8,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "bias": False,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def test_split_half_rotation_matches_hand_calculation_and_preserves_norm() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    angle = torch.tensor([[[[0.25, 0.5]]]])
    cosine = angle.cos()
    sine = angle.sin()

    actual = apply_rotary_emb(x, cosine, sine)
    expected = torch.tensor(
        [
            [
                [
                    [
                        math.cos(0.25) + 3.0 * math.sin(0.25),
                        2.0 * math.cos(0.5) + 4.0 * math.sin(0.5),
                        -math.sin(0.25) + 3.0 * math.cos(0.25),
                        -2.0 * math.sin(0.5) + 4.0 * math.cos(0.5),
                    ]
                ]
            ]
        ]
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1))


def test_rotary_position_zero_is_identity_and_relative_dot_products_shift() -> None:
    rotary = RotaryEmbedding(head_dim=4, max_seq_len=8, theta=10_000.0)
    first = torch.tensor([[[[1.0, -2.0, 0.5, 3.0]]]])
    second = torch.tensor([[[[-0.5, 4.0, 2.0, 1.0]]]])

    first_zero = rotary(first, torch.tensor([0]))
    first_one = rotary(first, torch.tensor([1]))
    first_three = rotary(first, torch.tensor([3]))
    second_two = rotary(second, torch.tensor([2]))
    second_four = rotary(second, torch.tensor([4]))

    torch.testing.assert_close(first_zero, first)
    torch.testing.assert_close(
        (first_one * second_two).sum(dim=-1),
        (first_three * second_four).sum(dim=-1),
    )


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"use_rope": True, "n_head": 2, "n_embd": 6}, "model.n_embd"),
        ({"use_rope": True, "rope_theta": 0.5}, "model.rope_theta"),
        ({"use_rope": True, "rope_theta": float("inf")}, "model.rope_theta"),
        ({"use_rope": True, "seq_len": 16_777_217}, "model.seq_len"),
    ],
)
def test_rope_rejects_unsupported_geometry_and_theta(
    overrides: dict[str, object], path: str
) -> None:
    with pytest.raises(ConfigValidationError) as error:
        _config(**overrides)
    assert error.value.path == path


def test_flag_off_preserves_learned_position_state_and_logits() -> None:
    torch.manual_seed(301)
    implicit = GPT(_config()).eval()
    torch.manual_seed(301)
    explicit = GPT(_config(use_rope=False, rope_theta=10_000.0)).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])

    assert isinstance(implicit.position_embedding, torch.nn.Embedding)
    assert set(implicit.state_dict()) == set(explicit.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(implicit(tokens), explicit(tokens), rtol=0, atol=0)


def test_rope_removes_only_position_parameters_and_tables_are_non_persistent() -> None:
    learned = GPT(_config())
    rope = GPT(_config(use_rope=True, rope_theta=10_000.0))

    assert rope.position_embedding is None
    assert set(learned.state_dict()) - set(rope.state_dict()) == {
        "position_embedding.weight"
    }
    assert set(rope.state_dict()) - set(learned.state_dict()) == set()
    assert all("rotary" not in key for key in rope.state_dict())
    assert estimate_gpt_model_size(rope.config).unique_parameters == (
        summarize_module_parameters(rope).unique_parameters
    )
    assert (
        summarize_module_parameters(learned).unique_parameters
        - summarize_module_parameters(rope).unique_parameters
        == rope.config.seq_len * rope.config.n_embd
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_tables_are_deterministic_non_persistent_and_dtype_safe(
    dtype: torch.dtype,
) -> None:
    first = RotaryEmbedding(head_dim=8, max_seq_len=16, theta=10_000.0)
    second = RotaryEmbedding(head_dim=8, max_seq_len=16, theta=10_000.0)
    assert first.state_dict() == {}
    torch.testing.assert_close(first.cosine, second.cosine, rtol=0, atol=0)
    torch.testing.assert_close(first.sine, second.sine, rtol=0, atol=0)

    x = torch.randn(2, 3, 4, 8, dtype=dtype, requires_grad=True)
    output = first(x, torch.tensor([2, 3, 4, 5]))
    output.square().mean().backward()

    assert output.dtype == dtype
    assert output.device == x.device
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rotary_rejects_positions_that_cannot_be_represented() -> None:
    rotary = RotaryEmbedding(head_dim=4, max_seq_len=4, theta=10_000.0)
    x = torch.ones(1, 1, 1, 4)

    with pytest.raises(ValueError, match="non-negative"):
        rotary(x, torch.tensor([-1]))
    with pytest.raises(ValueError, match="exceeds.*context"):
        rotary(x, torch.tensor([4]))
    with pytest.raises(ValueError, match="one position per token"):
        rotary(x, torch.tensor([0, 1]))


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_rope_prefill_and_every_cached_decode_match_full_prefix(backend: str) -> None:
    torch.manual_seed(307)
    model = GPT(
        _config(
            use_rope=True,
            rope_theta=10_000.0,
            attention_backend=backend,
        )
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cache = model.create_kv_cache(batch_size=1, capacity=tokens.shape[1])

    with torch.inference_mode():
        prefill = model(tokens[:, :3], kv_cache=cache)
        full = model(tokens[:, :3])
        torch.testing.assert_close(prefill, full, rtol=2e-5, atol=1e-6)
        for end in range(4, tokens.shape[1] + 1):
            decoded = model(tokens[:, end - 1 : end], kv_cache=cache)
            full = model(tokens[:, :end])
            torch.testing.assert_close(
                decoded[:, -1], full[:, -1], rtol=2e-5, atol=1e-6
            )


def test_rope_is_causal_and_round_trips_without_persistent_tables() -> None:
    torch.manual_seed(311)
    config = _config(use_rope=True, rope_theta=50_000.0)
    source = GPT(config).eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 7, 8]])

    with torch.inference_mode():
        prefix_logits = source(prefix)
        changed_logits = source(changed_future)
    torch.testing.assert_close(prefix_logits[:, :2], changed_logits[:, :2])

    restored = GPT(config).eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(prefix), prefix_logits, rtol=0, atol=0)
    assert config.parameter_compatibility_dict()["rope_theta"] == 50_000.0


def test_tiny_rope_model_has_finite_gradients_and_overfits_one_batch() -> None:
    torch.manual_seed(313)
    config = _config(
        vocab_size=16,
        seq_len=6,
        n_layer=1,
        n_head=2,
        n_embd=16,
        use_rope=True,
        rope_theta=10_000.0,
    )
    model = GPT(config)
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


def test_rope_resource_identity_is_parameter_free() -> None:
    estimate = estimate_gpt_model_size(
        _config(use_rope=True, rope_theta=50_000.0)
    ).to_dict()

    assert estimate["component_parameters"]["position_embeddings"] == 0
    assert estimate["position_encoding"] == {
        "parameter_free": True,
        "theta": 50_000.0,
        "type": "rope",
    }


def test_rope_readme_and_bounded_comparison_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    comparison_dir = PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-2-rope"
    report = (comparison_dir / "README.md").read_text(encoding="utf-8")
    payload = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (comparison_dir / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Rotary position embeddings" in readme
    assert "nanochat_split_half_negative_angle" in json.dumps(payload)
    assert "does not establish a quality or performance trend" in report
    assert payload["controls"]["rope_theta"] == 10_000.0
    assert payload["controls"]["changed_config_fields"] == [
        "model.use_rope",
        "run.name",
    ]
    assert payload["runs"]["learned_absolute"]["unique_parameters"] == 444_160
    assert payload["runs"]["rope"]["unique_parameters"] == 427_776
    assert payload["deltas"]["unique_parameters"] == -16_384
    assert payload["interpretation"]["all_training_telemetry_finite"] is True
    assert len(offline["runs"]) == 2
