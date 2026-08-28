"""Grouped-query attention geometry, compatibility, and bounded evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scratch_llm.attention import (
    CausalSelfAttention,
    expand_kv_heads,
    split_query_key_value,
)
from scratch_llm.attention_backends import (
    AttentionBackendResolution,
    AttentionBackendSelection,
    FlashAttentionProvider,
)
from scratch_llm.config import (
    ConfigValidationError,
    GPTConfig,
    ProjectConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.oom import diagnose_out_of_memory
from scratch_llm.diagnostics.resource_estimation import (
    estimate_gpt_model_size,
    estimate_training_resources,
)
from scratch_llm.model import GPT
from scratch_llm.training.telemetry import estimate_gpt_training_flops
from scratch_llm.diagnostics.accelerator_memory import collect_accelerator_memory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 24,
        "seq_len": 8,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 16,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "bias": False,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def _gqa_config(n_kv_head: int, **overrides: object) -> GPTConfig:
    return _config(
        n_kv_head=n_kv_head,
        use_gqa=n_kv_head < int(overrides.get("n_head", 4)),
        **overrides,
    )


def _reference_attention(
    module: CausalSelfAttention,
    inputs: torch.Tensor,
) -> torch.Tensor:
    batch, time, _ = inputs.shape
    projected = F.linear(
        inputs,
        module.qkv_projection.weight,
        module.qkv_projection.bias,
    )
    query, key, value = split_query_key_value(
        projected,
        n_head=module.n_head,
        n_kv_head=module.n_kv_head,
        head_dim=module.head_dim,
    )
    key, value = expand_kv_heads(
        key,
        value,
        query_head_count=module.n_head,
    )
    context = torch.empty_like(query)
    scale = math.sqrt(module.head_dim)
    for batch_index in range(batch):
        for head_index in range(module.n_head):
            for query_index in range(time):
                scores = (
                    query[batch_index, head_index, query_index]
                    @ key[batch_index, head_index, : query_index + 1].T
                ) / scale
                weights = torch.softmax(scores, dim=-1)
                context[batch_index, head_index, query_index] = (
                    weights[:, None] * value[batch_index, head_index, : query_index + 1]
                ).sum(dim=0)
    merged = context.transpose(1, 2).contiguous().view(batch, time, module.n_embd)
    return F.linear(merged, module.out_proj.weight, module.out_proj.bias)


def test_gqa_config_resolves_mha_default_and_validates_canonical_geometry() -> None:
    default = _config()
    assert default.n_kv_head == default.n_head
    assert default.use_gqa is False
    assert _gqa_config(2).use_gqa is True

    cases = [
        ({"n_kv_head": 0}, "model.n_kv_head"),
        ({"n_kv_head": 5}, "model.n_kv_head"),
        ({"n_kv_head": 3}, "model.n_head"),
        ({"n_kv_head": 2, "use_gqa": False}, "model.use_gqa"),
        ({"n_kv_head": 4, "use_gqa": True}, "model.use_gqa"),
    ]
    for overrides, path in cases:
        with pytest.raises(ConfigValidationError) as error:
            _config(**overrides)
        assert error.value.path == path


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_projection_shapes_and_group_expansion_cover_mha_gqa_and_mqa(
    n_kv_head: int,
) -> None:
    config = _gqa_config(n_kv_head)
    head_dim = config.n_embd // config.n_head
    projected_width = config.n_embd + 2 * n_kv_head * head_dim
    projected = torch.arange(2 * 3 * projected_width, dtype=torch.float32).view(
        2,
        3,
        projected_width,
    )

    query, key, value = split_query_key_value(
        projected,
        n_head=config.n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
    )
    expanded_key, expanded_value = expand_kv_heads(
        key,
        value,
        query_head_count=config.n_head,
    )

    assert query.shape == (2, config.n_head, 3, head_dim)
    assert key.shape == value.shape == (2, n_kv_head, 3, head_dim)
    assert expanded_key.shape == expanded_value.shape == query.shape
    group_size = config.n_head // n_kv_head
    for query_head in range(config.n_head):
        source_head = query_head // group_size
        torch.testing.assert_close(expanded_key[:, query_head], key[:, source_head])
        torch.testing.assert_close(
            expanded_value[:, query_head],
            value[:, source_head],
        )


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_gqa_forward_and_gradients_match_hand_built_reference(
    n_kv_head: int,
    backend: str,
) -> None:
    torch.manual_seed(653)
    module = CausalSelfAttention(
        _gqa_config(
            n_kv_head,
            n_layer=1,
            attention_backend=backend,
        )
    )
    actual_input = torch.randn(2, 5, 16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)

    actual = module(actual_input)
    reference = _reference_attention(module, reference_input)
    actual.square().mean().backward()
    reference.square().mean().backward()

    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=3e-5,
        atol=1e-6,
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_gqa_prepared_flash_receives_compact_kv_heads() -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_flash(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        captured.update(query=query, key=key, value=value)
        expanded_key, expanded_value = expand_kv_heads(
            key.transpose(1, 2),
            value.transpose(1, 2),
            query_head_count=query.shape[2],
        )
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            expanded_key,
            expanded_value,
            is_causal=True,
        ).transpose(1, 2)

    provider = FlashAttentionProvider(
        name="fa2",
        version="2.7.4",
        function=fake_flash,
        minimum_compute_capability=(8, 0),
        supported_dtypes=frozenset({torch.float16, torch.bfloat16}),
    )
    module = CausalSelfAttention(_gqa_config(2, n_layer=1, attention_backend="flash"))
    module.prepare_attention_backend(
        AttentionBackendResolution(
            AttentionBackendSelection(
                requested_backend="flash",
                effective_backend="flash",
                provider="fa2",
                provider_version="2.7.4",
            ),
            provider,
        )
    )

    output = module(torch.randn(2, 5, 16))

    assert output.shape == (2, 5, 16)
    assert captured["query"].shape == (2, 5, 4, 4)
    assert captured["key"].shape == captured["value"].shape == (2, 5, 2, 4)


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_compact_cache_bytes_and_cached_logits_match_full_forward(
    n_kv_head: int,
    backend: str,
) -> None:
    torch.manual_seed(659)
    config = _gqa_config(
        n_kv_head,
        attention_backend=backend,
        use_kv_cache=True,
    )
    model = GPT(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cache = model.create_kv_cache(batch_size=1, capacity=tokens.shape[1])

    assert cache.layer_shape == (
        1,
        n_kv_head,
        tokens.shape[1],
        config.n_embd // config.n_head,
    )
    assert cache.bytes_per_token == (
        2
        * config.n_layer
        * n_kv_head
        * (config.n_embd // config.n_head)
        * model.token_embedding.weight.element_size()
    )
    with torch.inference_mode():
        full = model(tokens)
        pieces = [model(tokens[:, :3], kv_cache=cache)]
        for index in range(3, tokens.shape[1]):
            pieces.append(model(tokens[:, index : index + 1], kv_cache=cache))
        cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached, full, rtol=3e-5, atol=2e-6)


def test_mha_construction_preserves_legacy_fused_projection_initialization() -> None:
    config = _config(n_layer=1, bias=True)
    torch.manual_seed(661)
    module = CausalSelfAttention(config)
    torch.manual_seed(661)
    legacy_qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
    legacy_out = nn.Linear(config.n_embd, config.n_embd, bias=True)

    torch.testing.assert_close(
        module.qkv_projection.weight, legacy_qkv.weight, rtol=0, atol=0
    )
    assert module.qkv_projection.bias is not None and legacy_qkv.bias is not None
    torch.testing.assert_close(
        module.qkv_projection.bias, legacy_qkv.bias, rtol=0, atol=0
    )
    torch.testing.assert_close(
        module.out_proj.weight, legacy_out.weight, rtol=0, atol=0
    )
    assert module.out_proj.bias is not None and legacy_out.bias is not None
    torch.testing.assert_close(module.out_proj.bias, legacy_out.bias, rtol=0, atol=0)


def test_legacy_fused_qkv_checkpoint_converts_losslessly_for_mha() -> None:
    torch.manual_seed(673)
    config = _config(n_layer=2)
    source = GPT(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = source(tokens)
    legacy_state = {
        name.replace(".attn.qkv_projection.", ".attn.qkv."): value.clone()
        for name, value in source.state_dict().items()
    }
    assert any(".attn.qkv." in name for name in legacy_state)

    restored = GPT(config).eval()
    result = restored.load_state_dict(legacy_state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert all(
        ".attn.qkv_projection." in name
        for name in restored.state_dict()
        if "qkv" in name
    )
    with torch.inference_mode():
        torch.testing.assert_close(restored(tokens), expected, rtol=0, atol=0)


def test_reduced_head_model_rejects_legacy_or_mismatched_projection_state() -> None:
    mha = GPT(_config(n_layer=1))
    legacy_state = {
        name.replace(".attn.qkv_projection.", ".attn.qkv."): value.clone()
        for name, value in mha.state_dict().items()
    }
    reduced = GPT(_gqa_config(2, n_layer=1))

    with pytest.raises(RuntimeError, match=r"legacy fused QKV.*n_kv_head=2"):
        reduced.load_state_dict(legacy_state, strict=True)
    with pytest.raises(RuntimeError, match=r"attention projection.*n_kv_head=2"):
        reduced.load_state_dict(mha.state_dict(), strict=True)


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_new_gqa_state_keys_round_trip(n_kv_head: int, tmp_path: Path) -> None:
    torch.manual_seed(677)
    config = _gqa_config(n_kv_head, n_layer=1)
    source = GPT(config).eval()
    path = tmp_path / f"gqa-{n_kv_head}.pt"
    torch.save(source.state_dict(), path)

    restored = GPT(config).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)

    assert any(".attn.qkv_projection.weight" in name for name in restored.state_dict())
    for name, expected in source.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_mha_gqa_and_mqa_tiny_overfit(n_kv_head: int) -> None:
    torch.manual_seed(683)
    model = GPT(
        _gqa_config(
            n_kv_head,
            vocab_size=16,
            seq_len=6,
            n_layer=1,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])

    for _ in range(60):
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.inference_mode():
        assert model(inputs, targets).item() < 0.1


def test_gqa_resource_flops_comparison_and_oom_identities_are_exact() -> None:
    mha = _config()
    gqa = _gqa_config(2)
    parameter_delta = 2 * mha.n_embd * (mha.n_embd // 2)
    mha_size = estimate_gpt_model_size(mha)
    gqa_size = estimate_gpt_model_size(gqa)

    assert mha_size.unique_parameters - gqa_size.unique_parameters == (
        mha.n_layer * parameter_delta
    )
    assert gqa_size.n_kv_head == 2
    assert gqa_size.use_gqa is True
    assert (
        estimate_gpt_training_flops(mha).executed_weight_elements
        - estimate_gpt_training_flops(gqa).executed_weight_elements
        == mha.n_layer * parameter_delta
    )
    project = ProjectConfig(
        model=GPTConfig(
            **{
                **gqa.to_dict(),
                "vocab_size": 32_768,
            }
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=gqa.seq_len,
            grad_accum_steps=1,
        ),
    )
    resources = estimate_training_resources(project)
    assert resources.model.n_kv_head == 2
    diagnostic = diagnose_out_of_memory(
        torch.OutOfMemoryError("synthetic"),
        config=project,
        memory=collect_accelerator_memory("cpu"),
    )
    assert diagnostic is not None
    assert diagnostic.attempt.n_kv_head == 2
    assert diagnostic.attempt.use_gqa is True


def test_gqa_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-7-gqa"
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Grouped-query attention" in readme
    assert "n_kv_head" in readme
    assert "experimental" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.n_kv_head",
        "model.use_gqa",
        "run.name",
    ]
    assert payload["runs"]["mha"]["n_kv_head"] == payload["controls"]["n_head"]
    assert payload["runs"]["gqa"]["n_kv_head"] < payload["controls"]["n_head"]
    assert payload["deltas"]["unique_parameters"] < 0
    assert (
        payload["runs"]["gqa"]["cache_bytes_per_token"] * 2
        == (payload["runs"]["mha"]["cache_bytes_per_token"])
    )
    assert payload["runs"]["gqa"]["cached_decode_ms_per_token_p50"] > 0
    assert payload["runs"]["mha"]["cached_decode_ms_per_token_p50"] > 0
    assert len(offline["runs"]) == 2
