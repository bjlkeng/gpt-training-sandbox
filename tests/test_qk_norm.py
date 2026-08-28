"""Parameter-free per-head query/key normalization and bounded evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import scratch_llm.attention as attention_module
from scratch_llm.attention import CausalSelfAttention, normalize_query_key
from scratch_llm.attention_backends import (
    AttentionBackendResolution,
    AttentionBackendSelection,
    FlashAttentionProvider,
)
from scratch_llm.config import ConfigValidationError, GPTConfig
from scratch_llm.diagnostics.resource_estimation import estimate_gpt_model_size
from scratch_llm.model import GPT, RMS_NORM_EPSILON


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 24,
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


def test_qk_norm_config_is_boolean_and_off_by_default() -> None:
    assert _config().use_qk_norm is False
    assert _config(use_qk_norm=True).use_qk_norm is True
    with pytest.raises(ConfigValidationError) as error:
        _config(use_qk_norm=1)
    assert error.value.path == "model.use_qk_norm"


def test_query_and_key_are_normalized_independently_per_head_and_token() -> None:
    query = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                [[10.0, 20.0, 30.0, 40.0], [0.5, 1.0, 1.5, 2.0]],
            ]
        ],
        requires_grad=True,
    )
    key = torch.tensor(
        [
            [
                [[2.0, -1.0, 0.5, 3.0], [8.0, 4.0, 2.0, 1.0]],
                [[-3.0, 6.0, -9.0, 12.0], [1.0, -2.0, 3.0, -4.0]],
            ]
        ],
        requires_grad=True,
    )

    normalized_query, normalized_key = normalize_query_key(query, key)
    expected_query = F.rms_norm(
        query,
        (query.shape[-1],),
        weight=None,
        eps=RMS_NORM_EPSILON,
    )
    expected_key = F.rms_norm(
        key,
        (key.shape[-1],),
        weight=None,
        eps=RMS_NORM_EPSILON,
    )
    (normalized_query.sum() + normalized_key.sum()).backward()

    torch.testing.assert_close(normalized_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(normalized_key, expected_key, rtol=0, atol=0)
    assert normalized_query.square().mean(dim=-1).sub(1).abs().max() < 5e-5
    assert normalized_key.square().mean(dim=-1).sub(1).abs().max() < 5e-5
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()


def test_qk_norm_runs_after_rope_and_before_scaled_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        seq_len=3,
        n_layer=1,
        use_rope=True,
        use_qk_norm=True,
        bias=True,
    )
    module = CausalSelfAttention(config)
    events: list[str] = []
    assert module.rotary is not None
    real_rotary = module.rotary.forward
    real_normalize = attention_module.normalize_query_key

    def record_rotary(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        events.append("rope")
        return real_rotary(x, positions)

    def record_normalize(
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events.append("qk_norm")
        return real_normalize(query, key)

    monkeypatch.setattr(module.rotary, "forward", record_rotary)
    monkeypatch.setattr(attention_module, "normalize_query_key", record_normalize)
    inputs = torch.tensor(
        [
            [
                [0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.4],
                [0.0, 0.5, -0.2, 0.1, 0.7, -0.3, 0.2, 0.4],
                [0.7, -0.3, 0.2, -0.4, 0.1, 0.6, -0.5, 0.2],
            ]
        ]
    )
    positions = torch.arange(3)

    actual = module(inputs, positions=positions)
    query, key, value = module.qkv(inputs).chunk(3, dim=-1)
    query = query.view(1, 3, module.n_head, module.head_dim).transpose(1, 2)
    key = key.view(1, 3, module.n_head, module.head_dim).transpose(1, 2)
    value = value.view(1, 3, module.n_head, module.head_dim).transpose(1, 2)
    query = real_rotary(query, positions)
    key = real_rotary(key, positions)
    query, key = real_normalize(query, key)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(module.head_dim)
    mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    context = (weights @ value).transpose(1, 2).contiguous().view(1, 3, 8)
    expected = module.out_proj(context)

    assert events[:3] == ["rope", "rope", "qk_norm"]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_disabled_qk_norm_preserves_state_and_exact_logits() -> None:
    torch.manual_seed(631)
    implicit = GPT(_config()).eval()
    torch.manual_seed(631)
    explicit = GPT(_config(use_qk_norm=False)).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])

    assert set(implicit.state_dict()) == set(explicit.state_dict())
    for name, expected in implicit.state_dict().items():
        torch.testing.assert_close(
            explicit.state_dict()[name], expected, rtol=0, atol=0
        )
    with torch.inference_mode():
        torch.testing.assert_close(implicit(tokens), explicit(tokens), rtol=0, atol=0)


def test_enabling_qk_norm_adds_no_parameters_buffers_or_state_keys() -> None:
    torch.manual_seed(633)
    disabled = GPT(_config(use_qk_norm=False))
    torch.manual_seed(633)
    enabled = GPT(_config(use_qk_norm=True))

    assert set(disabled.state_dict()) == set(enabled.state_dict())
    for name, expected in disabled.state_dict().items():
        torch.testing.assert_close(enabled.state_dict()[name], expected, rtol=0, atol=0)
    assert sum(p.numel() for p in disabled.parameters()) == sum(
        p.numel() for p in enabled.parameters()
    )
    assert [name for name, _ in disabled.named_buffers()] == [
        name for name, _ in enabled.named_buffers()
    ]
    assert enabled.config.parameter_compatibility_dict()["use_qk_norm"] is True


@pytest.mark.parametrize("sequence_length", [1, 8])
def test_qk_norm_manual_and_sdpa_forward_backward_parity(
    sequence_length: int,
) -> None:
    torch.manual_seed(637)
    manual = CausalSelfAttention(_config(use_qk_norm=True, attention_backend="manual"))
    sdpa = CausalSelfAttention(_config(use_qk_norm=True, attention_backend="sdpa"))
    sdpa.load_state_dict(manual.state_dict())
    manual_input = torch.randn(2, sequence_length, 8, requires_grad=True)
    sdpa_input = manual_input.detach().clone().requires_grad_(True)

    manual_output = manual(manual_input)
    sdpa_output = sdpa(sdpa_input)
    manual_output.square().mean().backward()
    sdpa_output.square().mean().backward()

    torch.testing.assert_close(sdpa_output, manual_output, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(sdpa_input.grad, manual_input.grad, rtol=3e-5, atol=1e-6)
    for (_, manual_parameter), (_, sdpa_parameter) in zip(
        manual.named_parameters(),
        sdpa.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            sdpa_parameter.grad,
            manual_parameter.grad,
            rtol=3e-5,
            atol=1e-6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qk_norm_cuda_reduced_precision_backend_parity(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    torch.manual_seed(639)
    manual = CausalSelfAttention(
        _config(use_qk_norm=True, attention_backend="manual")
    ).to(device="cuda", dtype=dtype)
    sdpa = CausalSelfAttention(_config(use_qk_norm=True, attention_backend="sdpa")).to(
        device="cuda", dtype=dtype
    )
    sdpa.load_state_dict(manual.state_dict())
    manual_input = torch.randn(
        2,
        8,
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


def test_prepared_flash_receives_normalized_qk_and_unchanged_values() -> None:
    captured: dict[str, object] = {}

    def fake_flash(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured.update(query=query.detach(), key=key.detach(), value=value.detach())
        captured.update(kwargs)
        return value

    provider = FlashAttentionProvider(
        name="fa2",
        version="2.7.4",
        function=fake_flash,
        minimum_compute_capability=(8, 0),
        supported_dtypes=frozenset({torch.float16, torch.bfloat16}),
    )
    module = CausalSelfAttention(
        _config(
            n_layer=1,
            use_qk_norm=True,
            attention_backend="flash",
        )
    )
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
    inputs = torch.randn(2, 5, 8)
    projected_query, projected_key, projected_value = module.qkv(inputs).chunk(
        3, dim=-1
    )

    module(inputs)

    query = captured["query"]
    key = captured["key"]
    value = captured["value"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(key, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    expected_query = projected_query.view(2, 5, 2, 4)
    expected_key = projected_key.view(2, 5, 2, 4)
    expected_query, expected_key = normalize_query_key(expected_query, expected_key)
    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, projected_value.view(2, 5, 2, 4))
    assert captured["softmax_scale"] is None


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("use_rope", [False, True])
def test_qk_norm_cached_decode_matches_full_forward(
    backend: str,
    use_rope: bool,
) -> None:
    torch.manual_seed(641)
    model = GPT(
        _config(
            n_layer=2,
            use_qk_norm=True,
            use_rope=use_rope,
            attention_backend=backend,
            use_kv_cache=True,
        )
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with torch.inference_mode():
        full = model(tokens)
        cache = model.create_kv_cache(batch_size=1, capacity=tokens.shape[1])
        pieces = [model(tokens[:, :3], kv_cache=cache)]
        for index in range(3, tokens.shape[1]):
            pieces.append(model(tokens[:, index : index + 1], kv_cache=cache))
        cached = torch.cat(pieces, dim=1)

    torch.testing.assert_close(cached, full, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_qk_norm_extreme_magnitudes_have_finite_outputs_and_gradients(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(643)
    module = CausalSelfAttention(
        _config(n_layer=1, use_qk_norm=True, attention_backend="manual")
    ).to(dtype=dtype)
    inputs = (torch.randn(2, 8, 8, dtype=dtype) * 1e12).requires_grad_(True)

    output = module(inputs)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


@pytest.mark.parametrize("use_qk_norm", [False, True])
def test_qk_norm_checkpoint_round_trip_preserves_logits_and_identity(
    tmp_path: Path,
    use_qk_norm: bool,
) -> None:
    torch.manual_seed(645)
    config = _config(use_qk_norm=use_qk_norm)
    source = GPT(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = source(tokens)
    checkpoint = tmp_path / f"qk-{use_qk_norm}.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = GPT(config).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)

    assert config.parameter_compatibility_dict()["use_qk_norm"] is use_qk_norm
    with torch.inference_mode():
        torch.testing.assert_close(restored(tokens), expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_qk_norm", [False, True])
def test_qk_norm_modes_tiny_overfit(use_qk_norm: bool) -> None:
    torch.manual_seed(647)
    model = GPT(
        _config(
            vocab_size=16,
            seq_len=6,
            n_layer=1,
            n_embd=16,
            use_qk_norm=use_qk_norm,
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


def test_qk_norm_resource_identity_has_no_parameter_delta() -> None:
    disabled = estimate_gpt_model_size(_config(use_qk_norm=False)).to_dict()
    enabled = estimate_gpt_model_size(_config(use_qk_norm=True)).to_dict()

    assert disabled["qk_norm"] is False
    assert enabled["qk_norm"] is True
    assert disabled["unique_parameters"] == enabled["unique_parameters"]
    assert disabled["component_parameters"] == enabled["component_parameters"]


def test_qk_norm_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-6-qk-norm"
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Query/key normalization" in readme
    assert "sharpening" in readme
    assert "logit softcap" in readme
    assert "experimental" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.use_qk_norm",
        "run.name",
    ]
    assert payload["deltas"]["unique_parameters"] == 0
    assert payload["runs"]["disabled"]["use_qk_norm"] is False
    assert payload["runs"]["enabled"]["use_qk_norm"] is True
    assert len(offline["runs"]) == 2
