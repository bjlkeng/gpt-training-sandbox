"""Sliding-window visibility, cache, backend, and evidence contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from scratch_llm.attention import (
    CausalSelfAttention,
    build_causal_attention_mask,
    expand_kv_heads,
    split_query_key_value,
)
from scratch_llm.attention_backends import (
    FLASH_WINDOW_UNSUPPORTED,
    AttentionBackendError,
    AttentionBackendRequest,
    AttentionBackendResolution,
    AttentionBackendSelection,
    FlashAttentionProvider,
    resolve_attention_backend,
    runtime_attention_request,
)
from scratch_llm.config import ConfigValidationError, GPTConfig
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.diagnostics.inference import (
    InferenceBenchmarkExecution,
    InferenceBenchmarkSettings,
    InferenceIteration,
    build_inference_benchmark,
)
from scratch_llm.diagnostics.resource_estimation import estimate_gpt_model_size
from scratch_llm.generation import GeneratedSequence
from scratch_llm.model import GPT
from scratch_llm.training.compilation import CompileSelection
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
        "sliding_window_pattern": "L",
        "sliding_window_size": 2,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def _reference_attention(
    module: CausalSelfAttention,
    inputs: torch.Tensor,
) -> torch.Tensor:
    batch, time, _ = inputs.shape
    query, key, value = split_query_key_value(
        F.linear(
            inputs,
            module.qkv_projection.weight,
            module.qkv_projection.bias,
        ),
        n_head=module.n_head,
        n_kv_head=module.n_kv_head,
        head_dim=module.head_dim,
    )
    key, value = expand_kv_heads(key, value, query_head_count=module.n_head)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(module.head_dim)
    positions = torch.arange(time)
    mask = build_causal_attention_mask(
        positions,
        positions,
        left_window=module.left_window,
    )
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    attended = weights @ value
    merged = attended.transpose(1, 2).contiguous().view(batch, time, module.n_embd)
    return F.linear(merged, module.out_proj.weight, module.out_proj.bias)


def test_window_config_tiles_pattern_and_forces_the_final_layer_full() -> None:
    baseline = _config(sliding_window_pattern="L")
    patterned = _config(n_layer=7, sliding_window_pattern="SLL")

    assert baseline.layer_attention_windows() == (None, None, None)
    assert patterned.layer_attention_windows() == (2, None, None, 2, None, None, None)
    assert patterned.attention_window_identity() == {
        "final_layer_forced_full": True,
        "pattern": "SLL",
        "resolved_layer_types": ["S", "L", "L", "S", "L", "L", "L"],
        "resolved_left_windows": [2, None, None, 2, None, None, None],
        "short_window_size": 2,
    }


@pytest.mark.parametrize(
    ("overrides", "path"),
    [
        ({"sliding_window_pattern": ""}, "model.sliding_window_pattern"),
        ({"sliding_window_pattern": "LSX"}, "model.sliding_window_pattern"),
        ({"sliding_window_pattern": "ls"}, "model.sliding_window_pattern"),
        ({"sliding_window_size": 0}, "model.sliding_window_size"),
        ({"sliding_window_size": 9}, "model.sliding_window_size"),
    ],
)
def test_window_config_rejects_invalid_contracts(
    overrides: dict[str, object],
    path: str,
) -> None:
    with pytest.raises(ConfigValidationError) as error:
        _config(**overrides)
    assert error.value.path == path


def test_hand_calculated_short_and_full_masks_have_exact_visibility() -> None:
    positions = torch.arange(6)
    full = build_causal_attention_mask(positions, positions, left_window=None)
    short = build_causal_attention_mask(positions, positions, left_window=2)

    torch.testing.assert_close(full, torch.tril(torch.ones(6, 6, dtype=torch.bool)))
    expected_short = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(short, expected_short)

    cached = build_causal_attention_mask(
        torch.tensor([5]),
        torch.tensor([3, 4, 5]),
        left_window=2,
    )
    torch.testing.assert_close(cached, torch.ones(1, 3, dtype=torch.bool))


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("sequence_length", [1, 2, 3, 6])
def test_short_window_forward_and_gradients_match_reference(
    backend: str,
    sequence_length: int,
) -> None:
    torch.manual_seed(701)
    module = CausalSelfAttention(
        _config(sliding_window_pattern="S", attention_backend=backend),
        layer_index=0,
    )
    actual_input = torch.randn(2, sequence_length, 16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)

    actual = module(actual_input)
    reference = _reference_attention(module, reference_input)
    actual.square().mean().backward()
    reference.square().mean().backward()

    torch.testing.assert_close(actual, reference, rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=5e-5,
        atol=2e-6,
    )


def test_default_full_window_preserves_exact_attention_initialization_and_logits() -> (
    None
):
    implicit = _config(sliding_window_pattern="L")
    explicit = _config(sliding_window_pattern="LLLL", sliding_window_size=7)
    torch.manual_seed(709)
    first = GPT(implicit).eval()
    torch.manual_seed(709)
    second = GPT(explicit).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])

    assert set(first.state_dict()) == set(second.state_dict())
    for name, expected in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], expected, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(first(tokens), second(tokens), rtol=0, atol=0)


def test_flash_receives_window_and_capability_fallback_is_explicit() -> None:
    captured: dict[str, object] = {}

    def fake_flash(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **arguments: object,
    ) -> torch.Tensor:
        captured.update(arguments)
        q = query.transpose(1, 2)
        k, v = expand_kv_heads(
            key.transpose(1, 2),
            value.transpose(1, 2),
            query_head_count=q.shape[1],
        )
        positions = torch.arange(q.shape[-2], device=q.device)
        mask = build_causal_attention_mask(positions, positions, left_window=2)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)

    provider = FlashAttentionProvider(
        name="fa2",
        version="2.7.4",
        function=fake_flash,
        minimum_compute_capability=(8, 0),
        supported_dtypes=frozenset({torch.float16, torch.bfloat16}),
        head_dimension_multiple=4,
    )
    config = _config(sliding_window_pattern="S", attention_backend="flash")
    module = CausalSelfAttention(config, layer_index=0)
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

    assert module(torch.randn(2, 5, 16)).shape == (2, 5, 16)
    assert captured["window_size"] == (2, 0)
    request = runtime_attention_request(
        config,
        torch.randn(1, 4, 2, 4),
        training=True,
    )
    assert request.window_size == (2, 0)

    unsupported = FlashAttentionProvider(
        **{**provider.__dict__, "supports_window": False}
    )
    runtime = AttentionBackendRequest(
        device_type="cuda",
        device_capability=(8, 6),
        dtype=torch.float16,
        head_dimension=8,
        training=True,
        requires_backward=True,
        dropout_p=0.0,
        window_size=(2, 0),
    )
    fallback = resolve_attention_backend(
        config,
        runtime,
        provider_loader=lambda *_: unsupported,
    )
    assert fallback.selection.effective_backend == "sdpa"
    assert fallback.selection.fallback_reason == FLASH_WINDOW_UNSUPPORTED
    with pytest.raises(AttentionBackendError, match=FLASH_WINDOW_UNSUPPORTED):
        resolve_attention_backend(
            _config(
                sliding_window_pattern="S",
                attention_backend="flash",
                attention_fallback_policy="error",
            ),
            runtime,
            provider_loader=lambda *_: unsupported,
        )


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_cached_decode_slices_short_layers_and_matches_full_logits(
    backend: str,
) -> None:
    torch.manual_seed(719)
    config = _config(
        sliding_window_pattern="S",
        attention_backend=backend,
        use_kv_cache=True,
    )
    model = GPT(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cache = model.create_kv_cache(batch_size=1, capacity=8)
    spans: list[tuple[int, int]] = []
    short_attention = model.blocks[0].attn
    method_name = f"_{backend}_attention"
    original = getattr(short_attention, method_name)

    def capture(
        _self: CausalSelfAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_start: int = 0,
        key_start: int = 0,
    ) -> torch.Tensor:
        spans.append((key_start, key.shape[-2]))
        return original(
            query,
            key,
            value,
            query_start=query_start,
            key_start=key_start,
        )

    setattr(short_attention, method_name, MethodType(capture, short_attention))
    with torch.inference_mode():
        full = model(tokens)
        pieces = [model(tokens[:, :3], kv_cache=cache)]
        for index in range(3, tokens.shape[1]):
            pieces.append(model(tokens[:, index : index + 1], kv_cache=cache))
        cached = torch.cat(pieces, dim=1)

    torch.testing.assert_close(cached, full, rtol=4e-5, atol=3e-6)
    assert spans[-1] == (3, 3)
    assert cache.metadata.layer_window_sizes == (2, 2, None)
    assert cache.metadata.capacity == 8
    assert cache.metadata.allocated_bytes == cache.metadata.bytes_per_token * 8


def test_flops_resources_and_inference_report_use_effective_layer_windows() -> None:
    full = _config(sliding_window_pattern="L")
    short = _config(sliding_window_pattern="S")
    full_flops = estimate_gpt_training_flops(full)
    short_flops = estimate_gpt_training_flops(short)
    expected_spans = (3, 3, 8)

    assert short_flops.layer_key_spans == expected_spans
    assert short_flops.attention_flops_per_token * 24 == (
        full_flops.attention_flops_per_token * sum(expected_spans)
    )
    resources = estimate_gpt_model_size(short).to_dict()
    attention = resources["attention"]
    assert isinstance(attention, dict)
    assert attention["window"] == short.attention_window_identity()

    sequence = GeneratedSequence(
        prompt_token_ids=(1, 2, 3, 4, 5),
        generated_token_ids=(6, 7),
        completion_reason="max_new_tokens",
        stop_token_id=None,
        sampled_token_count=2,
    )
    memory = AcceleratorMemorySnapshot(
        device=torch.device("cpu"),
        available=False,
        unavailable_reason="fixture",
    )
    naive = InferenceIteration(
        mode="naive",
        sequence=sequence,
        prompt_context_tokens=5,
        forward_query_lengths=(5, 6),
        prefill_seconds=0.01,
        time_to_first_token_seconds=0.02,
        decode_seconds=0.01,
        end_to_end_seconds=0.03,
        memory=memory,
    )
    cached = InferenceIteration(
        mode="cached",
        sequence=sequence,
        prompt_context_tokens=5,
        forward_query_lengths=(5, 1),
        prefill_seconds=0.01,
        time_to_first_token_seconds=0.02,
        decode_seconds=0.01,
        end_to_end_seconds=0.03,
        memory=memory,
    )
    settings = InferenceBenchmarkSettings(
        warmup_iterations=1,
        timed_iterations=1,
        max_new_tokens=2,
        temperature=0,
        top_k=None,
        top_p=None,
        seed=1,
    )
    execution = InferenceBenchmarkExecution(
        naive_iterations=(naive,),
        cached_iterations=(cached,),
        checkpoint_load_seconds=0.1,
        parameter_bytes=4096,
        cache_metadata={
            "allocated_bytes": 1536,
            "bytes_per_token": 192,
            "capacity": 8,
            "layer_window_sizes": [2, 2, None],
        },
        checkpoint_identity="sha256:" + "a" * 64,
        checkpoint_config_identity="sha256:" + "b" * 64,
        tokenizer_identity="fixture",
        hardware_identity={"device": "cpu"},
        cuda_identity={"available": False},
        pytorch_identity={"version": "fixture"},
        code_identity={"commit": "fixture", "tracked_dirty": False},
        device="cpu",
        dtype="float32",
        attention_selection=AttentionBackendSelection("manual", "manual"),
        compile_selection=CompileSelection(
            requested=False,
            effective=False,
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
            compile_duration_seconds=0,
        ),
    )
    report = build_inference_benchmark(
        short,
        settings=settings,
        execution=execution,
    ).to_dict()
    cached_report = report["modes"]["cached"]
    assert cached_report["cache"]["capacity"] == 8
    assert cached_report["cache"]["read_bytes_per_iteration"] == 768
    assert cached_report["cache"]["write_bytes_per_iteration"] == 192
    assert report["optimization_state"]["attention"]["window"] == (
        short.attention_window_identity()
    )


def test_short_window_state_round_trip_and_tiny_overfit() -> None:
    torch.manual_seed(727)
    config = _config(
        vocab_size=16,
        seq_len=6,
        n_layer=2,
        sliding_window_pattern="S",
    )
    source = GPT(config)
    restored = GPT(config)
    restored.load_state_dict(source.state_dict(), strict=True)
    assert set(source.state_dict()) == set(restored.state_dict())

    optimizer = torch.optim.AdamW(source.parameters(), lr=0.03, weight_decay=0.0)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])
    for _ in range(70):
        loss = source(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.inference_mode():
        assert source(inputs, targets).item() < 0.12


def test_sliding_window_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = (
        PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-8-sliding-window"
    )
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Sliding-window attention" in readme
    assert "experimental" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.sliding_window_pattern",
        "model.sliding_window_size",
        "run.name",
    ]
    assert payload["runs"]["full"]["resolved_layer_types"] == ["L", "L"]
    assert payload["runs"]["sliding"]["resolved_layer_types"] == ["S", "L"]
    assert (
        payload["runs"]["sliding"]["cache_read_bytes"]
        < (payload["runs"]["full"]["cache_read_bytes"])
    )
    assert len(offline["runs"]) == 2
