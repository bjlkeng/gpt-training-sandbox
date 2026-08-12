"""Capability, fallback, and math coverage for optional FlashAttention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import scratch_llm.attention as attention_module
import scratch_llm.attention_backends as backends
from scratch_llm.attention import CausalSelfAttention
from scratch_llm.attention_backends import (
    FLASH_BACKWARD_UNSUPPORTED,
    FLASH_CAPABILITY_UNSUPPORTED,
    FLASH_CAUSAL_UNSUPPORTED,
    FLASH_CUDA_UNAVAILABLE,
    FLASH_DEPENDENCY_UNAVAILABLE,
    FLASH_DROPOUT_UNSUPPORTED,
    FLASH_DTYPE_UNSUPPORTED,
    FLASH_HEAD_DIM_UNSUPPORTED,
    FLASH_KERNEL_UNAVAILABLE,
    FLASH_KV_CACHE_UNSUPPORTED,
    FLASH_TRAINING_UNSUPPORTED,
    FLASH_VERSION_UNSUPPORTED,
    FLASH_WINDOW_UNSUPPORTED,
    SDPA_UNAVAILABLE,
    AttentionBackendError,
    AttentionBackendRequest,
    FlashAttentionProvider,
    resolve_attention_backend,
    run_flash_attention,
)
from scratch_llm.config import GPTConfig
from scratch_llm.model import GPT


def _config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "seq_len": 8,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": True,
        "attention_backend": "flash",
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def _request(**overrides: object) -> AttentionBackendRequest:
    values: dict[str, object] = {
        "device_type": "cuda",
        "device_capability": (8, 6),
        "dtype": torch.float16,
        "head_dimension": 64,
        "training": True,
        "requires_backward": True,
        "dropout_p": 0.0,
    }
    values.update(overrides)
    return AttentionBackendRequest(**values)  # type: ignore[arg-type]


def _fake_flash_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    del softmax_scale, window_size
    attended = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=causal,
    )
    return attended.transpose(1, 2)


def _provider(**overrides: object) -> FlashAttentionProvider:
    values: dict[str, object] = {
        "name": "fa2",
        "version": "2.7.4",
        "function": _fake_flash_function,
        "minimum_compute_capability": (8, 0),
        "supported_dtypes": frozenset({torch.float16, torch.bfloat16}),
    }
    values.update(overrides)
    return FlashAttentionProvider(**values)  # type: ignore[arg-type]


def test_manual_and_sdpa_resolution_never_load_the_optional_provider() -> None:
    calls: list[tuple[str, tuple[int, int] | None]] = []

    def unexpected_loader(
        preference: str,
        capability: tuple[int, int] | None,
    ) -> FlashAttentionProvider:
        calls.append((preference, capability))
        raise AssertionError("provider must remain lazy")

    for backend in ("manual", "sdpa"):
        resolution = resolve_attention_backend(
            _config(attention_backend=backend),
            _request(),
            provider_loader=unexpected_loader,
        )
        assert resolution.selection.effective_backend == backend

    assert calls == []


def test_supported_fake_provider_reports_its_exact_identity() -> None:
    provider = _provider()
    resolution = resolve_attention_backend(
        _config(),
        _request(),
        provider_loader=lambda _preference, _capability: provider,
    )

    assert resolution.provider is provider
    assert resolution.selection.to_dict() == {
        "effective_backend": "flash",
        "fallback_reason": None,
        "provider": "fa2",
        "provider_version": "2.7.4",
        "requested_backend": "flash",
    }


@pytest.mark.parametrize(
    ("backend_request", "provider", "loader_error", "reason"),
    [
        (
            _request(device_type="cpu", device_capability=None),
            _provider(),
            False,
            FLASH_CUDA_UNAVAILABLE,
        ),
        (_request(dtype=torch.float32), _provider(), False, FLASH_DTYPE_UNSUPPORTED),
        (_request(head_dimension=63), _provider(), False, FLASH_HEAD_DIM_UNSUPPORTED),
        (_request(), _provider(version="2.2.9"), False, FLASH_VERSION_UNSUPPORTED),
        (
            _request(device_capability=(8, 0)),
            _provider(minimum_compute_capability=(9, 0)),
            False,
            FLASH_CAPABILITY_UNSUPPORTED,
        ),
        (
            _request(dtype=torch.bfloat16),
            _provider(supported_dtypes=frozenset({torch.float16})),
            False,
            FLASH_DTYPE_UNSUPPORTED,
        ),
        (
            _request(),
            _provider(supports_training=False),
            False,
            FLASH_TRAINING_UNSUPPORTED,
        ),
        (
            _request(),
            _provider(supports_backward=False),
            False,
            FLASH_BACKWARD_UNSUPPORTED,
        ),
        (
            _request(dropout_p=0.1),
            _provider(supports_dropout=False),
            False,
            FLASH_DROPOUT_UNSUPPORTED,
        ),
        (_request(), _provider(supports_causal=False), False, FLASH_CAUSAL_UNSUPPORTED),
        (
            _request(window_size=(128, 0)),
            _provider(supports_window=False),
            False,
            FLASH_WINDOW_UNSUPPORTED,
        ),
        (
            _request(use_kv_cache=True),
            _provider(supports_kv_cache=False),
            False,
            FLASH_KV_CACHE_UNSUPPORTED,
        ),
        (_request(), _provider(), True, FLASH_DEPENDENCY_UNAVAILABLE),
    ],
)
def test_every_capability_rejection_has_a_stable_sdpa_fallback_reason(
    backend_request: AttentionBackendRequest,
    provider: FlashAttentionProvider,
    loader_error: bool,
    reason: str,
) -> None:
    def load(
        _preference: str,
        _capability: tuple[int, int] | None,
    ) -> FlashAttentionProvider:
        if loader_error:
            raise ModuleNotFoundError("fixture")
        return provider

    resolution = resolve_attention_backend(
        _config(),
        backend_request,
        provider_loader=load,
    )

    assert resolution.selection.effective_backend == "sdpa"
    assert resolution.selection.fallback_reason == reason
    assert resolution.selection.provider is None


def test_fallback_reaches_manual_only_when_sdpa_is_unavailable() -> None:
    resolution = resolve_attention_backend(
        _config(),
        _request(device_type="cpu", device_capability=None),
        sdpa_available=False,
    )

    assert resolution.selection.effective_backend == "manual"
    assert resolution.selection.fallback_reason == (
        f"{FLASH_CUDA_UNAVAILABLE};{SDPA_UNAVAILABLE}"
    )


def test_strict_policy_rejects_before_loading_a_provider() -> None:
    called = False

    def provider_loader(
        _preference: str,
        _capability: tuple[int, int] | None,
    ) -> FlashAttentionProvider:
        nonlocal called
        called = True
        return _provider()

    with pytest.raises(AttentionBackendError, match=FLASH_CUDA_UNAVAILABLE):
        resolve_attention_backend(
            _config(attention_fallback_policy="error"),
            _request(device_type="cpu", device_capability=None),
            provider_loader=provider_loader,
        )

    assert called is False


def test_explicit_fa3_on_an_rtx_3090_class_device_falls_back_before_import() -> None:
    called = False

    def provider_loader(
        _preference: str,
        _capability: tuple[int, int] | None,
    ) -> FlashAttentionProvider:
        nonlocal called
        called = True
        return _provider(name="fa3")

    resolution = resolve_attention_backend(
        _config(flash_attention_provider="fa3"),
        _request(device_capability=(8, 6)),
        provider_loader=provider_loader,
    )

    assert resolution.selection.effective_backend == "sdpa"
    assert resolution.selection.fallback_reason == FLASH_CAPABILITY_UNSUPPORTED
    assert called is False


def test_flash_tensor_adapter_matches_sdpa_forward_and_backward() -> None:
    torch.manual_seed(71)
    provider = _provider()
    flash_tensors = [
        torch.randn(2, 3, 5, 8, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    sdpa_tensors = [
        tensor.detach().clone().requires_grad_(True) for tensor in flash_tensors
    ]

    flash_output = run_flash_attention(
        provider,
        *flash_tensors,
        dropout_p=0.0,
        causal=True,
    )
    sdpa_output = F.scaled_dot_product_attention(
        *sdpa_tensors,
        dropout_p=0.0,
        is_causal=True,
    )
    flash_output.square().mean().backward()
    sdpa_output.square().mean().backward()

    torch.testing.assert_close(flash_output, sdpa_output, rtol=1e-10, atol=1e-10)
    for flash_tensor, sdpa_tensor in zip(flash_tensors, sdpa_tensors, strict=True):
        torch.testing.assert_close(
            flash_tensor.grad,
            sdpa_tensor.grad,
            rtol=1e-10,
            atol=1e-10,
        )


def test_prepared_flash_provider_stays_out_of_the_compiled_forward_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    config = _config(attention_fallback_policy="error")
    resolution = resolve_attention_backend(
        config,
        _request(head_dimension=8),
        provider_loader=lambda _preference, _capability: provider,
    )
    module = CausalSelfAttention(config)
    module.prepare_attention_backend(resolution)

    def unexpected_resolution(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("prepared attention must not resolve inside forward")

    monkeypatch.setattr(
        attention_module,
        "resolve_attention_backend",
        unexpected_resolution,
    )
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    output = compiled(torch.randn(2, 6, 16))

    assert output.shape == (2, 6, 16)
    assert module.last_backend_selection == resolution.selection


def test_gpt_prepares_the_same_backend_for_every_decoder_block() -> None:
    provider = _provider()
    config = _config(n_layer=3)
    resolution = resolve_attention_backend(
        config,
        _request(head_dimension=8),
        provider_loader=lambda _preference, _capability: provider,
    )
    model = GPT(config)

    model.prepare_attention_backend(resolution)

    assert model.attention_backend_selection() == resolution.selection
    assert [block.attn.last_backend_selection for block in model.blocks] == [
        resolution.selection
    ] * 3


def test_prepared_flash_kernel_failure_falls_back_once() -> None:
    calls = 0

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        raise RuntimeError("kernel launch failed")

    provider = _provider(function=fail)
    config = _config()
    resolution = resolve_attention_backend(
        config,
        _request(head_dimension=8),
        provider_loader=lambda _preference, _capability: provider,
    )
    module = CausalSelfAttention(config)
    module.prepare_attention_backend(resolution)
    inputs = torch.randn(1, 4, 16)

    first = module(inputs)
    second = module(inputs)

    assert first.shape == second.shape == (1, 4, 16)
    assert calls == 1
    assert module.last_backend_selection.effective_backend == "sdpa"
    assert module.last_backend_selection.fallback_reason == FLASH_KERNEL_UNAVAILABLE


def test_flash_cpu_fallback_preserves_projection_keys_and_sdpa_math() -> None:
    torch.manual_seed(73)
    sdpa = CausalSelfAttention(_config(attention_backend="sdpa"))
    flash = CausalSelfAttention(_config())
    flash.load_state_dict(sdpa.state_dict(), strict=True)
    sdpa_input = torch.randn(2, 6, 16, requires_grad=True)
    flash_input = sdpa_input.detach().clone().requires_grad_(True)

    sdpa_output = sdpa(sdpa_input)
    flash_output = flash(flash_input)
    sdpa_output.square().mean().backward()
    flash_output.square().mean().backward()

    assert set(flash.state_dict()) == set(sdpa.state_dict())
    assert flash.last_backend_selection.effective_backend == "sdpa"
    assert flash.last_backend_selection.fallback_reason == FLASH_CUDA_UNAVAILABLE
    torch.testing.assert_close(flash_output, sdpa_output)
    torch.testing.assert_close(flash_input.grad, sdpa_input.grad)
    for flash_parameter, sdpa_parameter in zip(
        flash.parameters(),
        sdpa.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(flash_parameter.grad, sdpa_parameter.grad)


def test_kernel_failure_uses_the_same_observable_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        function=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("kernel launch failed")
        )
    )
    request = _request(head_dimension=8)
    monkeypatch.setattr(
        attention_module,
        "runtime_attention_request",
        lambda *_args, **_kwargs: request,
    )
    module = CausalSelfAttention(
        _config(),
        flash_provider_loader=lambda _preference, _capability: provider,
    )

    output = module(torch.randn(1, 4, 16))

    assert output.shape == (1, 4, 16)
    assert module.last_backend_selection.effective_backend == "sdpa"
    assert module.last_backend_selection.fallback_reason == FLASH_KERNEL_UNAVAILABLE


def test_default_loader_is_imported_only_by_a_supported_flash_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    module = type(
        "FakeFlashModule",
        (),
        {"__version__": "2.7.4", "flash_attn_func": _fake_flash_function},
    )()
    backends.load_flash_attention_provider.cache_clear()
    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda name: imported.append(name) or module,
    )
    monkeypatch.setattr(backends.metadata, "version", lambda _name: "2.7.4")

    resolve_attention_backend(_config(attention_backend="manual"), _request())
    resolve_attention_backend(_config(attention_backend="sdpa"), _request())
    resolve_attention_backend(_config(), _request())

    assert imported == ["flash_attn"]
    backends.load_flash_attention_provider.cache_clear()
