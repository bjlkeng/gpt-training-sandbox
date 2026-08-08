"""Lazy capability selection for optional attention kernels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import importlib
from importlib import metadata
import re
from typing import Protocol

import torch
import torch.nn.functional as F

from scratch_llm.config import GPTConfig, TrainDType


FLASH_DEPENDENCY_UNAVAILABLE = "flash_dependency_unavailable"
FLASH_VERSION_UNSUPPORTED = "flash_provider_version_unsupported"
FLASH_CUDA_UNAVAILABLE = "flash_cuda_unavailable"
FLASH_CAPABILITY_UNSUPPORTED = "flash_cuda_capability_unsupported"
FLASH_DTYPE_UNSUPPORTED = "flash_dtype_unsupported"
FLASH_HEAD_DIM_UNSUPPORTED = "flash_head_dimension_unsupported"
FLASH_TRAINING_UNSUPPORTED = "flash_training_unsupported"
FLASH_BACKWARD_UNSUPPORTED = "flash_backward_unsupported"
FLASH_DROPOUT_UNSUPPORTED = "flash_training_dropout_unsupported"
FLASH_CAUSAL_UNSUPPORTED = "flash_causal_unsupported"
FLASH_WINDOW_UNSUPPORTED = "flash_window_unsupported"
FLASH_KV_CACHE_UNSUPPORTED = "flash_kv_cache_unsupported"
FLASH_KERNEL_UNAVAILABLE = "flash_kernel_unavailable"
SDPA_UNAVAILABLE = "sdpa_unavailable"


class AttentionBackendError(RuntimeError):
    """An explicitly strict backend request cannot be honored."""


class FlashProviderLoader(Protocol):
    """Load one provider only after a flash backend has been requested."""

    def __call__(
        self,
        preference: str,
        capability: tuple[int, int] | None,
    ) -> FlashAttentionProvider: ...


@dataclass(frozen=True)
class FlashAttentionProvider:
    """One imported kernel plus the capabilities its adapter exposes."""

    name: str
    version: str
    function: Callable[..., torch.Tensor]
    minimum_compute_capability: tuple[int, int]
    supported_dtypes: frozenset[torch.dtype]
    maximum_head_dimension: int = 256
    head_dimension_multiple: int = 8
    supports_training: bool = True
    supports_backward: bool = True
    supports_dropout: bool = True
    supports_causal: bool = True
    supports_window: bool = True
    supports_kv_cache: bool = False


@dataclass(frozen=True)
class AttentionBackendRequest:
    """Runtime facts used for a deterministic capability decision."""

    device_type: str
    device_capability: tuple[int, int] | None
    dtype: torch.dtype
    head_dimension: int
    training: bool
    requires_backward: bool
    dropout_p: float
    causal: bool = True
    window_size: tuple[int, int] | None = None
    use_kv_cache: bool = False


@dataclass(frozen=True)
class AttentionBackendSelection:
    """Observable backend decision used by logs and benchmark identity."""

    requested_backend: str
    effective_backend: str
    fallback_reason: str | None = None
    provider: str | None = None
    provider_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "effective_backend": self.effective_backend,
            "fallback_reason": self.fallback_reason,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "requested_backend": self.requested_backend,
        }


@dataclass(frozen=True)
class AttentionBackendResolution:
    """A public decision plus the private loaded provider, when selected."""

    selection: AttentionBackendSelection
    provider: FlashAttentionProvider | None = None


def resolve_attention_backend(
    config: GPTConfig,
    request: AttentionBackendRequest,
    *,
    provider_loader: FlashProviderLoader | None = None,
    sdpa_available: bool | None = None,
) -> AttentionBackendResolution:
    """Resolve the requested backend without importing optional code eagerly."""

    config.validate()
    active_sdpa = (
        hasattr(F, "scaled_dot_product_attention")
        if sdpa_available is None
        else sdpa_available
    )
    if config.attention_backend == "manual":
        return _direct_resolution("manual")
    if config.attention_backend == "sdpa":
        if active_sdpa:
            return _direct_resolution("sdpa")
        return _fallback(config, SDPA_UNAVAILABLE, sdpa_available=False)

    reason = _request_rejection(config, request)
    if reason is not None:
        return _fallback(config, reason, sdpa_available=active_sdpa)

    loader = (
        load_flash_attention_provider if provider_loader is None else provider_loader
    )
    try:
        provider = loader(
            config.flash_attention_provider,
            request.device_capability,
        )
    except (ImportError, ModuleNotFoundError):
        return _fallback(
            config,
            FLASH_DEPENDENCY_UNAVAILABLE,
            sdpa_available=active_sdpa,
        )

    reason = _provider_rejection(provider, request)
    if reason is not None:
        return _fallback(config, reason, sdpa_available=active_sdpa)
    return AttentionBackendResolution(
        AttentionBackendSelection(
            requested_backend="flash",
            effective_backend="flash",
            provider=provider.name,
            provider_version=provider.version,
        ),
        provider,
    )


def preflight_attention_backend(
    config: GPTConfig,
    *,
    device: str | torch.device,
    dtype: TrainDType | torch.dtype,
    training: bool,
    requires_backward: bool | None = None,
    use_kv_cache: bool | None = None,
    window_size: tuple[int, int] | None = None,
    provider_loader: FlashProviderLoader | None = None,
    sdpa_available: bool | None = None,
) -> AttentionBackendResolution:
    """Resolve a model config from known run settings before work starts."""

    resolved_device = torch.device(device)
    capability = _device_capability(resolved_device)
    return resolve_attention_backend(
        config,
        AttentionBackendRequest(
            device_type=resolved_device.type,
            device_capability=capability,
            dtype=_torch_dtype(dtype),
            head_dimension=config.n_embd // config.n_head,
            training=training,
            requires_backward=training
            if requires_backward is None
            else requires_backward,
            dropout_p=config.dropout if training else 0.0,
            window_size=window_size,
            use_kv_cache=config.use_kv_cache if use_kv_cache is None else use_kv_cache,
        ),
        provider_loader=provider_loader,
        sdpa_available=sdpa_available,
    )


def run_flash_attention(
    provider: FlashAttentionProvider,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
    window_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Adapt internal ``(B, H, T, D)`` tensors to an upstream flash kernel."""

    arguments: dict[str, object] = {
        "dropout_p": dropout_p,
        "softmax_scale": None,
        "causal": causal,
    }
    if window_size is not None:
        arguments["window_size"] = window_size
    attended = provider.function(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        **arguments,
    )
    return attended.transpose(1, 2)


def runtime_attention_request(
    config: GPTConfig,
    q: torch.Tensor,
    *,
    training: bool,
    use_kv_cache: bool | None = None,
) -> AttentionBackendRequest:
    """Build the same capability request from projected query tensors."""

    capability = _device_capability(q.device)
    return AttentionBackendRequest(
        device_type=q.device.type,
        device_capability=capability,
        dtype=q.dtype,
        head_dimension=q.shape[-1],
        training=training,
        requires_backward=torch.is_grad_enabled() and q.requires_grad,
        dropout_p=config.dropout if training else 0.0,
        use_kv_cache=config.use_kv_cache if use_kv_cache is None else use_kv_cache,
    )


def kernel_fallback_resolution(
    config: GPTConfig,
    *,
    sdpa_available: bool | None = None,
) -> AttentionBackendResolution:
    """Resolve a provider runtime failure using the configured policy."""

    active_sdpa = (
        hasattr(F, "scaled_dot_product_attention")
        if sdpa_available is None
        else sdpa_available
    )
    return _fallback(config, FLASH_KERNEL_UNAVAILABLE, sdpa_available=active_sdpa)


def format_attention_selection(selection: AttentionBackendSelection) -> str:
    """Render one stable, compact progress/log line."""

    reason = selection.fallback_reason or "none"
    provider = selection.provider or "none"
    version = selection.provider_version or "none"
    return (
        f"Attention backend: requested={selection.requested_backend} "
        f"effective={selection.effective_backend} fallback_reason={reason} "
        f"provider={provider} provider_version={version}"
    )


def _direct_resolution(backend: str) -> AttentionBackendResolution:
    return AttentionBackendResolution(
        AttentionBackendSelection(
            requested_backend=backend,
            effective_backend=backend,
        )
    )


def _fallback(
    config: GPTConfig,
    reason: str,
    *,
    sdpa_available: bool,
) -> AttentionBackendResolution:
    if config.attention_fallback_policy == "error":
        raise AttentionBackendError(
            f"requested attention backend {config.attention_backend!r} is "
            f"unavailable: {reason}"
        )
    effective = "sdpa" if sdpa_available else "manual"
    fallback_reason = reason if sdpa_available else f"{reason};{SDPA_UNAVAILABLE}"
    return AttentionBackendResolution(
        AttentionBackendSelection(
            requested_backend=config.attention_backend,
            effective_backend=effective,
            fallback_reason=fallback_reason,
        )
    )


def _request_rejection(
    config: GPTConfig,
    request: AttentionBackendRequest,
) -> str | None:
    if request.device_type != "cuda" or request.device_capability is None:
        return FLASH_CUDA_UNAVAILABLE
    if config.flash_attention_provider == "fa3" and request.device_capability < (9, 0):
        return FLASH_CAPABILITY_UNSUPPORTED
    if request.dtype not in {torch.float16, torch.bfloat16}:
        return FLASH_DTYPE_UNSUPPORTED
    if request.head_dimension > 256 or request.head_dimension % 8:
        return FLASH_HEAD_DIM_UNSUPPORTED
    return None


def _provider_rejection(
    provider: FlashAttentionProvider,
    request: AttentionBackendRequest,
) -> str | None:
    if not _provider_version_supported(provider):
        return FLASH_VERSION_UNSUPPORTED
    if (
        request.device_capability is None
        or request.device_capability < provider.minimum_compute_capability
    ):
        return FLASH_CAPABILITY_UNSUPPORTED
    if request.dtype not in provider.supported_dtypes:
        return FLASH_DTYPE_UNSUPPORTED
    if (
        request.head_dimension > provider.maximum_head_dimension
        or request.head_dimension % provider.head_dimension_multiple
    ):
        return FLASH_HEAD_DIM_UNSUPPORTED
    if request.training and not provider.supports_training:
        return FLASH_TRAINING_UNSUPPORTED
    if request.requires_backward and not provider.supports_backward:
        return FLASH_BACKWARD_UNSUPPORTED
    if request.training and request.dropout_p and not provider.supports_dropout:
        return FLASH_DROPOUT_UNSUPPORTED
    if request.causal and not provider.supports_causal:
        return FLASH_CAUSAL_UNSUPPORTED
    if request.window_size is not None and not provider.supports_window:
        return FLASH_WINDOW_UNSUPPORTED
    if request.use_kv_cache and not provider.supports_kv_cache:
        return FLASH_KV_CACHE_UNSUPPORTED
    return None


def _provider_version_supported(provider: FlashAttentionProvider) -> bool:
    if provider.name == "fa3":
        return True
    numbers = tuple(int(value) for value in re.findall(r"\d+", provider.version)[:3])
    padded = numbers + (0,) * (3 - len(numbers))
    return padded >= (2, 3, 0)


def _torch_dtype(dtype: TrainDType | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(device)


@lru_cache(maxsize=3)
def load_flash_attention_provider(
    preference: str,
    capability: tuple[int, int] | None,
) -> FlashAttentionProvider:
    """Import FlashAttention only after selection asks for it."""

    candidates = (
        ("fa3", "fa2")
        if preference == "auto" and capability is not None and capability >= (9, 0)
        else (("fa2",) if preference == "auto" else (preference,))
    )
    last_error: ImportError | None = None
    for candidate in candidates:
        try:
            return _load_provider(candidate)
        except (ImportError, ModuleNotFoundError) as error:
            last_error = error
    raise ImportError(
        "no requested FlashAttention provider is installed"
    ) from last_error


def _load_provider(name: str) -> FlashAttentionProvider:
    if name == "fa3":
        module = importlib.import_module("flash_attn_interface")
        version = getattr(module, "__version__", "3-beta")
        return FlashAttentionProvider(
            name="fa3",
            version=str(version),
            function=getattr(module, "flash_attn_func"),
            minimum_compute_capability=(9, 0),
            supported_dtypes=frozenset({torch.float16, torch.bfloat16}),
            supports_dropout=False,
            supports_kv_cache=True,
        )

    module = importlib.import_module("flash_attn")
    try:
        installed_version = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        installed_version = str(getattr(module, "__version__", "0"))
    function = getattr(module, "flash_attn_func")
    return FlashAttentionProvider(
        name="fa2",
        version=installed_version,
        function=function,
        minimum_compute_capability=(8, 0),
        supported_dtypes=frozenset({torch.float16, torch.bfloat16}),
        supports_kv_cache=hasattr(module, "flash_attn_with_kvcache"),
    )
