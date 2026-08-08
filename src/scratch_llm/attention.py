"""Shared projections with manual, SDPA, and optional flash attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from scratch_llm.attention_backends import (
    AttentionBackendResolution,
    AttentionBackendSelection,
    FlashProviderLoader,
    kernel_fallback_resolution,
    resolve_attention_backend,
    run_flash_attention,
    runtime_attention_request,
)
from scratch_llm.config import GPTConfig
from scratch_llm.kv_cache import KVCacheTransaction


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention selected by one canonical backend setting."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        flash_provider_loader: FlashProviderLoader | None = None,
        layer_index: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.max_seq_len = config.seq_len
        self.attention_backend = config.attention_backend
        self._config = config
        self._flash_provider_loader = flash_provider_loader
        self.layer_index = layer_index
        self.last_backend_selection = AttentionBackendSelection(
            requested_backend=config.attention_backend,
            effective_backend=config.attention_backend,
        )

        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        self.causal_mask: torch.Tensor | None
        if self.attention_backend == "manual":
            causal_mask = torch.tril(
                torch.ones(config.seq_len, config.seq_len, dtype=torch.bool)
            )
            self.register_buffer("causal_mask", causal_mask, persistent=False)
        else:
            self.register_buffer("causal_mask", None, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCacheTransaction | None = None,
    ) -> torch.Tensor:
        """Apply causal self-attention to a ``(batch, time, channel)`` tensor."""

        if x.ndim != 3:
            raise ValueError(
                "attention input must have shape (batch, sequence, channels); "
                f"received {tuple(x.shape)}"
            )

        batch_size, sequence_length, channels = x.shape
        if channels != self.n_embd:
            raise ValueError(
                f"input channel dimension {channels} does not match "
                f"configured embedding dimension {self.n_embd}"
            )
        if sequence_length == 0:
            raise ValueError("attention input sequence must not be empty")
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured "
                f"context length {self.max_seq_len}"
            )

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(
            1, 2
        )
        query_start = 0
        if kv_cache is not None:
            if self.layer_index is None:
                raise ValueError("cached attention requires a configured layer index")
            query_start = kv_cache.start
            k, v = kv_cache.write(self.layer_index, k, v)

        if self.attention_backend == "manual":
            attended = self._manual_attention(q, k, v, query_start=query_start)
        elif self.attention_backend == "sdpa":
            attended = self._sdpa_attention(q, k, v, query_start=query_start)
        else:
            attended = self._flash_or_fallback_attention(
                q,
                k,
                v,
                query_start=query_start,
                use_kv_cache=kv_cache is not None,
            )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, channels)
        )
        return self.output_dropout(self.out_proj(attended))

    def _flash_or_fallback_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_start: int,
        use_kv_cache: bool,
    ) -> torch.Tensor:
        resolution = resolve_attention_backend(
            self._config,
            runtime_attention_request(
                self._config,
                q,
                training=self.training,
                use_kv_cache=use_kv_cache,
            ),
            provider_loader=self._flash_provider_loader,
        )
        self.last_backend_selection = resolution.selection
        if resolution.selection.effective_backend != "flash":
            return self._run_fallback(
                resolution,
                q,
                k,
                v,
                query_start=query_start,
            )
        if resolution.provider is None:  # pragma: no cover - resolver invariant.
            raise RuntimeError("flash selection lost its provider")
        try:
            return run_flash_attention(
                resolution.provider,
                q,
                k,
                v,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                causal=True,
            )
        except (NotImplementedError, RuntimeError, TypeError):
            fallback = kernel_fallback_resolution(self._config)
            self.last_backend_selection = fallback.selection
            return self._run_fallback(
                fallback,
                q,
                k,
                v,
                query_start=query_start,
            )

    def _run_fallback(
        self,
        resolution: AttentionBackendResolution,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_start: int,
    ) -> torch.Tensor:
        if resolution.selection.effective_backend == "sdpa":
            return self._sdpa_attention(q, k, v, query_start=query_start)
        return self._manual_attention(q, k, v, query_start=query_start)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_start: int = 0,
    ) -> torch.Tensor:
        query_length = q.shape[-2]
        key_length = k.shape[-2]
        if query_start == 0 and query_length == key_length:
            attention_mask = None
            is_causal = True
        else:
            query_positions = query_start + torch.arange(
                query_length,
                device=q.device,
            )
            key_positions = torch.arange(key_length, device=q.device)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            is_causal = False
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_start: int = 0,
    ) -> torch.Tensor:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_length = q.shape[-2]
        key_length = k.shape[-2]
        mask = self.causal_mask
        if query_start or query_length != key_length:
            query_positions = query_start + torch.arange(
                query_length,
                device=q.device,
            )
            key_positions = torch.arange(key_length, device=q.device)
            mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        elif mask is None:
            mask = torch.tril(
                torch.ones(
                    query_length,
                    key_length,
                    dtype=torch.bool,
                    device=q.device,
                )
            )
        else:
            mask = mask[:query_length, :key_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        return weights @ v
