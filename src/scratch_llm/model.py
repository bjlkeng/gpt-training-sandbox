"""Feed-forward and transformer building blocks for the baseline GPT."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scratch_llm.attention import CausalSelfAttention
from scratch_llm.attention_backends import (
    AttentionBackendResolution,
    AttentionBackendSelection,
    FlashProviderLoader,
)
from scratch_llm.config import GPTConfig
from scratch_llm.kv_cache import KVCache, KVCacheError, KVCacheTransaction


RMS_NORM_EPSILON = 1e-5


class RMSNorm(nn.Module):
    """Parameter-free native root-mean-square normalization over channels."""

    def __init__(self, channels: int, *, eps: float = RMS_NORM_EPSILON) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("RMSNorm channels must be a positive integer")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("RMSNorm epsilon must be a positive real number")
        self.channels = channels
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one floating-point tensor without learned affine state."""

        if x.ndim == 0:
            raise ValueError("RMSNorm input must have at least one dimension")
        if x.shape[-1] != self.channels:
            raise ValueError(
                f"RMSNorm input channel dimension {x.shape[-1]} does not match "
                f"configured channels {self.channels}"
            )
        if not x.is_floating_point():
            raise TypeError("RMSNorm input must use a floating-point dtype")
        return F.rms_norm(x, (self.channels,), weight=None, eps=self.eps)


def build_norm(config: GPTConfig) -> nn.Module:
    """Construct the selected normalization through one architecture boundary."""

    config.validate()
    if config.norm == "layernorm":
        return nn.LayerNorm(config.n_embd, bias=config.bias)
    if config.norm == "rmsnorm":
        return RMSNorm(config.n_embd)
    raise RuntimeError(f"unsupported validated normalization {config.norm!r}")


class MLP(nn.Module):
    """Expand each token internally, then restore the residual-stream width."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()

        self.n_embd = config.n_embd
        hidden_dim = config.mlp_ratio * config.n_embd
        self.in_proj = nn.Linear(
            config.n_embd,
            hidden_dim,
            bias=config.bias,
        )
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(
            hidden_dim,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to a ``(batch, time, channel)`` tensor."""

        if x.ndim != 3:
            raise ValueError(
                "MLP input must have shape (batch, sequence, channels); "
                f"received {tuple(x.shape)}"
            )
        if x.shape[-1] != self.n_embd:
            raise ValueError(
                f"input channel dimension {x.shape[-1]} does not match "
                f"configured embedding dimension {self.n_embd}"
            )

        return self.dropout(self.out_proj(self.activation(self.in_proj(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block with attention and MLP residuals."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        flash_provider_loader: FlashProviderLoader | None = None,
        layer_index: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()

        self.use_rope = config.use_rope
        self.ln_1 = build_norm(config)
        self.attn = CausalSelfAttention(
            config,
            flash_provider_loader=flash_provider_loader,
            layer_index=layer_index,
        )
        self.ln_2 = build_norm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        kv_cache: KVCacheTransaction | None = None,
    ) -> torch.Tensor:
        """Apply pre-normalized attention and feed-forward residual updates."""

        attention_residual = x
        normalized = self.ln_1(attention_residual)
        if self.use_rope:
            if positions is None:
                raise ValueError("rotary block requires absolute positions")
            attended = (
                self.attn(normalized, positions=positions)
                if kv_cache is None
                else self.attn(
                    normalized,
                    positions=positions,
                    kv_cache=kv_cache,
                )
            )
        else:
            attended = (
                self.attn(normalized)
                if kv_cache is None
                else self.attn(normalized, kv_cache=kv_cache)
            )
        x = attention_residual + attended

        mlp_residual = x
        return mlp_residual + self.mlp(self.ln_2(mlp_residual))


class GPT(nn.Module):
    """Decoder-only GPT assembled from learned embeddings and transformer blocks."""

    def __init__(
        self,
        config: GPTConfig,
        *,
        flash_provider_loader: FlashProviderLoader | None = None,
    ) -> None:
        super().__init__()
        config.validate()

        self.config = config
        self.max_seq_len = config.seq_len
        self.activation_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            None if config.use_rope else nn.Embedding(config.seq_len, config.n_embd)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    config,
                    flash_provider_loader=flash_provider_loader,
                    layer_index=layer_index,
                )
                for layer_index in range(config.n_layer)
            ]
        )
        self.ln_f = build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def attention_backend_selection(self) -> AttentionBackendSelection:
        """Return the common backend outcome observed by all decoder blocks."""

        selections: set[AttentionBackendSelection] = set()
        for block in self.blocks:
            if not isinstance(
                block, Block
            ):  # pragma: no cover - constructor invariant.
                raise RuntimeError("decoder block list contains a non-block module")
            selections.add(block.attn.last_backend_selection)
        if len(selections) != 1:  # pragma: no cover - identical blocks share facts.
            raise RuntimeError("decoder blocks observed mixed attention backends")
        return next(iter(selections))

    def prepare_attention_backend(
        self,
        resolution: AttentionBackendResolution,
    ) -> None:
        """Bind one preflight result across every decoder block."""

        if not isinstance(resolution, AttentionBackendResolution):
            raise TypeError("resolution must be an AttentionBackendResolution")
        for block in self.blocks:
            if not isinstance(
                block, Block
            ):  # pragma: no cover - constructor invariant.
                raise RuntimeError("decoder block list contains a non-block module")
            block.attn.prepare_attention_backend(resolution)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Select training-only non-reentrant block recomputation."""

        if not isinstance(enabled, bool):
            raise TypeError("activation checkpointing enabled must be a boolean")
        self.activation_checkpointing = enabled

    def create_kv_cache(
        self,
        *,
        batch_size: int,
        capacity: int | None = None,
    ) -> KVCache:
        """Allocate external cache storage matching this model's parameters."""

        active_capacity = self.max_seq_len if capacity is None else capacity
        if (
            isinstance(active_capacity, bool)
            or not isinstance(active_capacity, int)
            or active_capacity <= 0
        ):
            raise KVCacheError("cache capacity must be a positive integer")
        if active_capacity > self.max_seq_len:
            raise KVCacheError(
                f"cache capacity {active_capacity} exceeds model context "
                f"length {self.max_seq_len}"
            )
        reference = self.token_embedding.weight
        return KVCache(
            layer_count=len(self.blocks),
            batch_size=batch_size,
            kv_head_count=self.config.n_head,
            head_dimension=self.config.n_embd // self.config.n_head,
            capacity=active_capacity,
            device=reference.device,
            dtype=reference.dtype,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_reduction: str = "mean",
        *,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Return next-token logits or reduced loss for a batch of token IDs."""

        if token_ids.ndim != 2:
            raise ValueError(
                "GPT input must have shape (batch, sequence); "
                f"received {tuple(token_ids.shape)}"
            )
        sequence_length = token_ids.shape[1]
        if sequence_length == 0:
            raise ValueError("GPT input sequence must not be empty")
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured "
                f"context length {self.max_seq_len}"
            )
        transaction: KVCacheTransaction | None = None
        position_start = 0
        if kv_cache is not None:
            if self.training or torch.is_grad_enabled():
                raise KVCacheError(
                    "KV cache is inference-only; use eval() with no_grad or "
                    "inference_mode"
                )
            if targets is not None:
                raise KVCacheError("KV cache does not support training targets")
            if kv_cache.position + sequence_length > self.max_seq_len:
                raise KVCacheError(
                    f"cached position {kv_cache.position} plus {sequence_length} "
                    f"tokens exceeds model context length {self.max_seq_len}"
                )
            if kv_cache.metadata.layer_count != len(self.blocks):
                raise KVCacheError(
                    "cache layer_count mismatch: expected "
                    f"{len(self.blocks)}, got {kv_cache.metadata.layer_count}"
                )
            reference = self.token_embedding.weight
            transaction = kv_cache.begin(
                token_count=sequence_length,
                batch_size=token_ids.shape[0],
                kv_head_count=self.config.n_head,
                head_dimension=self.config.n_embd // self.config.n_head,
                device=reference.device,
                dtype=reference.dtype,
            )
            position_start = transaction.start
        try:
            positions = torch.arange(
                position_start,
                position_start + sequence_length,
                device=token_ids.device,
            )
            x = self.token_embedding(token_ids)
            if self.position_embedding is not None:
                x = x + self.position_embedding(positions)
            x = self.embedding_dropout(x)
            rotary_positions = positions if self.config.use_rope else None
            for block_index, block in enumerate(self.blocks):
                if (
                    self.activation_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    x = checkpoint(
                        self._run_checkpointed_block,
                        block,
                        block_index,
                        x,
                        rotary_positions,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                elif transaction is None:
                    x = (
                        block(x)
                        if rotary_positions is None
                        else block(x, positions=rotary_positions)
                    )
                else:
                    if not isinstance(block, Block):
                        raise KVCacheError(
                            "cached decoder block list contains a non-block module"
                        )
                    x = block(
                        x,
                        positions=rotary_positions,
                        kv_cache=transaction,
                    )
            logits = self.lm_head(self.ln_f(x))
        except Exception:
            if transaction is not None:
                transaction.rollback()
            raise
        if transaction is not None:
            transaction.commit()
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction == "none":
            return loss.reshape(targets.shape)
        return loss

    @staticmethod
    def _run_checkpointed_block(
        block: nn.Module,
        block_index: int,
        x: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        try:
            return block(x) if positions is None else block(x, positions=positions)
        except Exception as error:
            if type(error).__name__ == "_StopRecomputationError":
                raise
            raise RuntimeError(
                f"activation checkpoint block {block_index} failed during "
                f"forward or recomputation: {error}"
            ) from error
