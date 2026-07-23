"""Feed-forward and transformer building blocks for the baseline GPT."""

from __future__ import annotations

import torch
from torch import nn

from scratch_llm.attention import CausalSelfAttention
from scratch_llm.config import GPTConfig


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

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()

        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-normalized attention and feed-forward residual updates."""

        attention_residual = x
        x = attention_residual + self.attn(self.ln_1(attention_residual))

        mlp_residual = x
        return mlp_residual + self.mlp(self.ln_2(mlp_residual))


class GPT(nn.Module):
    """Decoder-only GPT assembled from learned embeddings and transformer blocks."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()

        self.config = config
        self.max_seq_len = config.seq_len
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.seq_len, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for a ``(batch, sequence)`` token tensor."""

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
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))
