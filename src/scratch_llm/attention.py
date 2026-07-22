"""Readable, manual causal self-attention for the baseline GPT."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from scratch_llm.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit lower-triangular mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.max_seq_len = config.seq_len

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

        causal_mask = torch.tril(
            torch.ones(config.seq_len, config.seq_len, dtype=torch.bool)
        )
        self.causal_mask: torch.Tensor
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))

        attended = weights @ v
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, channels)
        )
        return self.output_dropout(self.out_proj(attended))
