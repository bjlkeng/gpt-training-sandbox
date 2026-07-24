"""Shared no-cache autoregressive generation for decoder-only models."""

from __future__ import annotations

import math

import torch
from torch import nn


def generate(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Append sampled next-token IDs while cropping model inputs to context size."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    max_seq_len = getattr(model, "max_seq_len", None)
    if (
        not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or max_seq_len <= 0
    ):
        raise ValueError("model.max_seq_len must be a positive integer")
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError(f"token_ids must be a Tensor, got {type(token_ids).__name__}")
    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape (batch, sequence); "
            f"received {tuple(token_ids.shape)}"
        )
    if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
        raise ValueError("token_ids must contain at least one sequence and token")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError(
            f"max_new_tokens must be an integer, got {type(max_new_tokens).__name__}"
        )
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError(
            f"temperature must be a number, got {type(temperature).__name__}"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(
            f"temperature must be finite and non-negative, got {temperature}"
        )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    generated = token_ids.clone()
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=generated.device)
        generator.manual_seed(seed)

    module_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = generated[:, -max_seq_len:]
                logits = model(context)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(
                        "model must return a Tensor of next-token logits, "
                        f"got {type(logits).__name__}"
                    )
                expected_prefix = (context.shape[0], context.shape[1])
                if logits.ndim != 3 or logits.shape[:2] != expected_prefix:
                    raise ValueError(
                        "model logits must have shape (batch, sequence, vocab); "
                        f"received {tuple(logits.shape)}"
                    )
                if logits.shape[-1] == 0:
                    raise ValueError(
                        "model logits vocabulary dimension must not be empty"
                    )
                next_token_logits = logits[:, -1, :]
                if temperature == 0:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token_logits = next_token_logits / temperature
                    if top_k is not None:
                        k = min(top_k, next_token_logits.shape[-1])
                        values, indices = torch.topk(next_token_logits, k=k, dim=-1)
                        filtered_logits = torch.full_like(
                            next_token_logits,
                            -torch.inf,
                        )
                        filtered_logits.scatter_(dim=-1, index=indices, src=values)
                        next_token_logits = filtered_logits
                    probabilities = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    )
                generated = torch.cat((generated, next_token), dim=1)
    finally:
        for module, training_mode in module_modes:
            module.training = training_mode

    return generated


__all__ = ["generate"]
