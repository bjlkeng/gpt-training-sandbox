"""Shared no-cache autoregressive generation for decoder-only models."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import random
from typing import Literal, overload

import numpy as np
import torch
from torch import nn

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_integer,
    require_non_negative_integer,
    require_positive_integer,
)


CompletionReason = Literal["stop_token", "max_new_tokens"]
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


@dataclass(frozen=True)
class GeneratedSequence:
    """One prompt plus its variable-length visible generated token IDs."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    completion_reason: CompletionReason
    stop_token_id: int | None
    sampled_token_count: int

    def __post_init__(self) -> None:
        _validate_token_tuple(self.prompt_token_ids, name="prompt_token_ids")
        _validate_token_tuple(self.generated_token_ids, name="generated_token_ids")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if self.completion_reason not in ("stop_token", "max_new_tokens"):
            raise ValueError(
                "completion_reason must be 'stop_token' or 'max_new_tokens'"
            )
        require_non_negative_integer(
            self.sampled_token_count,
            name="sampled_token_count",
        )
        if self.completion_reason == "stop_token":
            if self.stop_token_id is None:
                raise ValueError("stop_token_id is required for stop_token completion")
            require_non_negative_integer(self.stop_token_id, name="stop_token_id")
            expected_sampled = len(self.generated_token_ids) + 1
        else:
            if self.stop_token_id is not None:
                raise ValueError(
                    "stop_token_id must be None for max_new_tokens completion"
                )
            expected_sampled = len(self.generated_token_ids)
        if self.sampled_token_count != expected_sampled:
            raise ValueError(
                "sampled_token_count does not match visible generated tokens "
                "and completion reason"
            )

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Return prompt plus visible generation, excluding a sampled stop."""

        return self.prompt_token_ids + self.generated_token_ids


@dataclass(frozen=True)
class GenerationBatchResult:
    """Variable-length generation results in original batch-row order."""

    sequences: tuple[GeneratedSequence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sequences, tuple) or not self.sequences:
            raise ValueError("sequences must be a non-empty tuple")
        if any(
            not isinstance(sequence, GeneratedSequence) for sequence in self.sequences
        ):
            raise TypeError("sequences must contain only GeneratedSequence values")


@overload
def generate(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    stop_token_ids: None = None,
) -> torch.Tensor: ...


@overload
def generate(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    stop_token_ids: Collection[int],
) -> GenerationBatchResult: ...


def generate(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    stop_token_ids: Collection[int] | None = None,
) -> torch.Tensor | GenerationBatchResult:
    """Generate tokens, returning stop metadata when a stop set is explicit.

    Calls that omit ``stop_token_ids`` retain the original rectangular Tensor
    interface. Supplying a collection returns variable-length per-row results;
    sampled stop tokens are recorded but excluded from visible token IDs.
    """

    result = generate_sequences(
        model,
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        stop_token_ids=() if stop_token_ids is None else stop_token_ids,
    )
    if stop_token_ids is not None:
        return result
    return torch.tensor(
        [sequence.token_ids for sequence in result.sequences],
        dtype=token_ids.dtype,
        device=token_ids.device,
    )


def generate_sequences(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    stop_token_ids: Collection[int] = (),
) -> GenerationBatchResult:
    """Generate variable-length rows with independent finished-row tracking."""

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
    if token_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"token_ids must use an integer dtype, got {token_ids.dtype}")
    if bool(token_ids.lt(0).any().item()):
        raise ValueError("token_ids must be non-negative")
    max_new_tokens = require_positive_integer(
        max_new_tokens,
        name="max_new_tokens",
    )
    temperature = require_finite_non_negative_real(
        temperature,
        name="temperature",
    )
    if top_k is not None:
        top_k = require_positive_integer(top_k, name="top_k")
    if seed is not None:
        seed = require_integer(seed, name="seed")
    normalized_stop_ids = _normalize_stop_token_ids(stop_token_ids)

    generated_rows = [row.clone() for row in token_ids]
    prompt_token_ids = tuple(
        tuple(int(token_id) for token_id in row.detach().cpu().tolist())
        for row in token_ids
    )
    sampled_token_counts = [0] * token_ids.shape[0]
    completion_reasons: list[CompletionReason] = ["max_new_tokens"] * token_ids.shape[0]
    observed_stop_ids: list[int | None] = [None] * token_ids.shape[0]
    finished = [False] * token_ids.shape[0]
    module_modes = [(module, module.training) for module in model.modules()]
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    cuda_rng_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized()
        else None
    )
    row_generators = _row_generators(
        token_ids.device,
        batch_size=token_ids.shape[0],
        seed=seed,
        caller_torch_rng_state=torch_rng_state,
    )
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                active_indices = [
                    index
                    for index, is_finished in enumerate(finished)
                    if not is_finished
                ]
                if not active_indices:
                    break
                context = torch.stack(
                    [generated_rows[index][-max_seq_len:] for index in active_indices]
                )
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
                if normalized_stop_ids and max(normalized_stop_ids) >= logits.shape[-1]:
                    raise ValueError(
                        "stop_token_ids must be within the model vocabulary "
                        f"of size {logits.shape[-1]}"
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
                    next_token = torch.stack(
                        [
                            torch.multinomial(
                                probabilities[active_row],
                                num_samples=1,
                                generator=row_generators[batch_index],
                            )
                            for active_row, batch_index in enumerate(active_indices)
                        ]
                    )
                for active_row, batch_index in enumerate(active_indices):
                    token_id = int(next_token[active_row, 0].item())
                    sampled_token_counts[batch_index] += 1
                    if token_id in normalized_stop_ids:
                        completion_reasons[batch_index] = "stop_token"
                        observed_stop_ids[batch_index] = token_id
                        finished[batch_index] = True
                    else:
                        generated_rows[batch_index] = torch.cat(
                            (
                                generated_rows[batch_index],
                                next_token[active_row],
                            )
                        )
    finally:
        for module, training_mode in module_modes:
            module.training = training_mode
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(list(cuda_rng_states))

    sequences = []
    prompt_length = token_ids.shape[1]
    for index, generated_row in enumerate(generated_rows):
        visible_ids = tuple(
            int(token_id)
            for token_id in generated_row[prompt_length:].detach().cpu().tolist()
        )
        sequences.append(
            GeneratedSequence(
                prompt_token_ids=prompt_token_ids[index],
                generated_token_ids=visible_ids,
                completion_reason=completion_reasons[index],
                stop_token_id=observed_stop_ids[index],
                sampled_token_count=sampled_token_counts[index],
            )
        )
    return GenerationBatchResult(tuple(sequences))


def _row_generators(
    device: torch.device,
    *,
    batch_size: int,
    seed: int | None,
    caller_torch_rng_state: torch.Tensor,
) -> tuple[torch.Generator, ...]:
    if seed is None:
        seed_source = torch.Generator(device="cpu")
        seed_source.set_state(caller_torch_rng_state)
        resolved_seed = int(
            torch.randint(
                0,
                2**63 - 1,
                (1,),
                dtype=torch.int64,
                generator=seed_source,
            ).item()
        )
    else:
        resolved_seed = seed

    generators = []
    for _ in range(batch_size):
        generator = torch.Generator(device=device)
        generator.manual_seed(resolved_seed)
        generators.append(generator)
    return tuple(generators)


def _normalize_stop_token_ids(
    stop_token_ids: Collection[int],
) -> frozenset[int]:
    if isinstance(stop_token_ids, (str, bytes)) or not isinstance(
        stop_token_ids,
        Collection,
    ):
        raise TypeError("stop_token_ids must be a collection of integers")
    normalized: set[int] = set()
    for position, token_id in enumerate(stop_token_ids):
        normalized.add(
            require_non_negative_integer(
                token_id,
                name=f"stop_token_ids item {position}",
            )
        )
    return frozenset(normalized)


def _validate_token_tuple(value: object, *, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    for position, token_id in enumerate(value):
        require_non_negative_integer(token_id, name=f"{name}[{position}]")


__all__ = [
    "CompletionReason",
    "GeneratedSequence",
    "GenerationBatchResult",
    "generate",
    "generate_sequences",
]
