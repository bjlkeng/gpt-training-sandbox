"""Deterministic fixed-prompt base sampling and Markdown artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Final

import torch
from torch import nn

from scratch_llm.generation import (
    CompletionReason,
    generate_sequences,
)
from scratch_llm.tokenizer import Tokenizer
from scratch_llm.utils import atomic_write, get_device


FIXED_BASE_SAMPLES_FORMAT: Final = "scratch_llm_fixed_base_samples"
FIXED_BASE_SAMPLES_FORMAT_VERSION: Final = 1
FIXED_BASE_PROMPTS: Final = (
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
)
FIXED_BASE_PROMPT_SET_IDENTITY: Final = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            {
                "format": "scratch_llm_fixed_base_prompt_set",
                "format_version": 1,
                "prompts": list(FIXED_BASE_PROMPTS),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
)
_MAX_SEED = 2**63 - len(FIXED_BASE_PROMPTS)


@dataclass(frozen=True)
class FixedBaseSamplingConfig:
    """Frozen generation settings for the public seven-prompt suite."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    seed: int = 42

    def __post_init__(self) -> None:
        _positive_integer(self.max_new_tokens, name="max_new_tokens")
        temperature = _finite_non_negative_real(
            self.temperature,
            name="temperature",
        )
        object.__setattr__(self, "temperature", temperature)
        if self.top_k is not None:
            _positive_integer(self.top_k, name="top_k")
        seed = _non_negative_integer(self.seed, name="seed")
        if seed > _MAX_SEED:
            raise ValueError(f"seed must be at most {_MAX_SEED}")

    def to_dict(self) -> dict[str, object]:
        """Return the exact deterministic generation protocol."""

        return {
            "max_new_tokens": self.max_new_tokens,
            "prompt_seed_strategy": "seed_plus_prompt_index",
            "seed": self.seed,
            "stop_tokens": "tokenizer_bos_only",
            "temperature": self.temperature,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class BaseSample:
    """One fixed prompt and its visible completion plus timing metadata."""

    prompt_index: int
    prompt: str
    prompt_token_count: int
    seed: int
    generated_token_ids: tuple[int, ...]
    sampled_token_count: int
    elapsed_seconds: float
    completion_reason: CompletionReason
    stop_token_id: int | None
    text: str

    def __post_init__(self) -> None:
        prompt_index = _non_negative_integer(
            self.prompt_index,
            name="prompt_index",
        )
        if prompt_index >= len(FIXED_BASE_PROMPTS):
            raise ValueError(
                f"prompt_index must be less than {len(FIXED_BASE_PROMPTS)}"
            )
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        _positive_integer(self.prompt_token_count, name="prompt_token_count")
        _non_negative_integer(self.seed, name="seed")
        _token_tuple(self.generated_token_ids, name="generated_token_ids")
        sampled_token_count = _positive_integer(
            self.sampled_token_count,
            name="sampled_token_count",
        )
        elapsed_seconds = _finite_positive_real(
            self.elapsed_seconds,
            name="elapsed_seconds",
        )
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)
        if self.completion_reason == "stop_token":
            if self.stop_token_id is None:
                raise ValueError("stop_token_id is required for stop_token completion")
            stop_token_id = _non_negative_integer(
                self.stop_token_id,
                name="stop_token_id",
            )
            if stop_token_id in self.generated_token_ids:
                raise ValueError(
                    "generated_token_ids must exclude the sampled stop token"
                )
            expected_sampled = len(self.generated_token_ids) + 1
        elif self.completion_reason == "max_new_tokens":
            if self.stop_token_id is not None:
                raise ValueError(
                    "stop_token_id must be None for max_new_tokens completion"
                )
            expected_sampled = len(self.generated_token_ids)
        else:
            raise ValueError(
                "completion_reason must be 'stop_token' or 'max_new_tokens'"
            )
        if sampled_token_count != expected_sampled:
            raise ValueError(
                "sampled_token_count does not match generated tokens and "
                "completion reason"
            )
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a string, got {type(self.text).__name__}")

    @property
    def generated_token_count(self) -> int:
        """Return visible generated tokens, excluding a sampled stop."""

        return len(self.generated_token_ids)

    @property
    def tokens_per_second(self) -> float:
        """Return sampled model steps per elapsed second."""

        return self.sampled_token_count / self.elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible sample fields."""

        return {
            "completion_reason": self.completion_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "generated_token_count": self.generated_token_count,
            "generated_token_ids": list(self.generated_token_ids),
            "prompt": self.prompt,
            "prompt_index": self.prompt_index,
            "prompt_token_count": self.prompt_token_count,
            "sampled_token_count": self.sampled_token_count,
            "seed": self.seed,
            "stop_token_id": self.stop_token_id,
            "text": self.text,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class BaseSamplesResult:
    """Immutable identity and output contract for all fixed base prompts."""

    checkpoint_identity: str
    tokenizer_identity: str
    prompt_set_identity: str
    generation_identity: str
    bos_token_id: int
    config: FixedBaseSamplingConfig
    samples: tuple[BaseSample, ...]

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_identity",
            "tokenizer_identity",
            "prompt_set_identity",
            "generation_identity",
        ):
            _non_empty_string(getattr(self, name), name=name)
        if self.prompt_set_identity != FIXED_BASE_PROMPT_SET_IDENTITY:
            raise ValueError(
                "prompt_set_identity does not match the frozen base prompt suite"
            )
        bos_token_id = _non_negative_integer(
            self.bos_token_id,
            name="bos_token_id",
        )
        if not isinstance(self.config, FixedBaseSamplingConfig):
            raise TypeError(
                "config must be a FixedBaseSamplingConfig, "
                f"got {type(self.config).__name__}"
            )
        expected_generation_identity = _generation_identity(
            self.config,
            bos_token_id=bos_token_id,
        )
        if self.generation_identity != expected_generation_identity:
            raise ValueError(
                "generation_identity does not match config and BOS stop token"
            )
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple")
        if len(self.samples) != len(FIXED_BASE_PROMPTS):
            raise ValueError(
                f"samples must contain exactly {len(FIXED_BASE_PROMPTS)} results"
            )
        for index, (prompt, sample) in enumerate(
            zip(FIXED_BASE_PROMPTS, self.samples, strict=True)
        ):
            if not isinstance(sample, BaseSample):
                raise TypeError("samples must contain only BaseSample values")
            if sample.prompt_index != index or sample.prompt != prompt:
                raise ValueError("samples must preserve frozen prompt order and text")
            if sample.seed != self.config.seed + index:
                raise ValueError("sample seed does not match seed_plus_prompt_index")
            if (
                sample.completion_reason == "stop_token"
                and sample.stop_token_id != bos_token_id
            ):
                raise ValueError("base samples may stop only on tokenizer BOS")

    @property
    def prompts(self) -> tuple[str, ...]:
        """Return prompts in their frozen public order."""

        return tuple(sample.prompt for sample in self.samples)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical reusable evaluation payload."""

        return {
            "checkpoint_identity": self.checkpoint_identity,
            "format": FIXED_BASE_SAMPLES_FORMAT,
            "format_version": FIXED_BASE_SAMPLES_FORMAT_VERSION,
            "generation": {
                "config": self.config.to_dict(),
                "identity": self.generation_identity,
                "stop_token_id": self.bos_token_id,
            },
            "prompt_set": {
                "identity": self.prompt_set_identity,
                "prompts": list(FIXED_BASE_PROMPTS),
            },
            "samples": [sample.to_dict() for sample in self.samples],
            "tokenizer_identity": self.tokenizer_identity,
        }


def generate_fixed_base_samples(
    model: nn.Module,
    tokenizer: Tokenizer,
    *,
    checkpoint_identity: str,
    config: FixedBaseSamplingConfig,
    device: str | torch.device,
    clock: Callable[[], float] = time.monotonic,
) -> BaseSamplesResult:
    """Generate the frozen seven-prompt suite without writing partial output."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    if not isinstance(config, FixedBaseSamplingConfig):
        raise TypeError(
            f"config must be a FixedBaseSamplingConfig, got {type(config).__name__}"
        )
    checkpoint_identity = _non_empty_string(
        checkpoint_identity,
        name="checkpoint_identity",
    )
    tokenizer_identity = _non_empty_string(
        tokenizer.get_identity(),
        name="tokenizer identity",
    )
    if not callable(clock):
        raise TypeError("clock must be callable")
    resolved_device = get_device(device)
    bos_token_id = _non_negative_integer(
        tokenizer.get_bos_token_id(),
        name="tokenizer BOS token ID",
    )
    if bos_token_id >= tokenizer.get_vocab_size():
        raise ValueError("tokenizer BOS token ID must be within its vocabulary")

    samples: list[BaseSample] = []
    for prompt_index, prompt in enumerate(FIXED_BASE_PROMPTS):
        prompt_token_ids = tuple(tokenizer.encode(prompt))
        _token_tuple(prompt_token_ids, name=f"prompt {prompt_index} token IDs")
        if not prompt_token_ids:
            raise ValueError(f"fixed prompt {prompt_index} encoded to no tokens")
        seed = config.seed + prompt_index
        prompt_tensor = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=resolved_device,
        )
        started_at = _clock_value(clock(), name="clock start")
        generated = generate_sequences(
            model,
            prompt_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            seed=seed,
            stop_token_ids={bos_token_id},
        )
        finished_at = _clock_value(clock(), name="clock end")
        elapsed_seconds = finished_at - started_at
        if elapsed_seconds <= 0:
            raise ValueError("clock must advance by a positive amount for each sample")
        sequence = generated.sequences[0]
        samples.append(
            BaseSample(
                prompt_index=prompt_index,
                prompt=prompt,
                prompt_token_count=len(prompt_token_ids),
                seed=seed,
                generated_token_ids=sequence.generated_token_ids,
                sampled_token_count=sequence.sampled_token_count,
                elapsed_seconds=elapsed_seconds,
                completion_reason=sequence.completion_reason,
                stop_token_id=sequence.stop_token_id,
                text=tokenizer.decode(sequence.generated_token_ids),
            )
        )

    return BaseSamplesResult(
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer_identity,
        prompt_set_identity=FIXED_BASE_PROMPT_SET_IDENTITY,
        generation_identity=_generation_identity(
            config,
            bos_token_id=bos_token_id,
        ),
        bos_token_id=bos_token_id,
        config=config,
        samples=tuple(samples),
    )


def write_base_samples_markdown(
    result: BaseSamplesResult,
    metrics_dir: str | os.PathLike[str],
) -> Path:
    """Atomically publish ``metrics/base_samples.md`` from one full result."""

    if not isinstance(result, BaseSamplesResult):
        raise TypeError(
            f"result must be a BaseSamplesResult, got {type(result).__name__}"
        )
    return atomic_write(
        Path(metrics_dir) / "base_samples.md",
        _render_markdown(result),
    )


def generate_and_write_fixed_base_samples(
    model: nn.Module,
    tokenizer: Tokenizer,
    metrics_dir: str | os.PathLike[str],
    *,
    checkpoint_identity: str,
    config: FixedBaseSamplingConfig,
    device: str | torch.device,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[BaseSamplesResult, Path]:
    """Generate every prompt successfully before atomically publishing Markdown."""

    result = generate_fixed_base_samples(
        model,
        tokenizer,
        checkpoint_identity=checkpoint_identity,
        config=config,
        device=device,
        clock=clock,
    )
    return result, write_base_samples_markdown(result, metrics_dir)


def _render_markdown(result: BaseSamplesResult) -> str:
    lines = [
        "# Fixed base samples",
        "",
        f"- Checkpoint identity: {_inline_code(result.checkpoint_identity)}",
        f"- Tokenizer identity: {_inline_code(result.tokenizer_identity)}",
        f"- Prompt-set identity: {_inline_code(result.prompt_set_identity)}",
        f"- Generation identity: {_inline_code(result.generation_identity)}",
        f"- BOS stop token ID: `{result.bos_token_id}`",
        (f"- Generation config: `{_compact_json(result.config.to_dict())}`"),
        "",
    ]
    for sample in result.samples:
        lines.extend(
            [
                f"## Sample {sample.prompt_index + 1}",
                "",
                "### Prompt",
                "",
                _fenced_text(sample.prompt),
                "",
                "### Completion",
                "",
                _fenced_text(sample.text),
                "",
                (
                    f"- Completion reason: `{sample.completion_reason}`; "
                    f"stop token ID: `{sample.stop_token_id}`"
                ),
                (
                    f"- Tokens: prompt `{sample.prompt_token_count}`, "
                    f"visible generated `{sample.generated_token_count}`, "
                    f"sampled `{sample.sampled_token_count}`"
                ),
                (
                    f"- Timing: `{sample.elapsed_seconds:.6f}` seconds; "
                    f"`{sample.tokens_per_second:.3f}` sampled tokens/second"
                ),
                f"- Seed: `{sample.seed}`",
                "",
            ]
        )
    return "\n".join(lines)


def _generation_identity(
    config: FixedBaseSamplingConfig,
    *,
    bos_token_id: int,
) -> str:
    return _identity(
        {
            "config": config.to_dict(),
            "format": "scratch_llm_fixed_base_generation",
            "format_version": 1,
            "stop_token_ids": [bos_token_id],
        }
    )


def _identity(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _compact_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _inline_code(value: str) -> str:
    maximum_run = _maximum_character_run(value, "`")
    delimiter = "`" * max(1, maximum_run + 1)
    padding = " " if value.startswith("`") or value.endswith("`") else ""
    return f"{delimiter}{padding}{value}{padding}{delimiter}"


def _fenced_text(value: str) -> str:
    fence = "`" * max(3, _maximum_character_run(value, "`") + 1)
    content = value if value.endswith("\n") else value + "\n"
    return f"{fence}text\n{content}{fence}"


def _maximum_character_run(value: str, character: str) -> int:
    maximum = 0
    current = 0
    for item in value:
        if item == character:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _clock_value(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _finite_positive_real(value: object, *, name: str) -> float:
    normalized = _finite_non_negative_real(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _finite_non_negative_real(value: object, *, name: str) -> float:
    normalized = _clock_value(value, name=name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    normalized = _non_negative_integer(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _non_negative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _token_tuple(value: object, *, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    for position, token_id in enumerate(value):
        _non_negative_integer(token_id, name=f"{name}[{position}]")


__all__ = [
    "FIXED_BASE_PROMPTS",
    "FIXED_BASE_PROMPT_SET_IDENTITY",
    "FIXED_BASE_SAMPLES_FORMAT",
    "FIXED_BASE_SAMPLES_FORMAT_VERSION",
    "BaseSample",
    "BaseSamplesResult",
    "FixedBaseSamplingConfig",
    "generate_and_write_fixed_base_samples",
    "generate_fixed_base_samples",
    "write_base_samples_markdown",
]
