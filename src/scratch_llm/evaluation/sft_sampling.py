"""Deterministic fixed public chat prompts and atomic SFT sample artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Final

import torch
from torch import nn

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_finite_positive_real,
    require_finite_real,
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.chat.conversation import Conversation, UserMessage
from scratch_llm.chat.rendering import CHAT_RENDERER_ID, render_completion_prompt
from scratch_llm.generation import CompletionReason, generate_sequences
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.utils import atomic_write, get_device


FIXED_SFT_SAMPLES_FORMAT: Final = "scratch_llm_fixed_sft_samples"
FIXED_SFT_SAMPLES_FORMAT_VERSION: Final = 1
FIXED_SFT_PROMPTS: Final = (
    "Explain gradient descent in simple terms.",
    "Write a Python function to reverse a string.",
    "Give me three project ideas for learning PyTorch.",
    "What is 17 * 23? Show your work.",
    "Return a JSON object with keys name, age, and city.",
)
FIXED_SFT_PROMPT_SET_IDENTITY: Final = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            {
                "format": "scratch_llm_fixed_sft_prompt_set",
                "format_version": 1,
                "prompts": list(FIXED_SFT_PROMPTS),
                "renderer_id": CHAT_RENDERER_ID,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
)
_MAX_SEED = 2**63 - len(FIXED_SFT_PROMPTS)


@dataclass(frozen=True, slots=True)
class FixedSFTSamplingConfig:
    """Frozen generation settings for the public five-prompt SFT suite."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    seed: int = 42

    def __post_init__(self) -> None:
        require_positive_integer(self.max_new_tokens, name="max_new_tokens")
        temperature = require_finite_non_negative_real(
            self.temperature,
            name="temperature",
        )
        object.__setattr__(self, "temperature", temperature)
        if self.top_k is not None:
            require_positive_integer(self.top_k, name="top_k")
        seed = require_non_negative_integer(self.seed, name="seed")
        if seed > _MAX_SEED:
            raise ValueError(f"seed must be at most {_MAX_SEED}")

    def to_dict(self) -> dict[str, object]:
        """Return the exact deterministic fixed-sample protocol."""

        return {
            "max_new_tokens": self.max_new_tokens,
            "prompt_seed_strategy": "seed_plus_prompt_index",
            "seed": self.seed,
            "stop_tokens": "assistant_end_then_bos_safety",
            "temperature": self.temperature,
            "top_k": self.top_k,
        }


@dataclass(frozen=True, slots=True)
class FixedSFTSample:
    """One public user prompt and its visible assistant completion."""

    prompt_index: int
    prompt: str
    prompt_token_ids: tuple[int, ...]
    seed: int
    generated_token_ids: tuple[int, ...]
    sampled_token_count: int
    elapsed_seconds: float
    completion_reason: CompletionReason
    stop_token_id: int | None
    text: str

    def __post_init__(self) -> None:
        prompt_index = require_non_negative_integer(
            self.prompt_index,
            name="prompt_index",
        )
        if prompt_index >= len(FIXED_SFT_PROMPTS):
            raise ValueError(f"prompt_index must be less than {len(FIXED_SFT_PROMPTS)}")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        _token_tuple(self.prompt_token_ids, name="prompt_token_ids", allow_empty=False)
        require_non_negative_integer(self.seed, name="seed")
        _token_tuple(
            self.generated_token_ids,
            name="generated_token_ids",
            allow_empty=True,
        )
        sampled_token_count = require_positive_integer(
            self.sampled_token_count,
            name="sampled_token_count",
        )
        elapsed_seconds = require_finite_positive_real(
            self.elapsed_seconds,
            name="elapsed_seconds",
        )
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)
        if self.completion_reason == "stop_token":
            if self.stop_token_id is None:
                raise ValueError("stop_token_id is required for stop_token completion")
            require_non_negative_integer(self.stop_token_id, name="stop_token_id")
            if self.stop_token_id in self.generated_token_ids:
                raise ValueError("generated_token_ids must exclude the sampled stop")
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
                "sampled_token_count does not match visible tokens and stop metadata"
            )
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a string, got {type(self.text).__name__}")

    @property
    def prompt_token_count(self) -> int:
        """Return the complete renderer-derived completion-prompt length."""

        return len(self.prompt_token_ids)

    @property
    def generated_token_count(self) -> int:
        """Return visible assistant tokens, excluding a sampled stop."""

        return len(self.generated_token_ids)

    @property
    def tokens_per_second(self) -> float:
        """Return sampled model steps per elapsed second."""

        return self.sampled_token_count / self.elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible sample metadata and public text."""

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


@dataclass(frozen=True, slots=True)
class FixedSFTSamplesResult:
    """Immutable identity and output contract for all fixed SFT prompts."""

    checkpoint_identity: str
    tokenizer_identity: str
    renderer_identity: str
    assistant_end_token_id: int
    bos_token_id: int
    config: FixedSFTSamplingConfig
    samples: tuple[FixedSFTSample, ...]

    def __post_init__(self) -> None:
        require_non_empty_string(
            self.checkpoint_identity,
            name="checkpoint_identity",
        )
        require_non_empty_string(
            self.tokenizer_identity,
            name="tokenizer_identity",
        )
        if self.renderer_identity != CHAT_RENDERER_ID:
            raise ValueError(f"renderer_identity must equal {CHAT_RENDERER_ID!r}")
        assistant_end_token_id = require_non_negative_integer(
            self.assistant_end_token_id,
            name="assistant_end_token_id",
        )
        bos_token_id = require_non_negative_integer(
            self.bos_token_id,
            name="bos_token_id",
        )
        if assistant_end_token_id == bos_token_id:
            raise ValueError("assistant_end and BOS stop token IDs must differ")
        if not isinstance(self.config, FixedSFTSamplingConfig):
            raise TypeError(
                "config must be a FixedSFTSamplingConfig, got "
                f"{type(self.config).__name__}"
            )
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple")
        if len(self.samples) != len(FIXED_SFT_PROMPTS):
            raise ValueError(
                f"samples must contain exactly {len(FIXED_SFT_PROMPTS)} results"
            )
        for index, (prompt, sample) in enumerate(
            zip(FIXED_SFT_PROMPTS, self.samples, strict=True)
        ):
            if not isinstance(sample, FixedSFTSample):
                raise TypeError("samples must contain only FixedSFTSample values")
            if sample.prompt_index != index or sample.prompt != prompt:
                raise ValueError("samples must preserve frozen prompt order and text")
            if sample.seed != self.config.seed + index:
                raise ValueError("sample seed does not match seed_plus_prompt_index")
            if (
                sample.completion_reason == "stop_token"
                and sample.stop_token_id not in {assistant_end_token_id, bos_token_id}
            ):
                raise ValueError("SFT samples may stop only on assistant_end or BOS")

    @property
    def prompt_set_identity(self) -> str:
        """Return the frozen public prompt-suite identity."""

        return FIXED_SFT_PROMPT_SET_IDENTITY

    @property
    def generation_identity(self) -> str:
        """Return the full renderer, sampling, and stop-policy identity."""

        return _identity(
            {
                "config": self.config.to_dict(),
                "format": "scratch_llm_fixed_sft_generation",
                "format_version": 1,
                "renderer_id": self.renderer_identity,
                "stop_token_ids": [
                    self.assistant_end_token_id,
                    self.bos_token_id,
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical reusable fixed-sample payload."""

        return {
            "checkpoint_identity": self.checkpoint_identity,
            "format": FIXED_SFT_SAMPLES_FORMAT,
            "format_version": FIXED_SFT_SAMPLES_FORMAT_VERSION,
            "generation": {
                "config": self.config.to_dict(),
                "identity": self.generation_identity,
                "stop_token_ids": [
                    self.assistant_end_token_id,
                    self.bos_token_id,
                ],
            },
            "prompt_set": {
                "identity": self.prompt_set_identity,
                "prompts": list(FIXED_SFT_PROMPTS),
            },
            "renderer_identity": self.renderer_identity,
            "samples": [sample.to_dict() for sample in self.samples],
            "tokenizer_identity": self.tokenizer_identity,
        }


def generate_fixed_sft_samples(
    model: nn.Module,
    tokenizer: Tokenizer,
    *,
    checkpoint_identity: str,
    config: FixedSFTSamplingConfig,
    device: str | torch.device,
    clock: Callable[[], float] = time.monotonic,
) -> FixedSFTSamplesResult:
    """Generate all fixed public prompts through the shared chat renderer."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    if not isinstance(config, FixedSFTSamplingConfig):
        raise TypeError(
            f"config must be a FixedSFTSamplingConfig, got {type(config).__name__}"
        )
    checkpoint_identity = require_non_empty_string(
        checkpoint_identity,
        name="checkpoint_identity",
    )
    if not callable(clock):
        raise TypeError("clock must be callable")
    resolved_device = get_device(device)
    assistant_end_token_id = tokenizer.encode_special("<|assistant_end|>")
    bos_token_id = tokenizer.get_bos_token_id()
    for name, token_id in (
        ("assistant_end", assistant_end_token_id),
        ("BOS", bos_token_id),
    ):
        require_non_negative_integer(token_id, name=f"{name} token ID")
        if token_id >= tokenizer.get_vocab_size():
            raise ValueError(f"tokenizer {name} token ID is outside its vocabulary")

    samples: list[FixedSFTSample] = []
    for prompt_index, prompt in enumerate(FIXED_SFT_PROMPTS):
        rendered = render_completion_prompt(
            Conversation(messages=(UserMessage(prompt),)),
            tokenizer,
        )
        seed = config.seed + prompt_index
        prompt_tensor = torch.tensor(
            [rendered.token_ids],
            dtype=torch.long,
            device=resolved_device,
        )
        started_at = require_finite_real(clock(), name="clock start")
        generated = generate_sequences(
            model,
            prompt_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            seed=seed,
            stop_token_ids={assistant_end_token_id, bos_token_id},
        )
        finished_at = require_finite_real(clock(), name="clock end")
        elapsed_seconds = finished_at - started_at
        if elapsed_seconds <= 0:
            raise ValueError("clock must advance by a positive amount for each sample")
        sequence = generated.sequences[0]
        samples.append(
            FixedSFTSample(
                prompt_index=prompt_index,
                prompt=prompt,
                prompt_token_ids=rendered.token_ids,
                seed=seed,
                generated_token_ids=sequence.generated_token_ids,
                sampled_token_count=sequence.sampled_token_count,
                elapsed_seconds=elapsed_seconds,
                completion_reason=sequence.completion_reason,
                stop_token_id=sequence.stop_token_id,
                text=tokenizer.decode(sequence.generated_token_ids),
            )
        )
    return FixedSFTSamplesResult(
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer.get_identity(),
        renderer_identity=CHAT_RENDERER_ID,
        assistant_end_token_id=assistant_end_token_id,
        bos_token_id=bos_token_id,
        config=config,
        samples=tuple(samples),
    )


def write_sft_samples_markdown(
    result: FixedSFTSamplesResult,
    metrics_dir: str | os.PathLike[str],
) -> Path:
    """Atomically publish ``metrics/sft_samples.md`` after all prompts succeed."""

    if not isinstance(result, FixedSFTSamplesResult):
        raise TypeError(
            f"result must be a FixedSFTSamplesResult, got {type(result).__name__}"
        )
    return atomic_write(
        Path(metrics_dir) / "sft_samples.md",
        _render_markdown(result),
    )


def _render_markdown(result: FixedSFTSamplesResult) -> str:
    lines = [
        "# Fixed SFT samples",
        "",
        f"- Checkpoint identity: {_inline_code(result.checkpoint_identity)}",
        f"- Tokenizer identity: {_inline_code(result.tokenizer_identity)}",
        f"- Renderer identity: {_inline_code(result.renderer_identity)}",
        f"- Prompt-set identity: {_inline_code(result.prompt_set_identity)}",
        f"- Generation identity: {_inline_code(result.generation_identity)}",
        (
            "- Stop token IDs: assistant_end "
            f"`{result.assistant_end_token_id}`, BOS `{result.bos_token_id}`"
        ),
        f"- Generation config: `{_compact_json(result.config.to_dict())}`",
        "",
    ]
    for sample in result.samples:
        lines.extend(
            [
                f"## Sample {sample.prompt_index + 1}",
                "",
                "### Public prompt",
                "",
                _fenced_text(sample.prompt),
                "",
                "### Assistant output",
                "",
                _fenced_text(sample.text),
                "",
                (
                    f"- Completion reason: `{sample.completion_reason}`; "
                    f"stop token ID: `{sample.stop_token_id}`"
                ),
                (
                    f"- Tokens: rendered prompt `{sample.prompt_token_count}`, "
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


def _token_tuple(
    values: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    for index, value in enumerate(values):
        require_non_negative_integer(value, name=f"{name}[{index}]")
    return values


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


__all__ = [
    "FIXED_SFT_PROMPTS",
    "FIXED_SFT_PROMPT_SET_IDENTITY",
    "FIXED_SFT_SAMPLES_FORMAT",
    "FIXED_SFT_SAMPLES_FORMAT_VERSION",
    "FixedSFTSample",
    "FixedSFTSamplesResult",
    "FixedSFTSamplingConfig",
    "generate_fixed_sft_samples",
    "write_sft_samples_markdown",
]
