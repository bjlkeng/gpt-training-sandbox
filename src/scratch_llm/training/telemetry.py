"""Pure FLOPs and utilization telemetry for baseline GPT training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_finite_positive_real,
    require_finite_real,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.config import GPTConfig, TrainConfig


TRAINING_FLOPS_FORMULA_ID: Final = "gpt_layer_window_training_v2"
_TRAINING_FLOPS_ASSUMPTIONS: Final = (
    "one multiply-accumulate is two FLOPs",
    "backward costs twice the modeled forward matrix multiplications",
    "full-context layers execute sequence-length score and value products",
    "short-window layers use their declared maximum visible key span",
    "embedding lookup, normalization, activation, bias, softmax, dropout, "
    "loss, clipping, optimizer, and scheduler FLOPs are excluded",
)


@dataclass(frozen=True)
class GPTTrainingFlopsEstimate:
    """Documented matrix-multiplication FLOPs for one baseline GPT token."""

    formula_id: str
    tie_weights: bool
    n_head: int
    n_kv_head: int
    use_gqa: bool
    sliding_window_pattern: str
    sliding_window_size: int
    layer_key_spans: tuple[int, ...]
    sequence_length: int
    executed_weight_elements: int
    linear_flops_per_token: int
    attention_flops_per_token: int
    flops_per_token: int
    assumptions: tuple[str, ...]

    def flops_for_tokens(self, processed_model_tokens: int) -> int:
        """Return training FLOPs for positions at the configured sequence length."""

        processed_model_tokens = require_positive_integer(
            processed_model_tokens,
            name="processed_model_tokens",
        )
        return processed_model_tokens * self.flops_per_token

    def to_dict(self) -> dict[str, object]:
        """Return the formula and its complete reproducibility assumptions."""

        return {
            "assumptions": list(self.assumptions),
            "attention_flops_per_token": self.attention_flops_per_token,
            "executed_weight_elements": self.executed_weight_elements,
            "flops_per_token": self.flops_per_token,
            "formula_id": self.formula_id,
            "linear_flops_per_token": self.linear_flops_per_token,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "sequence_length": self.sequence_length,
            "sliding_window": {
                "layer_key_spans": list(self.layer_key_spans),
                "pattern": self.sliding_window_pattern,
                "short_window_size": self.sliding_window_size,
            },
            "tie_weights": self.tie_weights,
            "use_gqa": self.use_gqa,
        }


@dataclass(frozen=True)
class PeakFlopsBasis:
    """Explicit hardware and arithmetic basis used as the MFU denominator."""

    flops_per_second: float
    description: str

    def __post_init__(self) -> None:
        flops_per_second = require_finite_positive_real(
            self.flops_per_second,
            name="flops_per_second",
        )
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        object.__setattr__(self, "flops_per_second", flops_per_second)
        object.__setattr__(self, "description", self.description.strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "flops_per_second": self.flops_per_second,
        }


@dataclass(frozen=True)
class TrainingStepTelemetry:
    """Immutable actual-work telemetry for one completed optimizer step."""

    processed_model_tokens: int
    supervised_target_tokens: int
    duration_seconds: float
    tokens_per_second: float
    step_flops: int
    total_training_flops: float
    total_training_time_seconds: float
    mfu: float | None
    peak_flops_basis: PeakFlopsBasis | None
    peak_memory_mib: float | None
    flops_estimate: GPTTrainingFlopsEstimate

    def __post_init__(self) -> None:
        processed_model_tokens = require_positive_integer(
            self.processed_model_tokens,
            name="processed_model_tokens",
        )
        require_non_negative_integer(
            self.supervised_target_tokens,
            name="supervised_target_tokens",
        )
        if self.supervised_target_tokens > processed_model_tokens:
            raise ValueError(
                "supervised_target_tokens cannot exceed processed_model_tokens"
            )
        duration_seconds = require_finite_positive_real(
            self.duration_seconds,
            name="duration_seconds",
        )
        tokens_per_second = require_finite_positive_real(
            self.tokens_per_second,
            name="tokens_per_second",
        )
        if not math.isclose(
            tokens_per_second,
            processed_model_tokens / duration_seconds,
            rel_tol=1e-12,
        ):
            raise ValueError(
                "tokens_per_second must use processed_model_tokens / duration_seconds"
            )
        step_flops = require_positive_integer(self.step_flops, name="step_flops")
        total_training_flops = require_finite_non_negative_real(
            self.total_training_flops,
            name="total_training_flops",
        )
        if total_training_flops < step_flops:
            raise ValueError("total_training_flops cannot be less than step_flops")
        total_training_time_seconds = require_finite_non_negative_real(
            self.total_training_time_seconds,
            name="total_training_time_seconds",
        )
        if total_training_time_seconds < duration_seconds:
            raise ValueError(
                "total_training_time_seconds cannot be less than duration_seconds"
            )
        if not isinstance(self.flops_estimate, GPTTrainingFlopsEstimate):
            raise TypeError(
                "flops_estimate must be a GPTTrainingFlopsEstimate, got "
                f"{type(self.flops_estimate).__name__}"
            )
        if self.flops_estimate.flops_for_tokens(processed_model_tokens) != step_flops:
            raise ValueError(
                "step_flops must match the estimator and processed_model_tokens"
            )
        if (self.mfu is None) != (self.peak_flops_basis is None):
            raise ValueError("mfu and peak_flops_basis must either both be set or null")
        if self.mfu is not None:
            if not isinstance(self.peak_flops_basis, PeakFlopsBasis):
                raise TypeError("peak_flops_basis must be a PeakFlopsBasis")
            mfu = require_finite_non_negative_real(self.mfu, name="mfu")
            expected_mfu = (
                step_flops / duration_seconds / self.peak_flops_basis.flops_per_second
            )
            if not math.isclose(mfu, expected_mfu, rel_tol=1e-12):
                raise ValueError(
                    "mfu must use measured FLOPs/second divided by the explicit peak"
                )
            object.__setattr__(self, "mfu", mfu)
        if self.peak_memory_mib is not None:
            object.__setattr__(
                self,
                "peak_memory_mib",
                require_finite_non_negative_real(
                    self.peak_memory_mib,
                    name="peak_memory_mib",
                ),
            )
        object.__setattr__(self, "duration_seconds", duration_seconds)
        object.__setattr__(self, "tokens_per_second", tokens_per_second)
        object.__setattr__(self, "total_training_flops", total_training_flops)
        object.__setattr__(
            self,
            "total_training_time_seconds",
            total_training_time_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete domain result without tracker-specific aliases."""

        return {
            "duration_seconds": self.duration_seconds,
            "flops_estimate": self.flops_estimate.to_dict(),
            "mfu": self.mfu,
            "peak_flops_basis": (
                None
                if self.peak_flops_basis is None
                else self.peak_flops_basis.to_dict()
            ),
            "peak_memory_mib": self.peak_memory_mib,
            "processed_model_tokens": self.processed_model_tokens,
            "step_flops": self.step_flops,
            "supervised_target_tokens": self.supervised_target_tokens,
            "tokens_per_second": self.tokens_per_second,
            "total_training_flops": self.total_training_flops,
            "total_training_time_seconds": self.total_training_time_seconds,
        }


def base_training_metrics(
    telemetry: TrainingStepTelemetry,
    *,
    loss: float,
    learning_rate_multiplier: float,
    grad_norm: float,
    epoch: float | None,
) -> dict[str, float | None]:
    """Map one completed base-training result to the public metric contract."""

    if not isinstance(telemetry, TrainingStepTelemetry):
        raise TypeError(
            f"telemetry must be a TrainingStepTelemetry, got {type(telemetry).__name__}"
        )
    metrics: dict[str, float | None] = {
        "train/loss": require_finite_real(loss, name="loss"),
        "train/lrm": require_finite_non_negative_real(
            learning_rate_multiplier,
            name="learning_rate_multiplier",
        ),
        "train/dt": telemetry.duration_seconds,
        "train/tok_per_sec": telemetry.tokens_per_second,
        "train/mfu": telemetry.mfu,
        "train/grad_norm": require_finite_non_negative_real(
            grad_norm,
            name="grad_norm",
        ),
        "total_training_flops": telemetry.total_training_flops,
        "total_training_time": telemetry.total_training_time_seconds,
    }
    if epoch is not None:
        metrics["train/epoch"] = require_finite_non_negative_real(
            epoch,
            name="epoch",
        )
    if telemetry.peak_memory_mib is not None:
        metrics["train/peak_memory_mib"] = telemetry.peak_memory_mib
    return metrics


def estimate_gpt_training_flops(config: GPTConfig) -> GPTTrainingFlopsEstimate:
    """Estimate dense forward-and-backward matmul FLOPs per processed token.

    Input embedding lookup does not execute a matrix multiplication. The output
    projection does, even when its weight aliases the input embedding, so
    weight tying changes storage but not this compute estimate.
    """

    if not isinstance(config, GPTConfig):
        raise TypeError(f"config must be a GPTConfig, got {type(config).__name__}")
    config.validate()

    channels = config.n_embd
    if config.n_kv_head is None:  # pragma: no cover - validated resolution.
        raise RuntimeError("validated config lost n_kv_head")
    kv_width = config.n_kv_head * (channels // config.n_head)
    per_layer_weights = (
        2 * channels**2 + 2 * channels * kv_width + 2 * config.mlp_ratio * channels**2
    )
    transformer_weights = config.n_layer * per_layer_weights
    output_weights = channels * config.vocab_size
    executed_weight_elements = transformer_weights + output_weights
    linear_flops_per_token = 6 * executed_weight_elements
    layer_key_spans = tuple(
        config.seq_len if window is None else min(config.seq_len, window + 1)
        for window in config.layer_attention_windows()
    )
    attention_flops_per_token = 12 * channels * sum(layer_key_spans)
    return GPTTrainingFlopsEstimate(
        formula_id=TRAINING_FLOPS_FORMULA_ID,
        tie_weights=config.tie_weights,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        use_gqa=config.use_gqa,
        sliding_window_pattern=config.sliding_window_pattern,
        sliding_window_size=config.sliding_window_size,
        layer_key_spans=layer_key_spans,
        sequence_length=config.seq_len,
        executed_weight_elements=executed_weight_elements,
        linear_flops_per_token=linear_flops_per_token,
        attention_flops_per_token=attention_flops_per_token,
        flops_per_token=linear_flops_per_token + attention_flops_per_token,
        assumptions=_TRAINING_FLOPS_ASSUMPTIONS,
    )


def peak_flops_basis_from_config(config: TrainConfig) -> PeakFlopsBasis | None:
    """Return the complete configured MFU basis, or explicit unavailability."""

    if not isinstance(config, TrainConfig):
        raise TypeError(f"config must be a TrainConfig, got {type(config).__name__}")
    config.validate()
    if config.mfu_peak_flops_per_second is None:
        return None
    if config.mfu_peak_flops_basis is None:  # pragma: no cover - validation pairs it.
        raise ValueError("validated MFU peak description is missing")
    return PeakFlopsBasis(
        flops_per_second=config.mfu_peak_flops_per_second,
        description=config.mfu_peak_flops_basis,
    )


__all__ = [
    "GPTTrainingFlopsEstimate",
    "PeakFlopsBasis",
    "TRAINING_FLOPS_FORMULA_ID",
    "TrainingStepTelemetry",
    "base_training_metrics",
    "estimate_gpt_training_flops",
    "peak_flops_basis_from_config",
]
