"""Pure preflight estimates for baseline GPT size, tokens, and training VRAM."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

from torch import nn

from scratch_llm._validation import (
    require_integer,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.config import GPTConfig, ProjectConfig
from scratch_llm.training.loop import derive_grad_accum_steps


RESOURCE_ESTIMATE_FORMAT: Final = "scratch_llm_training_resource_estimate"
RESOURCE_ESTIMATE_FORMAT_VERSION: Final = 7
_BYTES_PER_MIB = 1024**2
_MAX_SIGNED_64 = 2**63 - 1
_HEADROOM_NUMERATOR = 1
_HEADROOM_DENOMINATOR = 5
_MINIMUM_HEADROOM_BYTES = 512 * _BYTES_PER_MIB
_DTYPE_BYTES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
    }
)


@dataclass(frozen=True)
class ModuleParameterSummary:
    """Deduplicated actual module parameter counts by trainability."""

    unique_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int

    def __post_init__(self) -> None:
        for name in (
            "unique_parameters",
            "trainable_parameters",
            "non_trainable_parameters",
        ):
            require_non_negative_integer(getattr(self, name), name=name)
        if (
            self.trainable_parameters + self.non_trainable_parameters
            != self.unique_parameters
        ):
            raise ValueError(
                "trainable and non-trainable parameters must sum to unique parameters"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "non_trainable_parameters": self.non_trainable_parameters,
            "trainable_parameters": self.trainable_parameters,
            "unique_parameters": self.unique_parameters,
        }


@dataclass(frozen=True)
class GPTModelSizeEstimate:
    """Exact config-only parameter elements for the baseline GPT."""

    profile: str
    requested_depth: int | None
    requested_aspect_ratio: int | None
    requested_head_dim: int | None
    sequence_length: int
    layer_count: int
    head_count: int
    n_kv_head: int
    use_gqa: bool
    embedding_width: int
    norm: str
    activation: str
    qk_norm: bool
    position_encoding: str
    rope_theta: float | None
    tie_weights: bool
    token_embedding_parameters: int
    position_embedding_parameters: int
    transformer_block_parameters: int
    final_norm_parameters: int
    output_head_parameters: int
    unique_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int

    def __post_init__(self) -> None:
        if self.profile not in {"simple_gpt", "nanochat_depth"}:
            raise ValueError(f"unsupported model profile {self.profile!r}")
        if self.norm not in {"layernorm", "rmsnorm"}:
            raise ValueError(f"unsupported normalization {self.norm!r}")
        if self.activation not in {"gelu", "relu_squared"}:
            raise ValueError(f"unsupported activation {self.activation!r}")
        if not isinstance(self.qk_norm, bool):
            raise TypeError("qk_norm must be a boolean")
        if self.position_encoding not in {"learned_absolute", "rope"}:
            raise ValueError(
                f"unsupported position encoding {self.position_encoding!r}"
            )
        if self.position_encoding == "rope":
            if self.rope_theta is None or not math.isfinite(self.rope_theta):
                raise ValueError("RoPE estimates require a finite theta")
            if self.position_embedding_parameters != 0:
                raise ValueError("RoPE estimates cannot contain position parameters")
        elif self.rope_theta is not None:
            raise ValueError("learned position estimates cannot contain a RoPE theta")
        for name in (
            "sequence_length",
            "layer_count",
            "head_count",
            "n_kv_head",
            "embedding_width",
        ):
            require_positive_integer(getattr(self, name), name=name)
        if self.n_kv_head > self.head_count:
            raise ValueError("n_kv_head cannot exceed head_count")
        if self.head_count % self.n_kv_head != 0:
            raise ValueError("head_count must be divisible by n_kv_head")
        if not isinstance(self.use_gqa, bool):
            raise TypeError("use_gqa must be a boolean")
        if self.use_gqa is not (self.n_kv_head < self.head_count):
            raise ValueError("use_gqa must agree with reduced KV-head geometry")
        requested = (
            self.requested_depth,
            self.requested_aspect_ratio,
            self.requested_head_dim,
        )
        if self.profile == "simple_gpt":
            if any(value is not None for value in requested):
                raise ValueError("simple_gpt profile cannot have depth inputs")
        else:
            for name, value in zip(
                ("requested_depth", "requested_aspect_ratio", "requested_head_dim"),
                requested,
                strict=True,
            ):
                require_positive_integer(value, name=name)
        if not isinstance(self.tie_weights, bool):
            raise TypeError("tie_weights must be a boolean")
        for name in (
            "token_embedding_parameters",
            "position_embedding_parameters",
            "transformer_block_parameters",
            "final_norm_parameters",
            "output_head_parameters",
            "unique_parameters",
            "trainable_parameters",
            "non_trainable_parameters",
        ):
            require_non_negative_integer(getattr(self, name), name=name)
        require_positive_integer(self.unique_parameters, name="unique_parameters")
        require_positive_integer(self.trainable_parameters, name="trainable_parameters")
        if sum(self.component_parameters.values()) != self.unique_parameters:
            raise ValueError("model parameter components must sum to unique total")
        if (
            self.trainable_parameters + self.non_trainable_parameters
            != self.unique_parameters
        ):
            raise ValueError(
                "trainable and non-trainable parameters must sum to unique total"
            )
        if self.tie_weights and self.output_head_parameters != 0:
            raise ValueError("a tied output head must add no unique parameters")

    @property
    def component_parameters(self) -> Mapping[str, int]:
        """Return stable, non-overlapping parameter categories."""

        return MappingProxyType(
            {
                "final_norm": self.final_norm_parameters,
                "output_head": self.output_head_parameters,
                "position_embeddings": self.position_embedding_parameters,
                "token_embeddings": self.token_embedding_parameters,
                "transformer_blocks": self.transformer_block_parameters,
            }
        )

    @property
    def embedding_parameters(self) -> int:
        return self.token_embedding_parameters + self.position_embedding_parameters

    @property
    def embedding_fraction(self) -> float:
        return self.embedding_parameters / self.unique_parameters

    @property
    def embedding_dominated(self) -> bool:
        return self.embedding_parameters * 2 > self.unique_parameters

    @property
    def largest_component(self) -> str:
        return max(
            self.component_parameters,
            key=lambda name: self.component_parameters[name],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation": self.activation,
            "attention": {
                "n_head": self.head_count,
                "n_kv_head": self.n_kv_head,
                "use_gqa": self.use_gqa,
            },
            "component_parameters": dict(self.component_parameters),
            "embedding_dominated": self.embedding_dominated,
            "embedding_fraction": self.embedding_fraction,
            "embedding_parameters": self.embedding_parameters,
            "largest_component": self.largest_component,
            "normalization": {
                "parameter_free": self.norm == "rmsnorm",
                "type": self.norm,
            },
            "position_encoding": {
                "parameter_free": self.position_encoding == "rope",
                "theta": self.rope_theta,
                "type": self.position_encoding,
            },
            "qk_norm": self.qk_norm,
            "geometry": {
                "profile": self.profile,
                "requested": {
                    "aspect_ratio": self.requested_aspect_ratio,
                    "depth": self.requested_depth,
                    "head_dim": self.requested_head_dim,
                },
                "resolved": {
                    "n_embd": self.embedding_width,
                    "n_head": self.head_count,
                    "n_layer": self.layer_count,
                    "seq_len": self.sequence_length,
                },
            },
            "non_trainable_parameters": self.non_trainable_parameters,
            "tie_weights": self.tie_weights,
            "trainable_parameters": self.trainable_parameters,
            "unique_parameters": self.unique_parameters,
        }


@dataclass(frozen=True)
class TokenBudgetEstimate:
    """Exact processed-token budget plus config-unknowable target counts."""

    device_batch_size: int
    sequence_length: int
    grad_accum_steps: int
    configured_total_batch_size_tokens: int
    processed_model_tokens_per_microbatch: int
    processed_model_tokens_per_optimizer_step: int
    maximum_supervised_targets_per_microbatch: int
    maximum_supervised_targets_per_optimizer_step: int
    supervised_target_tokens_per_microbatch: None = None
    supervised_target_tokens_per_optimizer_step: None = None
    supervised_targets_are_data_and_mask_dependent: bool = True

    def __post_init__(self) -> None:
        for name in (
            "device_batch_size",
            "sequence_length",
            "grad_accum_steps",
            "configured_total_batch_size_tokens",
            "processed_model_tokens_per_microbatch",
            "processed_model_tokens_per_optimizer_step",
            "maximum_supervised_targets_per_microbatch",
            "maximum_supervised_targets_per_optimizer_step",
        ):
            require_positive_integer(getattr(self, name), name=name)
        if (
            self.processed_model_tokens_per_microbatch
            != self.device_batch_size * self.sequence_length
        ):
            raise ValueError(
                "processed microbatch tokens must equal batch size * sequence length"
            )
        if (
            self.processed_model_tokens_per_optimizer_step
            != self.processed_model_tokens_per_microbatch * self.grad_accum_steps
        ):
            raise ValueError(
                "processed optimizer-step tokens must equal microbatch tokens * "
                "grad accumulation"
            )
        if (
            self.processed_model_tokens_per_optimizer_step
            != self.configured_total_batch_size_tokens
        ):
            raise ValueError(
                "processed optimizer-step tokens must equal configured total"
            )
        if (
            self.maximum_supervised_targets_per_microbatch
            != self.processed_model_tokens_per_microbatch
            or self.maximum_supervised_targets_per_optimizer_step
            != self.processed_model_tokens_per_optimizer_step
        ):
            raise ValueError(
                "maximum supervised targets must equal processed-position bounds"
            )
        if (
            self.supervised_target_tokens_per_microbatch is not None
            or self.supervised_target_tokens_per_optimizer_step is not None
        ):
            raise ValueError(
                "actual supervised target counts cannot be derived from config"
            )
        if self.supervised_targets_are_data_and_mask_dependent is not True:
            raise ValueError(
                "supervised target counts must be marked data and mask dependent"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_total_batch_size_tokens": (
                self.configured_total_batch_size_tokens
            ),
            "device_batch_size": self.device_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "maximum_supervised_targets_per_microbatch": (
                self.maximum_supervised_targets_per_microbatch
            ),
            "maximum_supervised_targets_per_optimizer_step": (
                self.maximum_supervised_targets_per_optimizer_step
            ),
            "processed_model_tokens_per_microbatch": (
                self.processed_model_tokens_per_microbatch
            ),
            "processed_model_tokens_per_optimizer_step": (
                self.processed_model_tokens_per_optimizer_step
            ),
            "sequence_length": self.sequence_length,
            "supervised_target_tokens_per_microbatch": None,
            "supervised_target_tokens_per_optimizer_step": None,
            "supervised_targets_are_data_and_mask_dependent": True,
        }


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    """Itemized conservative bytes for one single-process training step."""

    dtype: str
    bytes_per_dtype_element: int
    device_batch_size: int
    sequence_length: int
    layer_count: int
    head_count: int
    embedding_width: int
    mlp_ratio: int
    vocabulary_size: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    activation_bytes: int
    logits_loss_workspace_bytes: int
    allocator_headroom_bytes: int
    compiled_graph_requested: bool = False
    activation_checkpointing_requested: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported estimate dtype {self.dtype!r}")
        if self.bytes_per_dtype_element != _DTYPE_BYTES[self.dtype]:
            raise ValueError("bytes_per_dtype_element does not match dtype")
        for name in (
            "bytes_per_dtype_element",
            "device_batch_size",
            "sequence_length",
            "layer_count",
            "head_count",
            "embedding_width",
            "mlp_ratio",
            "vocabulary_size",
            "parameter_bytes",
            "gradient_bytes",
            "optimizer_state_bytes",
            "activation_bytes",
            "logits_loss_workspace_bytes",
            "allocator_headroom_bytes",
        ):
            require_positive_integer(getattr(self, name), name=name)
        _bounded_signed_64(self.subtotal_bytes, name="memory subtotal bytes")
        _bounded_signed_64(self.total_bytes, name="memory total bytes")
        if not isinstance(self.compiled_graph_requested, bool):
            raise TypeError("compiled_graph_requested must be a boolean")
        if not isinstance(self.activation_checkpointing_requested, bool):
            raise TypeError("activation_checkpointing_requested must be a boolean")

    @property
    def component_bytes(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "activations": self.activation_bytes,
                "allocator_headroom": self.allocator_headroom_bytes,
                "gradients": self.gradient_bytes,
                "logits_loss_workspace": self.logits_loss_workspace_bytes,
                "optimizer_states": self.optimizer_state_bytes,
                "parameters": self.parameter_bytes,
            }
        )

    @property
    def subtotal_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.optimizer_state_bytes
            + self.activation_bytes
            + self.logits_loss_workspace_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.subtotal_bytes + self.allocator_headroom_bytes

    @property
    def total_mib(self) -> float:
        return self.total_bytes / _BYTES_PER_MIB

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumptions": {
                "activation_checkpointing_requested": (
                    self.activation_checkpointing_requested
                ),
                "activation_checkpointing_memory": (
                    "not credited in this conservative estimate; use observed "
                    "accelerator peak memory"
                ),
                "activation_formula": (
                    "(8 + 2 * mlp_ratio) saved hidden-width values per "
                    "token/layer at configured dtype, float32 materialized "
                    "attention scores and probabilities, final hidden state "
                    "at configured dtype, and one bool causal mask per layer"
                ),
                "allocator_headroom": (
                    "max(20% of modeled subtotal, 512 MiB) for allocator "
                    "fragmentation, framework/context allocations, kernels, "
                    "and per-tensor optimizer metadata"
                ),
                "attention": "manual_materialized_scores_and_probabilities",
                "attention_workspace_dtype": "float32_conservative",
                "automatic_mixed_precision": False,
                "compiled_graph_requested": self.compiled_graph_requested,
                "compiled_graph_workspace": (
                    "not separately modeled; use observed accelerator peak memory"
                ),
                "distributed_training": False,
                "gradient_dtype": self.dtype,
                "optimizer": "AdamW",
                "optimizer_moments_per_trainable_parameter": 2,
                "optimizer_state_dtype": "float32",
                "parameter_dtype": self.dtype,
                "workspace_formula": (
                    "configured-dtype logits plus same-shape float32 "
                    "loss/softmax workspace, int64 targets, and float32 "
                    "per-target losses"
                ),
            },
            "bytes_per_dtype_element": self.bytes_per_dtype_element,
            "classification": "conservative_estimate_not_observed",
            "components": {
                name: _byte_quantity(num_bytes)
                for name, num_bytes in self.component_bytes.items()
            },
            "dimensions": {
                "device_batch_size": self.device_batch_size,
                "embedding_width": self.embedding_width,
                "head_count": self.head_count,
                "layer_count": self.layer_count,
                "mlp_ratio": self.mlp_ratio,
                "sequence_length": self.sequence_length,
                "vocabulary_size": self.vocabulary_size,
            },
            "dtype": self.dtype,
            "headroom_fraction_of_subtotal": 0.20,
            "headroom_minimum_mib": 512,
            "subtotal": _byte_quantity(self.subtotal_bytes),
            "total": _byte_quantity(self.total_bytes),
        }


@dataclass(frozen=True)
class TrainingResourceEstimate:
    """Complete deterministic preflight result for one resolved config."""

    model: GPTModelSizeEstimate
    tokens: TokenBudgetEstimate
    memory: TrainingMemoryEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.model, GPTModelSizeEstimate):
            raise TypeError("model must be a GPTModelSizeEstimate")
        if not isinstance(self.tokens, TokenBudgetEstimate):
            raise TypeError("tokens must be a TokenBudgetEstimate")
        if not isinstance(self.memory, TrainingMemoryEstimate):
            raise TypeError("memory must be a TrainingMemoryEstimate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": RESOURCE_ESTIMATE_FORMAT,
            "format_version": RESOURCE_ESTIMATE_FORMAT_VERSION,
            "memory": self.memory.to_dict(),
            "model": self.model.to_dict(),
            "tokens": self.tokens.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class MemoryEstimateComparison:
    """Keep a planning estimate distinct from an optional observed snapshot."""

    estimated_total_bytes: int
    observed_device: str
    observed_available: bool
    observed_peak_allocated_bytes: int | None
    observed_peak_reserved_bytes: int | None
    observed_unavailable_reason: str | None
    estimate_minus_observed_peak_allocated_bytes: int | None
    estimate_minus_observed_peak_reserved_bytes: int | None

    def __post_init__(self) -> None:
        require_positive_integer(
            self.estimated_total_bytes,
            name="estimated_total_bytes",
        )
        if not isinstance(self.observed_device, str) or not self.observed_device:
            raise ValueError("observed_device must be a non-empty string")
        if not isinstance(self.observed_available, bool):
            raise TypeError("observed_available must be a boolean")
        observed_values = (
            self.observed_peak_allocated_bytes,
            self.observed_peak_reserved_bytes,
        )
        differences = (
            self.estimate_minus_observed_peak_allocated_bytes,
            self.estimate_minus_observed_peak_reserved_bytes,
        )
        if self.observed_available:
            if self.observed_unavailable_reason is not None:
                raise ValueError(
                    "observed_unavailable_reason must be absent when available"
                )
            for name, value in zip(
                (
                    "observed_peak_allocated_bytes",
                    "observed_peak_reserved_bytes",
                ),
                observed_values,
                strict=True,
            ):
                require_non_negative_integer(value, name=name)
            for name, value in zip(
                (
                    "estimate_minus_observed_peak_allocated_bytes",
                    "estimate_minus_observed_peak_reserved_bytes",
                ),
                differences,
                strict=True,
            ):
                require_integer(value, name=name)
        else:
            if (
                not isinstance(self.observed_unavailable_reason, str)
                or not self.observed_unavailable_reason
            ):
                raise ValueError(
                    "observed_unavailable_reason must explain unavailable data"
                )
            if any(value is not None for value in (*observed_values, *differences)):
                raise ValueError(
                    "observed peaks and differences must be absent when unavailable"
                )

    @property
    def observed_peak_allocated_mib(self) -> float | None:
        return _optional_mib(self.observed_peak_allocated_bytes)

    @property
    def observed_peak_reserved_mib(self) -> float | None:
        return _optional_mib(self.observed_peak_reserved_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison": "estimate_vs_observed_snapshot",
            "difference_bytes": {
                "estimate_minus_observed_peak_allocated": (
                    self.estimate_minus_observed_peak_allocated_bytes
                ),
                "estimate_minus_observed_peak_reserved": (
                    self.estimate_minus_observed_peak_reserved_bytes
                ),
            },
            "estimate": _byte_quantity(self.estimated_total_bytes),
            "observed": {
                "available": self.observed_available,
                "device": self.observed_device,
                "peak_allocated_bytes": self.observed_peak_allocated_bytes,
                "peak_allocated_mib": self.observed_peak_allocated_mib,
                "peak_reserved_bytes": self.observed_peak_reserved_bytes,
                "peak_reserved_mib": self.observed_peak_reserved_mib,
                "reason": self.observed_unavailable_reason,
            },
        }


def summarize_module_parameters(module: nn.Module) -> ModuleParameterSummary:
    """Count actual parameters once even when a module exposes tied aliases."""

    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
    seen: set[int] = set()
    trainable = 0
    non_trainable = 0
    for parameter in module.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count = _bounded_signed_64(
            parameter.numel(),
            name="module parameter count",
        )
        if parameter.requires_grad:
            trainable = _checked_sum(
                trainable,
                count,
                name="trainable module parameters",
            )
        else:
            non_trainable = _checked_sum(
                non_trainable,
                count,
                name="non-trainable module parameters",
            )
    return ModuleParameterSummary(
        unique_parameters=_checked_sum(
            trainable,
            non_trainable,
            name="unique module parameters",
        ),
        trainable_parameters=trainable,
        non_trainable_parameters=non_trainable,
    )


def estimate_gpt_model_size(config: GPTConfig) -> GPTModelSizeEstimate:
    """Return exact unique parameter elements without constructing a GPT."""

    if not isinstance(config, GPTConfig):
        raise TypeError(f"config must be a GPTConfig, got {type(config).__name__}")
    config.validate()
    _validate_baseline_model(config)
    channels = config.n_embd

    token_embeddings = _checked_product(
        config.vocab_size,
        channels,
        name="token embedding parameters",
    )
    position_embeddings = (
        0
        if config.use_rope
        else _checked_product(
            config.seq_len,
            channels,
            name="position embedding parameters",
        )
    )
    if config.n_kv_head is None:  # pragma: no cover - validated resolution.
        raise RuntimeError("validated config lost n_kv_head")
    head_dimension = channels // config.n_head
    kv_width = _checked_product(
        config.n_kv_head,
        head_dimension,
        name="KV projection width",
    )
    attention_matrix_parameters = _checked_sum(
        _checked_product(
            2,
            channels,
            channels,
            name="query and attention output matrix parameters",
        ),
        _checked_product(
            2,
            channels,
            kv_width,
            name="key and value matrix parameters",
        ),
        name="per-block attention matrix parameters",
    )
    mlp_matrix_parameters = _checked_product(
        2,
        config.mlp_ratio,
        channels,
        channels,
        name="per-block MLP matrix parameters",
    )
    block_matrix_parameters = _checked_sum(
        attention_matrix_parameters,
        mlp_matrix_parameters,
        name="per-block matrix parameters",
    )
    block_norm_parameters = (
        0
        if config.norm == "rmsnorm"
        else _checked_product(
            2,
            2 if config.bias else 1,
            channels,
            name="per-block normalization parameters",
        )
    )
    block_projection_bias_parameters = (
        _checked_sum(
            _checked_product(
                config.mlp_ratio + 3,
                channels,
                name="per-block query/output/MLP bias parameters",
            ),
            _checked_product(
                2,
                kv_width,
                name="per-block key/value bias parameters",
            ),
            name="per-block projection bias parameters",
        )
        if config.bias
        else 0
    )
    per_block_parameters = _checked_sum(
        block_matrix_parameters,
        block_norm_parameters,
        block_projection_bias_parameters,
        name="per-block parameters",
    )
    transformer_blocks = _checked_product(
        config.n_layer,
        per_block_parameters,
        name="transformer block parameters",
    )
    final_norm = (
        0
        if config.norm == "rmsnorm"
        else _checked_product(
            2 if config.bias else 1,
            channels,
            name="final normalization parameters",
        )
    )
    output_head = (
        0
        if config.tie_weights
        else _checked_product(
            config.vocab_size,
            channels,
            name="output head parameters",
        )
    )
    unique = _checked_sum(
        token_embeddings,
        position_embeddings,
        transformer_blocks,
        final_norm,
        output_head,
        name="unique GPT parameters",
    )
    return GPTModelSizeEstimate(
        profile=config.profile,
        requested_depth=config.depth,
        requested_aspect_ratio=config.aspect_ratio,
        requested_head_dim=config.head_dim,
        sequence_length=config.seq_len,
        layer_count=config.n_layer,
        head_count=config.n_head,
        n_kv_head=config.n_kv_head,
        use_gqa=config.use_gqa,
        embedding_width=config.n_embd,
        norm=config.norm,
        activation=config.activation,
        qk_norm=config.use_qk_norm,
        position_encoding="rope" if config.use_rope else "learned_absolute",
        rope_theta=config.rope_theta if config.use_rope else None,
        tie_weights=config.tie_weights,
        token_embedding_parameters=token_embeddings,
        position_embedding_parameters=position_embeddings,
        transformer_block_parameters=transformer_blocks,
        final_norm_parameters=final_norm,
        output_head_parameters=output_head,
        unique_parameters=unique,
        trainable_parameters=unique,
        non_trainable_parameters=0,
    )


def estimate_token_budget(
    *,
    device_batch_size: int,
    seq_len: int,
    total_batch_size_tokens: int,
    grad_accum_steps: int | str,
) -> TokenBudgetEstimate:
    """Use the training loop's exact accumulation derivation."""

    resolved_steps = derive_grad_accum_steps(
        device_batch_size=device_batch_size,
        seq_len=seq_len,
        total_batch_size_tokens=total_batch_size_tokens,
    )
    if grad_accum_steps != "auto":
        explicit_steps = require_positive_integer(
            grad_accum_steps,
            name="grad_accum_steps",
        )
        if explicit_steps != resolved_steps:
            raise ValueError(
                f"grad_accum_steps={explicit_steps} contradicts the exact "
                f"derived value {resolved_steps}"
            )
    microbatch_tokens = _checked_product(
        device_batch_size,
        seq_len,
        name="processed model tokens per microbatch",
    )
    optimizer_step_tokens = _checked_product(
        microbatch_tokens,
        resolved_steps,
        name="processed model tokens per optimizer step",
    )
    _bounded_signed_64(
        total_batch_size_tokens,
        name="configured total batch size tokens",
    )
    return TokenBudgetEstimate(
        device_batch_size=device_batch_size,
        sequence_length=seq_len,
        grad_accum_steps=resolved_steps,
        configured_total_batch_size_tokens=total_batch_size_tokens,
        processed_model_tokens_per_microbatch=microbatch_tokens,
        processed_model_tokens_per_optimizer_step=optimizer_step_tokens,
        maximum_supervised_targets_per_microbatch=microbatch_tokens,
        maximum_supervised_targets_per_optimizer_step=optimizer_step_tokens,
    )


def estimate_training_resources(
    config: ProjectConfig,
) -> TrainingResourceEstimate:
    """Estimate one resolved baseline run without constructing model/optimizer."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    config.validate()
    model = estimate_gpt_model_size(config.model)
    tokens = estimate_token_budget(
        device_batch_size=config.train.device_batch_size,
        seq_len=config.model.seq_len,
        total_batch_size_tokens=config.train.total_batch_size_tokens,
        grad_accum_steps=config.train.grad_accum_steps,
    )
    memory = _estimate_training_memory(config, model=model)
    return TrainingResourceEstimate(
        model=model,
        tokens=tokens,
        memory=memory,
    )


def compare_memory_estimate(
    estimate: TrainingMemoryEstimate,
    snapshot: AcceleratorMemorySnapshot,
) -> MemoryEstimateComparison:
    """Compare labeled estimate bytes with an optional observed peak snapshot."""

    if not isinstance(estimate, TrainingMemoryEstimate):
        raise TypeError("estimate must be a TrainingMemoryEstimate")
    if not isinstance(snapshot, AcceleratorMemorySnapshot):
        raise TypeError("snapshot must be an AcceleratorMemorySnapshot")
    if snapshot.available:
        assert snapshot.peak_allocated_bytes is not None
        assert snapshot.peak_reserved_bytes is not None
        peak_allocated = snapshot.peak_allocated_bytes
        peak_reserved = snapshot.peak_reserved_bytes
        reason = None
        allocated_difference = estimate.total_bytes - peak_allocated
        reserved_difference = estimate.total_bytes - peak_reserved
    else:
        peak_allocated = None
        peak_reserved = None
        reason = snapshot.unavailable_reason
        allocated_difference = None
        reserved_difference = None
    return MemoryEstimateComparison(
        estimated_total_bytes=estimate.total_bytes,
        observed_device=str(snapshot.device),
        observed_available=snapshot.available,
        observed_peak_allocated_bytes=peak_allocated,
        observed_peak_reserved_bytes=peak_reserved,
        observed_unavailable_reason=reason,
        estimate_minus_observed_peak_allocated_bytes=allocated_difference,
        estimate_minus_observed_peak_reserved_bytes=reserved_difference,
    )


def render_training_resource_estimate(
    result: TrainingResourceEstimate,
) -> str:
    """Render a concise human preflight that cannot be mistaken for telemetry."""

    if not isinstance(result, TrainingResourceEstimate):
        raise TypeError("result must be a TrainingResourceEstimate")
    model = result.model
    tokens = result.tokens
    memory = result.memory
    lines = [
        "Conservative training resource estimate (not observed usage):",
        (
            f"- Model parameters: unique={model.unique_parameters:,}, "
            f"trainable={model.trainable_parameters:,}, "
            f"non-trainable={model.non_trainable_parameters:,}, "
            f"largest component={model.largest_component}, "
            f"embedding fraction={model.embedding_fraction:.6f}"
        ),
        (
            "- Tokens: "
            f"{tokens.processed_model_tokens_per_microbatch:,} processed model "
            "tokens/microbatch; "
            f"{tokens.processed_model_tokens_per_optimizer_step:,} processed "
            f"model tokens/optimizer step across {tokens.grad_accum_steps} "
            "microbatches"
        ),
        "- Actual supervised target count: data/mask dependent, not config-derived",
        "- Memory components:",
    ]
    for name, num_bytes in memory.component_bytes.items():
        lines.append(
            f"  - {name}: {num_bytes:,} bytes ({num_bytes / _BYTES_PER_MIB:.3f} MiB)"
        )
    lines.extend(
        [
            (
                f"- Modeled subtotal: {memory.subtotal_bytes:,} bytes "
                f"({memory.subtotal_bytes / _BYTES_PER_MIB:.3f} MiB)"
            ),
            (
                f"- Estimated total with allocator/headroom: "
                f"{memory.total_bytes:,} bytes ({memory.total_mib:.3f} MiB)"
            ),
            (f"- Parameter/gradient dtype: {memory.dtype}; optimizer moments: float32"),
            "- automatic mixed precision: disabled",
            "- activation checkpointing: disabled",
            (
                "- This is a conservative planning estimate, not observed CUDA "
                "usage or an allocation guarantee."
            ),
        ]
    )
    return "\n".join(lines)


def _estimate_training_memory(
    config: ProjectConfig,
    *,
    model: GPTModelSizeEstimate,
) -> TrainingMemoryEstimate:
    dtype_bytes = _DTYPE_BYTES[config.train.dtype]
    batch = config.train.device_batch_size
    sequence = config.model.seq_len
    layers = config.model.n_layer
    heads = config.model.n_head
    channels = config.model.n_embd
    ratio = config.model.mlp_ratio
    vocab = config.model.vocab_size

    parameter_bytes = _checked_product(
        model.unique_parameters,
        dtype_bytes,
        name="parameter bytes",
    )
    gradient_bytes = _checked_product(
        model.trainable_parameters,
        dtype_bytes,
        name="gradient bytes",
    )
    optimizer_state_bytes = _checked_product(
        model.trainable_parameters,
        2,
        4,
        name="optimizer state bytes",
    )
    dense_activation_elements = _checked_product(
        layers,
        batch,
        sequence,
        channels,
        8 + 2 * ratio,
        name="dense activation elements",
    )
    attention_activation_elements = _checked_product(
        layers,
        2,
        batch,
        heads,
        sequence,
        sequence,
        name="attention activation elements",
    )
    final_hidden_elements = _checked_product(
        batch,
        sequence,
        channels,
        name="final hidden activation elements",
    )
    dense_and_final_activation_bytes = _checked_product(
        _checked_sum(
            dense_activation_elements,
            final_hidden_elements,
            name="configured-dtype activation elements",
        ),
        dtype_bytes,
        name="configured-dtype activation bytes",
    )
    attention_activation_bytes = _checked_product(
        attention_activation_elements,
        4,
        name="float32 attention activation bytes",
    )
    causal_mask_bytes = _checked_product(
        layers,
        sequence,
        sequence,
        name="causal mask bytes",
    )
    activation_bytes = _checked_sum(
        dense_and_final_activation_bytes,
        attention_activation_bytes,
        causal_mask_bytes,
        name="activation bytes",
    )
    logits_bytes = _checked_product(
        batch,
        sequence,
        vocab,
        dtype_bytes,
        name="logits bytes",
    )
    loss_workspace_bytes = _checked_product(
        batch,
        sequence,
        vocab,
        4,
        name="float32 loss workspace bytes",
    )
    target_and_loss_bytes = _checked_product(
        batch,
        sequence,
        8 + 4,
        name="target and per-target loss bytes",
    )
    logits_loss_workspace_bytes = _checked_sum(
        logits_bytes,
        loss_workspace_bytes,
        target_and_loss_bytes,
        name="logits/loss workspace bytes",
    )
    subtotal = _checked_sum(
        parameter_bytes,
        gradient_bytes,
        optimizer_state_bytes,
        activation_bytes,
        logits_loss_workspace_bytes,
        name="modeled memory subtotal bytes",
    )
    proportional_headroom = (
        subtotal * _HEADROOM_NUMERATOR + _HEADROOM_DENOMINATOR - 1
    ) // _HEADROOM_DENOMINATOR
    allocator_headroom_bytes = _bounded_signed_64(
        max(proportional_headroom, _MINIMUM_HEADROOM_BYTES),
        name="allocator headroom bytes",
    )
    _checked_sum(
        subtotal,
        allocator_headroom_bytes,
        name="estimated total memory bytes",
    )
    return TrainingMemoryEstimate(
        dtype=config.train.dtype,
        bytes_per_dtype_element=dtype_bytes,
        device_batch_size=batch,
        sequence_length=sequence,
        layer_count=layers,
        head_count=heads,
        embedding_width=channels,
        mlp_ratio=ratio,
        vocabulary_size=vocab,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_bytes=activation_bytes,
        logits_loss_workspace_bytes=logits_loss_workspace_bytes,
        allocator_headroom_bytes=allocator_headroom_bytes,
        compiled_graph_requested=config.train.compile,
        activation_checkpointing_requested=config.train.activation_checkpointing,
    )


def _validate_baseline_model(config: GPTConfig) -> None:
    unsupported = {
        "use_flash_attention": config.use_flash_attention,
        "use_kv_cache": config.use_kv_cache,
    }
    enabled = sorted(name for name, value in unsupported.items() if value)
    if enabled:
        raise ValueError(
            "resource estimation supports RMSNorm/RoPE/QK-norm/GQA GPTs "
            f"before later architecture switches; enabled switches={enabled}"
        )


def _byte_quantity(num_bytes: int) -> dict[str, int | float]:
    return {
        "bytes": num_bytes,
        "mib": num_bytes / _BYTES_PER_MIB,
    }


def _optional_mib(num_bytes: int | None) -> float | None:
    return None if num_bytes is None else num_bytes / _BYTES_PER_MIB


def _checked_product(*values: int, name: str) -> int:
    result = 1
    for value in values:
        normalized = require_non_negative_integer(value, name=name)
        result *= normalized
        _bounded_signed_64(result, name=name)
    return result


def _checked_sum(*values: int, name: str) -> int:
    result = 0
    for value in values:
        result += require_non_negative_integer(value, name=name)
        _bounded_signed_64(result, name=name)
    return result


def _bounded_signed_64(value: object, *, name: str) -> int:
    normalized = require_non_negative_integer(value, name=name)
    if normalized > _MAX_SIGNED_64:
        raise OverflowError(
            f"{name} exceeds the signed 64-bit planning limit "
            f"{_MAX_SIGNED_64}: {normalized}"
        )
    return normalized


__all__ = [
    "RESOURCE_ESTIMATE_FORMAT",
    "RESOURCE_ESTIMATE_FORMAT_VERSION",
    "GPTModelSizeEstimate",
    "MemoryEstimateComparison",
    "ModuleParameterSummary",
    "TokenBudgetEstimate",
    "TrainingMemoryEstimate",
    "TrainingResourceEstimate",
    "compare_memory_estimate",
    "estimate_gpt_model_size",
    "estimate_token_budget",
    "estimate_training_resources",
    "render_training_resource_estimate",
    "summarize_module_parameters",
]
