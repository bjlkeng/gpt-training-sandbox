"""Pure configuration-aware diagnostics for supported PyTorch OOM failures."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.config import ProjectConfig


@dataclass(frozen=True)
class OOMAttempt:
    """Resolved model and training settings active at the failed allocation."""

    device: str
    dtype: str
    model_profile: str
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    use_gqa: bool
    n_embd: int
    seq_len: int
    device_batch_size: int
    total_batch_size_tokens: int
    grad_accum_steps: int

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "device": self.device,
            "device_batch_size": self.device_batch_size,
            "dtype": self.dtype,
            "grad_accum_steps": self.grad_accum_steps,
            "model_profile": self.model_profile,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "n_layer": self.n_layer,
            "seq_len": self.seq_len,
            "total_batch_size_tokens": self.total_batch_size_tokens,
            "vocab_size": self.vocab_size,
            "use_gqa": self.use_gqa,
        }


@dataclass(frozen=True)
class OOMRecommendation:
    """One ordered, explicit, configuration-valid retry suggestion."""

    priority: int
    field: str
    current_value: int
    proposed_value: int
    cli_overrides: tuple[str, ...]
    reason: str
    preserves_total_batch_size_tokens: bool | None = None
    resulting_total_batch_size_tokens: int | None = None
    resulting_grad_accum_steps: int | None = None

    @property
    def cli_example(self) -> str:
        """Render the exact repeatable dotted-override arguments."""

        return " ".join(f"--override {override}" for override in self.cli_overrides)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "cli_example": self.cli_example,
            "cli_overrides": list(self.cli_overrides),
            "current_value": self.current_value,
            "field": self.field,
            "preserves_total_batch_size_tokens": (
                self.preserves_total_batch_size_tokens
            ),
            "priority": self.priority,
            "proposed_value": self.proposed_value,
            "reason": self.reason,
            "resulting_grad_accum_steps": self.resulting_grad_accum_steps,
            "resulting_total_batch_size_tokens": (
                self.resulting_total_batch_size_tokens
            ),
        }


@dataclass(frozen=True)
class OOMDiagnostic:
    """Immutable machine- and human-readable OOM diagnosis."""

    exception_type: str
    exception_message: str
    attempt: OOMAttempt
    memory: AcceleratorMemorySnapshot
    recommendations: tuple[OOMRecommendation, ...]
    schema_version: int = 2

    def to_dict(self) -> dict[str, object]:
        """Return the stable machine-readable diagnostic contract."""

        return {
            "attempt": self.attempt.to_dict(),
            "exception_message": self.exception_message,
            "exception_type": self.exception_type,
            "memory": _memory_to_dict(self.memory),
            "recommendations": [
                recommendation.to_dict() for recommendation in self.recommendations
            ],
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize canonical compact JSON without non-finite values."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def render(self) -> str:
        """Render one diagnostic with an embedded canonical JSON record."""

        lines = [
            (
                "Accelerator out of memory; the requested run was not changed "
                "or retried."
            ),
            f"Original error: {self.exception_message}",
            (
                "Attempted: "
                f"device={self.attempt.device}, dtype={self.attempt.dtype}, "
                f"batch={self.attempt.device_batch_size}, "
                f"sequence={self.attempt.seq_len}, "
                f"width={self.attempt.n_embd}, layers={self.attempt.n_layer}, "
                f"tokens/step={self.attempt.total_batch_size_tokens}"
            ),
        ]
        if self.memory.available:
            lines.append(
                "Memory bytes: "
                f"allocated={self.memory.allocated_bytes}, "
                f"reserved={self.memory.reserved_bytes}, "
                f"peak_allocated={self.memory.peak_allocated_bytes}, "
                f"peak_reserved={self.memory.peak_reserved_bytes}, "
                f"capacity={self.memory.capacity_bytes}"
            )
        else:
            lines.append(
                f"Memory snapshot unavailable: {self.memory.unavailable_reason}"
            )
        lines.append(f"OOM_DIAGNOSTIC_JSON={self.to_json()}")
        if self.recommendations:
            lines.append("Recommended reductions, in order:")
            for recommendation in self.recommendations:
                lines.append(
                    f"{recommendation.priority}. {recommendation.field}: "
                    f"{recommendation.current_value} -> "
                    f"{recommendation.proposed_value}; "
                    f"{recommendation.cli_example}"
                )
        else:
            lines.append(
                "No further positive integer reduction is available in the "
                "supported baseline fields."
            )
        return "\n".join(lines)


class PretrainingOOMError(RuntimeError):
    """Command-boundary failure carrying one structured OOM diagnostic."""

    def __init__(self, diagnostic: OOMDiagnostic) -> None:
        if not isinstance(diagnostic, OOMDiagnostic):
            raise TypeError(
                f"diagnostic must be an OOMDiagnostic, got {type(diagnostic).__name__}"
            )
        self.diagnostic = diagnostic
        super().__init__(diagnostic.render())


def diagnose_out_of_memory(
    error: BaseException,
    *,
    config: ProjectConfig,
    memory: AcceleratorMemorySnapshot,
) -> OOMDiagnostic | None:
    """Diagnose only supported PyTorch OOM exceptions without mutating inputs."""

    if not isinstance(error, torch.OutOfMemoryError):
        return None
    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    if not isinstance(memory, AcceleratorMemorySnapshot):
        raise TypeError(
            f"memory must be an AcceleratorMemorySnapshot, got {type(memory).__name__}"
        )
    config.validate()
    grad_accum_steps = config.train.total_batch_size_tokens // (
        config.train.device_batch_size * config.model.seq_len
    )
    if config.model.n_kv_head is None:  # pragma: no cover - validated resolution.
        raise RuntimeError("validated config lost n_kv_head")
    attempt = OOMAttempt(
        device=config.run.device,
        dtype=config.train.dtype,
        model_profile=config.model.profile,
        vocab_size=config.model.vocab_size,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_kv_head=config.model.n_kv_head,
        use_gqa=config.model.use_gqa,
        n_embd=config.model.n_embd,
        seq_len=config.model.seq_len,
        device_batch_size=config.train.device_batch_size,
        total_batch_size_tokens=config.train.total_batch_size_tokens,
        grad_accum_steps=grad_accum_steps,
    )
    recommendations = _recommendations(config)
    return OOMDiagnostic(
        exception_type=type(error).__qualname__,
        exception_message=str(error) or "PyTorch reported an out-of-memory failure",
        attempt=attempt,
        memory=memory,
        recommendations=recommendations,
    )


def _recommendations(config: ProjectConfig) -> tuple[OOMRecommendation, ...]:
    recommendations: list[OOMRecommendation] = []
    batch_size = config.train.device_batch_size
    seq_len = config.model.seq_len
    if batch_size > 1:
        recommendations.append(
            _token_shape_recommendation(
                config,
                priority=1,
                field="train.device_batch_size",
                current_value=batch_size,
                proposed_value=max(1, batch_size // 2),
                device_batch_size=max(1, batch_size // 2),
                seq_len=seq_len,
            )
        )
    if seq_len > 1:
        recommendations.append(
            _token_shape_recommendation(
                config,
                priority=2,
                field="model.seq_len",
                current_value=seq_len,
                proposed_value=max(1, seq_len // 2),
                device_batch_size=batch_size,
                seq_len=max(1, seq_len // 2),
            )
        )

    width = config.model.n_embd
    proposed_width = (width // 2 // config.model.n_head) * config.model.n_head
    if 0 < proposed_width < width:
        recommendations.append(
            OOMRecommendation(
                priority=3,
                field="model.n_embd",
                current_value=width,
                proposed_value=proposed_width,
                cli_overrides=(f"model.n_embd={proposed_width}",),
                reason=(
                    "reduce residual and MLP activation width while preserving "
                    f"divisibility by model.n_head={config.model.n_head}"
                ),
            )
        )

    layer_count = config.model.n_layer
    proposed_layers = max(1, layer_count // 2)
    if proposed_layers < layer_count:
        recommendations.append(
            OOMRecommendation(
                priority=4,
                field="model.n_layer",
                current_value=layer_count,
                proposed_value=proposed_layers,
                cli_overrides=(f"model.n_layer={proposed_layers}",),
                reason="reduce the number of transformer-block activations",
            )
        )
    return tuple(recommendations)


def _token_shape_recommendation(
    config: ProjectConfig,
    *,
    priority: int,
    field: str,
    current_value: int,
    proposed_value: int,
    device_batch_size: int,
    seq_len: int,
) -> OOMRecommendation:
    requested_total = config.train.total_batch_size_tokens
    tokens_per_microbatch = device_batch_size * seq_len
    grad_accum_steps, remainder = divmod(requested_total, tokens_per_microbatch)
    preserves_total = remainder == 0
    if not preserves_total:
        grad_accum_steps = max(1, grad_accum_steps)
    resulting_total = tokens_per_microbatch * grad_accum_steps

    overrides = [f"{field}={proposed_value}"]
    if not preserves_total:
        overrides.append(f"train.total_batch_size_tokens={resulting_total}")
    overrides.append(f"train.grad_accum_steps={grad_accum_steps}")
    if preserves_total:
        reason = (
            f"preserve train.total_batch_size_tokens={requested_total} with "
            f"train.grad_accum_steps={grad_accum_steps}"
        )
    else:
        reason = (
            f"train.total_batch_size_tokens={requested_total} is not divisible "
            f"by the proposed microbatch size {tokens_per_microbatch}; use the "
            f"explicit valid budget {resulting_total} with "
            f"train.grad_accum_steps={grad_accum_steps}"
        )
    return OOMRecommendation(
        priority=priority,
        field=field,
        current_value=current_value,
        proposed_value=proposed_value,
        cli_overrides=tuple(overrides),
        reason=reason,
        preserves_total_batch_size_tokens=preserves_total,
        resulting_total_batch_size_tokens=resulting_total,
        resulting_grad_accum_steps=grad_accum_steps,
    )


def _memory_to_dict(memory: AcceleratorMemorySnapshot) -> dict[str, object]:
    return {
        "allocated_bytes": memory.allocated_bytes,
        "available": memory.available,
        "capacity_bytes": memory.capacity_bytes,
        "device": str(memory.device),
        "peak_allocated_bytes": memory.peak_allocated_bytes,
        "peak_reserved_bytes": memory.peak_reserved_bytes,
        "reserved_bytes": memory.reserved_bytes,
        "unavailable_reason": memory.unavailable_reason,
    }


__all__ = [
    "OOMAttempt",
    "OOMDiagnostic",
    "OOMRecommendation",
    "PretrainingOOMError",
    "diagnose_out_of_memory",
]
