"""Protocol-neutral bits-per-byte arithmetic and coverage results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import random
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_finite_unit_interval,
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.utils import get_device


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
class BPBAccumulation:
    """Final token/byte counts and summed cross-entropy nats."""

    processed_model_tokens: int
    counted_target_tokens: int
    counted_target_bytes: int
    total_nats: float

    def __post_init__(self) -> None:
        for name in (
            "processed_model_tokens",
            "counted_target_tokens",
            "counted_target_bytes",
        ):
            require_non_negative_integer(getattr(self, name), name=name)
        if self.counted_target_tokens == 0:
            raise ValueError("counted_target_tokens must be positive")
        if self.counted_target_bytes == 0:
            raise ValueError("counted_target_bytes must be positive")
        if self.processed_model_tokens < self.counted_target_tokens:
            raise ValueError(
                "processed_model_tokens must be greater than or equal to "
                "counted_target_tokens"
            )
        total_nats = require_finite_non_negative_real(
            self.total_nats,
            name="total_nats",
        )
        object.__setattr__(self, "total_nats", total_nats)

    @property
    def bpb(self) -> float:
        """Return summed nats converted to bits per counted raw byte."""

        return self.total_nats / math.log(2) / self.counted_target_bytes

    def to_dict(self) -> dict[str, int | float]:
        """Return stable JSON-compatible arithmetic fields."""

        return {
            "bpb": self.bpb,
            "counted_target_bytes": self.counted_target_bytes,
            "counted_target_tokens": self.counted_target_tokens,
            "processed_model_tokens": self.processed_model_tokens,
            "total_nats": self.total_nats,
        }


class BPBAccumulator:
    """Incrementally sum losses and raw bytes from explicitly counted targets."""

    def __init__(self, token_bytes: Tensor) -> None:
        self._token_bytes = _validated_token_bytes(token_bytes)
        self._processed_model_tokens = 0
        self._counted_target_tokens = 0
        self._counted_target_bytes = 0
        self._nats_partials: list[float] = []

    @property
    def processed_model_tokens(self) -> int:
        """Return model positions processed across accepted chunks."""

        return self._processed_model_tokens

    @property
    def counted_target_tokens(self) -> int:
        """Return ordinary supervised targets accepted across chunks."""

        return self._counted_target_tokens

    @property
    def counted_target_bytes(self) -> int:
        """Return raw bytes represented by accepted ordinary targets."""

        return self._counted_target_bytes

    @property
    def total_nats(self) -> float:
        """Return accurately summed cross-entropy nats."""

        return math.fsum(self._nats_partials)

    def snapshot(self) -> tuple[int, int, int, tuple[float, ...]]:
        """Return internal scalar state for diagnostics and atomicity tests."""

        return (
            self._processed_model_tokens,
            self._counted_target_tokens,
            self._counted_target_bytes,
            tuple(self._nats_partials),
        )

    def update(
        self,
        losses_nats: Tensor,
        targets: Tensor,
        *,
        supervision_mask: Tensor | None = None,
        processed_model_tokens: int | None = None,
    ) -> None:
        """Validate and atomically add one unreduced-loss chunk.

        A position contributes only when its target is non-negative, its
        explicit mask is true, and its target token has a positive byte length.
        """

        _validate_loss_and_target_tensors(losses_nats, targets)
        mask = _validated_supervision_mask(supervision_mask, targets=targets)
        if processed_model_tokens is None:
            processed_count = targets.numel()
        else:
            processed_count = require_non_negative_integer(
                processed_model_tokens,
                name="processed_model_tokens",
            )
            if processed_count < targets.numel():
                raise ValueError(
                    "processed_model_tokens must be at least the number of "
                    f"loss positions ({targets.numel()})"
                )

        non_negative_targets = targets >= 0
        if bool(non_negative_targets.any().item()):
            maximum_target = int(targets[non_negative_targets].max().item())
            if maximum_target >= self._token_bytes.numel():
                raise ValueError(
                    f"target ID {maximum_target} is out of range for token_bytes "
                    f"with size {self._token_bytes.numel()}"
                )

        token_bytes = self._token_bytes.to(device=targets.device)
        target_byte_lengths = torch.zeros_like(targets, dtype=torch.int64)
        if bool(non_negative_targets.any().item()):
            target_byte_lengths[non_negative_targets] = token_bytes[
                targets[non_negative_targets].to(torch.long)
            ]
        counted = non_negative_targets & mask & target_byte_lengths.gt(0)
        counted_losses = (
            losses_nats.detach()[counted].to(device="cpu", dtype=torch.float64).tolist()
        )
        counted_byte_values = (
            target_byte_lengths[counted].detach().to(device="cpu").tolist()
        )

        candidate_partials = self._nats_partials.copy()
        for loss in counted_losses:
            _add_fsum_value(candidate_partials, float(loss))
        candidate_nats = math.fsum(candidate_partials)
        if not math.isfinite(candidate_nats):
            raise ValueError("accumulated total_nats must remain finite")

        counted_tokens = len(counted_losses)
        counted_bytes = sum(int(value) for value in counted_byte_values)
        self._processed_model_tokens += processed_count
        self._counted_target_tokens += counted_tokens
        self._counted_target_bytes += counted_bytes
        self._nats_partials = candidate_partials

    def finalize(self) -> BPBAccumulation:
        """Return an immutable result, rejecting an undefined zero-byte BPB."""

        if self._counted_target_bytes == 0:
            raise ValueError("cannot compute BPB with zero counted target bytes")
        return BPBAccumulation(
            processed_model_tokens=self._processed_model_tokens,
            counted_target_tokens=self._counted_target_tokens,
            counted_target_bytes=self._counted_target_bytes,
            total_nats=self.total_nats,
        )


def accumulate_bpb(
    losses_nats: Tensor,
    targets: Tensor,
    token_bytes: Tensor,
    *,
    supervision_mask: Tensor | None = None,
    processed_model_tokens: int | None = None,
) -> BPBAccumulation:
    """Compute BPB arithmetic for one validated unreduced-loss tensor."""

    accumulator = BPBAccumulator(token_bytes)
    accumulator.update(
        losses_nats,
        targets,
        supervision_mask=supervision_mask,
        processed_model_tokens=processed_model_tokens,
    )
    return accumulator.finalize()


def evaluate_bpb_batches(
    model: nn.Module,
    batches: Iterable[Sequence[Tensor]],
    token_bytes: Tensor,
    *,
    device: str | torch.device,
) -> BPBAccumulation:
    """Evaluate protocol-selected batches without changing training state.

    Each batch contains ``(inputs, targets)`` or
    ``(inputs, targets, supervision_mask)``. Selection, packing, and document
    retention remain the caller's protocol-specific responsibility.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    resolved_device = get_device(device)
    accumulator = BPBAccumulator(token_bytes)
    module_modes = [(module, module.training) for module in model.modules()]
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    cuda_rng_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if resolved_device.type == "cuda"
        else None
    )

    try:
        try:
            batch_iterator = iter(batches)
        except TypeError as error:
            raise TypeError(
                f"batches must be iterable, got {type(batches).__name__}"
            ) from error
        model.eval()
        with torch.inference_mode():
            for batch_index, batch in enumerate(batch_iterator):
                if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                    raise TypeError(
                        f"evaluation batch {batch_index} must contain inputs, "
                        "targets, and an optional supervision mask"
                    )
                if any(not isinstance(value, Tensor) for value in batch):
                    raise TypeError(
                        f"evaluation batch {batch_index} values must all be Tensors"
                    )
                inputs, targets = batch[:2]
                supervision_mask = batch[2] if len(batch) == 3 else None
                if inputs.shape != targets.shape:
                    raise ValueError(
                        f"evaluation batch {batch_index} input and target shapes "
                        f"must match; got {tuple(inputs.shape)} and "
                        f"{tuple(targets.shape)}"
                    )
                device_inputs = inputs.to(resolved_device)
                device_targets = targets.to(resolved_device)
                device_mask = (
                    None
                    if supervision_mask is None
                    else supervision_mask.to(resolved_device)
                )
                losses = model(
                    device_inputs,
                    device_targets,
                    loss_reduction="none",
                )
                if not isinstance(losses, Tensor):
                    raise TypeError(
                        "model must return a Tensor of unreduced losses, "
                        f"got {type(losses).__name__}"
                    )
                accumulator.update(
                    losses,
                    device_targets,
                    supervision_mask=device_mask,
                    processed_model_tokens=inputs.numel(),
                )
        return accumulator.finalize()
    finally:
        for module, training_mode in module_modes:
            module.training = training_mode
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(list(cuda_rng_states))


@dataclass(frozen=True)
class BaseValidationResult:
    """Immutable protocol identity, corpus coverage, and BPB result."""

    protocol_id: str
    protocol_version: int
    reference_commit: str | None
    reference_config: Mapping[str, Any]
    checkpoint_identity: str
    tokenizer_identity: str
    validation_manifest_identity: str
    source_documents: int
    source_tokens: int
    source_bytes: int
    processed_model_tokens: int
    counted_target_tokens: int
    counted_target_bytes: int
    unique_source_tokens: int
    unique_source_bytes: int
    source_token_retention: float
    source_byte_retention: float
    total_nats: float
    bpb: float

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "checkpoint_identity",
            "tokenizer_identity",
            "validation_manifest_identity",
        ):
            require_non_empty_string(getattr(self, name), name=name)
        require_positive_integer(self.protocol_version, name="protocol_version")
        if self.reference_commit is not None:
            require_non_empty_string(
                self.reference_commit,
                name="reference_commit",
            )
        frozen_reference_config = _freeze_json_mapping(
            self.reference_config,
            name="reference_config",
        )
        object.__setattr__(
            self,
            "reference_config",
            frozen_reference_config,
        )

        for name in (
            "source_documents",
            "source_tokens",
            "source_bytes",
            "processed_model_tokens",
            "counted_target_tokens",
            "counted_target_bytes",
            "unique_source_tokens",
            "unique_source_bytes",
        ):
            require_non_negative_integer(getattr(self, name), name=name)
        if self.counted_target_tokens == 0:
            raise ValueError("counted_target_tokens must be positive")
        if self.counted_target_bytes == 0:
            raise ValueError("counted_target_bytes must be positive")
        if self.source_documents == 0:
            raise ValueError(
                "source_documents must be positive when source coverage is present"
            )
        if self.unique_source_tokens > self.source_tokens:
            raise ValueError("unique_source_tokens must not exceed source_tokens")
        if self.unique_source_bytes > self.source_bytes:
            raise ValueError("unique_source_bytes must not exceed source_bytes")
        if self.counted_target_tokens < self.unique_source_tokens:
            raise ValueError(
                "counted_target_tokens must be greater than or equal to "
                "unique_source_tokens"
            )
        if self.counted_target_bytes < self.unique_source_bytes:
            raise ValueError(
                "counted_target_bytes must be greater than or equal to "
                "unique_source_bytes"
            )
        if self.processed_model_tokens < self.counted_target_tokens:
            raise ValueError(
                "processed_model_tokens must be greater than or equal to "
                "counted_target_tokens"
            )

        source_token_retention = require_finite_unit_interval(
            self.source_token_retention,
            name="source_token_retention",
        )
        source_byte_retention = require_finite_unit_interval(
            self.source_byte_retention,
            name="source_byte_retention",
        )
        expected_token_retention = self.unique_source_tokens / self.source_tokens
        expected_byte_retention = self.unique_source_bytes / self.source_bytes
        if not math.isclose(
            source_token_retention,
            expected_token_retention,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "source_token_retention does not match unique/source token counts"
            )
        if not math.isclose(
            source_byte_retention,
            expected_byte_retention,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "source_byte_retention does not match unique/source byte counts"
            )
        total_nats = require_finite_non_negative_real(
            self.total_nats,
            name="total_nats",
        )
        bpb = require_finite_non_negative_real(self.bpb, name="bpb")
        expected_bpb = total_nats / math.log(2) / self.counted_target_bytes
        if not math.isclose(
            bpb,
            expected_bpb,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "bpb does not match total_nats / log(2) / counted_target_bytes"
            )
        object.__setattr__(self, "source_token_retention", source_token_retention)
        object.__setattr__(self, "source_byte_retention", source_byte_retention)
        object.__setattr__(self, "total_nats", total_nats)
        object.__setattr__(self, "bpb", bpb)

    @property
    def discarded_source_tokens(self) -> int:
        """Return source tokens never retained by this protocol."""

        return self.source_tokens - self.unique_source_tokens

    @property
    def discarded_source_bytes(self) -> int:
        """Return source bytes never retained by this protocol."""

        return self.source_bytes - self.unique_source_bytes

    @classmethod
    def from_accumulation(
        cls,
        accumulation: BPBAccumulation,
        *,
        protocol_id: str,
        protocol_version: int,
        reference_commit: str | None,
        reference_config: Mapping[str, Any],
        checkpoint_identity: str,
        tokenizer_identity: str,
        validation_manifest_identity: str,
        source_documents: int,
        source_tokens: int,
        source_bytes: int,
        unique_source_tokens: int,
        unique_source_bytes: int,
    ) -> BaseValidationResult:
        """Combine arithmetic with caller-owned protocol coverage metadata."""

        if not isinstance(accumulation, BPBAccumulation):
            raise TypeError(
                "accumulation must be a BPBAccumulation, "
                f"got {type(accumulation).__name__}"
            )
        source_tokens = require_positive_integer(
            source_tokens,
            name="source_tokens",
        )
        source_bytes = require_positive_integer(
            source_bytes,
            name="source_bytes",
        )
        return cls(
            protocol_id=protocol_id,
            protocol_version=protocol_version,
            reference_commit=reference_commit,
            reference_config=reference_config,
            checkpoint_identity=checkpoint_identity,
            tokenizer_identity=tokenizer_identity,
            validation_manifest_identity=validation_manifest_identity,
            source_documents=source_documents,
            source_tokens=source_tokens,
            source_bytes=source_bytes,
            processed_model_tokens=accumulation.processed_model_tokens,
            counted_target_tokens=accumulation.counted_target_tokens,
            counted_target_bytes=accumulation.counted_target_bytes,
            unique_source_tokens=unique_source_tokens,
            unique_source_bytes=unique_source_bytes,
            source_token_retention=unique_source_tokens / source_tokens,
            source_byte_retention=unique_source_bytes / source_bytes,
            total_nats=accumulation.total_nats,
            bpb=accumulation.bpb,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON report schema in human-facing field order."""

        return {
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "reference_commit": self.reference_commit,
            "reference_config": _thaw_json(self.reference_config),
            "checkpoint_identity": self.checkpoint_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "validation_manifest_identity": self.validation_manifest_identity,
            "source_documents": self.source_documents,
            "source_tokens": self.source_tokens,
            "source_bytes": self.source_bytes,
            "processed_model_tokens": self.processed_model_tokens,
            "counted_target_tokens": self.counted_target_tokens,
            "counted_target_bytes": self.counted_target_bytes,
            "unique_source_tokens": self.unique_source_tokens,
            "unique_source_bytes": self.unique_source_bytes,
            "source_token_retention": self.source_token_retention,
            "source_byte_retention": self.source_byte_retention,
            "total_nats": self.total_nats,
            "bpb": self.bpb,
        }

    def to_json(self) -> str:
        """Return canonical UTF-8 JSON with no non-finite number spellings."""

        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _validated_token_bytes(token_bytes: Tensor) -> Tensor:
    if not isinstance(token_bytes, Tensor):
        raise TypeError(
            f"token_bytes must be a Tensor, got {type(token_bytes).__name__}"
        )
    if token_bytes.ndim != 1:
        raise ValueError(
            f"token_bytes must be one-dimensional, got shape {tuple(token_bytes.shape)}"
        )
    if token_bytes.numel() == 0:
        raise ValueError("token_bytes must not be empty")
    if token_bytes.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"token_bytes must have an integer dtype, got {token_bytes.dtype}"
        )
    normalized = token_bytes.detach().to(device="cpu", dtype=torch.int64).clone()
    if bool(normalized.lt(0).any().item()):
        raise ValueError("token_bytes values must be non-negative")
    return normalized


def _add_fsum_value(partials: list[float], value: float) -> None:
    """Add one finite value to a non-overlapping floating-point expansion."""

    write_index = 0
    for partial in partials:
        if abs(value) < abs(partial):
            value, partial = partial, value
        combined = value + partial
        remainder = partial - (combined - value)
        if remainder:
            partials[write_index] = remainder
            write_index += 1
        value = combined
    partials[write_index:] = [value]


def _validate_loss_and_target_tensors(
    losses_nats: Tensor,
    targets: Tensor,
) -> None:
    if not isinstance(losses_nats, Tensor):
        raise TypeError(
            f"losses_nats must be a Tensor, got {type(losses_nats).__name__}"
        )
    if not isinstance(targets, Tensor):
        raise TypeError(f"targets must be a Tensor, got {type(targets).__name__}")
    if losses_nats.shape != targets.shape:
        raise ValueError(
            "loss and target shapes must match; "
            f"got {tuple(losses_nats.shape)} and {tuple(targets.shape)}"
        )
    if losses_nats.device != targets.device:
        raise ValueError("losses_nats and targets must be on the same device")
    if not losses_nats.dtype.is_floating_point:
        raise TypeError(
            f"losses_nats must have a floating-point dtype, got {losses_nats.dtype}"
        )
    detached_losses = losses_nats.detach()
    if not bool(torch.isfinite(detached_losses).all().item()):
        raise ValueError("losses_nats values must all be finite")
    if bool(detached_losses.lt(0).any().item()):
        raise ValueError("losses_nats values must be non-negative")
    if targets.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"targets must have an integer dtype, got {targets.dtype}")


def _validated_supervision_mask(
    supervision_mask: Tensor | None,
    *,
    targets: Tensor,
) -> Tensor:
    if supervision_mask is None:
        return torch.ones_like(targets, dtype=torch.bool)
    if not isinstance(supervision_mask, Tensor):
        raise TypeError(
            "supervision_mask must be a Tensor or None, "
            f"got {type(supervision_mask).__name__}"
        )
    if supervision_mask.shape != targets.shape:
        raise ValueError(
            "supervision mask shape must match targets; "
            f"got {tuple(supervision_mask.shape)} and {tuple(targets.shape)}"
        )
    if supervision_mask.device != targets.device:
        raise ValueError("supervision_mask and targets must be on the same device")
    if supervision_mask.dtype != torch.bool:
        raise TypeError(
            f"supervision mask must have dtype torch.bool, got {supervision_mask.dtype}"
        )
    return supervision_mask


def _freeze_json_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, Any]:
    frozen = _freeze_json(value, name=name)
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return frozen


def _freeze_json(value: object, *, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{name} object keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json(value[key], name=f"{name}.{key}")
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(
        f"{name} must contain only JSON-compatible values, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "BPBAccumulation",
    "BPBAccumulator",
    "BaseValidationResult",
    "accumulate_bpb",
    "evaluate_bpb_batches",
]
