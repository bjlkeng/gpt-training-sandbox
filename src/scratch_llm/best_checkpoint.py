"""Protocol-pinned validation state for atomic best checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final

from scratch_llm.bpb import BaseValidationResult
from scratch_llm.full_document_bpb import (
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_PROTOCOL_VERSION,
)
from scratch_llm.nanochat_bpb import (
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_PROTOCOL_VERSION,
    NANOCHAT_REFERENCE_COMMIT,
)


BEST_CHECKPOINT_RANKING_PROTOCOL_ID: Final = NANOCHAT_COMPAT_PROTOCOL_ID
VALIDATION_IDENTITY_FORMAT: Final = "scratch_llm_base_validation_identity_v1"
_VALIDATION_STATE_KEYS = frozenset(
    {
        "current_compatibility_bpb",
        "current_full_document_bpb",
        "minimum_compatibility_bpb",
        "minimum_full_document_bpb",
        "ranking_protocol_id",
        "validation_identity",
        "validation_step",
    }
)


class BestCheckpointError(RuntimeError):
    """Validation state cannot safely continue the saved ranking history."""


@dataclass(frozen=True)
class PeriodicValidationResult:
    """One compatibility result plus its complete-corpus companion."""

    compatibility: BaseValidationResult
    full_document: BaseValidationResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.compatibility, BaseValidationResult):
            raise TypeError(
                "compatibility must be a BaseValidationResult, got "
                f"{type(self.compatibility).__name__}"
            )
        if self.compatibility.protocol_id != NANOCHAT_COMPAT_PROTOCOL_ID:
            raise ValueError(
                f"compatibility protocol must be {NANOCHAT_COMPAT_PROTOCOL_ID!r}"
            )
        _finite_non_negative(
            self.compatibility.bpb,
            name="compatibility bpb",
        )
        if self.full_document is None:
            return
        if not isinstance(self.full_document, BaseValidationResult):
            raise TypeError(
                "full_document must be a BaseValidationResult or None, got "
                f"{type(self.full_document).__name__}"
            )
        if self.full_document.protocol_id != FULL_DOCUMENT_PROTOCOL_ID:
            raise ValueError(
                f"full-document protocol must be {FULL_DOCUMENT_PROTOCOL_ID!r}"
            )
        _finite_non_negative(
            self.full_document.bpb,
            name="full-document bpb",
        )
        for name in (
            "checkpoint_identity",
            "tokenizer_identity",
            "validation_manifest_identity",
        ):
            if getattr(self.compatibility, name) != getattr(
                self.full_document,
                name,
            ):
                label = name.replace("_", " ")
                raise ValueError(f"compatibility and full-document {label} must match")

    @property
    def complete(self) -> bool:
        return self.full_document is not None

    @property
    def validation_identity(self) -> str:
        """Hash every stable evaluator choice while excluding checkpoint values."""

        return base_validation_identity(
            tokenizer_identity=self.compatibility.tokenizer_identity,
            validation_manifest_identity=(
                self.compatibility.validation_manifest_identity
            ),
            compatibility_reference_config=self.compatibility.to_dict()[
                "reference_config"
            ],
            full_document_reference_config=(
                None
                if self.full_document is None
                else self.full_document.to_dict()["reference_config"]
            ),
        )


@dataclass(frozen=True)
class ValidationCheckpointState:
    """Current and minimum BPB values persisted in resumable checkpoints."""

    ranking_protocol_id: str
    validation_identity: str
    validation_step: int
    current_compatibility_bpb: float
    minimum_compatibility_bpb: float
    current_full_document_bpb: float | None
    minimum_full_document_bpb: float | None

    def __post_init__(self) -> None:
        for name in ("ranking_protocol_id", "validation_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.validation_step, int)
            or isinstance(self.validation_step, bool)
            or self.validation_step < 0
        ):
            raise ValueError("validation_step must be a non-negative integer")
        current_compatibility_bpb = _finite_non_negative(
            self.current_compatibility_bpb,
            name="current_compatibility_bpb",
        )
        minimum_compatibility_bpb = _finite_non_negative(
            self.minimum_compatibility_bpb,
            name="minimum_compatibility_bpb",
        )
        if minimum_compatibility_bpb > current_compatibility_bpb:
            raise ValueError(
                "minimum_compatibility_bpb cannot exceed current_compatibility_bpb"
            )
        has_current_full = self.current_full_document_bpb is not None
        has_minimum_full = self.minimum_full_document_bpb is not None
        if has_current_full != has_minimum_full:
            raise ValueError(
                "current and minimum full-document BPB must both be set or null"
            )
        if has_current_full:
            current_full_document_bpb = _finite_non_negative(
                self.current_full_document_bpb,
                name="current_full_document_bpb",
            )
            minimum_full_document_bpb = _finite_non_negative(
                self.minimum_full_document_bpb,
                name="minimum_full_document_bpb",
            )
            if minimum_full_document_bpb > current_full_document_bpb:
                raise ValueError(
                    "minimum_full_document_bpb cannot exceed current_full_document_bpb"
                )
            object.__setattr__(
                self,
                "current_full_document_bpb",
                current_full_document_bpb,
            )
            object.__setattr__(
                self,
                "minimum_full_document_bpb",
                minimum_full_document_bpb,
            )
        object.__setattr__(
            self,
            "current_compatibility_bpb",
            current_compatibility_bpb,
        )
        object.__setattr__(
            self,
            "minimum_compatibility_bpb",
            minimum_compatibility_bpb,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "current_compatibility_bpb": self.current_compatibility_bpb,
            "current_full_document_bpb": self.current_full_document_bpb,
            "minimum_compatibility_bpb": self.minimum_compatibility_bpb,
            "minimum_full_document_bpb": self.minimum_full_document_bpb,
            "ranking_protocol_id": self.ranking_protocol_id,
            "validation_identity": self.validation_identity,
            "validation_step": self.validation_step,
        }

    @classmethod
    def from_dict(cls, value: object) -> ValidationCheckpointState:
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise BestCheckpointError(
                "checkpoint validation state must be an object with string keys"
            )
        if set(value) != _VALIDATION_STATE_KEYS:
            missing = sorted(_VALIDATION_STATE_KEYS - set(value))
            unexpected = sorted(set(value) - _VALIDATION_STATE_KEYS)
            raise BestCheckpointError(
                "checkpoint validation fields do not match the current format; "
                f"missing={missing}, unexpected={unexpected}"
            )
        try:
            ranking_protocol_id = value["ranking_protocol_id"]
            validation_identity = value["validation_identity"]
            validation_step = value["validation_step"]
            if not isinstance(ranking_protocol_id, str):
                raise TypeError("ranking_protocol_id must be a string")
            if not isinstance(validation_identity, str):
                raise TypeError("validation_identity must be a string")
            if not isinstance(validation_step, int) or isinstance(
                validation_step,
                bool,
            ):
                raise TypeError("validation_step must be an integer")
            return cls(
                ranking_protocol_id=ranking_protocol_id,
                validation_identity=validation_identity,
                validation_step=validation_step,
                current_compatibility_bpb=_real(
                    value["current_compatibility_bpb"],
                    name="current_compatibility_bpb",
                ),
                minimum_compatibility_bpb=_real(
                    value["minimum_compatibility_bpb"],
                    name="minimum_compatibility_bpb",
                ),
                current_full_document_bpb=_optional_real(
                    value["current_full_document_bpb"],
                    name="current_full_document_bpb",
                ),
                minimum_full_document_bpb=_optional_real(
                    value["minimum_full_document_bpb"],
                    name="minimum_full_document_bpb",
                ),
            )
        except BestCheckpointError:
            raise
        except (TypeError, ValueError) as error:
            raise BestCheckpointError(
                f"checkpoint contains invalid validation state: {error}"
            ) from error


@dataclass(frozen=True)
class ValidationDecision:
    """Whether one validation was accepted and strictly improved the ranking."""

    state: ValidationCheckpointState | None
    accepted: bool
    improved: bool
    reason: str | None = None


def advance_validation_state(
    previous: ValidationCheckpointState | None,
    validation: PeriodicValidationResult,
    *,
    validation_step: int,
) -> ValidationDecision:
    """Advance current/minimum values under the pinned compatibility ranking."""

    if previous is not None and not isinstance(previous, ValidationCheckpointState):
        raise TypeError(
            "previous must be a ValidationCheckpointState or None, got "
            f"{type(previous).__name__}"
        )
    if not isinstance(validation, PeriodicValidationResult):
        raise TypeError(
            "validation must be a PeriodicValidationResult, got "
            f"{type(validation).__name__}"
        )
    if (
        not isinstance(validation_step, int)
        or isinstance(validation_step, bool)
        or validation_step < 0
    ):
        raise ValueError("validation_step must be a non-negative integer")
    if not validation.complete:
        return ValidationDecision(
            state=previous,
            accepted=False,
            improved=False,
            reason=f"{FULL_DOCUMENT_PROTOCOL_ID} result is unavailable",
        )
    if previous is not None:
        if previous.ranking_protocol_id != BEST_CHECKPOINT_RANKING_PROTOCOL_ID:
            raise BestCheckpointError(
                "checkpoint ranking protocol changed: "
                f"{previous.ranking_protocol_id!r} != "
                f"{BEST_CHECKPOINT_RANKING_PROTOCOL_ID!r}"
            )
        if previous.validation_identity != validation.validation_identity:
            raise BestCheckpointError(
                "checkpoint validation identity changed: "
                f"{previous.validation_identity!r} != "
                f"{validation.validation_identity!r}"
            )
        if validation_step <= previous.validation_step:
            raise BestCheckpointError(
                "validation_step must advance beyond the checkpoint state"
            )

    full_document = validation.full_document
    if full_document is None:  # pragma: no cover - complete checked above.
        raise RuntimeError("complete validation lost its full-document result")
    compatibility_bpb = validation.compatibility.bpb
    full_document_bpb = full_document.bpb
    improved = (
        previous is None or compatibility_bpb < previous.minimum_compatibility_bpb
    )
    state = ValidationCheckpointState(
        ranking_protocol_id=BEST_CHECKPOINT_RANKING_PROTOCOL_ID,
        validation_identity=validation.validation_identity,
        validation_step=validation_step,
        current_compatibility_bpb=compatibility_bpb,
        minimum_compatibility_bpb=(
            compatibility_bpb
            if previous is None
            else min(compatibility_bpb, previous.minimum_compatibility_bpb)
        ),
        current_full_document_bpb=full_document_bpb,
        minimum_full_document_bpb=(
            full_document_bpb
            if previous is None or previous.minimum_full_document_bpb is None
            else min(full_document_bpb, previous.minimum_full_document_bpb)
        ),
    )
    return ValidationDecision(
        state=state,
        accepted=True,
        improved=improved,
    )


def base_validation_identity(
    *,
    tokenizer_identity: str,
    validation_manifest_identity: str,
    compatibility_reference_config: Mapping[str, object],
    full_document_reference_config: Mapping[str, object] | None,
) -> str:
    """Hash the pinned ranking protocol, evaluator configs, and data identity."""

    for name, value in (
        ("tokenizer_identity", tokenizer_identity),
        ("validation_manifest_identity", validation_manifest_identity),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if not isinstance(compatibility_reference_config, Mapping):
        raise TypeError("compatibility_reference_config must be a mapping")
    if full_document_reference_config is not None and not isinstance(
        full_document_reference_config,
        Mapping,
    ):
        raise TypeError("full_document_reference_config must be a mapping or None")
    payload = {
        "format": VALIDATION_IDENTITY_FORMAT,
        "ranking_protocol_id": BEST_CHECKPOINT_RANKING_PROTOCOL_ID,
        "tokenizer_identity": tokenizer_identity,
        "validation_manifest_identity": validation_manifest_identity,
        "compatibility_protocol": {
            "protocol_id": NANOCHAT_COMPAT_PROTOCOL_ID,
            "protocol_version": NANOCHAT_COMPAT_PROTOCOL_VERSION,
            "reference_commit": NANOCHAT_REFERENCE_COMMIT,
            "reference_config": dict(compatibility_reference_config),
        },
        "full_document_protocol": (
            None
            if full_document_reference_config is None
            else {
                "protocol_id": FULL_DOCUMENT_PROTOCOL_ID,
                "protocol_version": FULL_DOCUMENT_PROTOCOL_VERSION,
                "reference_commit": None,
                "reference_config": dict(full_document_reference_config),
            }
        ),
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"validation identity inputs must contain finite JSON values: {error}"
        ) from error
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _real(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _optional_real(value: object, *, name: str) -> float | None:
    return None if value is None else _real(value, name=name)


def _finite_non_negative(value: object, *, name: str) -> float:
    numeric = _real(value, name=name)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


__all__ = [
    "BEST_CHECKPOINT_RANKING_PROTOCOL_ID",
    "BestCheckpointError",
    "PeriodicValidationResult",
    "ValidationCheckpointState",
    "ValidationDecision",
    "advance_validation_state",
    "base_validation_identity",
]
