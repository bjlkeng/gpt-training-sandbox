"""Tests for compatibility-ranked best-checkpoint state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from scratch_llm.best_checkpoint import (
    BEST_CHECKPOINT_RANKING_PROTOCOL_ID,
    BestCheckpointError,
    PeriodicValidationResult,
    ValidationCheckpointState,
    advance_validation_state,
)
from scratch_llm.bpb import BPBAccumulation, BaseValidationResult
from scratch_llm.full_document_bpb import (
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_PROTOCOL_VERSION,
)
from scratch_llm.nanochat_bpb import (
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_PROTOCOL_VERSION,
    NANOCHAT_REFERENCE_COMMIT,
)


def _result(
    protocol_id: str,
    *,
    bpb: float,
    checkpoint_identity: str = "checkpoint:step",
    tokenizer_identity: str = "tokenizer:fixture",
    manifest_identity: str = "manifest:fixture",
) -> BaseValidationResult:
    is_compatibility = protocol_id == NANOCHAT_COMPAT_PROTOCOL_ID
    return BaseValidationResult.from_accumulation(
        BPBAccumulation(
            processed_model_tokens=4,
            counted_target_tokens=2,
            counted_target_bytes=2,
            total_nats=bpb * math.log(2) * 2,
        ),
        protocol_id=protocol_id,
        protocol_version=(
            NANOCHAT_COMPAT_PROTOCOL_VERSION
            if is_compatibility
            else FULL_DOCUMENT_PROTOCOL_VERSION
        ),
        reference_commit=NANOCHAT_REFERENCE_COMMIT if is_compatibility else None,
        reference_config={"fixture": protocol_id},
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer_identity,
        validation_manifest_identity=manifest_identity,
        source_documents=1,
        source_tokens=2,
        source_bytes=2,
        unique_source_tokens=2,
        unique_source_bytes=2,
    )


def _periodic(
    compatibility_bpb: float,
    full_document_bpb: float,
    *,
    checkpoint_identity: str = "checkpoint:step",
) -> PeriodicValidationResult:
    return PeriodicValidationResult(
        compatibility=_result(
            NANOCHAT_COMPAT_PROTOCOL_ID,
            bpb=compatibility_bpb,
            checkpoint_identity=checkpoint_identity,
        ),
        full_document=_result(
            FULL_DOCUMENT_PROTOCOL_ID,
            bpb=full_document_bpb,
            checkpoint_identity=checkpoint_identity,
        ),
    )


def test_strict_compatibility_curve_updates_current_values_and_both_minima() -> None:
    previous: ValidationCheckpointState | None = None
    decisions = []
    for step, (compatibility_bpb, full_document_bpb) in enumerate(
        (
            (2.0, 3.0),
            (1.5, 2.8),
            (1.5, 2.4),
            (1.8, 2.6),
            (1.2, 2.5),
        ),
        start=1,
    ):
        decision = advance_validation_state(
            previous,
            _periodic(
                compatibility_bpb,
                full_document_bpb,
                checkpoint_identity=f"checkpoint:{step}",
            ),
            validation_step=step,
        )
        decisions.append(decision)
        previous = decision.state

    assert [decision.accepted for decision in decisions] == [True] * 5
    assert [decision.improved for decision in decisions] == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert previous is not None
    assert previous.ranking_protocol_id == BEST_CHECKPOINT_RANKING_PROTOCOL_ID
    assert previous.validation_step == 5
    assert previous.current_compatibility_bpb == 1.2
    assert previous.minimum_compatibility_bpb == 1.2
    assert previous.current_full_document_bpb == 2.5
    assert previous.minimum_full_document_bpb == 2.4
    assert previous.validation_identity == decisions[0].state.validation_identity
    assert (
        previous.to_dict()
        == ValidationCheckpointState.from_dict(previous.to_dict()).to_dict()
    )

    with pytest.raises(FrozenInstanceError):
        previous.minimum_compatibility_bpb = 0.0  # type: ignore[misc]


def test_partial_validation_cannot_create_or_advance_ranking_state() -> None:
    partial = PeriodicValidationResult(
        compatibility=_result(NANOCHAT_COMPAT_PROTOCOL_ID, bpb=1.0),
        full_document=None,
    )

    initial = advance_validation_state(None, partial, validation_step=1)
    existing = advance_validation_state(
        None,
        _periodic(2.0, 3.0),
        validation_step=1,
    ).state
    repeated = advance_validation_state(existing, partial, validation_step=2)

    assert initial.accepted is False
    assert initial.improved is False
    assert initial.state is None
    assert initial.reason == "full_documents_v1 result is unavailable"
    assert repeated.accepted is False
    assert repeated.improved is False
    assert repeated.state is existing


def test_protocol_identity_checkpoint_identity_and_finite_bpb_are_strict() -> None:
    with pytest.raises(ValueError, match="compatibility protocol"):
        PeriodicValidationResult(
            compatibility=_result(FULL_DOCUMENT_PROTOCOL_ID, bpb=1.0),
            full_document=_result(FULL_DOCUMENT_PROTOCOL_ID, bpb=1.0),
        )

    with pytest.raises(ValueError, match="checkpoint identity"):
        PeriodicValidationResult(
            compatibility=_result(
                NANOCHAT_COMPAT_PROTOCOL_ID,
                bpb=1.0,
                checkpoint_identity="checkpoint:a",
            ),
            full_document=_result(
                FULL_DOCUMENT_PROTOCOL_ID,
                bpb=1.0,
                checkpoint_identity="checkpoint:b",
            ),
        )

    non_finite = _result(NANOCHAT_COMPAT_PROTOCOL_ID, bpb=1.0)
    object.__setattr__(non_finite, "bpb", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        PeriodicValidationResult(
            compatibility=non_finite,
            full_document=_result(FULL_DOCUMENT_PROTOCOL_ID, bpb=1.0),
        )


def test_resume_rejects_changed_ranking_or_validation_identity() -> None:
    original = advance_validation_state(
        None,
        _periodic(2.0, 3.0),
        validation_step=1,
    ).state
    assert original is not None

    changed_identity = _periodic(1.0, 2.0)
    object.__setattr__(
        changed_identity.compatibility,
        "reference_config",
        {"fixture": "changed"},
    )
    with pytest.raises(BestCheckpointError, match="validation identity"):
        advance_validation_state(
            original,
            changed_identity,
            validation_step=2,
        )

    payload = original.to_dict()
    payload["ranking_protocol_id"] = "full_documents_v1"
    changed_ranking = ValidationCheckpointState.from_dict(payload)
    with pytest.raises(BestCheckpointError, match="ranking protocol"):
        advance_validation_state(
            changed_ranking,
            _periodic(1.0, 2.0),
            validation_step=2,
        )


def test_checkpoint_validation_state_rejects_malformed_json_values() -> None:
    state = ValidationCheckpointState(
        ranking_protocol_id=BEST_CHECKPOINT_RANKING_PROTOCOL_ID,
        validation_identity="sha256:fixture",
        validation_step=1,
        current_compatibility_bpb=2.0,
        minimum_compatibility_bpb=1.5,
        current_full_document_bpb=None,
        minimum_full_document_bpb=None,
    )
    payload = state.to_dict()

    with pytest.raises(BestCheckpointError, match="must be an object"):
        ValidationCheckpointState.from_dict([])

    missing_field = dict(payload)
    del missing_field["validation_identity"]
    with pytest.raises(BestCheckpointError, match="fields do not match"):
        ValidationCheckpointState.from_dict(missing_field)

    empty_protocol = dict(payload)
    empty_protocol["ranking_protocol_id"] = " "
    with pytest.raises(BestCheckpointError, match="non-empty string"):
        ValidationCheckpointState.from_dict(empty_protocol)

    boolean_step = dict(payload)
    boolean_step["validation_step"] = True
    with pytest.raises(BestCheckpointError, match="must be an integer"):
        ValidationCheckpointState.from_dict(boolean_step)

    boolean_bpb = dict(payload)
    boolean_bpb["current_compatibility_bpb"] = True
    with pytest.raises(BestCheckpointError, match="must be a number"):
        ValidationCheckpointState.from_dict(boolean_bpb)
