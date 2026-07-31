"""Backend-neutral tracking-state and W&B resume-policy tests."""

from __future__ import annotations

import pytest

from scratch_llm.tracking_state import (
    TrackingState,
    TrackingStateError,
    resolve_wandb_resume_state,
)


def test_tracking_state_round_trips_exact_backend_and_remote_id() -> None:
    state = TrackingState(backend="wandb", run_id="run_123-abc")

    assert state.to_dict() == {
        "backend": "wandb",
        "run_id": "run_123-abc",
    }
    assert TrackingState.from_dict(state.to_dict()) == state


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "fields"),
        ({"backend": "wandb", "run_id": ""}, "run_id"),
        ({"backend": "wandb", "run_id": "contains space"}, "run_id"),
        ({"backend": 1, "run_id": "valid"}, "backend"),
    ],
)
def test_tracking_state_rejects_malformed_checkpoint_values(
    value: object,
    message: str,
) -> None:
    with pytest.raises(TrackingStateError, match=message):
        TrackingState.from_dict(value)


def test_unchanged_identity_defaults_to_same_saved_wandb_run() -> None:
    state = TrackingState(backend="wandb", run_id="source-id")

    selected = resolve_wandb_resume_state(
        state,
        source_run_name="run",
        source_output_dir="runs",
        current_run_name="run",
        current_output_dir="runs",
        behavior=None,
    )

    assert selected == state


def test_changed_identity_requires_explicit_same_or_fork_choice() -> None:
    state = TrackingState(backend="wandb", run_id="source-id")
    arguments = {
        "source_run_name": "source",
        "source_output_dir": "runs",
        "current_run_name": "resumed",
        "current_output_dir": "runs",
    }

    with pytest.raises(ValueError, match="--wandb-resume.*same.*fork"):
        resolve_wandb_resume_state(state, behavior=None, **arguments)

    assert (
        resolve_wandb_resume_state(
            state,
            behavior="same",
            **arguments,
        )
        == state
    )
    assert (
        resolve_wandb_resume_state(
            state,
            behavior="fork",
            **arguments,
        )
        is None
    )


def test_same_run_requires_saved_compatible_wandb_identity() -> None:
    arguments = {
        "source_run_name": "source",
        "source_output_dir": "runs",
        "current_run_name": "resumed",
        "current_output_dir": "runs",
        "behavior": "same",
    }

    with pytest.raises(ValueError, match="does not contain.*W&B.*run"):
        resolve_wandb_resume_state(None, **arguments)
    with pytest.raises(ValueError, match="backend"):
        resolve_wandb_resume_state(
            TrackingState(backend="other", run_id="remote"),
            **arguments,
        )


def test_fork_requires_a_new_local_run_identity() -> None:
    with pytest.raises(ValueError, match="fork.*run.name.*output_dir"):
        resolve_wandb_resume_state(
            TrackingState(backend="wandb", run_id="source"),
            source_run_name="run",
            source_output_dir="runs",
            current_run_name="run",
            current_output_dir="runs",
            behavior="fork",
        )
