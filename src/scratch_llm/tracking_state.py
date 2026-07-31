"""Small backend-neutral remote-tracking state for checkpoint resume."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal


_TRACKING_STATE_KEYS = frozenset({"backend", "run_id"})
_REMOTE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
WandbResumeBehavior = Literal["same", "fork"]


class TrackingStateError(ValueError):
    """Serialized tracking state is malformed or incompatible."""


@dataclass(frozen=True)
class TrackingState:
    """Identify one remote backend run without backend-specific arguments."""

    backend: str
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")
        if (
            not isinstance(self.run_id, str)
            or _REMOTE_ID_PATTERN.fullmatch(self.run_id) is None
        ):
            raise ValueError(
                "run_id must contain only letters, numbers, underscores, and hyphens"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrackingState:
        if not isinstance(value, dict) or set(value) != _TRACKING_STATE_KEYS:
            raise TrackingStateError(
                "checkpoint tracking state fields must be backend and run_id"
            )
        backend = value["backend"]
        run_id = value["run_id"]
        if not isinstance(backend, str):
            raise TrackingStateError("checkpoint tracking backend must be a string")
        if not isinstance(run_id, str):
            raise TrackingStateError("checkpoint tracking run_id must be a string")
        try:
            return cls(backend=backend, run_id=run_id)
        except ValueError as error:
            raise TrackingStateError(
                f"checkpoint contains invalid tracking state: {error}"
            ) from error


def resolve_wandb_resume_state(
    checkpoint_state: TrackingState | None,
    *,
    source_run_name: str,
    source_output_dir: str,
    current_run_name: str,
    current_output_dir: str,
    behavior: WandbResumeBehavior | None,
) -> TrackingState | None:
    """Select same-run or fork behavior for one W&B-enabled resume."""

    for name, value in (
        ("source_run_name", source_run_name),
        ("source_output_dir", source_output_dir),
        ("current_run_name", current_run_name),
        ("current_output_dir", current_output_dir),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if behavior not in {None, "same", "fork"}:
        raise ValueError("behavior must be 'same', 'fork', or None")

    identity_changed = (
        source_run_name != current_run_name
        or Path(source_output_dir).resolve() != Path(current_output_dir).resolve()
    )
    selected_behavior = behavior
    if selected_behavior is None:
        if identity_changed:
            raise ValueError(
                "W&B-enabled resume changes run.name or run.output_dir; choose "
                "--wandb-resume same to continue the saved remote run or "
                "--wandb-resume fork to create a new remote run"
            )
        selected_behavior = "same"

    if selected_behavior == "fork":
        if not identity_changed:
            raise ValueError(
                "a W&B run fork must change run.name or run.output_dir so the "
                "new local run state cannot overwrite the source run"
            )
        return None

    if checkpoint_state is None:
        raise ValueError(
            "resume checkpoint does not contain a W&B remote run identity; "
            "choose --wandb-resume fork with a new local run identity"
        )
    if checkpoint_state.backend != "wandb":
        raise ValueError(
            "resume checkpoint tracking backend is incompatible with W&B: "
            f"{checkpoint_state.backend!r}"
        )
    return checkpoint_state


__all__ = [
    "TrackingState",
    "TrackingStateError",
    "WandbResumeBehavior",
    "resolve_wandb_resume_state",
]
