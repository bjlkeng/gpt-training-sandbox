"""Runtime selection for training-only GPT block activation checkpointing."""

from __future__ import annotations

from dataclasses import dataclass

from scratch_llm.model import GPT


@dataclass(frozen=True)
class ActivationCheckpointSelection:
    """Observable activation-retention choice."""

    requested: bool
    effective: bool
    block_boundary: bool = True
    use_reentrant: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "block_boundary": self.block_boundary,
            "effective": self.effective,
            "requested": self.requested,
            "use_reentrant": self.use_reentrant,
        }


def configure_activation_checkpointing(
    model: GPT,
    *,
    enabled: bool,
) -> ActivationCheckpointSelection:
    """Configure one canonical GPT without changing its module structure."""

    if not isinstance(model, GPT):
        raise TypeError(f"model must be a GPT, got {type(model).__name__}")
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a boolean")
    model.set_activation_checkpointing(enabled)
    return ActivationCheckpointSelection(
        requested=enabled,
        effective=enabled,
    )


def format_activation_checkpoint_selection(
    selection: ActivationCheckpointSelection,
) -> str:
    """Render one stable progress line."""

    return (
        f"Activation checkpointing: requested={selection.requested} "
        f"effective={selection.effective} block_boundary={selection.block_boundary} "
        f"use_reentrant={selection.use_reentrant}"
    )
