"""Shared BPB conformance fixture for both validation protocols."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BPBConformanceFixture:
    """Oversized Unicode source plus labeled target-position edge cases."""

    context_length: int
    documents: tuple[str, ...]
    token_bytes: tuple[int, ...]
    losses_nats: tuple[float, ...]
    targets: tuple[int, ...]
    supervision_mask: tuple[bool, ...]
    position_kinds: tuple[str, ...]

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fresh tensors so tests cannot mutate shared fixture state."""

        return (
            torch.tensor(self.losses_nats, dtype=torch.float64),
            torch.tensor(self.targets, dtype=torch.long),
            torch.tensor(self.token_bytes, dtype=torch.long),
            torch.tensor(self.supervision_mask, dtype=torch.bool),
        )


BPB_CONFORMANCE_FIXTURE = BPBConformanceFixture(
    context_length=8,
    documents=(
        "ordinary ASCII document",
        "🧗🏽‍♀️ café 漢字 — oversized Unicode document\n" * 4,
    ),
    # IDs 0/1/2 are one-, two-, and three-byte ordinary tokens. ID 3 is BOS.
    token_bytes=(1, 2, 3, 0),
    losses_nats=(0.5, 1.0, 99.0, 1.5, 88.0, 77.0),
    targets=(0, 1, 3, 2, 1, -1),
    supervision_mask=(True, True, True, False, False, True),
    position_kinds=(
        "ordinary",
        "ordinary_multibyte",
        "special",
        "carried_context",
        "padding",
        "negative_target",
    ),
)


__all__ = ["BPB_CONFORMANCE_FIXTURE", "BPBConformanceFixture"]
