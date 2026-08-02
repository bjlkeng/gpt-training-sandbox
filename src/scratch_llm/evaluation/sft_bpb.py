"""Assistant-content bits-per-byte evaluation for finite SFT validation data."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import json
import math
from typing import Final, Protocol

import torch
from torch import Tensor, nn

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_non_empty_string,
    require_non_negative_integer,
    require_optional_positive_integer,
    require_positive_integer,
)
from scratch_llm.chat.loader import SFTBatchInfo
from scratch_llm.chat.rendering import CHAT_RENDERER_ID
from scratch_llm.evaluation.bpb import (
    BPBAccumulation,
    BPBAccumulator,
    evaluate_bpb_batches,
)


SFT_ASSISTANT_BPB_PROTOCOL_ID: Final = "sft_assistant_bpb_v1"
SFT_ASSISTANT_BPB_PROTOCOL_VERSION: Final = 1


class SFTValidationError(ValueError):
    """SFT validation data or protocol settings cannot produce a valid result."""


class SFTValidationLoader(Protocol):
    """Minimal fresh finite loader boundary consumed by SFT evaluation."""

    repeat: bool
    global_step: int

    @property
    def last_batch_info(self) -> SFTBatchInfo:
        """Return provenance for the most recently emitted whole batch."""

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return one whole SFT batch or raise ``StopIteration``."""


@dataclass(frozen=True, slots=True)
class SFTAssistantBPBResult:
    """Immutable protocol identity, actual coverage, and assistant-only BPB."""

    protocol_id: str
    protocol_version: int
    checkpoint_identity: str
    tokenizer_identity: str
    renderer_identity: str
    validation_mixture_identity: str
    batch_budget: int | None
    evaluated_batches: int
    source_conversations: int
    processed_model_tokens: int
    supervised_target_tokens: int
    supervised_target_bytes: int
    total_nats: float
    bpb: float

    def __post_init__(self) -> None:
        if self.protocol_id != SFT_ASSISTANT_BPB_PROTOCOL_ID:
            raise ValueError(
                f"protocol_id must equal {SFT_ASSISTANT_BPB_PROTOCOL_ID!r}"
            )
        if self.protocol_version != SFT_ASSISTANT_BPB_PROTOCOL_VERSION:
            raise ValueError(
                f"protocol_version must equal {SFT_ASSISTANT_BPB_PROTOCOL_VERSION}"
            )
        for name in (
            "checkpoint_identity",
            "tokenizer_identity",
            "validation_mixture_identity",
        ):
            require_non_empty_string(getattr(self, name), name=name)
        if self.renderer_identity != CHAT_RENDERER_ID:
            raise ValueError(f"renderer_identity must equal {CHAT_RENDERER_ID!r}")
        batch_budget = require_optional_positive_integer(
            self.batch_budget,
            name="batch_budget",
        )
        evaluated_batches = require_positive_integer(
            self.evaluated_batches,
            name="evaluated_batches",
        )
        if batch_budget is not None and evaluated_batches > batch_budget:
            raise ValueError("evaluated_batches must not exceed batch_budget")
        source_conversations = require_positive_integer(
            self.source_conversations,
            name="source_conversations",
        )
        processed_model_tokens = require_positive_integer(
            self.processed_model_tokens,
            name="processed_model_tokens",
        )
        supervised_target_tokens = require_positive_integer(
            self.supervised_target_tokens,
            name="supervised_target_tokens",
        )
        supervised_target_bytes = require_positive_integer(
            self.supervised_target_bytes,
            name="supervised_target_bytes",
        )
        if processed_model_tokens < supervised_target_tokens:
            raise ValueError(
                "processed_model_tokens must be greater than or equal to "
                "supervised_target_tokens"
            )
        total_nats = require_finite_non_negative_real(
            self.total_nats,
            name="total_nats",
        )
        bpb = require_finite_non_negative_real(self.bpb, name="bpb")
        expected_bpb = total_nats / math.log(2) / supervised_target_bytes
        if not math.isclose(
            bpb,
            expected_bpb,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "bpb does not match total_nats / log(2) / supervised_target_bytes"
            )
        object.__setattr__(self, "batch_budget", batch_budget)
        object.__setattr__(self, "evaluated_batches", evaluated_batches)
        object.__setattr__(self, "source_conversations", source_conversations)
        object.__setattr__(
            self,
            "processed_model_tokens",
            processed_model_tokens,
        )
        object.__setattr__(
            self,
            "supervised_target_tokens",
            supervised_target_tokens,
        )
        object.__setattr__(
            self,
            "supervised_target_bytes",
            supervised_target_bytes,
        )
        object.__setattr__(self, "total_nats", total_nats)
        object.__setattr__(self, "bpb", bpb)

    @classmethod
    def from_accumulation(
        cls,
        accumulation: BPBAccumulation,
        *,
        checkpoint_identity: str,
        tokenizer_identity: str,
        validation_mixture_identity: str,
        batch_budget: int | None,
        evaluated_batches: int,
        source_conversations: int,
    ) -> SFTAssistantBPBResult:
        """Combine shared BPB arithmetic with SFT protocol coverage."""

        if not isinstance(accumulation, BPBAccumulation):
            raise TypeError(
                "accumulation must be a BPBAccumulation, "
                f"got {type(accumulation).__name__}"
            )
        return cls(
            protocol_id=SFT_ASSISTANT_BPB_PROTOCOL_ID,
            protocol_version=SFT_ASSISTANT_BPB_PROTOCOL_VERSION,
            checkpoint_identity=checkpoint_identity,
            tokenizer_identity=tokenizer_identity,
            renderer_identity=CHAT_RENDERER_ID,
            validation_mixture_identity=validation_mixture_identity,
            batch_budget=batch_budget,
            evaluated_batches=evaluated_batches,
            source_conversations=source_conversations,
            processed_model_tokens=accumulation.processed_model_tokens,
            supervised_target_tokens=accumulation.counted_target_tokens,
            supervised_target_bytes=accumulation.counted_target_bytes,
            total_nats=accumulation.total_nats,
            bpb=accumulation.bpb,
        )

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Return the stable JSON-compatible result schema."""

        return {
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "checkpoint_identity": self.checkpoint_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "renderer_identity": self.renderer_identity,
            "validation_mixture_identity": self.validation_mixture_identity,
            "batch_budget": self.batch_budget,
            "evaluated_batches": self.evaluated_batches,
            "source_conversations": self.source_conversations,
            "processed_model_tokens": self.processed_model_tokens,
            "supervised_target_tokens": self.supervised_target_tokens,
            "supervised_target_bytes": self.supervised_target_bytes,
            "total_nats": self.total_nats,
            "bpb": self.bpb,
        }

    def to_json(self) -> str:
        """Return canonical UTF-8 JSON without non-finite spellings."""

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


class _BudgetedSFTBatches:
    """Expose complete loader batches while recording actual conversation coverage."""

    def __init__(
        self,
        loader: SFTValidationLoader,
        *,
        max_batches: int | None,
    ) -> None:
        self.loader = loader
        self.max_batches = max_batches
        self.evaluated_batches = 0
        self.source_conversations = 0

    def __iter__(self) -> Iterator[Sequence[Tensor]]:
        while self.max_batches is None or self.evaluated_batches < self.max_batches:
            try:
                batch = self.loader.next_batch()
            except StopIteration:
                return
            info = self.loader.last_batch_info
            if not isinstance(info, SFTBatchInfo):
                raise SFTValidationError(
                    "validation loader last_batch_info must be an SFTBatchInfo"
                )
            conversation_count = sum(
                len(row_identities) for row_identities in info.row_item_identities
            )
            if conversation_count <= 0:
                raise SFTValidationError(
                    "validation batch provenance must contain a source conversation"
                )
            self.evaluated_batches += 1
            self.source_conversations += conversation_count
            yield batch


def evaluate_sft_assistant_bpb(
    model: nn.Module,
    validation_loader: SFTValidationLoader,
    token_bytes: Tensor,
    *,
    checkpoint_identity: str,
    tokenizer_identity: str,
    validation_mixture_identity: str,
    device: str | torch.device,
    max_batches: int | None = None,
) -> SFTAssistantBPBResult:
    """Evaluate positive-byte assistant targets over fresh finite whole batches."""

    for name, value in (
        ("checkpoint_identity", checkpoint_identity),
        ("tokenizer_identity", tokenizer_identity),
        ("validation_mixture_identity", validation_mixture_identity),
    ):
        require_non_empty_string(value, name=name)
    max_batches = require_optional_positive_integer(
        max_batches,
        name="max_batches",
    )
    _validate_fresh_finite_loader(validation_loader)
    budgeted_batches = _BudgetedSFTBatches(
        validation_loader,
        max_batches=max_batches,
    )
    try:
        accumulation = evaluate_bpb_batches(
            model,
            budgeted_batches,
            token_bytes,
            device=device,
        )
    except ValueError as error:
        if "zero counted target bytes" not in str(error):
            raise
        if budgeted_batches.evaluated_batches == 0:
            raise SFTValidationError(
                "SFT validation data yielded no batches"
            ) from error
        raise SFTValidationError(
            "SFT validation data yielded zero counted assistant-content bytes"
        ) from error
    return SFTAssistantBPBResult.from_accumulation(
        accumulation,
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer_identity,
        validation_mixture_identity=validation_mixture_identity,
        batch_budget=max_batches,
        evaluated_batches=budgeted_batches.evaluated_batches,
        source_conversations=budgeted_batches.source_conversations,
    )


@dataclass(frozen=True, slots=True)
class SFTAssistantBPBCallback:
    """Artifact-free trainer callback that evaluates an in-memory named step."""

    model: nn.Module
    validation_loader_factory: Callable[[], SFTValidationLoader]
    token_bytes: Tensor
    checkpoint_identity_prefix: str
    tokenizer_identity: str
    validation_mixture_identity: str
    device: str | torch.device
    max_batches: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"model must be an nn.Module, got {type(self.model).__name__}"
            )
        if not callable(self.validation_loader_factory):
            raise TypeError("validation_loader_factory must be callable")
        for name in (
            "checkpoint_identity_prefix",
            "tokenizer_identity",
            "validation_mixture_identity",
        ):
            require_non_empty_string(getattr(self, name), name=name)
        max_batches = require_optional_positive_integer(
            self.max_batches,
            name="max_batches",
        )
        BPBAccumulator(self.token_bytes)
        object.__setattr__(
            self,
            "token_bytes",
            self.token_bytes.detach().to(device="cpu").clone(),
        )
        object.__setattr__(self, "max_batches", max_batches)

    def __call__(self, step: int) -> SFTAssistantBPBResult:
        """Evaluate a new finite validation view at non-negative ``step``."""

        step = require_non_negative_integer(step, name="step")
        loader = self.validation_loader_factory()
        return evaluate_sft_assistant_bpb(
            self.model,
            loader,
            self.token_bytes,
            checkpoint_identity=f"{self.checkpoint_identity_prefix}#step:{step}",
            tokenizer_identity=self.tokenizer_identity,
            validation_mixture_identity=self.validation_mixture_identity,
            device=self.device,
            max_batches=self.max_batches,
        )


def _validate_fresh_finite_loader(loader: object) -> None:
    if not callable(getattr(loader, "next_batch", None)):
        raise TypeError("validation_loader must expose next_batch()")
    if getattr(loader, "repeat", None) is not False:
        raise SFTValidationError(
            "SFT validation requires a finite loader with repeat=False"
        )
    try:
        global_step = require_non_negative_integer(
            getattr(loader, "global_step"),
            name="validation_loader.global_step",
        )
    except AttributeError as error:
        raise TypeError("validation_loader must expose global_step") from error
    if global_step != 0:
        raise SFTValidationError(
            "SFT validation requires a fresh loader with global_step=0"
        )


__all__ = [
    "SFT_ASSISTANT_BPB_PROTOCOL_ID",
    "SFT_ASSISTANT_BPB_PROTOCOL_VERSION",
    "SFTAssistantBPBCallback",
    "SFTAssistantBPBResult",
    "SFTValidationError",
    "SFTValidationLoader",
    "evaluate_sft_assistant_bpb",
]
