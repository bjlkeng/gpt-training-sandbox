"""Mode orchestration for standalone base-model evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from scratch_llm._validation import (
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.evaluation.sampling import BaseSamplesResult
from scratch_llm.best_checkpoint import PeriodicValidationResult
from scratch_llm.evaluation.core.results import CoreEvaluationResult


BaseEvaluationMode: TypeAlias = Literal["bpb", "sample", "core"]
BaseEvaluationRunKind: TypeAlias = Literal["bounded", "full"]
BaseEvaluationBpbRunner: TypeAlias = Callable[[], PeriodicValidationResult]
BaseEvaluationSampleRunner: TypeAlias = Callable[[], BaseSamplesResult]
BaseEvaluationCoreRunner: TypeAlias = Callable[
    [int | None],
    CoreEvaluationResult,
]

_SUPPORTED_MODES = frozenset({"bpb", "sample", "core"})


class BaseEvaluationError(ValueError):
    """A standalone base-evaluation request is invalid or inconsistent."""


class BaseEvaluationUnavailableError(BaseEvaluationError):
    """A requested evaluation mode has no registered implementation yet."""


@dataclass(frozen=True)
class BaseEvaluationContext:
    """Immutable identities and scope shared by every requested mode."""

    checkpoint_identity: str
    checkpoint_step: int
    config_identity: str
    tokenizer_identity: str
    validation_manifest_identity: str | None
    run_kind: BaseEvaluationRunKind
    max_per_task: int | None

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_identity",
            "config_identity",
            "tokenizer_identity",
        ):
            require_non_empty_string(getattr(self, name), name=name)
        require_non_negative_integer(self.checkpoint_step, name="checkpoint_step")
        if self.validation_manifest_identity is not None:
            require_non_empty_string(
                self.validation_manifest_identity,
                name="validation_manifest_identity",
            )
        if self.run_kind not in ("bounded", "full"):
            raise BaseEvaluationError("run_kind must be 'bounded' or 'full'")
        if self.max_per_task is not None:
            require_positive_integer(self.max_per_task, name="max_per_task")


@dataclass(frozen=True)
class CompletedBaseEvaluation:
    """All requested modes completed against one frozen evaluation context."""

    context: BaseEvaluationContext
    requested_modes: tuple[BaseEvaluationMode, ...]
    completed_modes: tuple[BaseEvaluationMode, ...]
    validation: PeriodicValidationResult | None
    samples: BaseSamplesResult | None
    core_result: CoreEvaluationResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.context, BaseEvaluationContext):
            raise TypeError("context must be a BaseEvaluationContext")
        _validate_modes(self.requested_modes)
        _validate_modes(self.completed_modes)
        if self.completed_modes != self.requested_modes:
            raise BaseEvaluationError(
                "completed_modes must exactly match requested_modes and order"
            )
        self._validate_bpb_result()
        self._validate_sample_result()
        self._validate_core_result()

    def _validate_bpb_result(self) -> None:
        requested = "bpb" in self.requested_modes
        if not requested:
            if self.validation is not None:
                raise BaseEvaluationError(
                    "validation must be absent when bpb was not requested"
                )
            return
        if not isinstance(self.validation, PeriodicValidationResult):
            raise BaseEvaluationError(
                "bpb completion requires a PeriodicValidationResult"
            )
        full_document = self.validation.full_document
        if full_document is None:
            raise BaseEvaluationError(
                "bpb completion requires compatibility and full-document results"
            )
        for result in (self.validation.compatibility, full_document):
            if result.checkpoint_identity != self.context.checkpoint_identity:
                raise BaseEvaluationError(
                    "BPB checkpoint identity does not match the evaluation context"
                )
            if result.tokenizer_identity != self.context.tokenizer_identity:
                raise BaseEvaluationError(
                    "BPB tokenizer identity does not match the evaluation context"
                )
            if (
                result.validation_manifest_identity
                != self.context.validation_manifest_identity
            ):
                raise BaseEvaluationError(
                    "BPB validation manifest identity does not match the "
                    "evaluation context"
                )

    def _validate_sample_result(self) -> None:
        requested = "sample" in self.requested_modes
        if not requested:
            if self.samples is not None:
                raise BaseEvaluationError(
                    "samples must be absent when sample was not requested"
                )
            return
        if not isinstance(self.samples, BaseSamplesResult):
            raise BaseEvaluationError("sample completion requires a BaseSamplesResult")
        if self.samples.checkpoint_identity != self.context.checkpoint_identity:
            raise BaseEvaluationError(
                "sample checkpoint identity does not match the evaluation context"
            )
        if self.samples.tokenizer_identity != self.context.tokenizer_identity:
            raise BaseEvaluationError(
                "sample tokenizer identity does not match the evaluation context"
            )

    def _validate_core_result(self) -> None:
        requested = "core" in self.requested_modes
        if not requested:
            if self.core_result is not None:
                raise BaseEvaluationError(
                    "core_result must be absent when core was not requested"
                )
            return
        if self.core_result is None:
            raise BaseEvaluationError("core completion requires a result object")
        if not isinstance(self.core_result, CoreEvaluationResult):
            raise TypeError("core_result must be a CoreEvaluationResult")
        if self.core_result.checkpoint_identity != self.context.checkpoint_identity:
            raise BaseEvaluationError(
                "CORE checkpoint identity does not match the evaluation context"
            )
        if self.core_result.tokenizer_identity != self.context.tokenizer_identity:
            raise BaseEvaluationError(
                "CORE tokenizer identity does not match the evaluation context"
            )
        if self.core_result.run_kind != self.context.run_kind:
            raise BaseEvaluationError(
                "CORE run kind does not match the evaluation context"
            )
        if self.core_result.max_per_task != self.context.max_per_task:
            raise BaseEvaluationError(
                "CORE max_per_task does not match the evaluation context"
            )


def normalize_base_evaluation_modes(value: str) -> tuple[BaseEvaluationMode, ...]:
    """Normalize one comma-separated mode list while preserving first order."""

    if not isinstance(value, str):
        raise TypeError(
            f"evaluation modes must be a string, got {type(value).__name__}"
        )
    raw_modes = value.split(",")
    normalized: list[BaseEvaluationMode] = []
    seen: set[str] = set()
    for raw_mode in raw_modes:
        mode = raw_mode.strip().lower()
        if not mode:
            raise BaseEvaluationError("evaluation modes must not contain empty values")
        if mode not in _SUPPORTED_MODES:
            options = ", ".join(sorted(_SUPPORTED_MODES))
            raise BaseEvaluationError(
                f"unknown evaluation mode {mode!r}; expected one of: {options}"
            )
        if mode not in seen:
            normalized.append(mode)  # type: ignore[arg-type]
            seen.add(mode)
    if not normalized:  # pragma: no cover - split always returns one item.
        raise BaseEvaluationError("at least one evaluation mode is required")
    return tuple(normalized)


def execute_base_evaluation_modes(
    modes: Sequence[BaseEvaluationMode],
    *,
    context: BaseEvaluationContext,
    bpb_runner: BaseEvaluationBpbRunner | None,
    sample_runner: BaseEvaluationSampleRunner | None,
    core_runner: BaseEvaluationCoreRunner | None,
) -> CompletedBaseEvaluation:
    """Preflight capabilities, then execute each supplied mode runner once."""

    requested_modes = tuple(modes)
    _validate_modes(requested_modes)
    if not isinstance(context, BaseEvaluationContext):
        raise TypeError("context must be a BaseEvaluationContext")
    runners: dict[BaseEvaluationMode, Callable[..., object] | None] = {
        "bpb": bpb_runner,
        "sample": sample_runner,
        "core": core_runner,
    }
    for mode in requested_modes:
        if runners[mode] is None:
            if mode == "core":
                raise BaseEvaluationUnavailableError(
                    "CORE evaluation is unavailable until the Milestone 5 "
                    "runner is registered"
                )
            raise BaseEvaluationUnavailableError(
                f"evaluation mode {mode!r} has no registered runner"
            )

    validation: PeriodicValidationResult | None = None
    samples: BaseSamplesResult | None = None
    core_result: CoreEvaluationResult | None = None
    completed: list[BaseEvaluationMode] = []
    for mode in requested_modes:
        if mode == "bpb":
            assert bpb_runner is not None
            validation = bpb_runner()
            if not isinstance(validation, PeriodicValidationResult):
                raise TypeError("bpb_runner must return a PeriodicValidationResult")
        elif mode == "sample":
            assert sample_runner is not None
            samples = sample_runner()
            if not isinstance(samples, BaseSamplesResult):
                raise TypeError("sample_runner must return a BaseSamplesResult")
        else:
            assert core_runner is not None
            core_result = core_runner(context.max_per_task)
            if not isinstance(core_result, CoreEvaluationResult):
                raise TypeError("core_runner must return a CoreEvaluationResult")
        completed.append(mode)

    return CompletedBaseEvaluation(
        context=context,
        requested_modes=requested_modes,
        completed_modes=tuple(completed),
        validation=validation,
        samples=samples,
        core_result=core_result,
    )


def _validate_modes(modes: tuple[BaseEvaluationMode, ...]) -> None:
    if not isinstance(modes, tuple) or not modes:
        raise BaseEvaluationError("evaluation modes must be a non-empty tuple")
    seen: set[str] = set()
    for mode in modes:
        if mode not in _SUPPORTED_MODES:
            raise BaseEvaluationError(f"unknown evaluation mode {mode!r}")
        if mode in seen:
            raise BaseEvaluationError(f"evaluation mode {mode!r} is duplicated")
        seen.add(mode)


__all__ = [
    "BaseEvaluationContext",
    "BaseEvaluationCoreRunner",
    "BaseEvaluationError",
    "BaseEvaluationMode",
    "BaseEvaluationRunKind",
    "BaseEvaluationUnavailableError",
    "CompletedBaseEvaluation",
    "execute_base_evaluation_modes",
    "normalize_base_evaluation_modes",
]
