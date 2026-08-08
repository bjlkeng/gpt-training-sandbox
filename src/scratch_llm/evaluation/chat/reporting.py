"""Immutable completed chat evaluation and atomic JSON reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

from scratch_llm._validation import (
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.evaluation.chat.categorical import CategoricalTaskResult
from scratch_llm.evaluation.chat.chatcore import (
    CHATCORE_PROTOCOL_ID,
    CHATCORE_PROTOCOL_VERSION,
    CHATCORE_TASK_ORDER,
    ChatCoreResult,
    ChatCoreTaskName,
    ChatCoreTaskResult,
)
from scratch_llm.evaluation.chat.diagnostics import FixedSFTDiagnostics
from scratch_llm.evaluation.chat.generative import (
    GenerativeEvaluationConfig,
    GenerativeTaskResult,
)
from scratch_llm.evaluation.sft_sampling import (
    FIXED_SFT_PROMPTS,
    FIXED_SFT_PROMPT_SET_IDENTITY,
    FixedSFTSamplingConfig,
)
from scratch_llm.identity import canonical_json_identity
from scratch_llm.utils import save_json


CHAT_EVALUATION_REPORT_FORMAT: Final = "scratch_llm_chat_evaluation"
CHAT_EVALUATION_REPORT_FORMAT_VERSION: Final = 1
CHAT_EVALUATION_REPORT_RELATIVE_PATH: Final = Path("metrics/chat_eval.json")

ChatTaskResult: TypeAlias = CategoricalTaskResult | GenerativeTaskResult
ChatEvaluationScopeKind: TypeAlias = Literal[
    "full",
    "bounded",
    "partial",
    "bounded_partial",
]

_TASK_INDEX = {name: index for index, name in enumerate(CHATCORE_TASK_ORDER)}
_CATEGORICAL_TASKS = frozenset(CHATCORE_TASK_ORDER[:3])


class ChatEvaluationError(ValueError):
    """A requested or completed chat evaluation is internally inconsistent."""


class ChatEvaluationReportConflictError(RuntimeError):
    """A run already owns a different canonical completed chat report."""


def normalize_chat_task_names(
    value: str | tuple[str, ...],
) -> tuple[ChatCoreTaskName, ...]:
    """Return a non-empty unique task filter in canonical ChatCORE order."""

    raw_names = (
        tuple(part.strip() for part in value.split(","))
        if isinstance(value, str)
        else value
    )
    if (
        not isinstance(raw_names, tuple)
        or not raw_names
        or any(not isinstance(name, str) or not name for name in raw_names)
    ):
        raise ChatEvaluationError("tasks must contain at least one task name")
    if len(set(raw_names)) != len(raw_names):
        raise ChatEvaluationError("tasks must not contain duplicates")
    unknown = tuple(name for name in raw_names if name not in _TASK_INDEX)
    if unknown:
        supported = ", ".join(CHATCORE_TASK_ORDER)
        raise ChatEvaluationError(
            f"unsupported chat task {unknown[0]!r}; supported tasks are: {supported}"
        )
    selected = set(raw_names)
    return tuple(name for name in CHATCORE_TASK_ORDER if name in selected)


@dataclass(frozen=True, slots=True)
class ChatEvaluationSettings:
    """Frozen execution settings shared by orchestration and its report."""

    task_names: tuple[ChatCoreTaskName, ...]
    batch_size: int
    max_problems: int | None
    generation: GenerativeEvaluationConfig
    fixed_sampling: FixedSFTSamplingConfig
    allow_generated_code_execution: bool
    executor_identity: str | None

    def __post_init__(self) -> None:
        normalized = normalize_chat_task_names(cast(tuple[str, ...], self.task_names))
        if self.task_names != normalized:
            raise ChatEvaluationError("task_names must use canonical task order")
        try:
            require_positive_integer(self.batch_size, name="batch_size")
            if self.max_problems is not None:
                require_positive_integer(self.max_problems, name="max_problems")
        except (TypeError, ValueError) as error:
            raise ChatEvaluationError(str(error)) from error
        if not isinstance(self.generation, GenerativeEvaluationConfig):
            raise TypeError("generation must be a GenerativeEvaluationConfig")
        if not isinstance(self.fixed_sampling, FixedSFTSamplingConfig):
            raise TypeError("fixed_sampling must be a FixedSFTSamplingConfig")
        if not isinstance(self.allow_generated_code_execution, bool):
            raise TypeError("allow_generated_code_execution must be a boolean")
        if self.executor_identity is not None:
            try:
                require_non_empty_string(
                    self.executor_identity,
                    name="executor_identity",
                )
            except (TypeError, ValueError) as error:
                raise ChatEvaluationError(str(error)) from error
        selects_humaneval = "HumanEval" in self.task_names
        if not selects_humaneval and (
            self.allow_generated_code_execution or self.executor_identity is not None
        ):
            raise ChatEvaluationError(
                "generated-code execution settings require HumanEval"
            )
        if selects_humaneval and (
            self.allow_generated_code_execution != (self.executor_identity is not None)
        ):
            raise ChatEvaluationError(
                "HumanEval consent and executor identity must be configured together"
            )

    @property
    def full(self) -> bool:
        return self.task_names == CHATCORE_TASK_ORDER and self.max_problems is None

    @property
    def kind(self) -> ChatEvaluationScopeKind:
        selected_all = self.task_names == CHATCORE_TASK_ORDER
        bounded = self.max_problems is not None
        if selected_all and not bounded:
            return "full"
        if selected_all:
            return "bounded"
        if bounded:
            return "bounded_partial"
        return "partial"

    @property
    def missing_tasks(self) -> tuple[ChatCoreTaskName, ...]:
        selected = set(self.task_names)
        return tuple(name for name in CHATCORE_TASK_ORDER if name not in selected)

    def scope_dict(self) -> dict[str, object]:
        """Return the exact publication scope, distinct from task completion."""

        return {
            "bounded": self.max_problems is not None,
            "full": self.full,
            "kind": self.kind,
            "max_problems": self.max_problems,
            "missing_tasks": list(self.missing_tasks),
            "selected_tasks": list(self.task_names),
            "task_count": len(self.task_names),
        }

    def to_dict(self) -> dict[str, object]:
        """Return deterministic settings without configuration aliases."""

        return {
            "batch_size": self.batch_size,
            "fixed_prompts": self.fixed_sampling.to_dict(),
            "generated_code_execution": {
                "allowed": self.allow_generated_code_execution,
                "executor_identity": self.executor_identity,
            },
            "generative": self.generation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CompletedChatEvaluation:
    """One all-requested-tasks completion marker ready for publication."""

    config_identity: str
    checkpoint_identity: str
    checkpoint_step: int
    tokenizer_identity: str
    settings: ChatEvaluationSettings
    task_results: tuple[ChatTaskResult, ...]
    diagnostics: FixedSFTDiagnostics
    chatcore: ChatCoreResult = field(init=False)

    def __post_init__(self) -> None:
        try:
            for name in (
                "config_identity",
                "checkpoint_identity",
                "tokenizer_identity",
            ):
                require_non_empty_string(getattr(self, name), name=name)
            require_non_negative_integer(self.checkpoint_step, name="checkpoint_step")
        except (TypeError, ValueError) as error:
            raise ChatEvaluationError(str(error)) from error
        if not isinstance(self.settings, ChatEvaluationSettings):
            raise TypeError("settings must be a ChatEvaluationSettings")
        if not isinstance(self.task_results, tuple) or not self.task_results:
            raise ChatEvaluationError("task_results must be a non-empty tuple")
        if any(
            not isinstance(result, (CategoricalTaskResult, GenerativeTaskResult))
            for result in self.task_results
        ):
            raise TypeError(
                "task_results must contain categorical or generative task results"
            )
        names = tuple(result.task_name for result in self.task_results)
        if names != self.settings.task_names:
            raise ChatEvaluationError(
                "task_results must match the requested task order exactly"
            )
        if any(
            result.checkpoint_identity != self.checkpoint_identity
            for result in self.task_results
        ):
            raise ChatEvaluationError("task checkpoint identity does not match report")
        if any(
            result.tokenizer_identity != self.tokenizer_identity
            for result in self.task_results
        ):
            raise ChatEvaluationError("task tokenizer identity does not match report")
        expected_kind = "bounded" if self.settings.max_problems is not None else "full"
        if any(
            result.run_kind != expected_kind
            or result.max_problems != self.settings.max_problems
            for result in self.task_results
        ):
            raise ChatEvaluationError("task scope does not match requested scope")
        for result in self.task_results:
            _evaluation_type(result)
            if (
                isinstance(result, GenerativeTaskResult)
                and result.config != self.settings.generation
            ):
                raise ChatEvaluationError(
                    f"{result.task_name} generation settings do not match report"
                )
        if not isinstance(self.diagnostics, FixedSFTDiagnostics):
            raise TypeError("diagnostics must be a FixedSFTDiagnostics")
        if self.diagnostics.checkpoint_identity != self.checkpoint_identity:
            raise ChatEvaluationError(
                "diagnostic checkpoint identity does not match report"
            )
        if self.diagnostics.tokenizer_identity != self.tokenizer_identity:
            raise ChatEvaluationError(
                "diagnostic tokenizer identity does not match report"
            )
        if self.diagnostics.renderer_identity != self.task_results[0].renderer_identity:
            raise ChatEvaluationError(
                "diagnostic renderer identity does not match task results"
            )
        if (
            self.diagnostics.prompt_set_identity != FIXED_SFT_PROMPT_SET_IDENTITY
            or self.diagnostics.sample_count != len(FIXED_SFT_PROMPTS)
        ):
            raise ChatEvaluationError(
                "diagnostics must cover the canonical fixed SFT prompt set"
            )
        if "HumanEval" in names:
            if self.settings.allow_generated_code_execution is not True:
                raise ChatEvaluationError(
                    "completed HumanEval requires explicit consent to generated-code "
                    "execution"
                )
            human_result = self.task_results[names.index("HumanEval")]
            if not isinstance(human_result, GenerativeTaskResult):
                raise ChatEvaluationError("HumanEval must use generative evaluation")
            if (
                self.settings.executor_identity is None
                or human_result.scoring_identity != self.settings.executor_identity
            ):
                raise ChatEvaluationError(
                    "HumanEval result must match the configured executor identity"
                )
        object.__setattr__(
            self,
            "chatcore",
            ChatCoreResult(
                tuple(_chatcore_task(result) for result in self.task_results)
            ),
        )

    @property
    def chatcore_metric(self) -> float | None:
        """Return the five-task aggregate only for the canonical full scope."""

        return self.chatcore.chatcore_metric if self.settings.full else None

    @property
    def chatcore_cat(self) -> float | None:
        """Return the categorical aggregate only for the canonical full scope."""

        return self.chatcore.chatcore_cat if self.settings.full else None

    def _body(self) -> dict[str, object]:
        aggregate = self.chatcore
        return {
            "chatcore": {
                "chatcore_cat": self.chatcore_cat,
                "chatcore_metric": self.chatcore_metric,
                "protocol_id": CHATCORE_PROTOCOL_ID,
                "protocol_version": CHATCORE_PROTOCOL_VERSION,
            },
            "context": {
                "checkpoint": {
                    "identity": self.checkpoint_identity,
                    "step": self.checkpoint_step,
                    "training_stage": "sft",
                },
                "config_identity": self.config_identity,
                "renderer_identity": self.diagnostics.renderer_identity,
                "tokenizer_identity": self.tokenizer_identity,
            },
            "format": CHAT_EVALUATION_REPORT_FORMAT,
            "format_version": CHAT_EVALUATION_REPORT_FORMAT_VERSION,
            "response_diagnostics": self.diagnostics.to_dict(),
            "scope": self.settings.scope_dict(),
            "settings": self.settings.to_dict(),
            "status": "completed",
            "tasks": [
                {
                    "details": _deterministic_task_details(result),
                    "evaluation_type": _evaluation_type(result),
                    "score": score.to_dict(),
                }
                for result, score in zip(
                    self.task_results,
                    aggregate.tasks,
                    strict=True,
                )
            ],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the canonical content-free completed JSON payload."""

        payload = self._body()
        payload["result_identity"] = canonical_json_identity(payload)
        return payload


def write_chat_evaluation_report(
    completed: CompletedChatEvaluation,
    *,
    run_dir: str | Path,
) -> Path:
    """Atomically install one compatible completed report under a prepared run."""

    if not isinstance(completed, CompletedChatEvaluation):
        raise TypeError("completed must be a CompletedChatEvaluation")
    resolved_run_dir = Path(run_dir)
    if not resolved_run_dir.is_dir():
        raise ChatEvaluationError(
            f"run_dir must be an existing directory: {resolved_run_dir}"
        )
    path = resolved_run_dir / CHAT_EVALUATION_REPORT_RELATIVE_PATH
    payload = completed.to_dict()
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ChatEvaluationReportConflictError(
                f"existing chat evaluation report is not valid JSON: {path}"
            ) from error
        if canonical_json_identity(existing) != canonical_json_identity(payload):
            raise ChatEvaluationReportConflictError(
                f"{path} already contains a different completed chat evaluation"
            )
        return path
    return save_json(payload, path)


def _chatcore_task(result: ChatTaskResult) -> ChatCoreTaskResult:
    try:
        task_name = cast(ChatCoreTaskName, result.task_name)
        return ChatCoreTaskResult(
            task_name=task_name,
            status="completed",
            passed_count=result.passed_count,
            total_count=result.evaluated_count,
            accuracy=result.accuracy,
            checkpoint_identity=result.checkpoint_identity,
            tokenizer_identity=result.tokenizer_identity,
            renderer_identity=result.renderer_identity,
            run_kind=result.run_kind,
            max_problems=result.max_problems,
        )
    except (TypeError, ValueError) as error:
        raise ChatEvaluationError(
            f"could not normalize {result.task_name!r} for ChatCORE: {error}"
        ) from error


def _evaluation_type(result: ChatTaskResult) -> str:
    if result.task_name == "HumanEval":
        return "code_execution"
    if result.task_name in _CATEGORICAL_TASKS:
        if not isinstance(result, CategoricalTaskResult):
            raise ChatEvaluationError(
                f"{result.task_name} must use categorical evaluation"
            )
        return "categorical"
    if not isinstance(result, GenerativeTaskResult):
        raise ChatEvaluationError(f"{result.task_name} must use generative evaluation")
    return "generative"


def _deterministic_task_details(result: ChatTaskResult) -> dict[str, object]:
    payload = result.to_dict()
    # Wall-clock duration is useful live telemetry, not reproducible evaluation data.
    payload.pop("elapsed_seconds", None)
    return payload


__all__ = [
    "CHAT_EVALUATION_REPORT_FORMAT",
    "CHAT_EVALUATION_REPORT_FORMAT_VERSION",
    "CHAT_EVALUATION_REPORT_RELATIVE_PATH",
    "ChatEvaluationError",
    "ChatEvaluationReportConflictError",
    "ChatEvaluationSettings",
    "ChatTaskResult",
    "CompletedChatEvaluation",
    "normalize_chat_task_names",
    "write_chat_evaluation_report",
]
