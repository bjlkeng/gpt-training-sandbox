"""Single-device composition of existing chat evaluators and reporting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from scratch_llm.config import ProjectConfig
from scratch_llm.evaluation.chat.arc import load_arc_task
from scratch_llm.evaluation.chat.categorical import (
    CategoricalTask,
    evaluate_categorical_task,
)
from scratch_llm.evaluation.chat.diagnostics import compute_fixed_sft_diagnostics
from scratch_llm.evaluation.chat.execution import CodeExecutor
from scratch_llm.evaluation.chat.generative import (
    GenerativeTask,
    evaluate_generative_task,
)
from scratch_llm.evaluation.chat.gsm8k import (
    load_gsm8k_task,
    score_gsm8k_completion,
)
from scratch_llm.evaluation.chat.humaneval import (
    HUMANEVAL_EXECUTION_WARNING,
    evaluate_humaneval_task,
    load_humaneval_task,
)
from scratch_llm.evaluation.chat.mmlu import load_mmlu_task
from scratch_llm.evaluation.chat.reporting import (
    ChatEvaluationError,
    ChatEvaluationSettings,
    ChatTaskResult,
    CompletedChatEvaluation,
    write_chat_evaluation_report,
)
from scratch_llm.evaluation.sft_sampling import generate_fixed_sft_samples
from scratch_llm.identity import file_identity, project_config_identity
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.training.checkpoint import (
    load_checkpoint_metadata,
    load_model_checkpoint,
)
from scratch_llm.training.rng_state import preserve_global_rng_state
from scratch_llm.utils import get_device


ChatTask = CategoricalTask | GenerativeTask
ChatTaskProgress = Callable[[ChatTaskResult], None]


@dataclass(frozen=True, slots=True)
class ChatEvaluationRun:
    """One completed immutable result and its canonical local report path."""

    completed: CompletedChatEvaluation
    report_path: Path


def evaluate_checkpoint_chat_model(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    cache_root: str | Path,
    settings: ChatEvaluationSettings,
    run_dir: str | Path,
    executor: CodeExecutor | None = None,
    progress: ChatTaskProgress | None = None,
) -> ChatEvaluationRun:
    """Evaluate every requested task once, then atomically publish on success."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    config.validate()
    if not isinstance(settings, ChatEvaluationSettings):
        raise TypeError("settings must be a ChatEvaluationSettings")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    resolved_checkpoint = Path(checkpoint_path)
    resolved_cache_root = Path(cache_root)
    resolved_run_dir = Path(run_dir)
    if not resolved_checkpoint.is_file():
        raise ChatEvaluationError(
            f"checkpoint_path must be a regular file: {resolved_checkpoint}"
        )
    if not resolved_cache_root.is_dir():
        raise ChatEvaluationError(
            f"cache_root must be an existing directory: {resolved_cache_root}"
        )
    if not resolved_run_dir.is_dir():
        raise ChatEvaluationError(
            f"run_dir must be an existing directory: {resolved_run_dir}"
        )

    _validate_code_execution(settings, executor)
    device = get_device(config.run.device)
    metadata = load_checkpoint_metadata(resolved_checkpoint)
    _validate_sft_checkpoint(
        requested=config,
        checkpoint_config=metadata.config,
        training_stage=metadata.training_stage,
    )
    checkpoint_identity = file_identity(resolved_checkpoint)

    # Validate every local source before allocating and restoring the model.
    tasks = tuple(
        _load_task(task_name, resolved_cache_root) for task_name in settings.task_names
    )
    loaded = load_model_checkpoint(resolved_checkpoint, device=device)
    _validate_sft_checkpoint(
        requested=config,
        checkpoint_config=loaded.config,
        training_stage=loaded.training_stage,
    )
    if loaded.step != metadata.step:
        raise ChatEvaluationError(
            "loaded checkpoint step does not match its preflight metadata"
        )
    model = loaded.model
    tokenizer = loaded.tokenizer
    if not isinstance(model, nn.Module):
        raise TypeError("checkpoint model must be a torch.nn.Module")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("checkpoint tokenizer must implement Tokenizer")

    modes = tuple((module, module.training) for module in model.modules())
    results: list[ChatTaskResult] = []
    with preserve_global_rng_state(device):
        try:
            model.eval()
            for task in tasks:
                result = _evaluate_task(
                    model,
                    tokenizer,
                    task,
                    checkpoint_identity=checkpoint_identity,
                    settings=settings,
                    device=device,
                    executor=executor,
                )
                results.append(result)
                if progress is not None:
                    progress(result)
            fixed_samples = generate_fixed_sft_samples(
                model,
                tokenizer,
                checkpoint_identity=checkpoint_identity,
                config=settings.fixed_sampling,
                device=device,
            )
            diagnostics = compute_fixed_sft_diagnostics(fixed_samples)
        finally:
            for module, training in modes:
                module.training = training

    completed = CompletedChatEvaluation(
        config_identity=project_config_identity(config),
        checkpoint_identity=checkpoint_identity,
        checkpoint_step=loaded.step,
        tokenizer_identity=tokenizer.get_identity(),
        settings=settings,
        task_results=tuple(results),
        diagnostics=diagnostics,
    )
    report_path = write_chat_evaluation_report(
        completed,
        run_dir=resolved_run_dir,
    )
    return ChatEvaluationRun(completed=completed, report_path=report_path)


def _validate_code_execution(
    settings: ChatEvaluationSettings,
    executor: CodeExecutor | None,
) -> None:
    if "HumanEval" not in settings.task_names:
        if executor is not None or settings.executor_identity is not None:
            raise ChatEvaluationError(
                "a code executor is only valid when HumanEval is selected"
            )
        return
    if settings.allow_generated_code_execution is not True:
        raise ChatEvaluationError(
            "HumanEval requires the explicit generated-code execution opt-in; "
            f"{HUMANEVAL_EXECUTION_WARNING}"
        )
    if executor is None:
        raise ChatEvaluationError("HumanEval requires a configured code executor")
    identity = getattr(executor, "identity", None)
    if identity != settings.executor_identity:
        raise ChatEvaluationError(
            "code executor identity does not match evaluation settings"
        )
    if not callable(getattr(executor, "execute", None)):
        raise ChatEvaluationError("code executor must provide callable execute")


def _validate_sft_checkpoint(
    *,
    requested: ProjectConfig,
    checkpoint_config: ProjectConfig,
    training_stage: object,
) -> None:
    if training_stage != "sft":
        raise ChatEvaluationError(
            f"chat evaluation requires an SFT checkpoint, got {training_stage!r}"
        )
    if not isinstance(checkpoint_config, ProjectConfig):
        raise TypeError("checkpoint config must be a ProjectConfig")
    differences: list[str] = []
    if checkpoint_config.model != requested.model:
        differences.append("model")
    if checkpoint_config.tokenizer != requested.tokenizer:
        differences.append("tokenizer")
    if differences:
        raise ChatEvaluationError(
            "checkpoint does not match the resolved evaluation config for: "
            + ", ".join(differences)
        )


def _load_task(task_name: str, cache_root: Path) -> ChatTask:
    if task_name in {"ARC-Easy", "ARC-Challenge"}:
        return load_arc_task(cache_root, task_name)
    if task_name == "MMLU":
        return load_mmlu_task(cache_root)
    if task_name == "GSM8K":
        return load_gsm8k_task(cache_root)
    if task_name == "HumanEval":
        return load_humaneval_task(cache_root)
    raise ChatEvaluationError(f"unsupported chat task {task_name!r}")


def _evaluate_task(
    model: nn.Module,
    tokenizer: Tokenizer,
    task: ChatTask,
    *,
    checkpoint_identity: str,
    settings: ChatEvaluationSettings,
    device: torch.device,
    executor: CodeExecutor | None,
) -> ChatTaskResult:
    if isinstance(task, CategoricalTask):
        return evaluate_categorical_task(
            model,
            tokenizer,
            task,
            checkpoint_identity=checkpoint_identity,
            batch_size=settings.batch_size,
            max_problems=settings.max_problems,
            device=device,
        )
    if task.name == "GSM8K":
        return evaluate_generative_task(
            model,
            tokenizer,
            task,
            score_gsm8k_completion,
            checkpoint_identity=checkpoint_identity,
            config=settings.generation,
            max_problems=settings.max_problems,
            device=device,
        )
    if task.name == "HumanEval" and executor is not None:
        return evaluate_humaneval_task(
            model,
            tokenizer,
            task,
            executor,
            allow_generated_code_execution=(settings.allow_generated_code_execution),
            checkpoint_identity=checkpoint_identity,
            config=settings.generation,
            max_problems=settings.max_problems,
            device=device,
        )
    raise ChatEvaluationError(f"unsupported loaded chat task {task.name!r}")


__all__ = [
    "ChatEvaluationRun",
    "ChatTaskProgress",
    "evaluate_checkpoint_chat_model",
]
