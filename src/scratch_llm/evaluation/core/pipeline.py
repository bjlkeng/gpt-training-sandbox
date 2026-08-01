"""Model, bundle, prompt, and scoring composition for CORE evaluation."""

from __future__ import annotations

from collections.abc import Callable
import time

import torch
from torch import nn

from scratch_llm._validation import require_non_empty_string, require_positive_integer
from scratch_llm.evaluation.core.bundle import CoreBundle, CoreTask
from scratch_llm.evaluation.core.results import (
    CoreEvaluationError,
    CoreEvaluationResult,
    CoreReferenceComparison,
    CoreTaskResult,
)
from scratch_llm.evaluation.core.examples import (
    CoreTaskExamples,
    load_core_task_examples,
)
from scratch_llm.evaluation.core.prompting import build_core_token_batch
from scratch_llm.evaluation.core.scoring import (
    prepare_core_evaluation_cases,
    score_core_token_batch,
)
from scratch_llm.tokenizer import Tokenizer
from scratch_llm.utils import get_device


def evaluate_core_bundle(
    model: nn.Module,
    tokenizer: Tokenizer,
    bundle: CoreBundle,
    *,
    checkpoint_identity: str,
    max_per_task: int | None,
    device: str | torch.device,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[CoreTaskResult], None] | None = None,
) -> CoreEvaluationResult:
    """Evaluate every configured task and return one immutable result."""

    _validate_runtime_inputs(
        model,
        tokenizer=tokenizer,
        bundle=bundle,
        checkpoint_identity=checkpoint_identity,
        clock=clock,
        progress=progress,
    )
    _validate_evaluation_scope(bundle, max_per_task=max_per_task)
    resolved_device = get_device(device)
    max_seq_len = _model_max_seq_len(model)
    task_results = []
    for task in bundle.tasks:
        examples = load_core_task_examples(bundle, task)
        result = _evaluate_task(
            model,
            tokenizer,
            task,
            examples,
            max_per_task=max_per_task,
            max_seq_len=max_seq_len,
            device=resolved_device,
            clock=clock,
        )
        task_results.append(result)
        if progress is not None:
            progress(result)
    return CoreEvaluationResult(
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer.get_identity(),
        bundle_identity=bundle.identity,
        config_identity=bundle.config_identity,
        metadata_identity=bundle.metadata_identity,
        run_kind="bounded" if max_per_task is not None else "full",
        max_per_task=max_per_task,
        tasks=tuple(task_results),
        references=tuple(
            CoreReferenceComparison(reference.model_id, reference.core_metric)
            for reference in bundle.reference_results
        ),
        elapsed_seconds=sum(result.elapsed_seconds for result in task_results),
    )


def _evaluate_task(
    model: nn.Module,
    tokenizer: Tokenizer,
    task: CoreTask,
    examples: CoreTaskExamples,
    *,
    max_per_task: int | None,
    max_seq_len: int,
    device: torch.device,
    clock: Callable[[], float],
) -> CoreTaskResult:
    cases = prepare_core_evaluation_cases(
        task,
        examples,
        max_per_task=max_per_task,
    )
    started_at = clock()
    correct = 0
    for case in cases:
        batch = build_core_token_batch(
            task,
            case.example,
            case.fewshot_examples,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        correct += int(
            score_core_token_batch(
                model,
                batch,
                pad_token_id=tokenizer.get_bos_token_id(),
                device=device,
            )
        )
    elapsed_seconds = clock() - started_at
    return CoreTaskResult(
        label=task.label,
        task_type=task.task_type,
        num_fewshot=task.num_fewshot,
        random_baseline_percent=task.random_baseline_percent,
        correct_examples=correct,
        evaluated_examples=len(cases),
        available_examples=len(examples),
        elapsed_seconds=elapsed_seconds,
        data_identity=examples.identity,
    )


def _validate_runtime_inputs(
    model: nn.Module,
    *,
    tokenizer: Tokenizer,
    bundle: CoreBundle,
    checkpoint_identity: str,
    clock: Callable[[], float],
    progress: Callable[[CoreTaskResult], None] | None,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement Tokenizer")
    if not isinstance(bundle, CoreBundle):
        raise TypeError("bundle must be a CoreBundle")
    require_non_empty_string(checkpoint_identity, name="checkpoint_identity")
    if not callable(clock):
        raise TypeError("clock must be callable")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")


def _validate_evaluation_scope(
    bundle: CoreBundle,
    *,
    max_per_task: int | None,
) -> None:
    if max_per_task is None:
        return
    try:
        limit = require_positive_integer(max_per_task, name="max_per_task")
    except (TypeError, ValueError) as error:
        raise CoreEvaluationError(str(error)) from error
    required = max(task.num_fewshot + 1 for task in bundle.tasks)
    if limit < required:
        raise CoreEvaluationError(
            f"max_per_task must be at least {required} for the bundle's few-shot tasks"
        )


def _model_max_seq_len(model: nn.Module) -> int:
    value = getattr(model, "max_seq_len", None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("model.max_seq_len must be a positive integer")
    return value


__all__ = ["evaluate_core_bundle"]
