"""Deterministic example selection and model scoring for CORE tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from scratch_llm._validation import (
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.evaluation.core.bundle import CoreTask
from scratch_llm.evaluation.core.results import CoreEvaluationError
from scratch_llm.evaluation.core.examples import (
    CoreExample,
    CoreTaskExamples,
)
from scratch_llm.evaluation.core.prompting import CoreTokenBatch
from scratch_llm.utils import get_device


_TASK_SHUFFLE_SEED = 1337
_FEWSHOT_SEED_BASE = 1234


class CoreScoringError(CoreEvaluationError):
    """A bounded pool or model output cannot satisfy CORE scoring."""


@dataclass(frozen=True)
class CoreEvaluationCase:
    """One shuffled task example with its deterministic few-shot context."""

    shuffled_index: int
    example: CoreExample
    fewshot_examples: tuple[CoreExample, ...]


def prepare_core_evaluation_cases(
    task: CoreTask,
    examples: CoreTaskExamples,
    *,
    max_per_task: int | None,
) -> tuple[CoreEvaluationCase, ...]:
    """Reproduce nanochat seed-1337 shuffling and per-index few-shot seeds."""

    if not isinstance(task, CoreTask):
        raise TypeError("task must be a CoreTask")
    if not isinstance(examples, CoreTaskExamples):
        raise TypeError("examples must be CoreTaskExamples")
    if max_per_task is not None:
        max_per_task = require_positive_integer(
            max_per_task,
            name="max_per_task",
        )
    pool = list(examples)
    random.Random(_TASK_SHUFFLE_SEED).shuffle(pool)
    if max_per_task is not None:
        pool = pool[:max_per_task]
    required_pool_size = task.num_fewshot + 1
    if len(pool) < required_pool_size:
        raise CoreScoringError(
            f"CORE task {task.label!r} requires at least {required_pool_size} "
            "examples after max_per_task to select few-shot context"
        )
    cases = []
    for index, example in enumerate(pool):
        available_indices = [
            candidate for candidate in range(len(pool)) if candidate != index
        ]
        fewshot_indices = random.Random(_FEWSHOT_SEED_BASE + index).sample(
            available_indices,
            task.num_fewshot,
        )
        cases.append(
            CoreEvaluationCase(
                shuffled_index=index,
                example=example,
                fewshot_examples=tuple(
                    pool[candidate] for candidate in fewshot_indices
                ),
            )
        )
    return tuple(cases)


def score_core_token_batch(
    model: nn.Module,
    batch: CoreTokenBatch,
    *,
    pad_token_id: int,
    device: str | torch.device,
) -> bool:
    """Score one rendered example by the pinned categorical or exact rule."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(batch, CoreTokenBatch):
        raise TypeError("batch must be a CoreTokenBatch")
    pad_token_id = require_non_negative_integer(
        pad_token_id,
        name="pad_token_id",
    )
    resolved_device = get_device(device)
    input_ids = _stack_token_rows(
        batch.token_ids,
        pad_token_id=pad_token_id,
        device=resolved_device,
    )
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(input_ids)
        _validate_logits(logits, input_ids=input_ids)
        if batch.task_type == "language_modeling":
            return _language_modeling_is_correct(
                logits,
                input_ids=input_ids,
                start=batch.start_indices[0],
                end=batch.end_indices[0],
            )
        return _categorical_is_correct(logits, input_ids=input_ids, batch=batch)
    finally:
        for module, training in modes:
            module.training = training


def _categorical_is_correct(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    batch: CoreTokenBatch,
) -> bool:
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape(shifted_targets.shape)
    mean_losses = []
    for row, (start, end) in enumerate(
        zip(batch.start_indices, batch.end_indices, strict=True)
    ):
        continuation_losses = losses[row, start - 1 : end - 1]
        if continuation_losses.numel() == 0:
            # The pinned upstream common-prefix rule leaves an empty span when
            # one option is a complete token prefix of another. Its empty
            # mean is NaN, and Python's stable min keeps/ignores that option
            # according to its original position. Preserve that behavior so
            # the production bundle can be evaluated exactly as configured.
            mean_losses.append(math.nan)
            continue
        mean_loss = float(continuation_losses.mean().item())
        if not math.isfinite(mean_loss):
            raise CoreScoringError("model produced a non-finite CORE continuation loss")
        mean_losses.append(mean_loss)
    predicted_index = min(range(len(mean_losses)), key=mean_losses.__getitem__)
    assert batch.gold_index is not None
    return predicted_index == batch.gold_index


def _language_modeling_is_correct(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    start: int,
    end: int,
) -> bool:
    predictions = logits.argmax(dim=-1)
    predicted_tokens = predictions[0, start - 1 : end - 1]
    actual_tokens = input_ids[0, start:end]
    return bool(torch.equal(predicted_tokens, actual_tokens))


def _stack_token_rows(
    rows: tuple[tuple[int, ...], ...],
    *,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    width = max(len(row) for row in rows)
    batch = torch.full(
        (len(rows), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        batch[index, : len(row)] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
    return batch


def _validate_logits(logits: object, *, input_ids: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise CoreScoringError("model must return a Tensor of CORE logits")
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise CoreScoringError(
            "model CORE logits must have shape (batch, sequence, vocabulary)"
        )
    if logits.shape[-1] == 0:
        raise CoreScoringError("model CORE logits vocabulary must not be empty")
    if int(input_ids.max().item()) >= logits.shape[-1]:
        raise CoreScoringError("CORE token id exceeds the model logits vocabulary")


__all__ = [
    "CoreEvaluationCase",
    "CoreScoringError",
    "prepare_core_evaluation_cases",
    "score_core_token_batch",
]
