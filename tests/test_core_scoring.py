"""Tests for CORE example selection and model-backed scoring."""

from __future__ import annotations

import torch
from torch import nn

import pytest

from scratch_llm.evaluation.core.bundle import CoreTask
from scratch_llm.evaluation.core.examples import (
    CoreTaskExamples,
    LanguageModelingExample,
    MultipleChoiceExample,
)
from scratch_llm.evaluation.core.prompting import build_core_token_batch
from scratch_llm.evaluation.core.scoring import (
    CoreScoringError,
    prepare_core_evaluation_cases,
    score_core_token_batch,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer


class _PreferredTokenModel(nn.Module):
    def __init__(self, token_id: int) -> None:
        super().__init__()
        self.token_id = token_id
        self.max_seq_len = 64

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*token_ids.shape, 265), device=token_ids.device)
        logits[..., self.token_id] = 10.0
        return logits


class _OracleNextTokenModel(nn.Module):
    max_seq_len = 64

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*token_ids.shape, 265), device=token_ids.device)
        next_tokens = token_ids[:, 1:]
        logits[:, :-1].scatter_(2, next_tokens.unsqueeze(-1), 10.0)
        return logits


def _task(*, task_type: str, num_fewshot: int = 0) -> CoreTask:
    return CoreTask(
        label="fixture",
        task_type=task_type,  # type: ignore[arg-type]
        dataset_member="eval_bundle/eval_data/fixture.jsonl",
        num_fewshot=num_fewshot,
        continuation_delimiter=" ",
        random_baseline_percent=25.0,
    )


def test_score_core_token_batch_uses_mean_continuation_loss_and_restores_mode() -> None:
    tokenizer = ByteTokenizer()
    batch = build_core_token_batch(
        _task(task_type="multiple_choice"),
        MultipleChoiceExample("Pick", ("A", "B"), 1),
        (),
        tokenizer=tokenizer,
        max_seq_len=64,
    )
    model = _PreferredTokenModel(ord("B"))
    model.train()

    assert score_core_token_batch(
        model,
        batch,
        pad_token_id=tokenizer.get_bos_token_id(),
        device="cpu",
    )
    assert model.training is True


def test_score_core_token_batch_matches_upstream_prefix_option_fallback() -> None:
    tokenizer = ByteTokenizer()
    batch = build_core_token_batch(
        _task(task_type="multiple_choice"),
        MultipleChoiceExample("Pick", ("answer plus", "answer"), 0),
        (),
        tokenizer=tokenizer,
        max_seq_len=64,
    )

    assert score_core_token_batch(
        _PreferredTokenModel(ord("x")),
        batch,
        pad_token_id=tokenizer.get_bos_token_id(),
        device="cpu",
    )


def test_score_core_token_batch_requires_every_language_model_token() -> None:
    tokenizer = ByteTokenizer()
    batch = build_core_token_batch(
        _task(task_type="language_modeling"),
        LanguageModelingExample("prefix", "answer"),
        (),
        tokenizer=tokenizer,
        max_seq_len=64,
    )

    assert score_core_token_batch(
        _OracleNextTokenModel(),
        batch,
        pad_token_id=tokenizer.get_bos_token_id(),
        device="cpu",
    )
    assert not score_core_token_batch(
        _PreferredTokenModel(ord("a")),
        batch,
        pad_token_id=tokenizer.get_bos_token_id(),
        device="cpu",
    )


def test_prepare_core_evaluation_cases_is_seeded_and_uses_the_bounded_pool() -> None:
    task = _task(task_type="multiple_choice", num_fewshot=2)
    examples = CoreTaskExamples(
        examples=tuple(
            MultipleChoiceExample(f"Question {index}", ("A", "B"), index % 2)
            for index in range(12)
        ),
        identity="sha256:" + "1" * 64,
    )

    first = prepare_core_evaluation_cases(task, examples, max_per_task=5)
    second = prepare_core_evaluation_cases(task, examples, max_per_task=5)

    assert first == second
    assert len(first) == 5
    assert all(len(case.fewshot_examples) == 2 for case in first)
    bounded_examples = {case.example for case in first}
    assert all(set(case.fewshot_examples) <= bounded_examples for case in first)
    assert all(case.example not in case.fewshot_examples for case in first)


def test_prepare_core_evaluation_cases_rejects_too_small_a_fewshot_pool() -> None:
    task = _task(task_type="multiple_choice", num_fewshot=2)
    examples = CoreTaskExamples(
        examples=tuple(
            MultipleChoiceExample(f"Question {index}", ("A", "B"), 0)
            for index in range(4)
        ),
        identity="sha256:" + "1" * 64,
    )

    with pytest.raises(CoreScoringError, match="at least 3"):
        prepare_core_evaluation_cases(task, examples, max_per_task=2)
