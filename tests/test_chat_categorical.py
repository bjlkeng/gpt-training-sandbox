"""Tests for the shared answer-letter-logit chat evaluator."""

from __future__ import annotations

import torch
from torch import nn
import pytest

from scratch_llm.chat.conversation import Conversation, UserMessage
from scratch_llm.evaluation.chat.categorical import (
    CHAT_CATEGORICAL_CONTEXT_POLICY_ID,
    CategoricalEvaluationError,
    CategoricalExample,
    CategoricalTask,
    evaluate_categorical_task,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer


class _PositionLogitModel(nn.Module):
    max_seq_len = 512

    def __init__(
        self,
        *,
        answer_positions: tuple[int, ...],
        preferred_token_ids: tuple[int, ...],
        distractor_token_ids: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.answer_positions = answer_positions
        self.preferred_token_ids = preferred_token_ids
        self.distractor_token_ids = distractor_token_ids
        self.probe = nn.Dropout()
        self.inputs: list[torch.Tensor] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.inputs.append(token_ids.detach().cpu().clone())
        logits = torch.zeros((*token_ids.shape, 265), device=token_ids.device)
        for row, answer_position in enumerate(self.answer_positions):
            logits[row, answer_position, self.preferred_token_ids[row]] = 4.0
            if answer_position != token_ids.shape[1] - 1:
                logits[row, -1, self.distractor_token_ids[row]] = 8.0
        return logits


class _ZeroLogitModel(nn.Module):
    max_seq_len = 512

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*token_ids.shape, 265), device=token_ids.device)


class _BadLogitModel(nn.Module):
    max_seq_len = 512

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(token_ids.shape, device=token_ids.device)


class _NonFiniteLogitModel(nn.Module):
    max_seq_len = 512

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*token_ids.shape, 265), device=token_ids.device)
        logits[..., ord("A")] = torch.nan
        return logits


def _example(
    prompt: str,
    *,
    labels: tuple[str, ...],
    answer: str,
    source_row: int,
) -> CategoricalExample:
    return CategoricalExample(
        conversation=Conversation(messages=(UserMessage(prompt),)),
        labels=labels,
        answer=answer,
        source_row=source_row,
        identity=f"example-{source_row}",
    )


def _task(*examples: CategoricalExample) -> CategoricalTask:
    return CategoricalTask(
        name="fixture",
        examples=examples,
        source_identity="source",
        dataset_identity="dataset",
        order_identity="order",
    )


def test_categorical_eval_uses_each_true_prompt_end_with_bos_right_padding() -> None:
    tokenizer = ByteTokenizer()
    examples = (
        _example("short", labels=("A", "B"), answer="B", source_row=3),
        _example(
            "a much longer prompt",
            labels=("A", "B", "C"),
            answer="C",
            source_row=7,
        ),
    )
    from scratch_llm.chat.rendering import render_completion_prompt

    prompt_lengths = tuple(
        len(render_completion_prompt(example.conversation, tokenizer).token_ids)
        for example in examples
    )
    model = _PositionLogitModel(
        answer_positions=tuple(length - 1 for length in prompt_lengths),
        preferred_token_ids=(ord("B"), ord("C")),
        distractor_token_ids=(ord("A"), ord("A")),
    )
    model.train()
    model.probe.eval()

    result = evaluate_categorical_task(
        model,
        tokenizer,
        _task(*examples),
        checkpoint_identity="checkpoint",
        batch_size=2,
        max_problems=None,
        device="cpu",
    )

    assert result.passed_count == 2
    assert result.accuracy == 1.0
    assert result.run_kind == "full"
    assert result.to_dict()["identities"] == {
        "checkpoint": "checkpoint",
        "dataset": "dataset",
        "order": "order",
        "renderer": "scratch_llm_chat_renderer_v1",
        "source": "source",
        "tokenizer": tokenizer.get_identity(),
    }
    assert model.training is True
    assert model.probe.training is False
    batch = model.inputs[0]
    assert tuple(batch.shape) == (2, max(prompt_lengths))
    assert torch.all(batch[0, prompt_lengths[0] :] == tokenizer.get_bos_token_id())


def test_categorical_eval_ties_choose_the_first_declared_label_deterministically() -> (
    None
):
    example = _example(
        "choose",
        labels=("B", "A", "C"),
        answer="B",
        source_row=0,
    )

    first = evaluate_categorical_task(
        _ZeroLogitModel(),
        ByteTokenizer(),
        _task(example),
        checkpoint_identity="checkpoint",
        batch_size=1,
        max_problems=1,
        device="cpu",
        clock=iter((1.0, 2.5)).__next__,
    )
    second = evaluate_categorical_task(
        _ZeroLogitModel(),
        ByteTokenizer(),
        _task(example),
        checkpoint_identity="checkpoint",
        batch_size=1,
        max_problems=1,
        device="cpu",
        clock=iter((1.0, 2.5)).__next__,
    )

    assert first == second
    assert first.passed_count == 1
    assert first.run_kind == "bounded"
    assert first.max_problems == 1


def test_categorical_eval_excludes_overlength_prompts_without_cropping() -> None:
    tokenizer = ByteTokenizer()
    examples = (
        _example("short", labels=("A", "B"), answer="A", source_row=3),
        _example("x" * 40, labels=("A", "B"), answer="B", source_row=7),
        _example("also short", labels=("A", "B"), answer="B", source_row=11),
    )
    from scratch_llm.chat.rendering import render_completion_prompt

    prompts = tuple(
        render_completion_prompt(example.conversation, tokenizer).token_ids
        for example in examples
    )
    max_seq_len = max(len(prompts[0]), len(prompts[2]))
    assert len(prompts[1]) > max_seq_len
    model = _PositionLogitModel(
        answer_positions=(len(prompts[0]) - 1, len(prompts[2]) - 1),
        preferred_token_ids=(ord("A"), ord("B")),
        distractor_token_ids=(ord("B"), ord("A")),
    )
    model.max_seq_len = max_seq_len

    result = evaluate_categorical_task(
        model,
        tokenizer,
        _task(*examples),
        checkpoint_identity="checkpoint",
        batch_size=2,
        max_problems=None,
        device="cpu",
        clock=iter((1.0, 2.0)).__next__,
    )

    assert result.passed_count == 2
    assert result.evaluated_count == 2
    assert result.available_count == 3
    assert result.excluded_overlength_count == 1
    assert result.run_kind == "full"
    assert result.to_dict()["counts"] == {
        "available": 3,
        "evaluated": 2,
        "excluded_overlength": 1,
        "passed": 2,
        "selected": 3,
    }
    assert result.to_dict()["prompt_preflight"] == {
        "excluded_examples": [
            {
                "example_identity": "example-7",
                "prompt_token_count": len(prompts[1]),
                "source_row": 7,
            }
        ],
        "model_max_seq_len": max_seq_len,
        "policy_id": CHAT_CATEGORICAL_CONTEXT_POLICY_ID,
        "policy_version": 1,
    }
    assert len(model.inputs) == 1
    assert tuple(model.inputs[0][0, : len(prompts[0])].tolist()) == prompts[0]
    assert tuple(model.inputs[0][1, : len(prompts[2])].tolist()) == prompts[2]


def test_categorical_eval_rejects_when_every_selected_prompt_is_overlength() -> None:
    model = _PositionLogitModel(
        answer_positions=(),
        preferred_token_ids=(),
        distractor_token_ids=(),
    )
    model.max_seq_len = 1

    with pytest.raises(CategoricalEvaluationError, match="no selected prompts fit"):
        evaluate_categorical_task(
            model,
            ByteTokenizer(),
            _task(
                _example(
                    "too long",
                    labels=("A", "B"),
                    answer="A",
                    source_row=0,
                )
            ),
            checkpoint_identity="checkpoint",
            batch_size=1,
            max_problems=None,
            device="cpu",
        )

    assert model.inputs == []


def test_categorical_eval_rejects_multi_token_labels_before_model_execution() -> None:
    model = _ZeroLogitModel()
    example = _example(
        "choose",
        labels=("AA", "B"),
        answer="B",
        source_row=0,
    )

    with pytest.raises(CategoricalEvaluationError, match="must encode as one token"):
        evaluate_categorical_task(
            model,
            ByteTokenizer(),
            _task(example),
            checkpoint_identity="checkpoint",
            batch_size=1,
            max_problems=None,
            device="cpu",
        )


def test_categorical_eval_restores_mode_when_model_output_is_invalid() -> None:
    model = _BadLogitModel()
    model.train()

    with pytest.raises(CategoricalEvaluationError, match="shape"):
        evaluate_categorical_task(
            model,
            ByteTokenizer(),
            _task(
                _example(
                    "choose",
                    labels=("A", "B"),
                    answer="A",
                    source_row=0,
                )
            ),
            checkpoint_identity="checkpoint",
            batch_size=1,
            max_problems=None,
            device="cpu",
        )

    assert model.training is True


def test_categorical_eval_rejects_non_finite_candidate_logits() -> None:
    with pytest.raises(CategoricalEvaluationError, match="non-finite"):
        evaluate_categorical_task(
            _NonFiniteLogitModel(),
            ByteTokenizer(),
            _task(
                _example(
                    "choose",
                    labels=("A", "B"),
                    answer="A",
                    source_row=0,
                )
            ),
            checkpoint_identity="checkpoint",
            batch_size=1,
            max_problems=None,
            device="cpu",
        )
