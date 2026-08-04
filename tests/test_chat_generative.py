"""Tests for the shared deterministic generative chat-task runner."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from scratch_llm.chat.conversation import Conversation, UserMessage
from scratch_llm.chat.rendering import render_completion_prompt
import scratch_llm.evaluation.chat.generative as generative
from scratch_llm.evaluation.chat.generative import (
    GenerativeEvaluationConfig,
    GenerativeEvaluationError,
    GenerativeProblem,
    GenerativeTask,
    derive_generative_sample_seed,
    evaluate_generative_task,
)
from scratch_llm.generation import GenerationBatchResult, GeneratedSequence
from scratch_llm.tokenization.tokenizer import ByteTokenizer


class _StopPolicyModel(nn.Module):
    max_seq_len = 512

    def __init__(self, tokenizer: ByteTokenizer) -> None:
        super().__init__()
        self.assistant_start = tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = tokenizer.encode_special("<|assistant_end|>")
        self.bos = tokenizer.get_bos_token_id()
        self.probe = nn.Dropout()
        self.initial_contexts: list[torch.Tensor] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*token_ids.shape, 265),
            -torch.inf,
            device=token_ids.device,
        )
        for row, values in enumerate(token_ids.detach().cpu().tolist()):
            assistant_start = len(values) - 1 - values[::-1].index(self.assistant_start)
            generated_count = len(values) - assistant_start - 1
            if generated_count == 0:
                self.initial_contexts.append(token_ids[row].detach().cpu().clone())
            prompt = bytes(value for value in values[:assistant_start] if value < 256)
            if b"end" in prompt:
                sequence = (*b"#### 2", self.assistant_end)
            elif b"bos" in prompt:
                sequence = (self.bos,)
            else:
                sequence = (ord("x"),) * 32
            logits[row, -1, sequence[generated_count]] = 0
        return logits


class _RngConsumingStopPolicyModel(_StopPolicyModel):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        random.random()
        np.random.random()
        return super().forward(token_ids)


def _problem(prompt: str, source_row: int) -> GenerativeProblem:
    return GenerativeProblem(
        conversation=Conversation(messages=(UserMessage(prompt),)),
        source_row=source_row,
        identity=f"problem-{source_row}",
    )


def _task(*problems: GenerativeProblem) -> GenerativeTask:
    return GenerativeTask(
        name="fixture",
        problems=problems,
        source_identity="source",
        dataset_identity="dataset",
        order_identity="order",
    )


def test_generative_runner_uses_one_seeded_batch_per_problem_and_records_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = ByteTokenizer()
    model = _StopPolicyModel(tokenizer)
    model.train()
    model.probe.eval()
    task = _task(_problem("end", 0), _problem("bos", 1), _problem("max", 2))
    config = GenerativeEvaluationConfig(
        num_samples=2,
        max_new_tokens=7,
        temperature=0,
        top_k=5,
        seed=42,
    )
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    shared_generate = generative.generate_sequences

    def record_generation(
        active_model: nn.Module,
        token_ids: torch.Tensor,
        **kwargs: object,
    ) -> GenerationBatchResult:
        row_seeds = kwargs["row_seeds"]
        assert isinstance(row_seeds, tuple)
        calls.append((tuple(token_ids.shape), row_seeds))
        return shared_generate(active_model, token_ids, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(generative, "generate_sequences", record_generation)

    result = evaluate_generative_task(
        model,
        tokenizer,
        task,
        lambda _problem, completion: completion == "#### 2",
        checkpoint_identity="checkpoint",
        config=config,
        max_problems=None,
        device="cpu",
    )

    assert len(calls) == len(task.problems)
    assert all(shape[0] == config.num_samples for shape, _seeds in calls)
    seeds = tuple(seed for _shape, row_seeds in calls for seed in row_seeds)
    assert len(seeds) == len(set(seeds)) == 6
    assert result.passed_count == 1
    assert result.evaluated_count == 3
    assert result.total_sample_count == 6
    assert result.accuracy == 1 / 3
    assert result.stop_counts == {
        "assistant_end": 2,
        "bos": 2,
        "max_new_tokens": 2,
    }
    assert result.run_kind == "full"
    assert result.to_dict()["generation"]["config"] == config.to_dict()
    assert "#### 2" not in json.dumps(result.to_dict())
    assert model.training is True
    assert model.probe.training is False
    expected_prompt = render_completion_prompt(task.problems[0].conversation, tokenizer)
    assert tuple(model.initial_contexts[0].tolist()) == expected_prompt.token_ids

    repeated = evaluate_generative_task(
        _StopPolicyModel(tokenizer),
        tokenizer,
        task,
        lambda _problem, completion: completion == "#### 2",
        checkpoint_identity="checkpoint",
        config=config,
        max_problems=None,
        device="cpu",
    )
    assert repeated == result


def test_generative_runner_marks_explicit_limits_as_bounded() -> None:
    tokenizer = ByteTokenizer()
    result = evaluate_generative_task(
        _StopPolicyModel(tokenizer),
        tokenizer,
        _task(_problem("bos", 0), _problem("max", 1)),
        lambda _problem, _completion: False,
        checkpoint_identity="checkpoint",
        config=GenerativeEvaluationConfig(
            num_samples=1,
            max_new_tokens=2,
            temperature=0,
            seed=3,
        ),
        max_problems=1,
        device="cpu",
    )

    assert result.run_kind == "bounded"
    assert result.max_problems == 1
    assert result.evaluated_count == 1
    assert result.available_count == 2


def test_generative_runner_rejects_zero_samples_and_incomplete_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="num_samples must be positive"):
        GenerativeEvaluationConfig(num_samples=0)

    tokenizer = ByteTokenizer()

    def incomplete_generation(
        _model: nn.Module,
        token_ids: torch.Tensor,
        **_kwargs: object,
    ) -> GenerationBatchResult:
        prompt = tuple(int(value) for value in token_ids[0].tolist())
        return GenerationBatchResult(
            (
                GeneratedSequence(
                    prompt_token_ids=prompt,
                    generated_token_ids=(),
                    completion_reason="max_new_tokens",
                    stop_token_id=None,
                    sampled_token_count=0,
                ),
            )
        )

    monkeypatch.setattr(generative, "generate_sequences", incomplete_generation)
    with pytest.raises(GenerativeEvaluationError, match="expected 2 samples"):
        evaluate_generative_task(
            _StopPolicyModel(tokenizer),
            tokenizer,
            _task(_problem("max", 0)),
            lambda _problem, _completion: False,
            checkpoint_identity="checkpoint",
            config=GenerativeEvaluationConfig(
                num_samples=2,
                max_new_tokens=2,
                temperature=0,
            ),
            max_problems=None,
            device="cpu",
        )


def test_generative_runner_restores_rng_and_modes_when_scoring_fails() -> None:
    tokenizer = ByteTokenizer()
    model = _RngConsumingStopPolicyModel(tokenizer).train()
    model.probe.eval()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    def fail_scoring(_problem: GenerativeProblem, _completion: str) -> bool:
        raise RuntimeError("scoring failed")

    with pytest.raises(GenerativeEvaluationError, match="scoring failed"):
        evaluate_generative_task(
            model,
            tokenizer,
            _task(_problem("end", 0)),
            fail_scoring,
            checkpoint_identity="checkpoint",
            config=GenerativeEvaluationConfig(
                num_samples=2,
                max_new_tokens=7,
                temperature=0,
            ),
            max_problems=None,
            device="cpu",
        )

    assert model.training is True
    assert model.probe.training is False
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_generative_seed_derivation_is_stable_and_sample_specific() -> None:
    first = derive_generative_sample_seed(
        base_seed=42,
        order_identity="order",
        problem_identity="problem",
        problem_index=3,
        sample_index=0,
    )
    repeated = derive_generative_sample_seed(
        base_seed=42,
        order_identity="order",
        problem_identity="problem",
        problem_index=3,
        sample_index=0,
    )
    next_sample = derive_generative_sample_seed(
        base_seed=42,
        order_identity="order",
        problem_identity="problem",
        problem_index=3,
        sample_index=1,
    )

    assert first == repeated
    assert first != next_sample
    assert 0 <= first < 2**63
