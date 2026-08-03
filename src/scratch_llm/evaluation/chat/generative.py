"""Shared deterministic sampling and pass-any scoring for chat tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from typing import Final, Literal, TypeAlias

import torch
from torch import nn

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.chat.conversation import Conversation, UserMessage
from scratch_llm.chat.rendering import CHAT_RENDERER_ID, render_completion_prompt
from scratch_llm.evaluation.chat.protocol import CHAT_EVAL_REFERENCE_COMMIT
from scratch_llm.generation import (
    CompletionReason,
    GenerationBatchResult,
    generate_sequences,
)
from scratch_llm.identity import canonical_json_identity
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.utils import get_device


CHAT_GENERATIVE_PROTOCOL_ID: Final = "nanochat_chat_generative_v1"
CHAT_GENERATIVE_PROTOCOL_VERSION: Final = 1
GENERATIVE_SEED_STRATEGY: Final = "sha256_order_problem_sample_v1"
_MAX_SEED: Final = 2**63 - 1
GenerativeRunKind: TypeAlias = Literal["bounded", "full"]
GenerativeScorer: TypeAlias = Callable[["GenerativeProblem", str], bool]


class GenerativeEvaluationError(ValueError):
    """A generative task, completion, or result cannot be scored safely."""


@dataclass(frozen=True, slots=True)
class GenerativeEvaluationConfig:
    """Deterministic settings shared by every generative chat task."""

    num_samples: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_k: int | None = 50
    seed: int = 42

    def __post_init__(self) -> None:
        require_positive_integer(self.num_samples, name="num_samples")
        require_positive_integer(self.max_new_tokens, name="max_new_tokens")
        temperature = require_finite_non_negative_real(
            self.temperature,
            name="temperature",
        )
        object.__setattr__(self, "temperature", temperature)
        if self.top_k is not None:
            require_positive_integer(self.top_k, name="top_k")
        seed = require_non_negative_integer(self.seed, name="seed")
        if seed > _MAX_SEED:
            raise ValueError(f"seed must be at most {_MAX_SEED}")

    def to_dict(self) -> dict[str, object]:
        """Return the stable public generation settings."""

        return {
            "max_new_tokens": self.max_new_tokens,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "seed_strategy": GENERATIVE_SEED_STRATEGY,
            "stop_tokens": "assistant_end_then_bos_safety",
            "temperature": self.temperature,
            "top_k": self.top_k,
        }


@dataclass(frozen=True, slots=True)
class GenerativeProblem:
    """One user-only prompt with stable source metadata."""

    conversation: Conversation
    source_row: int
    identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if not isinstance(self.conversation.messages[-1], UserMessage):
            raise GenerativeEvaluationError(
                "generative completion conversations must end with a user message"
            )
        try:
            require_non_negative_integer(self.source_row, name="source_row")
            require_non_empty_string(self.identity, name="identity")
        except (TypeError, ValueError) as error:
            raise GenerativeEvaluationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class GenerativeTask:
    """One complete, deterministically ordered generative dataset view."""

    name: str
    problems: tuple[GenerativeProblem, ...]
    source_identity: str
    dataset_identity: str
    order_identity: str

    def __post_init__(self) -> None:
        try:
            for name in (
                "name",
                "source_identity",
                "dataset_identity",
                "order_identity",
            ):
                require_non_empty_string(getattr(self, name), name=name)
        except (TypeError, ValueError) as error:
            raise GenerativeEvaluationError(str(error)) from error
        if not isinstance(self.problems, tuple) or not self.problems:
            raise GenerativeEvaluationError("problems must be a non-empty tuple")
        if any(not isinstance(problem, GenerativeProblem) for problem in self.problems):
            raise TypeError("problems must contain only GenerativeProblem values")
        identities = tuple(problem.identity for problem in self.problems)
        if len(set(identities)) != len(identities):
            raise GenerativeEvaluationError("problem identities must be unique")
        source_rows = tuple(problem.source_row for problem in self.problems)
        if len(set(source_rows)) != len(source_rows):
            raise GenerativeEvaluationError("problem source rows must be unique")


@dataclass(frozen=True, slots=True)
class GenerativeSampleResult:
    """Content-free scoring and termination metadata for one completion."""

    problem_index: int
    sample_index: int
    seed: int
    passed: bool
    generated_token_count: int
    sampled_token_count: int
    completion_reason: CompletionReason
    stop_token_id: int | None
    completion_identity: str

    def __post_init__(self) -> None:
        require_non_negative_integer(self.problem_index, name="problem_index")
        require_non_negative_integer(self.sample_index, name="sample_index")
        require_non_negative_integer(self.seed, name="seed")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        generated_count = require_non_negative_integer(
            self.generated_token_count,
            name="generated_token_count",
        )
        sampled_count = require_positive_integer(
            self.sampled_token_count,
            name="sampled_token_count",
        )
        if self.completion_reason == "stop_token":
            require_non_negative_integer(self.stop_token_id, name="stop_token_id")
            expected_count = generated_count + 1
        elif self.completion_reason == "max_new_tokens":
            if self.stop_token_id is not None:
                raise GenerativeEvaluationError(
                    "max_new_tokens samples must not set stop_token_id"
                )
            expected_count = generated_count
        else:
            raise GenerativeEvaluationError(
                "completion_reason must be 'stop_token' or 'max_new_tokens'"
            )
        if sampled_count != expected_count:
            raise GenerativeEvaluationError(
                "sampled_token_count does not match completion metadata"
            )
        require_non_empty_string(
            self.completion_identity,
            name="completion_identity",
        )

    def to_dict(self) -> dict[str, object]:
        """Return stable metadata without raw completion content."""

        return {
            "completion_identity": self.completion_identity,
            "completion_reason": self.completion_reason,
            "generated_token_count": self.generated_token_count,
            "passed": self.passed,
            "sample_index": self.sample_index,
            "sampled_token_count": self.sampled_token_count,
            "seed": self.seed,
            "stop_token_id": self.stop_token_id,
        }


@dataclass(frozen=True, slots=True)
class GenerativeProblemResult:
    """Pass-any outcome and all sample metadata for one problem."""

    problem_index: int
    problem_identity: str
    source_row: int
    passed: bool
    samples: tuple[GenerativeSampleResult, ...]

    def __post_init__(self) -> None:
        require_non_negative_integer(self.problem_index, name="problem_index")
        require_non_empty_string(self.problem_identity, name="problem_identity")
        require_non_negative_integer(self.source_row, name="source_row")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise GenerativeEvaluationError("samples must be a non-empty tuple")
        if any(
            not isinstance(sample, GenerativeSampleResult) for sample in self.samples
        ):
            raise TypeError("samples must contain only GenerativeSampleResult values")
        for sample_index, sample in enumerate(self.samples):
            if (
                sample.problem_index != self.problem_index
                or sample.sample_index != sample_index
            ):
                raise GenerativeEvaluationError(
                    "sample problem/sample indices do not match their result order"
                )
        if self.passed != any(sample.passed for sample in self.samples):
            raise GenerativeEvaluationError("problem passed must use pass-any scoring")

    def to_dict(self) -> dict[str, object]:
        """Return stable per-problem result metadata."""

        return {
            "passed": self.passed,
            "problem_identity": self.problem_identity,
            "problem_index": self.problem_index,
            "samples": [sample.to_dict() for sample in self.samples],
            "source_row": self.source_row,
        }


@dataclass(frozen=True, slots=True)
class GenerativeTaskResult:
    """One complete bounded or full generative task result."""

    task_name: str
    checkpoint_identity: str
    tokenizer_identity: str
    source_identity: str
    dataset_identity: str
    order_identity: str
    run_kind: GenerativeRunKind
    max_problems: int | None
    available_count: int
    assistant_end_token_id: int
    bos_token_id: int
    config: GenerativeEvaluationConfig
    problems: tuple[GenerativeProblemResult, ...]
    renderer_identity: str = CHAT_RENDERER_ID

    def __post_init__(self) -> None:
        try:
            for name in (
                "task_name",
                "checkpoint_identity",
                "tokenizer_identity",
                "source_identity",
                "dataset_identity",
                "order_identity",
            ):
                require_non_empty_string(getattr(self, name), name=name)
            require_positive_integer(self.available_count, name="available_count")
            assistant_end = require_non_negative_integer(
                self.assistant_end_token_id,
                name="assistant_end_token_id",
            )
            bos = require_non_negative_integer(self.bos_token_id, name="bos_token_id")
        except (TypeError, ValueError) as error:
            raise GenerativeEvaluationError(str(error)) from error
        if assistant_end == bos:
            raise GenerativeEvaluationError(
                "assistant_end and BOS stop token IDs must differ"
            )
        if self.renderer_identity != CHAT_RENDERER_ID:
            raise GenerativeEvaluationError(
                f"renderer_identity must equal {CHAT_RENDERER_ID!r}"
            )
        if not isinstance(self.config, GenerativeEvaluationConfig):
            raise TypeError("config must be a GenerativeEvaluationConfig")
        if not isinstance(self.problems, tuple) or not self.problems:
            raise GenerativeEvaluationError("problems must be a non-empty tuple")
        if any(
            not isinstance(problem, GenerativeProblemResult)
            for problem in self.problems
        ):
            raise TypeError("problems must contain only GenerativeProblemResult values")
        if len(self.problems) > self.available_count:
            raise GenerativeEvaluationError(
                "evaluated problems must not exceed available_count"
            )
        for problem_index, problem in enumerate(self.problems):
            if problem.problem_index != problem_index:
                raise GenerativeEvaluationError(
                    "problem indices must match deterministic result order"
                )
            if len(problem.samples) != self.config.num_samples:
                raise GenerativeEvaluationError(
                    "every problem must contain config.num_samples results"
                )
            for sample in problem.samples:
                if (
                    sample.completion_reason == "stop_token"
                    and sample.stop_token_id
                    not in {
                        assistant_end,
                        bos,
                    }
                ):
                    raise GenerativeEvaluationError(
                        "samples may stop only on assistant_end or BOS"
                    )
        if self.run_kind == "full":
            if self.max_problems is not None:
                raise GenerativeEvaluationError(
                    "full results must not set max_problems"
                )
            if len(self.problems) != self.available_count:
                raise GenerativeEvaluationError(
                    "full results must evaluate every available problem"
                )
        elif self.run_kind == "bounded":
            try:
                require_positive_integer(self.max_problems, name="max_problems")
            except (TypeError, ValueError) as error:
                raise GenerativeEvaluationError(str(error)) from error
        else:
            raise GenerativeEvaluationError("run_kind must be 'bounded' or 'full'")

    @property
    def evaluated_count(self) -> int:
        return len(self.problems)

    @property
    def passed_count(self) -> int:
        return sum(problem.passed for problem in self.problems)

    @property
    def total_sample_count(self) -> int:
        return sum(len(problem.samples) for problem in self.problems)

    @property
    def accuracy(self) -> float:
        return self.passed_count / self.evaluated_count

    @property
    def generation_identity(self) -> str:
        return canonical_json_identity(
            {
                "config": self.config.to_dict(),
                "format": "scratch_llm_chat_generative_generation_v1",
                "renderer_identity": self.renderer_identity,
                "stop_token_ids": [
                    self.assistant_end_token_id,
                    self.bos_token_id,
                ],
            }
        )

    @property
    def stop_counts(self) -> dict[str, int]:
        samples = tuple(
            sample for problem in self.problems for sample in problem.samples
        )
        return {
            "assistant_end": sum(
                sample.stop_token_id == self.assistant_end_token_id
                for sample in samples
            ),
            "bos": sum(sample.stop_token_id == self.bos_token_id for sample in samples),
            "max_new_tokens": sum(
                sample.completion_reason == "max_new_tokens" for sample in samples
            ),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete result without raw prompts or completions."""

        return {
            "accuracy": self.accuracy,
            "counts": {
                "available_problems": self.available_count,
                "evaluated_problems": self.evaluated_count,
                "passed_problems": self.passed_count,
                "samples": self.total_sample_count,
                "samples_per_problem": self.config.num_samples,
            },
            "generation": {
                "config": self.config.to_dict(),
                "identity": self.generation_identity,
                "stop_counts": self.stop_counts,
                "stop_token_ids": [
                    self.assistant_end_token_id,
                    self.bos_token_id,
                ],
            },
            "identities": {
                "checkpoint": self.checkpoint_identity,
                "dataset": self.dataset_identity,
                "order": self.order_identity,
                "renderer": self.renderer_identity,
                "source": self.source_identity,
                "tokenizer": self.tokenizer_identity,
            },
            "problems": [problem.to_dict() for problem in self.problems],
            "protocol_id": CHAT_GENERATIVE_PROTOCOL_ID,
            "protocol_version": CHAT_GENERATIVE_PROTOCOL_VERSION,
            "reference_commit": CHAT_EVAL_REFERENCE_COMMIT,
            "scope": {
                "bounded": self.run_kind == "bounded",
                "max_problems": self.max_problems,
                "run_kind": self.run_kind,
            },
            "task_name": self.task_name,
        }


def derive_generative_sample_seed(
    *,
    base_seed: int,
    order_identity: str,
    problem_identity: str,
    problem_index: int,
    sample_index: int,
) -> int:
    """Derive one stable seed independent of batch and sample counts."""

    base_seed = require_non_negative_integer(base_seed, name="base_seed")
    if base_seed > _MAX_SEED:
        raise ValueError(f"base_seed must be at most {_MAX_SEED}")
    order_identity = require_non_empty_string(
        order_identity,
        name="order_identity",
    )
    problem_identity = require_non_empty_string(
        problem_identity,
        name="problem_identity",
    )
    problem_index = require_non_negative_integer(
        problem_index,
        name="problem_index",
    )
    sample_index = require_non_negative_integer(
        sample_index,
        name="sample_index",
    )
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "order_identity": order_identity,
            "problem_identity": problem_identity,
            "problem_index": problem_index,
            "sample_index": sample_index,
            "strategy": GENERATIVE_SEED_STRATEGY,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED


def evaluate_generative_task(
    model: nn.Module,
    tokenizer: Tokenizer,
    task: GenerativeTask,
    score_completion: GenerativeScorer,
    *,
    checkpoint_identity: str,
    config: GenerativeEvaluationConfig,
    max_problems: int | None,
    device: str | torch.device,
) -> GenerativeTaskResult:
    """Generate one seeded sample batch per problem and apply pass-any scoring."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement Tokenizer")
    if not isinstance(task, GenerativeTask):
        raise TypeError("task must be a GenerativeTask")
    if not callable(score_completion):
        raise TypeError("score_completion must be callable")
    if not isinstance(config, GenerativeEvaluationConfig):
        raise TypeError("config must be a GenerativeEvaluationConfig")
    try:
        checkpoint_identity = require_non_empty_string(
            checkpoint_identity,
            name="checkpoint_identity",
        )
        if max_problems is not None:
            max_problems = require_positive_integer(
                max_problems,
                name="max_problems",
            )
    except (TypeError, ValueError) as error:
        raise GenerativeEvaluationError(str(error)) from error
    resolved_device = get_device(device)
    assistant_end_token_id = tokenizer.encode_special("<|assistant_end|>")
    bos_token_id = tokenizer.get_bos_token_id()
    _validate_stop_tokens(
        tokenizer,
        assistant_end_token_id=assistant_end_token_id,
        bos_token_id=bos_token_id,
    )
    problems = (
        task.problems
        if max_problems is None
        else task.problems[: min(max_problems, len(task.problems))]
    )
    problem_results = []
    for problem_index, problem in enumerate(problems):
        rendered = render_completion_prompt(problem.conversation, tokenizer)
        prompt = torch.tensor(
            [rendered.token_ids],
            dtype=torch.long,
            device=resolved_device,
        ).repeat(config.num_samples, 1)
        row_seeds = tuple(
            derive_generative_sample_seed(
                base_seed=config.seed,
                order_identity=task.order_identity,
                problem_identity=problem.identity,
                problem_index=problem_index,
                sample_index=sample_index,
            )
            for sample_index in range(config.num_samples)
        )
        generated = generate_sequences(
            model,
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            row_seeds=row_seeds,
            stop_token_ids={assistant_end_token_id, bos_token_id},
        )
        if not isinstance(generated, GenerationBatchResult):
            raise GenerativeEvaluationError(
                "shared generation did not return a GenerationBatchResult"
            )
        if len(generated.sequences) != config.num_samples:
            raise GenerativeEvaluationError(
                f"expected {config.num_samples} samples for problem "
                f"{problem_index}, got {len(generated.sequences)}"
            )
        sample_results = []
        for sample_index, (seed, sequence) in enumerate(
            zip(row_seeds, generated.sequences, strict=True)
        ):
            if sequence.prompt_token_ids != rendered.token_ids:
                raise GenerativeEvaluationError(
                    "shared generation returned a conflicting prompt identity"
                )
            completion = tokenizer.decode(sequence.generated_token_ids)
            try:
                passed = score_completion(problem, completion)
            except Exception as error:
                raise GenerativeEvaluationError(
                    f"{task.name} problem {problem_index} scoring failed: {error}"
                ) from error
            if not isinstance(passed, bool):
                raise GenerativeEvaluationError(
                    "score_completion must return a boolean"
                )
            sample_results.append(
                GenerativeSampleResult(
                    problem_index=problem_index,
                    sample_index=sample_index,
                    seed=seed,
                    passed=passed,
                    generated_token_count=len(sequence.generated_token_ids),
                    sampled_token_count=sequence.sampled_token_count,
                    completion_reason=sequence.completion_reason,
                    stop_token_id=sequence.stop_token_id,
                    completion_identity=canonical_json_identity(
                        {"generated_token_ids": list(sequence.generated_token_ids)}
                    ),
                )
            )
        immutable_samples = tuple(sample_results)
        problem_results.append(
            GenerativeProblemResult(
                problem_index=problem_index,
                problem_identity=problem.identity,
                source_row=problem.source_row,
                passed=any(sample.passed for sample in immutable_samples),
                samples=immutable_samples,
            )
        )
    return GenerativeTaskResult(
        task_name=task.name,
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer.get_identity(),
        source_identity=task.source_identity,
        dataset_identity=task.dataset_identity,
        order_identity=task.order_identity,
        run_kind="bounded" if max_problems is not None else "full",
        max_problems=max_problems,
        available_count=len(task.problems),
        assistant_end_token_id=assistant_end_token_id,
        bos_token_id=bos_token_id,
        config=config,
        problems=tuple(problem_results),
    )


def _validate_stop_tokens(
    tokenizer: Tokenizer,
    *,
    assistant_end_token_id: int,
    bos_token_id: int,
) -> None:
    vocab_size = tokenizer.get_vocab_size()
    for name, token_id in (
        ("assistant_end", assistant_end_token_id),
        ("BOS", bos_token_id),
    ):
        try:
            token_id = require_non_negative_integer(token_id, name=f"{name} token ID")
        except (TypeError, ValueError) as error:
            raise GenerativeEvaluationError(str(error)) from error
        if token_id >= vocab_size:
            raise GenerativeEvaluationError(
                f"tokenizer {name} token ID is outside its vocabulary"
            )
    if assistant_end_token_id == bos_token_id:
        raise GenerativeEvaluationError(
            "assistant_end and BOS stop token IDs must differ"
        )


__all__ = [
    "CHAT_GENERATIVE_PROTOCOL_ID",
    "CHAT_GENERATIVE_PROTOCOL_VERSION",
    "GENERATIVE_SEED_STRATEGY",
    "GenerativeEvaluationConfig",
    "GenerativeEvaluationError",
    "GenerativeProblem",
    "GenerativeProblemResult",
    "GenerativeRunKind",
    "GenerativeSampleResult",
    "GenerativeScorer",
    "GenerativeTask",
    "GenerativeTaskResult",
    "derive_generative_sample_seed",
    "evaluate_generative_task",
]
