"""Deterministic answer-letter-logit scoring for categorical chat tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from types import MappingProxyType
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
from scratch_llm.chat.rendering import (
    CHAT_RENDERER_ID,
    render_completion_prompt,
    render_multiple_choice_prompt,
)
from scratch_llm.evaluation.chat.protocol import CHAT_EVAL_REFERENCE_COMMIT
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.utils import get_device


CHAT_CATEGORICAL_PROTOCOL_ID: Final = "nanochat_chat_categorical_v1"
CHAT_CATEGORICAL_PROTOCOL_VERSION: Final = 1
CHAT_CATEGORICAL_CONTEXT_POLICY_ID: Final = (
    "scratch_llm_exclude_overlength_categorical_prompts_v1"
)
CHAT_CATEGORICAL_CONTEXT_POLICY_VERSION: Final = 1
CHAT_CATEGORICAL_REFERENCE_FILES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scripts/chat_eval.py": (
            "394d8d5ab1e8a9bbe08fec5cd3a03fffec0cdf8c610a29a947a2ab801f488927"
        ),
        "tasks/common.py": (
            "cb2e997fe151acd2b6f026caa32b7dc7a9779c48fa2a2886d33ac7e21b16ff23"
        ),
    }
)
CategoricalRunKind: TypeAlias = Literal["bounded", "full"]


class CategoricalEvaluationError(ValueError):
    """A categorical task or model output cannot be scored safely."""


@dataclass(frozen=True, slots=True)
class CategoricalExample:
    """One user-only prompt with its declared answer-letter choices."""

    conversation: Conversation
    labels: tuple[str, ...]
    answer: str
    source_row: int
    identity: str
    group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if not isinstance(self.conversation.messages[-1], UserMessage):
            raise CategoricalEvaluationError(
                "categorical completion conversations must end with a user message"
            )
        if not isinstance(self.labels, tuple) or not self.labels:
            raise CategoricalEvaluationError("labels must be a non-empty tuple")
        for index, label in enumerate(self.labels):
            try:
                require_non_empty_string(label, name=f"labels[{index}]")
            except (TypeError, ValueError) as error:
                raise CategoricalEvaluationError(str(error)) from error
        if len(set(self.labels)) != len(self.labels):
            raise CategoricalEvaluationError("labels must be unique")
        if self.answer not in self.labels:
            raise CategoricalEvaluationError("answer must be present in labels")
        try:
            require_non_negative_integer(self.source_row, name="source_row")
            require_non_empty_string(self.identity, name="identity")
            if self.group is not None:
                require_non_empty_string(self.group, name="group")
        except (TypeError, ValueError) as error:
            raise CategoricalEvaluationError(str(error)) from error


@dataclass(frozen=True, slots=True)
class CategoricalTask:
    """One complete, deterministically ordered categorical dataset view."""

    name: str
    examples: tuple[CategoricalExample, ...]
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
            raise CategoricalEvaluationError(str(error)) from error
        if not isinstance(self.examples, tuple) or not self.examples:
            raise CategoricalEvaluationError("examples must be a non-empty tuple")
        if any(
            not isinstance(example, CategoricalExample) for example in self.examples
        ):
            raise TypeError("examples must contain only CategoricalExample values")
        identities = tuple(example.identity for example in self.examples)
        if len(set(identities)) != len(identities):
            raise CategoricalEvaluationError("example identities must be unique")
        source_rows = tuple(example.source_row for example in self.examples)
        if len(set(source_rows)) != len(source_rows):
            raise CategoricalEvaluationError("example source rows must be unique")
        grouped = tuple(example.group is not None for example in self.examples)
        if any(grouped) and not all(grouped):
            raise CategoricalEvaluationError(
                "categorical task examples must be either all grouped or all ungrouped"
            )


@dataclass(frozen=True, slots=True)
class CategoricalGroupResult:
    """Passed and evaluated counts for one optional categorical task group."""

    name: str
    passed_count: int
    evaluated_count: int

    def __post_init__(self) -> None:
        try:
            require_non_empty_string(self.name, name="name")
            require_non_negative_integer(self.passed_count, name="passed_count")
            require_positive_integer(self.evaluated_count, name="evaluated_count")
        except (TypeError, ValueError) as error:
            raise CategoricalEvaluationError(str(error)) from error
        if self.passed_count > self.evaluated_count:
            raise CategoricalEvaluationError(
                "group passed_count must not exceed evaluated_count"
            )

    @property
    def accuracy(self) -> float:
        """Return group accuracy derived from exact integer counts."""

        return self.passed_count / self.evaluated_count

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible group counts."""

        return {
            "accuracy": self.accuracy,
            "evaluated": self.evaluated_count,
            "name": self.name,
            "passed": self.passed_count,
        }


@dataclass(frozen=True, slots=True)
class CategoricalPromptExclusion:
    """Content-free identity and length for one prompt excluded at preflight."""

    example_identity: str
    source_row: int
    prompt_token_count: int

    def __post_init__(self) -> None:
        try:
            require_non_empty_string(
                self.example_identity,
                name="example_identity",
            )
            require_non_negative_integer(self.source_row, name="source_row")
            require_positive_integer(
                self.prompt_token_count,
                name="prompt_token_count",
            )
        except (TypeError, ValueError) as error:
            raise CategoricalEvaluationError(str(error)) from error

    def to_dict(self) -> dict[str, object]:
        """Return stable content-free exclusion metadata."""

        return {
            "example_identity": self.example_identity,
            "prompt_token_count": self.prompt_token_count,
            "source_row": self.source_row,
        }


@dataclass(frozen=True, slots=True)
class CategoricalTaskResult:
    """Completed counts and identities for one categorical task run."""

    task_name: str
    checkpoint_identity: str
    tokenizer_identity: str
    source_identity: str
    dataset_identity: str
    order_identity: str
    run_kind: CategoricalRunKind
    max_problems: int | None
    passed_count: int
    evaluated_count: int
    available_count: int
    elapsed_seconds: float
    model_max_seq_len: int
    prompt_exclusions: tuple[CategoricalPromptExclusion, ...] = ()
    groups: tuple[CategoricalGroupResult, ...] = ()
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
            require_non_negative_integer(self.passed_count, name="passed_count")
            require_positive_integer(self.evaluated_count, name="evaluated_count")
            require_positive_integer(self.available_count, name="available_count")
            require_positive_integer(
                self.model_max_seq_len,
                name="model_max_seq_len",
            )
            require_finite_non_negative_real(
                self.elapsed_seconds,
                name="elapsed_seconds",
            )
        except (TypeError, ValueError) as error:
            raise CategoricalEvaluationError(str(error)) from error
        if self.renderer_identity != CHAT_RENDERER_ID:
            raise CategoricalEvaluationError(
                f"renderer_identity must equal {CHAT_RENDERER_ID!r}"
            )
        if self.passed_count > self.evaluated_count:
            raise CategoricalEvaluationError(
                "passed_count must not exceed evaluated_count"
            )
        if self.evaluated_count > self.available_count:
            raise CategoricalEvaluationError(
                "evaluated_count must not exceed available_count"
            )
        if not isinstance(self.prompt_exclusions, tuple) or any(
            not isinstance(exclusion, CategoricalPromptExclusion)
            for exclusion in self.prompt_exclusions
        ):
            raise TypeError(
                "prompt_exclusions must be a tuple of CategoricalPromptExclusion values"
            )
        exclusion_identities = tuple(
            exclusion.example_identity for exclusion in self.prompt_exclusions
        )
        exclusion_rows = tuple(
            exclusion.source_row for exclusion in self.prompt_exclusions
        )
        if len(set(exclusion_identities)) != len(exclusion_identities):
            raise CategoricalEvaluationError(
                "prompt exclusion example identities must be unique"
            )
        if len(set(exclusion_rows)) != len(exclusion_rows):
            raise CategoricalEvaluationError(
                "prompt exclusion source rows must be unique"
            )
        if any(
            exclusion.prompt_token_count <= self.model_max_seq_len
            for exclusion in self.prompt_exclusions
        ):
            raise CategoricalEvaluationError(
                "excluded prompt lengths must exceed model_max_seq_len"
            )
        if self.run_kind == "full":
            if self.max_problems is not None:
                raise CategoricalEvaluationError(
                    "full results must not set max_problems"
                )
            if self.selected_count != self.available_count:
                raise CategoricalEvaluationError(
                    "full results must account for every available problem"
                )
        elif self.run_kind == "bounded":
            try:
                bounded_max_problems = require_positive_integer(
                    self.max_problems,
                    name="max_problems",
                )
            except (TypeError, ValueError) as error:
                raise CategoricalEvaluationError(str(error)) from error
            if self.evaluated_count > bounded_max_problems:
                raise CategoricalEvaluationError(
                    "bounded evaluated_count must not exceed max_problems"
                )
            if self.selected_count > self.available_count:
                raise CategoricalEvaluationError(
                    "bounded selected_count must not exceed available_count"
                )
            if (
                self.evaluated_count < bounded_max_problems
                and self.selected_count != self.available_count
            ):
                raise CategoricalEvaluationError(
                    "bounded results below max_problems must account for every "
                    "available problem"
                )
        else:
            raise CategoricalEvaluationError("run_kind must be 'bounded' or 'full'")
        if not isinstance(self.groups, tuple) or any(
            not isinstance(group, CategoricalGroupResult) for group in self.groups
        ):
            raise TypeError("groups must be a tuple of CategoricalGroupResult values")
        group_names = tuple(group.name for group in self.groups)
        if group_names != tuple(sorted(group_names)) or len(set(group_names)) != len(
            group_names
        ):
            raise CategoricalEvaluationError(
                "groups must have unique names in sorted order"
            )
        if self.groups and (
            sum(group.passed_count for group in self.groups) != self.passed_count
            or sum(group.evaluated_count for group in self.groups)
            != self.evaluated_count
        ):
            raise CategoricalEvaluationError(
                "group counts must reconcile with overall counts"
            )

    @property
    def accuracy(self) -> float:
        """Return exact accuracy derived from integer counts."""

        return self.passed_count / self.evaluated_count

    @property
    def selected_count(self) -> int:
        """Return source rows consumed to satisfy the requested evaluation scope."""

        return self.evaluated_count + self.excluded_overlength_count

    @property
    def excluded_overlength_count(self) -> int:
        """Return prompts rejected intact because they exceed the model window."""

        return len(self.prompt_exclusions)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible completed-result representation."""

        return {
            "accuracy": self.accuracy,
            "counts": {
                "available": self.available_count,
                "evaluated": self.evaluated_count,
                "excluded_overlength": self.excluded_overlength_count,
                "passed": self.passed_count,
                "selected": self.selected_count,
            },
            "elapsed_seconds": self.elapsed_seconds,
            "groups": [group.to_dict() for group in self.groups],
            "identities": {
                "checkpoint": self.checkpoint_identity,
                "dataset": self.dataset_identity,
                "order": self.order_identity,
                "renderer": self.renderer_identity,
                "source": self.source_identity,
                "tokenizer": self.tokenizer_identity,
            },
            "protocol_id": CHAT_CATEGORICAL_PROTOCOL_ID,
            "protocol_version": CHAT_CATEGORICAL_PROTOCOL_VERSION,
            "prompt_preflight": {
                "excluded_examples": [
                    exclusion.to_dict() for exclusion in self.prompt_exclusions
                ],
                "model_max_seq_len": self.model_max_seq_len,
                "policy_id": CHAT_CATEGORICAL_CONTEXT_POLICY_ID,
                "policy_version": CHAT_CATEGORICAL_CONTEXT_POLICY_VERSION,
            },
            "reference_commit": CHAT_EVAL_REFERENCE_COMMIT,
            "reference_files": dict(CHAT_CATEGORICAL_REFERENCE_FILES),
            "scope": {
                "bounded": self.run_kind == "bounded",
                "max_problems": self.max_problems,
                "run_kind": self.run_kind,
            },
            "task_name": self.task_name,
        }


def evaluate_categorical_task(
    model: nn.Module,
    tokenizer: Tokenizer,
    task: CategoricalTask,
    *,
    checkpoint_identity: str,
    batch_size: int,
    max_problems: int | None,
    device: str | torch.device,
    clock: Callable[[], float] = time.monotonic,
) -> CategoricalTaskResult:
    """Score declared answer letters at every row's true prompt end."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement Tokenizer")
    if not isinstance(task, CategoricalTask):
        raise TypeError("task must be a CategoricalTask")
    try:
        checkpoint_identity = require_non_empty_string(
            checkpoint_identity,
            name="checkpoint_identity",
        )
        batch_size = require_positive_integer(batch_size, name="batch_size")
        if max_problems is not None:
            max_problems = require_positive_integer(
                max_problems,
                name="max_problems",
            )
    except (TypeError, ValueError) as error:
        raise CategoricalEvaluationError(str(error)) from error
    if not callable(clock):
        raise TypeError("clock must be callable")
    max_seq_len = getattr(model, "max_seq_len", None)
    if (
        not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or max_seq_len <= 0
    ):
        raise CategoricalEvaluationError("model.max_seq_len must be positive")

    selected_example_list: list[CategoricalExample] = []
    selected_prompt_list: list[tuple[int, ...]] = []
    fitting_prompt_count = 0
    for example in task.examples:
        prompt = render_completion_prompt(example.conversation, tokenizer).token_ids
        selected_example_list.append(example)
        selected_prompt_list.append(prompt)
        if len(prompt) <= max_seq_len:
            fitting_prompt_count += 1
        if max_problems is not None and fitting_prompt_count == max_problems:
            break
    selected_examples = tuple(selected_example_list)
    selected_prompts = tuple(selected_prompt_list)
    selected_label_token_ids = _encode_declared_labels(
        selected_examples,
        tokenizer=tokenizer,
    )
    prompt_exclusions = tuple(
        CategoricalPromptExclusion(
            example_identity=example.identity,
            source_row=example.source_row,
            prompt_token_count=len(prompt),
        )
        for example, prompt in zip(
            selected_examples,
            selected_prompts,
            strict=True,
        )
        if len(prompt) > max_seq_len
    )
    prepared = tuple(
        (example, prompt, label_token_ids)
        for example, prompt, label_token_ids in zip(
            selected_examples,
            selected_prompts,
            selected_label_token_ids,
            strict=True,
        )
        if len(prompt) <= max_seq_len
    )
    if not prepared:
        raise CategoricalEvaluationError(
            "no selected prompts fit model.max_seq_len "
            f"{max_seq_len}; excluded {len(prompt_exclusions)} overlength prompts"
        )
    examples = tuple(item[0] for item in prepared)
    rendered = tuple(item[1] for item in prepared)
    label_token_ids = tuple(item[2] for item in prepared)
    resolved_device = get_device(device)
    bos_token_id = tokenizer.get_bos_token_id()
    modes = tuple((module, module.training) for module in model.modules())
    passed_count = 0
    group_counts: dict[str, tuple[int, int]] = {}
    started_at = clock()
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                stop = min(start + batch_size, len(examples))
                prompt_batch = rendered[start:stop]
                input_ids = _right_pad_prompts(
                    prompt_batch,
                    pad_token_id=bos_token_id,
                    device=resolved_device,
                )
                logits = model(input_ids)
                _validate_logits(logits, input_ids=input_ids)
                for row, example in enumerate(examples[start:stop]):
                    candidate_ids = label_token_ids[start + row]
                    answer_position = len(prompt_batch[row]) - 1
                    candidate_logits = logits[row, answer_position, candidate_ids]
                    if not bool(torch.isfinite(candidate_logits).all().item()):
                        raise CategoricalEvaluationError(
                            "model produced non-finite categorical candidate logits"
                        )
                    predicted_index = int(candidate_logits.argmax().item())
                    passed = int(example.labels[predicted_index] == example.answer)
                    passed_count += passed
                    if example.group is not None:
                        group_passed, group_evaluated = group_counts.get(
                            example.group,
                            (0, 0),
                        )
                        group_counts[example.group] = (
                            group_passed + passed,
                            group_evaluated + 1,
                        )
    finally:
        for module, training in modes:
            module.training = training
    elapsed_seconds = clock() - started_at
    return CategoricalTaskResult(
        task_name=task.name,
        checkpoint_identity=checkpoint_identity,
        tokenizer_identity=tokenizer.get_identity(),
        source_identity=task.source_identity,
        dataset_identity=task.dataset_identity,
        order_identity=task.order_identity,
        run_kind="bounded" if max_problems is not None else "full",
        max_problems=max_problems,
        passed_count=passed_count,
        evaluated_count=len(examples),
        available_count=len(task.examples),
        elapsed_seconds=elapsed_seconds,
        model_max_seq_len=max_seq_len,
        prompt_exclusions=prompt_exclusions,
        groups=tuple(
            CategoricalGroupResult(
                name=name,
                passed_count=counts[0],
                evaluated_count=counts[1],
            )
            for name, counts in sorted(group_counts.items())
        ),
    )


def _encode_declared_labels(
    examples: tuple[CategoricalExample, ...],
    *,
    tokenizer: Tokenizer,
) -> tuple[tuple[int, ...], ...]:
    cache: dict[str, int] = {}
    encoded_examples = []
    for example in examples:
        encoded_labels = []
        for label in example.labels:
            if label not in cache:
                encoded = tokenizer.encode(label)
                if len(encoded) != 1:
                    raise CategoricalEvaluationError(
                        f"declared answer label {label!r} must encode as one token"
                    )
                cache[label] = encoded[0]
            encoded_labels.append(cache[label])
        if len(set(encoded_labels)) != len(encoded_labels):
            raise CategoricalEvaluationError(
                "declared answer labels must encode as distinct token IDs"
            )
        encoded_examples.append(tuple(encoded_labels))
    return tuple(encoded_examples)


def _right_pad_prompts(
    prompts: tuple[tuple[int, ...], ...],
    *,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    width = max(len(prompt) for prompt in prompts)
    batch = torch.full(
        (len(prompts), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for row, prompt in enumerate(prompts):
        batch[row, : len(prompt)] = torch.tensor(
            prompt,
            dtype=torch.long,
            device=device,
        )
    return batch


def _validate_logits(logits: object, *, input_ids: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise CategoricalEvaluationError("model must return a Tensor of logits")
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise CategoricalEvaluationError(
            "model logits must have shape (batch, sequence, vocabulary)"
        )
    if logits.shape[-1] <= int(input_ids.max().item()):
        raise CategoricalEvaluationError(
            "input token ID exceeds the model logits vocabulary"
        )


__all__ = [
    "CHAT_CATEGORICAL_CONTEXT_POLICY_ID",
    "CHAT_CATEGORICAL_CONTEXT_POLICY_VERSION",
    "CHAT_CATEGORICAL_PROTOCOL_ID",
    "CHAT_CATEGORICAL_PROTOCOL_VERSION",
    "CHAT_CATEGORICAL_REFERENCE_FILES",
    "CHAT_EVAL_REFERENCE_COMMIT",
    "CategoricalEvaluationError",
    "CategoricalExample",
    "CategoricalGroupResult",
    "CategoricalPromptExclusion",
    "CategoricalRunKind",
    "CategoricalTask",
    "CategoricalTaskResult",
    "evaluate_categorical_task",
    "render_multiple_choice_prompt",
]
