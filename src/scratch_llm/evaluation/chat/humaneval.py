"""Opt-in HumanEval-style evaluation over the shared generative runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import keyword
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from scratch_llm._validation import (
    require_non_empty_string,
    require_non_negative_integer,
)
from scratch_llm.chat.conversation import Conversation, UserMessage
from scratch_llm.data.hub import (
    CachedHubParquetDataset,
    HubDatasetSpec,
    load_hub_parquet_cache,
)
from scratch_llm.evaluation.chat.cache import read_cached_parquet_rows
from scratch_llm.evaluation.chat.execution import CodeExecutor
from scratch_llm.evaluation.chat.generative import (
    GenerativeEvaluationConfig,
    GenerativeProblem,
    GenerativeScore,
    GenerativeTask,
    GenerativeTaskResult,
    evaluate_generative_task,
)
from scratch_llm.evaluation.chat.protocol import CHAT_EVAL_REFERENCE_COMMIT
from scratch_llm.identity import canonical_json_identity
from scratch_llm.tokenization.tokenizer import Tokenizer


HUMANEVAL_TASK_NAME: Final = "HumanEval"
HUMANEVAL_SHUFFLE_SEED: Final = 42
HUMANEVAL_REFERENCE_FILE_SHA256: Final = (
    "2ee62f4da86d6c2e9aea4b41cee1969a7da97036fd0e5d0152af600825dfd613"
)
HUMANEVAL_EXECUTION_REFERENCE_FILE_SHA256: Final = (
    "122ce1457100a5cb20fbe9666a57006df3b0fe78c014240d2f87c35ccade2c1a"
)
HUMANEVAL_EXECUTION_WARNING: Final = (
    "Generated Python executes locally in a resource-limited subprocess that is "
    "not safe for malicious or adversarial code."
)


class HumanEvalDatasetError(ValueError):
    """The HumanEval source or cached task view is invalid."""


class HumanEvalDatasetRowError(HumanEvalDatasetError):
    """One HumanEval row cannot be normalized safely."""


class HumanEvalExecutionDisabledError(PermissionError):
    """Generated-code execution was not explicitly enabled by the operator."""


@dataclass(frozen=True, slots=True)
class HumanEvalProblem(GenerativeProblem):
    """One user prompt plus trusted tests and an entry point, never a solution."""

    required_imports: str
    test_program: str
    entry_point: str

    def __post_init__(self) -> None:
        GenerativeProblem.__post_init__(self)
        if not isinstance(self.required_imports, str):
            raise HumanEvalDatasetRowError("required_imports must be a string")
        try:
            require_non_empty_string(self.test_program, name="test_program")
        except (TypeError, ValueError) as error:
            raise HumanEvalDatasetRowError(str(error)) from error
        if not _valid_entry_point(self.entry_point):
            raise HumanEvalDatasetRowError(
                "entry_point must be a valid Python identifier"
            )


def get_humaneval_dataset_spec() -> HubDatasetSpec:
    """Return the pinned openai/openai_humaneval test cache contract."""

    return HubDatasetSpec(
        dataset="humaneval",
        repository="openai/openai_humaneval",
        subset="openai_humaneval",
        split="test",
        adapter_version="humaneval_chat_v1",
        reference_commit=CHAT_EVAL_REFERENCE_COMMIT,
        required_columns=(
            "prompt",
            "canonical_solution",
            "test",
            "entry_point",
        ),
    )


def extract_leading_imports(prompt: str) -> str:
    """Extract trusted leading imports using the pinned nanochat rule."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    imports = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            break
    return "\n".join(imports)


def extract_humaneval_program(completion: str) -> str:
    """Return the first complete python/untyped fence, else trimmed plain text."""

    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    lines = completion.splitlines()
    index = 0
    while index < len(lines):
        opening = lines[index].strip()
        if not opening.startswith("```"):
            index += 1
            continue
        language = opening[3:].strip()
        closing = next(
            (
                candidate
                for candidate in range(index + 1, len(lines))
                if lines[candidate].strip() == "```"
            ),
            None,
        )
        if closing is None:
            break
        if language in {"", "python"}:
            return "\n".join(lines[index + 1 : closing]).strip()
        index = closing + 1
    return completion.strip()


def normalize_humaneval_row(
    row: Mapping[str, object],
    *,
    source_row: int,
    source_identity: str,
    context: str,
) -> HumanEvalProblem:
    """Normalize a strict row while retaining no executable reference solution."""

    try:
        if not isinstance(row, Mapping):
            raise HumanEvalDatasetRowError("row must be an object")
        prompt = _non_empty_string(row.get("prompt"), label="prompt")
        canonical_solution = _non_empty_string(
            row.get("canonical_solution"),
            label="canonical_solution",
        )
        test_program = _non_empty_string(row.get("test"), label="test")
        entry_point = _non_empty_string(row.get("entry_point"), label="entry_point")
        if not _valid_entry_point(entry_point):
            raise HumanEvalDatasetRowError(
                "entry_point must be a valid Python identifier"
            )
        try:
            source_row = require_non_negative_integer(
                source_row,
                name="source_row",
            )
            source_identity = require_non_empty_string(
                source_identity,
                name="source_identity",
            )
        except (TypeError, ValueError) as error:
            raise HumanEvalDatasetRowError(str(error)) from error

        required_imports = extract_leading_imports(prompt)
        identity = canonical_json_identity(
            {
                "canonical_solution_identity": canonical_json_identity(
                    canonical_solution
                ),
                "entry_point": entry_point,
                "prompt": prompt,
                "required_imports": required_imports,
                "source_identity": source_identity,
                "source_row": source_row,
                "test_identity": canonical_json_identity(test_program),
            }
        )
        return HumanEvalProblem(
            conversation=Conversation(messages=(UserMessage(prompt),)),
            source_row=source_row,
            identity=identity,
            required_imports=required_imports,
            test_program=test_program,
            entry_point=entry_point,
        )
    except HumanEvalDatasetRowError as error:
        raise _with_context(context, error) from error


def assemble_humaneval_program(
    problem: HumanEvalProblem,
    completion: str,
) -> str:
    """Assemble trusted imports/tests around generated code, never the solution."""

    if not isinstance(problem, HumanEvalProblem):
        raise TypeError("problem must be a HumanEvalProblem")
    completion_program = extract_humaneval_program(completion)
    sections = (
        problem.required_imports,
        completion_program,
        problem.test_program.strip(),
        f"check({problem.entry_point})",
    )
    return "\n\n".join(section for section in sections if section)


def score_humaneval_completion(
    problem: GenerativeProblem,
    completion: str,
    executor: CodeExecutor,
) -> GenerativeScore:
    """Execute one assembled completion and return only its normalized outcome."""

    if not isinstance(problem, HumanEvalProblem):
        raise TypeError("problem must be a HumanEvalProblem")
    _validate_executor(executor)
    execution = executor.execute(assemble_humaneval_program(problem, completion))
    return GenerativeScore(execution.success, execution.status)


def build_humaneval_task(cache: CachedHubParquetDataset) -> GenerativeTask:
    """Materialize deterministic seed-42 problems from a verified test cache."""

    if not isinstance(cache, CachedHubParquetDataset):
        raise TypeError("cache must be a CachedHubParquetDataset")
    expected_spec = get_humaneval_dataset_spec()
    if cache.spec != expected_spec:
        raise HumanEvalDatasetError(
            "cache spec does not match the pinned HumanEval test contract"
        )
    rows = read_cached_parquet_rows(cache)
    permutation = tuple(
        int(index)
        for index in np.random.default_rng(HUMANEVAL_SHUFFLE_SEED).permutation(
            len(rows)
        )
    )
    problems = tuple(
        normalize_humaneval_row(
            rows[source_row],
            source_row=source_row,
            source_identity=cache.source_identity,
            context=(
                f"{cache.spec.repository}/{cache.spec.subset}/"
                f"{cache.spec.split} row {source_row}"
            ),
        )
        for source_row in permutation
    )
    order_identity = canonical_json_identity(
        {
            "dataset_identity": cache.source_identity,
            "problem_identities": [problem.identity for problem in problems],
            "reference_file_sha256": HUMANEVAL_REFERENCE_FILE_SHA256,
            "seed": HUMANEVAL_SHUFFLE_SEED,
        }
    )
    return GenerativeTask(
        name=HUMANEVAL_TASK_NAME,
        problems=problems,
        source_identity=cache.spec.source_identity,
        dataset_identity=cache.source_identity,
        order_identity=order_identity,
    )


def load_humaneval_task(cache_root: str | Path) -> GenerativeTask:
    """Load the prepared HumanEval test cache without a network fallback."""

    spec = get_humaneval_dataset_spec()
    return build_humaneval_task(load_hub_parquet_cache(spec, cache_root))


def evaluate_humaneval_task(
    model: nn.Module,
    tokenizer: Tokenizer,
    task: GenerativeTask,
    executor: CodeExecutor,
    *,
    allow_generated_code_execution: bool,
    checkpoint_identity: str,
    config: GenerativeEvaluationConfig,
    max_problems: int | None,
    device: str | torch.device,
) -> GenerativeTaskResult:
    """Run shared generation only after explicit local-execution consent."""

    if allow_generated_code_execution is not True:
        raise HumanEvalExecutionDisabledError(
            "HumanEval generated-code execution is disabled; pass the explicit "
            f"opt-in only after accepting that {HUMANEVAL_EXECUTION_WARNING}"
        )
    execution_identity = _validate_executor(executor)
    if not isinstance(task, GenerativeTask) or task.name != HUMANEVAL_TASK_NAME:
        raise TypeError("task must be a HumanEval GenerativeTask")
    if any(not isinstance(problem, HumanEvalProblem) for problem in task.problems):
        raise HumanEvalDatasetError(
            "HumanEval task must contain only HumanEvalProblem values"
        )
    return evaluate_generative_task(
        model,
        tokenizer,
        task,
        lambda problem, completion: score_humaneval_completion(
            problem,
            completion,
            executor,
        ),
        checkpoint_identity=checkpoint_identity,
        config=config,
        max_problems=max_problems,
        device=device,
        scoring_identity=execution_identity,
    )


def _valid_entry_point(value: object) -> bool:
    return (
        isinstance(value, str) and value.isidentifier() and not keyword.iskeyword(value)
    )


def _validate_executor(executor: CodeExecutor) -> str:
    identity = getattr(executor, "identity", None)
    execute = getattr(executor, "execute", None)
    try:
        identity = require_non_empty_string(identity, name="executor.identity")
    except (TypeError, ValueError) as error:
        raise TypeError(str(error)) from error
    if not callable(execute):
        raise TypeError("executor.execute must be callable")
    return identity


def _non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise HumanEvalDatasetRowError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise HumanEvalDatasetRowError(f"{label} must be non-empty")
    return value


def _with_context(context: str, error: Exception) -> HumanEvalDatasetRowError:
    try:
        context = require_non_empty_string(context, name="context")
    except (TypeError, ValueError) as context_error:
        raise ValueError(str(context_error)) from context_error
    return HumanEvalDatasetRowError(f"{context}: {error}")


__all__ = [
    "HUMANEVAL_EXECUTION_REFERENCE_FILE_SHA256",
    "HUMANEVAL_EXECUTION_WARNING",
    "HUMANEVAL_REFERENCE_FILE_SHA256",
    "HUMANEVAL_SHUFFLE_SEED",
    "HUMANEVAL_TASK_NAME",
    "HumanEvalDatasetError",
    "HumanEvalDatasetRowError",
    "HumanEvalExecutionDisabledError",
    "HumanEvalProblem",
    "assemble_humaneval_program",
    "build_humaneval_task",
    "evaluate_humaneval_task",
    "extract_humaneval_program",
    "extract_leading_imports",
    "get_humaneval_dataset_spec",
    "load_humaneval_task",
    "normalize_humaneval_row",
    "score_humaneval_completion",
]
