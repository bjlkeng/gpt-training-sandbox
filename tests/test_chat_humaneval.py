"""Tests for the opt-in HumanEval-style generative task adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from torch import nn

from scratch_llm.chat.conversation import UserMessage
from scratch_llm.data.hub import publish_local_parquet_cache
from scratch_llm.evaluation.chat.execution import (
    CodeExecutionResult,
    LocalPythonExecutionConfig,
    LocalPythonExecutor,
)
from scratch_llm.evaluation.chat.generative import GenerativeEvaluationConfig
from scratch_llm.evaluation.chat.humaneval import (
    HUMANEVAL_EXECUTION_WARNING,
    HUMANEVAL_SHUFFLE_SEED,
    HumanEvalDatasetRowError,
    HumanEvalExecutionDisabledError,
    assemble_humaneval_program,
    build_humaneval_task,
    evaluate_humaneval_task,
    extract_humaneval_program,
    extract_leading_imports,
    get_humaneval_dataset_spec,
    load_humaneval_task,
    normalize_humaneval_row,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer


class _CompletionModel(nn.Module):
    max_seq_len = 1024

    def __init__(self, tokenizer: ByteTokenizer) -> None:
        super().__init__()
        self.assistant_start = tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = tokenizer.encode_special("<|assistant_end|>")
        self.forward_count = 0

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        logits = torch.full(
            (*token_ids.shape, 265),
            -torch.inf,
            device=token_ids.device,
        )
        sequence = (*b"pass", self.assistant_end)
        for row, values in enumerate(token_ids.detach().cpu().tolist()):
            start = len(values) - 1 - values[::-1].index(self.assistant_start)
            generated_count = len(values) - start - 1
            logits[row, -1, sequence[generated_count]] = 0
        return logits


class _FakeExecutor:
    identity = "sha256:fake-execution-policy"

    def __init__(self, statuses: tuple[str, ...]) -> None:
        self.statuses = list(statuses)
        self.programs: list[str] = []

    def execute(self, program: str) -> CodeExecutionResult:
        self.programs.append(program)
        status = self.statuses.pop(0)
        return CodeExecutionResult(
            status=status,  # type: ignore[arg-type]
            stdout="fake stdout with private content",
            stderr="",
            return_code=0 if status == "passed" else 1,
        )


def _row(index: int) -> dict[str, str]:
    return {
        "canonical_solution": "\n    return 'REFERENCE_ONLY'\n",
        "entry_point": f"function_{index}",
        "prompt": (
            "# trusted dataset prompt\n"
            "import math\n"
            "from typing import Iterable\n\n"
            f"def function_{index}(values: Iterable[int]):\n"
            '    """Return a value."""\n'
        ),
        "test": f"def check(candidate):\n    assert candidate is function_{index}\n",
    }


def _cache(tmp_path: Path, rows: list[dict[str, str]]):
    parquet_path = tmp_path / "humaneval.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    return publish_local_parquet_cache(
        get_humaneval_dataset_spec(),
        tmp_path / "cache",
        (parquet_path,),
    )


def test_humaneval_spec_pins_the_named_test_cache_contract() -> None:
    spec = get_humaneval_dataset_spec()

    assert spec.repository == "openai/openai_humaneval"
    assert spec.subset == "openai_humaneval"
    assert spec.split == "test"
    assert spec.required_columns == (
        "prompt",
        "canonical_solution",
        "test",
        "entry_point",
    )
    assert spec.reference_commit == ("92d63d4e8bb4df75c3b71618f31ddde2378b2bcd")


def test_normalize_humaneval_row_preserves_prompt_tests_imports_and_identity() -> None:
    row = _row(3)

    problem = normalize_humaneval_row(
        row,
        source_row=8,
        source_identity="source",
        context="fixture row 8",
    )

    assert problem.source_row == 8
    assert problem.entry_point == "function_3"
    assert problem.test_program == row["test"]
    assert problem.required_imports == "import math\nfrom typing import Iterable"
    assert problem.identity.startswith("sha256:")
    assert problem.conversation.messages == (UserMessage(row["prompt"]),)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("  def plain():\n    pass  ", "def plain():\n    pass"),
        (
            "Explanation\n```python\ndef first():\n    return 1\n```\nAfter",
            "def first():\n    return 1",
        ),
        ("```\ndef untyped():\n    return 2\n```", "def untyped():\n    return 2"),
        (
            "```javascript\nnope\n```\n```python\ndef chosen():\n    pass\n```",
            "def chosen():\n    pass",
        ),
    ],
)
def test_extract_humaneval_program_uses_first_python_or_untyped_block(
    completion: str,
    expected: str,
) -> None:
    assert extract_humaneval_program(completion) == expected


def test_program_assembly_adds_imports_tests_and_check_without_reference_solution() -> (
    None
):
    problem = normalize_humaneval_row(
        _row(1),
        source_row=1,
        source_identity="source",
        context="fixture row 1",
    )

    program = assemble_humaneval_program(problem, "def function_1(values):\n    pass")

    assert program.startswith("import math\nfrom typing import Iterable\n\n")
    assert "def function_1(values):\n    pass" in program
    assert problem.test_program in program
    assert program.endswith("check(function_1)")
    assert "REFERENCE_ONLY" not in program
    assert extract_leading_imports(_row(1)["prompt"]) == problem.required_imports


def test_assembled_controlled_fixture_runs_through_the_local_executor() -> None:
    row = {
        "canonical_solution": "\n    return value * value\n",
        "entry_point": "square",
        "prompt": 'import math\n\ndef square(value):\n    """Square value."""\n',
        "test": "def check(candidate):\n    assert candidate(4) == 16\n",
    }
    problem = normalize_humaneval_row(
        row,
        source_row=0,
        source_identity="controlled-source",
        context="controlled row 0",
    )
    executor = LocalPythonExecutor(LocalPythonExecutionConfig(timeout_seconds=1.0))

    result = executor.execute(
        assemble_humaneval_program(
            problem,
            "def square(value):\n    return value * value",
        )
    )

    assert result.status == "passed"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt", " ", "prompt must be non-empty"),
        ("canonical_solution", 2, "canonical_solution must be a string"),
        ("test", "", "test must be non-empty"),
        ("entry_point", "not-valid()", "valid Python identifier"),
    ],
)
def test_normalize_humaneval_row_rejects_malformed_rows(
    field: str,
    value: object,
    message: str,
) -> None:
    row: dict[str, object] = dict(_row(0))
    row[field] = value

    with pytest.raises(HumanEvalDatasetRowError, match=message):
        normalize_humaneval_row(
            row,
            source_row=0,
            source_identity="source",
            context="fixture row 0",
        )


def test_humaneval_task_uses_seed_42_order_and_offline_cache(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(10)]
    cache = _cache(tmp_path, rows)

    first = build_humaneval_task(cache)
    repeated = load_humaneval_task(tmp_path / "cache")

    expected_order = tuple(
        int(index)
        for index in np.random.default_rng(HUMANEVAL_SHUFFLE_SEED).permutation(10)
    )
    assert tuple(problem.source_row for problem in first.problems) == expected_order
    assert repeated == first
    assert first.name == "HumanEval"
    assert first.source_identity == cache.spec.source_identity
    assert first.dataset_identity == cache.source_identity


def test_humaneval_refuses_before_generation_or_execution_without_opt_in(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    model = _CompletionModel(tokenizer)
    executor = _FakeExecutor(("passed",))
    task = build_humaneval_task(_cache(tmp_path, [_row(0)]))

    with pytest.raises(
        HumanEvalExecutionDisabledError,
        match="not safe for malicious or adversarial code",
    ):
        evaluate_humaneval_task(
            model,
            tokenizer,
            task,
            executor,
            allow_generated_code_execution=False,
            checkpoint_identity="checkpoint",
            config=GenerativeEvaluationConfig(
                num_samples=1,
                max_new_tokens=5,
                temperature=0,
            ),
            max_problems=None,
            device="cpu",
        )

    assert model.forward_count == 0
    assert executor.programs == []
    assert "not safe for malicious or adversarial code" in HUMANEVAL_EXECUTION_WARNING


def test_humaneval_uses_shared_pass_any_and_content_free_execution_outcomes(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    model = _CompletionModel(tokenizer)
    task = build_humaneval_task(_cache(tmp_path, [_row(0), _row(1)]))
    executor = _FakeExecutor(("test_failure", "passed", "syntax_error", "timeout"))

    result = evaluate_humaneval_task(
        model,
        tokenizer,
        task,
        executor,
        allow_generated_code_execution=True,
        checkpoint_identity="checkpoint",
        config=GenerativeEvaluationConfig(
            num_samples=2,
            max_new_tokens=5,
            temperature=0,
            seed=9,
        ),
        max_problems=None,
        device="cpu",
    )

    assert result.passed_count == 1
    assert result.evaluated_count == 2
    assert result.total_sample_count == 4
    assert result.scoring_identity == executor.identity
    assert result.score_outcome_counts == {
        "passed": 1,
        "syntax_error": 1,
        "test_failure": 1,
        "timeout": 1,
    }
    assert len(executor.programs) == 4
    for problem_index, problem in enumerate(task.problems):
        programs = executor.programs[problem_index * 2 : problem_index * 2 + 2]
        assert all(
            program.endswith(f"check({problem.entry_point})") for program in programs
        )
    assert all("REFERENCE_ONLY" not in program for program in executor.programs)
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "fake stdout with private content" not in serialized
    assert "REFERENCE_ONLY" not in serialized
    assert executor.identity in serialized
