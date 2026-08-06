"""Tests for explicitly opted-in local generated-code execution."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

import scratch_llm.evaluation.chat.execution as execution
from scratch_llm.evaluation.chat.execution import (
    LocalPythonExecutionConfig,
    LocalPythonExecutor,
)


def _executor(
    *,
    timeout_seconds: float = 1.0,
    maximum_memory_bytes: int = 128 * 1024 * 1024,
    maximum_output_bytes: int = 4096,
) -> LocalPythonExecutor:
    return LocalPythonExecutor(
        LocalPythonExecutionConfig(
            timeout_seconds=timeout_seconds,
            maximum_memory_bytes=maximum_memory_bytes,
            maximum_output_bytes=maximum_output_bytes,
        )
    )


def test_local_executor_runs_harmless_code_with_scrubbed_env_and_closed_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCRATCH_LLM_SECRET", "must-not-leak")

    result = _executor().execute(
        "import os, sys\n"
        "print(os.environ.get('SCRATCH_LLM_SECRET'))\n"
        "print(repr(sys.stdin.read()))"
    )

    assert result.success is True
    assert result.status == "passed"
    assert result.stdout == "None\n''\n"
    assert result.stderr == ""
    assert result.return_code == 0


@pytest.mark.parametrize(
    ("program", "status"),
    [
        ("assert False, 'failed test'", "test_failure"),
        ("def broken(:\n    pass", "syntax_error"),
        ("raise ValueError('runtime')", "runtime_error"),
    ],
)
def test_local_executor_distinguishes_python_failures(
    program: str,
    status: str,
) -> None:
    result = _executor().execute(program)

    assert result.success is False
    assert result.status == status
    assert result.return_code != 0


def test_local_executor_kills_timeout_and_caps_captured_output() -> None:
    timed_out = _executor(timeout_seconds=0.1).execute("while True:\n    pass")
    excessive_output = _executor(maximum_output_bytes=256).execute("print('x' * 10000)")

    assert timed_out.status == "timeout"
    assert timed_out.return_code is None
    assert excessive_output.status == "output_limit"
    assert len(excessive_output.stdout.encode("utf-8")) <= 256


@pytest.mark.skipif(
    os.name != "posix" or sys.platform == "darwin",
    reason="the local executor's memory rlimit requires supported POSIX",
)
def test_local_executor_reports_memory_limit() -> None:
    result = _executor(maximum_memory_bytes=64 * 1024 * 1024).execute(
        "payload = bytearray(256 * 1024 * 1024)"
    )

    assert result.status == "memory_limit"
    assert result.success is False


def test_local_executor_uses_and_cleans_a_fresh_temporary_directory() -> None:
    result = _executor().execute(
        "from pathlib import Path\n"
        "Path('created.txt').write_text('temporary', encoding='utf-8')\n"
        "print(Path.cwd())"
    )

    execution_directory = Path(result.stdout.strip())
    assert result.success is True
    assert not execution_directory.exists()


def test_local_executor_distinguishes_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_start(*_args: object, **_kwargs: object) -> None:
        raise OSError("could not start interpreter")

    monkeypatch.setattr(execution.subprocess, "run", fail_to_start)

    result = _executor().execute("pass")

    assert result.status == "infrastructure_error"
    assert result.return_code is None
    assert "could not start interpreter" in result.stderr


def test_execution_serialization_is_content_free_and_identity_is_stable() -> None:
    first_executor = _executor()
    repeated_executor = _executor()
    result = first_executor.execute("print('private output')")

    assert first_executor.identity == repeated_executor.identity
    assert result.to_dict() == {
        "return_code": 0,
        "status": "passed",
        "stderr_bytes": 0,
        "stdout_bytes": len("private output\n"),
        "success": True,
    }
    assert "private output" not in str(result.to_dict())
