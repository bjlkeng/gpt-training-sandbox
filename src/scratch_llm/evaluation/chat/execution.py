"""Opt-in local Python execution with bounded accidental-damage controls.

This is not a security sandbox. Generated code can use Python's dynamic features,
native extensions, or network access to bypass process-level guards. It is intended
only for trusted, non-adversarial evaluation after explicit operator consent.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import signal
import subprocess
import sys
import tempfile
from typing import BinaryIO, Final, Literal, Protocol, TypeAlias

from scratch_llm._validation import (
    require_finite_positive_real,
    require_positive_integer,
)
from scratch_llm.identity import canonical_json_identity


LOCAL_PYTHON_EXECUTOR_ID: Final = "scratch_llm_local_python_executor_v1"
_INFRASTRUCTURE_SENTINEL: Final = "__SCRATCH_LLM_EXECUTOR_SETUP_FAILED__"
_SIGXFSZ: Final = getattr(signal, "SIGXFSZ", None)
CodeExecutionStatus: TypeAlias = Literal[
    "passed",
    "test_failure",
    "syntax_error",
    "runtime_error",
    "timeout",
    "memory_limit",
    "output_limit",
    "infrastructure_error",
]
_EXECUTION_STATUSES: Final = frozenset(
    {
        "passed",
        "test_failure",
        "syntax_error",
        "runtime_error",
        "timeout",
        "memory_limit",
        "output_limit",
        "infrastructure_error",
    }
)

_GUARD = r"""
import builtins
import os
import shutil
import subprocess
import sys

try:
    if os.name != "posix" or sys.platform == "darwin":
        raise RuntimeError("memory rlimits are unavailable on this platform")
    import resource
    _maximum_memory_bytes = __MAXIMUM_MEMORY_BYTES__
    for _resource_name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_STACK"):
        _resource = getattr(resource, _resource_name)
        resource.setrlimit(_resource, (_maximum_memory_bytes, _maximum_memory_bytes))
    _maximum_output_bytes = __MAXIMUM_OUTPUT_BYTES__
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (_maximum_output_bytes, _maximum_output_bytes),
    )
except BaseException:
    sys.__stderr__.write("__SCRATCH_LLM_EXECUTOR_SETUP_FAILED__\n")
    raise

os.environ["OMP_NUM_THREADS"] = "1"
builtins.exit = None
builtins.quit = None
builtins.help = None
for _name in (
    "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
    "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
    "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
    "lchflags", "lchmod", "lchown", "chdir",
):
    if hasattr(os, _name):
        setattr(os, _name, None)
for _name in ("rmtree", "move", "chown"):
    if hasattr(shutil, _name):
        setattr(shutil, _name, None)
subprocess.Popen = None
"""


@dataclass(frozen=True, slots=True)
class LocalPythonExecutionConfig:
    """Resource limits for one fresh local Python subprocess."""

    timeout_seconds: float = 5.0
    maximum_memory_bytes: int = 256 * 1024 * 1024
    maximum_output_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        timeout_seconds = require_finite_positive_real(
            self.timeout_seconds,
            name="timeout_seconds",
        )
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        require_positive_integer(
            self.maximum_memory_bytes,
            name="maximum_memory_bytes",
        )
        require_positive_integer(
            self.maximum_output_bytes,
            name="maximum_output_bytes",
        )

    def to_dict(self) -> dict[str, object]:
        """Return the exact local execution policy settings."""

        return {
            "maximum_memory_bytes": self.maximum_memory_bytes,
            "maximum_output_bytes_per_stream": self.maximum_output_bytes,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class CodeExecutionResult:
    """One bounded subprocess outcome with captured output kept in memory only."""

    status: CodeExecutionStatus
    stdout: str
    stderr: str
    return_code: int | None

    def __post_init__(self) -> None:
        if self.status not in _EXECUTION_STATUSES:
            raise ValueError(f"unsupported execution status {self.status!r}")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("stdout and stderr must be strings")
        if self.return_code is not None and (
            not isinstance(self.return_code, int) or isinstance(self.return_code, bool)
        ):
            raise TypeError("return_code must be an integer or None")
        if self.status == "passed" and self.return_code != 0:
            raise ValueError("passed execution must have return_code 0")
        if self.status != "passed" and self.return_code == 0:
            raise ValueError("failed execution must not have return_code 0")

    @property
    def success(self) -> bool:
        """Return whether the program and appended tests completed successfully."""

        return self.status == "passed"

    def to_dict(self) -> dict[str, object]:
        """Return content-free metadata without captured stdout or stderr."""

        return {
            "return_code": self.return_code,
            "status": self.status,
            "stderr_bytes": len(self.stderr.encode("utf-8")),
            "stdout_bytes": len(self.stdout.encode("utf-8")),
            "success": self.success,
        }


class CodeExecutor(Protocol):
    """Injected boundary for evaluating one assembled generated program."""

    @property
    def identity(self) -> str:
        """Return the immutable execution-policy identity."""

    def execute(self, program: str) -> CodeExecutionResult:
        """Execute one program and return a normalized outcome."""


@dataclass(frozen=True, slots=True)
class LocalPythonExecutor:
    """Run trusted, non-adversarial code in a fresh resource-limited process."""

    config: LocalPythonExecutionConfig = LocalPythonExecutionConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.config, LocalPythonExecutionConfig):
            raise TypeError("config must be a LocalPythonExecutionConfig")

    @property
    def identity(self) -> str:
        """Return a stable identity for the interpreter and guard policy."""

        return canonical_json_identity(
            {
                "config": self.config.to_dict(),
                "executor_id": LOCAL_PYTHON_EXECUTOR_ID,
                "python_implementation": sys.implementation.name,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            }
        )

    def execute(self, program: str) -> CodeExecutionResult:
        """Execute code locally after the caller has obtained explicit consent."""

        if not isinstance(program, str):
            raise TypeError("program must be a string")
        command = [sys.executable, "-I", "-c", _guarded_program(program, self.config)]
        try:
            with tempfile.TemporaryDirectory(prefix="scratch-llm-humaneval-") as cwd:
                with (
                    tempfile.TemporaryFile(mode="w+b", dir=cwd) as stdout_file,
                    tempfile.TemporaryFile(mode="w+b", dir=cwd) as stderr_file,
                ):
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=cwd,
                            env={"PATH": os.defpath},
                            stdin=subprocess.DEVNULL,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            timeout=self.config.timeout_seconds,
                            check=False,
                            start_new_session=True,
                        )
                    except subprocess.TimeoutExpired:
                        return CodeExecutionResult(
                            status="timeout",
                            stdout=_read_output(stdout_file),
                            stderr=_read_output(stderr_file),
                            return_code=None,
                        )
                    stdout = _read_output(stdout_file)
                    stderr = _read_output(stderr_file)
        except OSError as error:
            return CodeExecutionResult(
                status="infrastructure_error",
                stdout="",
                stderr=str(error),
                return_code=None,
            )

        status = _classify_status(
            completed.returncode,
            stderr,
        )
        return CodeExecutionResult(
            status=status,
            stdout=stdout,
            stderr=stderr,
            return_code=completed.returncode,
        )


def _guarded_program(
    program: str,
    config: LocalPythonExecutionConfig,
) -> str:
    guard = _GUARD.replace(
        "__MAXIMUM_MEMORY_BYTES__",
        str(config.maximum_memory_bytes),
    ).replace(
        "__MAXIMUM_OUTPUT_BYTES__",
        str(config.maximum_output_bytes),
    )
    return (
        guard
        + f"\nexec(compile({program!r}, '<generated>', 'exec'), {{'__name__': '__main__'}})\n"
    )


def _classify_status(
    return_code: int,
    stderr: str,
) -> CodeExecutionStatus:
    if (
        (_SIGXFSZ is not None and return_code == -_SIGXFSZ)
        or "File too large" in stderr
        or "file too large" in stderr
    ):
        return "output_limit"
    if _INFRASTRUCTURE_SENTINEL in stderr:
        return "infrastructure_error"
    if return_code == 0:
        return "passed"
    if "MemoryError" in stderr:
        return "memory_limit"
    if "SyntaxError" in stderr:
        return "syntax_error"
    if "AssertionError" in stderr:
        return "test_failure"
    return "runtime_error"


def _read_output(stream: BinaryIO) -> str:
    stream.seek(0)
    return stream.read().decode("utf-8", errors="replace")


__all__ = [
    "LOCAL_PYTHON_EXECUTOR_ID",
    "CodeExecutionResult",
    "CodeExecutionStatus",
    "CodeExecutor",
    "LocalPythonExecutionConfig",
    "LocalPythonExecutor",
]
