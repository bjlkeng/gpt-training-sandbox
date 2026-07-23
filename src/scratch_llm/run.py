"""Run-directory layout, resolved configuration snapshots, and basic logging."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from scratch_llm.config import ProjectConfig, RunConfig


_RUN_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_MANAGED_HANDLER_ATTRIBUTE = "_scratch_llm_run_handler"


class RunConflictError(RuntimeError):
    """A run already has a different resolved configuration snapshot."""


@dataclass(frozen=True)
class RunPaths:
    """Filesystem paths owned by one named run."""

    run_dir: Path
    metrics_dir: Path
    checkpoints_dir: Path
    config_path: Path
    log_path: Path


@dataclass(frozen=True)
class RunContext:
    """Prepared run paths and the logger configured for that run."""

    paths: RunPaths
    logger: logging.Logger


def _get_run_config(config: ProjectConfig | RunConfig) -> RunConfig:
    if isinstance(config, ProjectConfig):
        return config.run
    if isinstance(config, RunConfig):
        return config
    raise TypeError(
        f"config must be a ProjectConfig or RunConfig, got {type(config).__name__}"
    )


def get_run_dir(config: ProjectConfig | RunConfig) -> Path:
    """Return the named run directory below the configured output base."""

    run_config = _get_run_config(config)
    run_config.validate()
    return Path(run_config.output_dir) / run_config.name


def _existing_snapshot_matches(path: Path, expected: str) -> bool:
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    if existing != expected:
        raise RunConflictError(
            f"{path} already contains a different resolved configuration"
        )
    return True


def _install_config_snapshot(path: Path, snapshot: str) -> Path:
    """Atomically install ``snapshot`` without replacing an existing file."""

    if _existing_snapshot_matches(path, snapshot):
        return path

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            mode="w",
            encoding="utf-8",
            newline="",
        ) as temporary_file:
            file_descriptor = -1
            temporary_file.write(snapshot)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        try:
            os.link(temporary_path, path)
        except FileExistsError:
            _existing_snapshot_matches(path, snapshot)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        temporary_path.unlink(missing_ok=True)

    return path


def prepare_run(config: ProjectConfig) -> RunPaths:
    """Create one run's directories and resolved configuration snapshot."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    config.validate()

    run_dir = get_run_dir(config)
    paths = RunPaths(
        run_dir=run_dir,
        metrics_dir=run_dir / "metrics",
        checkpoints_dir=run_dir / "checkpoints",
        config_path=run_dir / "config.yaml",
        log_path=run_dir / "run.log",
    )
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _install_config_snapshot(paths.config_path, config.to_yaml())
    return paths


def configure_logging(
    log_path: str | os.PathLike[str],
    *,
    logger_name: str = "scratch_llm",
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one console and one run-local file handler idempotently."""

    if not isinstance(logger_name, str) or not logger_name.strip():
        raise ValueError("logger_name must be a non-empty string")

    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_RUN_LOG_FORMAT)

    console_handler = logging.StreamHandler(stream)
    try:
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(
            destination,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
    except BaseException:
        console_handler.close()
        raise

    for handler in (console_handler, file_handler):
        setattr(handler, _MANAGED_HANDLER_ATTRIBUTE, True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.disabled = False
    logger.propagate = False

    for existing_handler in tuple(logger.handlers):
        if getattr(existing_handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(existing_handler)
            existing_handler.close()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def bootstrap_run(
    config: ProjectConfig,
    *,
    logger_name: str = "scratch_llm",
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
) -> RunContext:
    """Prepare a run and initialize its console/file logger."""

    paths = prepare_run(config)
    logger = configure_logging(
        paths.log_path,
        logger_name=logger_name,
        level=level,
        stream=stream,
    )
    return RunContext(paths=paths, logger=logger)


__all__ = [
    "RunConflictError",
    "RunContext",
    "RunPaths",
    "bootstrap_run",
    "configure_logging",
    "get_run_dir",
    "prepare_run",
]
