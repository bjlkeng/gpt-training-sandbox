"""Tests for run-directory bootstrap and basic logging."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pytest

from scratch_llm.config import (
    ConfigValidationError,
    ProjectConfig,
    RunConfig,
    load_config,
)
from scratch_llm.run import RunConflictError, bootstrap_run, get_run_dir, prepare_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_config(
    output_dir: Path,
    *,
    name: str = "experiment-01",
    seed: int = 1337,
) -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(
            name=name,
            seed=seed,
            device="cpu",
            output_dir=str(output_dir),
        )
    )


def test_prepare_run_creates_layout_and_resolved_snapshot_idempotently(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path / "runs")

    first = prepare_run(config)
    snapshot = first.config_path.read_text(encoding="utf-8")
    second = prepare_run(config)

    assert first == second
    assert first.run_dir == tmp_path / "runs" / config.run.name
    assert first.metrics_dir == first.run_dir / "metrics"
    assert first.checkpoints_dir == first.run_dir / "checkpoints"
    assert first.metrics_dir.is_dir()
    assert first.checkpoints_dir.is_dir()
    assert first.config_path == first.run_dir / "config.yaml"
    assert first.config_path.read_text(encoding="utf-8") == snapshot
    assert load_config(first.config_path) == config
    assert not list(first.run_dir.glob(".config.yaml.*.tmp"))


def test_smoke_config_targets_the_canonical_named_run_directory() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")

    assert get_run_dir(config) == Path("runs") / config.run.name


def test_prepare_run_rejects_a_conflicting_snapshot_without_overwriting(
    tmp_path: Path,
) -> None:
    first_config = _project_config(tmp_path / "runs", seed=7)
    paths = prepare_run(first_config)
    original_snapshot = paths.config_path.read_bytes()
    conflicting_config = _project_config(tmp_path / "runs", seed=8)

    with pytest.raises(RunConflictError, match=r"config\.yaml"):
        prepare_run(conflicting_config)

    assert paths.config_path.read_bytes() == original_snapshot
    assert load_config(paths.config_path) == first_config
    assert not list(paths.run_dir.glob(".config.yaml.*.tmp"))


def test_prepare_run_installs_only_a_complete_snapshot_and_cleans_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project_config(tmp_path / "runs")
    expected_path = tmp_path / "runs" / config.run.name / "config.yaml"
    observed_snapshot = ""

    def fail_install(source: object, destination: object) -> None:
        nonlocal observed_snapshot
        source_path = Path(source)  # type: ignore[arg-type]
        observed_snapshot = source_path.read_text(encoding="utf-8")
        assert Path(destination) == expected_path  # type: ignore[arg-type]
        assert not expected_path.exists()
        raise OSError("snapshot install failed")

    monkeypatch.setattr("scratch_llm.run.os.link", fail_install)

    with pytest.raises(OSError, match="snapshot install failed"):
        prepare_run(config)

    assert "run:\n" in observed_snapshot
    assert "name: experiment-01\n" in observed_snapshot
    assert not expected_path.exists()
    assert not list(expected_path.parent.glob(".config.yaml.*.tmp"))


@pytest.mark.parametrize(
    "name",
    [
        ".",
        "..",
        "../escape",
        "/absolute",
        "nested/run",
        r"nested\run",
        ".hidden",
        "white space",
        "line\nbreak",
    ],
)
def test_run_config_rejects_unsafe_run_names(name: str) -> None:
    with pytest.raises(ConfigValidationError, match=r"^run\.name:.*safe"):
        RunConfig(name=name)


@pytest.mark.parametrize("name", ["smoke", "tiny-20m", "trial_02", "model.v2"])
def test_run_config_accepts_portable_run_names(name: str) -> None:
    assert RunConfig(name=name).name == name


def test_bootstrap_run_logs_once_to_console_and_run_file_on_repeated_setup(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path / "runs")
    console = StringIO()
    logger_name = f"scratch_llm.test_run.{tmp_path.name}"
    logger = logging.getLogger(logger_name)

    try:
        first = bootstrap_run(config, logger_name=logger_name, stream=console)
        second = bootstrap_run(config, logger_name=logger_name, stream=console)

        assert first.paths == second.paths
        assert first.logger is second.logger is logger
        assert len(logger.handlers) == 2
        assert (
            sum(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
            == 1
        )

        logger.info("one message")
        for handler in logger.handlers:
            handler.flush()

        assert console.getvalue().count("one message") == 1
        log_text = first.paths.log_path.read_text(encoding="utf-8")
        assert log_text.count("one message") == 1
    finally:
        for handler in tuple(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
