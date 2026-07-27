"""Tests for tracked raw and tokenized data preparation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from scratch_llm.config import (
    ProjectConfig,
    RunConfig,
    TrackingConfig,
    WandbConfig,
)
from scratch_llm.data_preparation import (
    DataPreparationError,
    prepare_tracked_tokenized_parquet_shards,
)
from scratch_llm.run import prepare_run
from scratch_llm.tokenizer import ByteTokenizer
from scratch_llm.tracking import Tracker, build_tracker
from scratch_llm.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"


class _SpyTracker(Tracker):
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append((metrics, step))

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.artifacts.append((path, name, type))

    def finish(self) -> None:
        pass


class _FailFirstArtifactTracker(_SpyTracker):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def log_artifact(self, path: str, name: str, type: str) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("artifact tracker interrupted")
        super().log_artifact(path, name, type)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_completed_data_preparation_logs_exact_metrics_and_manifest_artifacts(
    tmp_path: Path,
) -> None:
    tracker = _SpyTracker()
    clock_values = iter((10.0, 12.5))

    result = prepare_tracked_tokenized_parquet_shards(
        PARQUET_FIXTURE_DIR,
        tmp_path / "tokenized",
        tokenizer=ByteTokenizer(),
        tracker=tracker,
        run_dir=tmp_path / "run",
        num_train_shards=2,
        clock=lambda: next(clock_values),
    )

    assert result.metrics == {
        "data/train_shards": 2,
        "data/val_shards": 1,
        "data/train_docs": 6,
        "data/val_docs": 3,
        "data/train_chars": 137,
        "data/val_chars": 55,
        "data/tokenized_train_tokens": 147,
        "data/tokenized_val_tokens": 63,
        "data/shard_write_seconds": 2.5,
    }
    assert tracker.metrics == [(result.metrics, None)]
    assert tracker.artifacts == [
        ("artifacts/data_stats.json", "data_stats", "dataset"),
        (
            "artifacts/tokenized_shard_manifest.json",
            "tokenized_shard_manifest",
            "dataset",
        ),
    ]
    assert result.data_stats_path.is_file()
    assert result.tokenized_manifest_path.is_file()
    assert all(".bin" not in path for path, _, _ in tracker.artifacts)
    assert all(".parquet" not in path for path, _, _ in tracker.artifacts)
    assert json.loads(result.data_stats_path.read_text()) == result.statistics.to_dict()
    assert (
        json.loads(result.tokenized_manifest_path.read_text())
        == result.manifest.to_dict()
    )
    assert result.reused_tokenized_data is False
    stable_artifacts = (
        result.data_stats_path.read_bytes(),
        result.tokenized_manifest_path.read_bytes(),
    )

    resumed = prepare_tracked_tokenized_parquet_shards(
        PARQUET_FIXTURE_DIR,
        tmp_path / "tokenized",
        tokenizer=ByteTokenizer(),
        tracker=tracker,
        run_dir=tmp_path / "run",
        num_train_shards=2,
        clock=lambda: (_ for _ in ()).throw(
            AssertionError("generic retry rewrote completed shards")
        ),
    )

    assert resumed.reused_tokenized_data is True
    assert resumed.metrics == result.metrics
    assert len(tracker.metrics) == 1
    assert len(tracker.artifacts) == 2
    assert (
        resumed.data_stats_path.read_bytes(),
        resumed.tokenized_manifest_path.read_bytes(),
    ) == stable_artifacts


def test_local_run_tracker_retries_without_duplicate_or_contradictory_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig(
        run=RunConfig(
            name="data-retry",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        )
    )
    paths = prepare_run(config)
    tokenized_dir = tmp_path / "tokenized"
    first_tracker = build_tracker(config, paths, stage="data_preparation")
    with first_tracker:
        first = prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tokenized_dir,
            tokenizer=ByteTokenizer(),
            tracker=first_tracker,
            run_dir=paths.run_dir,
            num_train_shards=2,
            clock=iter((4.0, 5.25)).__next__,
        )

    def fail_rewrite(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("completed tokenized data was rewritten")

    monkeypatch.setattr(
        "scratch_llm.data_preparation.write_tokenized_parquet_shards",
        fail_rewrite,
    )
    resumed_tracker = build_tracker(config, paths, stage="data_preparation")
    with resumed_tracker:
        resumed = prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tokenized_dir,
            tokenizer=ByteTokenizer(),
            tracker=resumed_tracker,
            run_dir=paths.run_dir,
            num_train_shards=2,
            clock=lambda: (_ for _ in ()).throw(
                AssertionError("completed retry read the clock")
            ),
        )

    records = _read_jsonl(paths.metrics_dir / "metrics.jsonl")
    data_metrics = [
        record
        for record in records
        if record.get("record_type") == "metrics"
        and "data/train_shards" in record["metrics"]
    ]
    data_artifacts = [
        record
        for record in records
        if record.get("record_type") == "artifact" and record.get("type") == "dataset"
    ]

    assert first.metrics == resumed.metrics
    assert first.reused_tokenized_data is False
    assert resumed.reused_tokenized_data is True
    assert len(data_metrics) == 1
    assert len(data_artifacts) == 2
    assert all(record["path"].startswith("artifacts/") for record in data_artifacts)
    assert all("event_id" in record for record in [*data_metrics, *data_artifacts])
    assert [record["record_type"] for record in records].count("config") == 1


class _FakeWandbConfig:
    def update(
        self,
        config: dict[str, Any],
        *,
        allow_val_change: bool,
    ) -> None:
        del config, allow_val_change


class _FakeWandbArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.paths: list[str] = []

    def add_file(self, path: str) -> None:
        self.paths.append(path)


class _FakeWandbRun:
    def __init__(self) -> None:
        self.config = _FakeWandbConfig()
        self.logs: list[dict[str, Any]] = []
        self.artifacts: list[_FakeWandbArtifact] = []

    def log(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        assert kwargs == {}
        self.logs.append(metrics)

    def log_artifact(self, artifact: _FakeWandbArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        pass


@pytest.mark.parametrize("log_dataset_artifacts", [False, True])
def test_wandb_dataset_gate_never_changes_local_manifest_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log_dataset_artifacts: bool,
) -> None:
    run = _FakeWandbRun()
    fake_wandb = ModuleType("wandb")
    setattr(fake_wandb, "init", lambda **kwargs: run)
    setattr(fake_wandb, "Artifact", _FakeWandbArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = ProjectConfig(
        run=RunConfig(
            name=f"dataset-gate-{log_dataset_artifacts}",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(
            wandb=WandbConfig(
                enabled=True,
                mode="offline",
                dir=str(tmp_path / "wandb"),
                log_dataset_artifacts=log_dataset_artifacts,
            )
        ),
    )
    paths = prepare_run(config)
    tracker = build_tracker(config, paths, stage="data_preparation")

    with tracker:
        result = prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tmp_path / "tokenized",
            tokenizer=ByteTokenizer(),
            tracker=tracker,
            run_dir=paths.run_dir,
            num_train_shards=2,
            clock=iter((1.0, 2.0)).__next__,
        )

    local_artifacts = [
        record
        for record in _read_jsonl(paths.metrics_dir / "metrics.jsonl")
        if record.get("record_type") == "artifact"
    ]
    assert [record["path"] for record in local_artifacts] == [
        "artifacts/data_stats.json",
        "artifacts/tokenized_shard_manifest.json",
    ]
    assert run.logs == [dict(result.metrics)]
    if log_dataset_artifacts:
        assert [
            (artifact.name, artifact.type, artifact.paths) for artifact in run.artifacts
        ] == [
            (
                "data_stats",
                "dataset",
                [str(paths.run_dir / "artifacts" / "data_stats.json")],
            ),
            (
                "tokenized_shard_manifest",
                "dataset",
                [str(paths.run_dir / "artifacts" / "tokenized_shard_manifest.json")],
            ),
        ]
    else:
        assert run.artifacts == []


def test_disabled_wandb_path_runs_without_importing_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_import(name: str) -> Any:
        raise AssertionError(f"disabled data preparation imported {name}")

    monkeypatch.setattr("scratch_llm.tracking.import_module", reject_import)
    config = ProjectConfig(
        run=RunConfig(
            name="no-wandb",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(wandb=WandbConfig(enabled=True, mode="disabled")),
    )
    paths = prepare_run(config)
    tracker = build_tracker(config, paths, stage="data_preparation")

    with tracker:
        prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tmp_path / "tokenized",
            tokenizer=ByteTokenizer(),
            tracker=tracker,
            run_dir=paths.run_dir,
            num_train_shards=2,
            clock=iter((0.0, 1.0)).__next__,
        )

    records = _read_jsonl(paths.metrics_dir / "metrics.jsonl")
    assert [record["record_type"] for record in records] == [
        "config",
        "metrics",
        "artifact",
        "artifact",
    ]


def test_failed_shard_write_logs_nothing_and_leaves_no_artifact_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _SpyTracker()

    def fail_write(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("shard write interrupted")

    monkeypatch.setattr(
        "scratch_llm.data_preparation.write_tokenized_parquet_shards",
        fail_write,
    )

    with pytest.raises(OSError, match="shard write interrupted"):
        prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tmp_path / "tokenized",
            tokenizer=ByteTokenizer(),
            tracker=tracker,
            run_dir=tmp_path / "run",
            num_train_shards=2,
            clock=lambda: 1.0,
        )

    assert tracker.metrics == []
    assert tracker.artifacts == []
    assert list((tmp_path / "run" / "artifacts").iterdir()) == []


def test_readme_documents_local_payload_and_remote_artifact_policy() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "`prepare_tracked_tokenized_parquet_shards`" in readme
    assert "nine roadmap `data/*` names" in readme
    assert "tokenized payloads are never registered" in readme
    assert "`tracking.wandb.log_dataset_artifacts` is true" in readme
    assert "does not append duplicate or contradictory totals" in readme


def test_interrupted_artifact_write_reuses_durable_shards_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _SpyTracker()
    real_save_json = save_json

    def fail_manifest_artifact(value: Any, path: str | Path) -> Path:
        if Path(path).name == "tokenized_shard_manifest.json":
            raise OSError("manifest artifact interrupted")
        return real_save_json(value, path)

    monkeypatch.setattr(
        "scratch_llm.data_preparation.save_json",
        fail_manifest_artifact,
    )
    tokenized_dir = tmp_path / "tokenized"

    with pytest.raises(OSError, match="manifest artifact interrupted"):
        prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tokenized_dir,
            tokenizer=ByteTokenizer(),
            tracker=tracker,
            run_dir=tmp_path / "run",
            num_train_shards=2,
            clock=iter((1.0, 2.0)).__next__,
        )

    assert (tokenized_dir / "manifest.json").is_file()
    assert list(tokenized_dir.glob("*.bin"))
    assert (tmp_path / "run" / "artifacts" / "data_stats.json").is_file()
    assert not (
        tmp_path / "run" / "artifacts" / ".data_preparation_state.json"
    ).exists()
    assert tracker.metrics == []
    assert tracker.artifacts == []

    monkeypatch.setattr(
        "scratch_llm.data_preparation.save_json",
        real_save_json,
    )
    result = prepare_tracked_tokenized_parquet_shards(
        PARQUET_FIXTURE_DIR,
        tokenized_dir,
        tokenizer=ByteTokenizer(),
        tracker=tracker,
        run_dir=tmp_path / "run",
        num_train_shards=2,
        clock=lambda: (_ for _ in ()).throw(
            AssertionError("retry rewrote durable tokenized shards")
        ),
    )

    assert result.reused_tokenized_data is True
    assert result.metrics["data/shard_write_seconds"] == 0.0
    assert len(tracker.metrics) == 1
    assert len(tracker.artifacts) == 2


def test_partial_tracker_failure_resumes_missing_events_without_double_counting(
    tmp_path: Path,
) -> None:
    tracker = _FailFirstArtifactTracker()
    tokenized_dir = tmp_path / "tokenized"

    with pytest.raises(RuntimeError, match="artifact tracker interrupted"):
        prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tokenized_dir,
            tokenizer=ByteTokenizer(),
            tracker=tracker,
            run_dir=tmp_path / "run",
            num_train_shards=2,
            clock=iter((1.0, 2.0)).__next__,
        )

    assert len(tracker.metrics) == 1
    assert tracker.artifacts == []
    resumed = prepare_tracked_tokenized_parquet_shards(
        PARQUET_FIXTURE_DIR,
        tokenized_dir,
        tokenizer=ByteTokenizer(),
        tracker=tracker,
        run_dir=tmp_path / "run",
        num_train_shards=2,
        clock=lambda: (_ for _ in ()).throw(
            AssertionError("partial tracking retry rewrote shards")
        ),
    )

    assert resumed.reused_tokenized_data is True
    assert len(tracker.metrics) == 1
    assert len(tracker.artifacts) == 2


def test_contradictory_saved_totals_fail_before_appending_retry_records(
    tmp_path: Path,
) -> None:
    tracker = _SpyTracker()
    prepare_tracked_tokenized_parquet_shards(
        PARQUET_FIXTURE_DIR,
        tmp_path / "tokenized",
        tokenizer=ByteTokenizer(),
        tracker=tracker,
        run_dir=tmp_path / "run",
        num_train_shards=2,
        clock=iter((1.0, 2.0)).__next__,
    )
    tracker.metrics.clear()
    tracker.artifacts.clear()
    state_path = tmp_path / "run" / "artifacts" / ".data_preparation_state.json"
    state = json.loads(state_path.read_text())
    state["metrics"]["data/train_docs"] = 999
    save_json(state, state_path)

    with pytest.raises(DataPreparationError, match="metrics contradict"):
        prepare_tracked_tokenized_parquet_shards(
            PARQUET_FIXTURE_DIR,
            tmp_path / "tokenized",
            tokenizer=ByteTokenizer(),
            tracker=tracker,
            run_dir=tmp_path / "run",
            num_train_shards=2,
            clock=lambda: 0.0,
        )

    assert tracker.metrics == []
    assert tracker.artifacts == []
