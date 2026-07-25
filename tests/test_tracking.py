"""Tests for the experiment-tracking contract and local backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scratch_llm.tracking import (
    CompositeTracker,
    JsonlTracker,
    NullTracker,
    Tracker,
    WandbTracker,
    build_tracker,
)
from scratch_llm.config import (
    ProjectConfig,
    RunConfig,
    TrackingConfig,
    WandbConfig,
)
from scratch_llm.run import prepare_run


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_null_tracker_implements_the_contract_without_side_effects(
    tmp_path: Path,
) -> None:
    tracker = NullTracker()

    tracker.log({"loss": 1.25, "nested": {"enabled": True}}, step=0)
    tracker.log_config({"run": {"name": "smoke"}})
    tracker.log_artifact(
        str(tmp_path / "missing.pt"),
        name="checkpoint",
        type="model",
    )
    tracker.finish()
    tracker.finish()

    assert isinstance(tracker, Tracker)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs", "expected"),
    [
        (
            "log",
            (
                {
                    "loss": 1.25,
                    "enabled": False,
                    "tags": ["tiny", "café"],
                    "extra": None,
                },
            ),
            {},
            {
                "record_type": "metrics",
                "metrics": {
                    "loss": 1.25,
                    "enabled": False,
                    "tags": ["tiny", "café"],
                    "extra": None,
                },
            },
        ),
        (
            "log",
            ({"loss": 0.0},),
            {"step": 0},
            {
                "record_type": "metrics",
                "metrics": {"loss": 0.0},
                "step": 0,
            },
        ),
        (
            "log_config",
            ({"run": {"name": "smoke", "seed": 1337}},),
            {},
            {
                "record_type": "config",
                "config": {"run": {"name": "smoke", "seed": 1337}},
            },
        ),
        (
            "log_artifact",
            ("runs/smoke/model.pt", "final-checkpoint", "model"),
            {},
            {
                "record_type": "artifact",
                "path": "runs/smoke/model.pt",
                "name": "final-checkpoint",
                "type": "model",
            },
        ),
    ],
)
def test_jsonl_tracker_writes_valid_flushed_records(
    tmp_path: Path,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    destination = tmp_path / "nested" / "metrics.jsonl"
    tracker = JsonlTracker(destination)

    getattr(tracker, method_name)(*args, **kwargs)

    assert destination.parent.is_dir()
    assert _read_jsonl(destination) == [expected]
    tracker.finish()


def test_jsonl_tracker_appends_across_sessions_and_finish_is_idempotent(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "metrics.jsonl"
    destination.write_text(
        '{"record_type":"metrics","metrics":{"loss":2.0},"step":1}\n',
        encoding="utf-8",
    )

    first = JsonlTracker(destination)
    first.log({"loss": 1.0}, step=2)
    first.finish()
    first.finish()

    second = JsonlTracker(destination)
    second.log({"loss": 0.5})
    second.finish()
    second.finish()

    assert _read_jsonl(destination) == [
        {
            "record_type": "metrics",
            "metrics": {"loss": 2.0},
            "step": 1,
        },
        {
            "record_type": "metrics",
            "metrics": {"loss": 1.0},
            "step": 2,
        },
        {
            "record_type": "metrics",
            "metrics": {"loss": 0.5},
        },
    ]
    assert destination.read_bytes().endswith(b"\n")


def test_jsonl_tracker_rejects_invalid_json_without_appending_a_partial_record(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "metrics.jsonl"
    tracker = JsonlTracker(destination)

    with pytest.raises(ValueError, match="valid JSON"):
        tracker.log({"loss": float("nan")})

    assert destination.read_bytes() == b""
    tracker.finish()


class _RecordingTracker(NullTracker):
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str]],
        *,
        failure: Exception | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.failure = failure
        self.fail_on = fail_on

    def _record(self, method: str) -> None:
        self.events.append((self.name, method))
        if self.failure is not None and (
            self.fail_on is None or self.fail_on == method
        ):
            raise self.failure

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._record("log")

    def log_config(self, config: dict[str, Any]) -> None:
        self._record("log_config")

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self._record("log_artifact")

    def finish(self) -> None:
        self._record("finish")


def test_composite_tracker_forwards_the_lifecycle_in_stable_order() -> None:
    events: list[tuple[str, str]] = []
    first = _RecordingTracker("first", events)
    second = _RecordingTracker("second", events)
    tracker = CompositeTracker(first, second)

    tracker.log({"loss": 1.0}, step=3)
    tracker.log_config({"run": {"name": "smoke"}})
    tracker.log_artifact("model.pt", name="checkpoint", type="model")
    tracker.finish()
    tracker.finish()

    assert isinstance(tracker, Tracker)
    assert events == [
        ("first", "log"),
        ("second", "log"),
        ("first", "log_config"),
        ("second", "log_config"),
        ("first", "log_artifact"),
        ("second", "log_artifact"),
        ("first", "finish"),
        ("second", "finish"),
    ]


def test_composite_tracker_preserves_local_records_and_surfaces_first_failure(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    local_path = tmp_path / "metrics.jsonl"
    local = JsonlTracker(local_path)
    remote_failure = RuntimeError("remote tracker unavailable")
    remote = _RecordingTracker(
        "remote",
        events,
        failure=remote_failure,
        fail_on="log",
    )
    final = _RecordingTracker(
        "final",
        events,
        failure=ValueError("later tracker failure"),
        fail_on="log",
    )
    tracker = CompositeTracker(local, remote, final)

    with pytest.raises(RuntimeError, match="remote tracker unavailable") as caught:
        tracker.log({"loss": 0.75}, step=2)

    assert caught.value is remote_failure
    assert events == [("remote", "log"), ("final", "log")]
    assert _read_jsonl(local_path) == [
        {
            "record_type": "metrics",
            "metrics": {"loss": 0.75},
            "step": 2,
        }
    ]
    tracker.finish()


@pytest.mark.parametrize(
    ("enabled", "mode"),
    [
        (False, "online"),
        (True, "disabled"),
    ],
)
def test_wandb_tracker_disabled_does_not_import_wandb(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    mode: str,
) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    import_attempts: list[str] = []
    real_import = __import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "wandb" or name.startswith("wandb."):
            import_attempts.append(name)
            raise AssertionError("disabled tracking imported wandb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    tracker = WandbTracker(
        project="scratch-llm",
        enabled=enabled,
        mode=mode,
    )

    tracker.log({"loss": 1.0}, step=1)
    tracker.log_config({"run": {"name": "local-only"}})
    tracker.log_artifact("model.pt", name="checkpoint", type="model")
    tracker.finish()
    tracker.finish()

    assert import_attempts == []


class _FakeWandbConfig:
    def __init__(self) -> None:
        self.updates: list[tuple[dict[str, Any], bool]] = []

    def update(
        self,
        config: dict[str, Any],
        *,
        allow_val_change: bool,
    ) -> None:
        self.updates.append((config, allow_val_change))


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
        self.logs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.artifacts: list[_FakeWandbArtifact] = []
        self.finish_calls = 0

    def log(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        self.logs.append((metrics, kwargs))

    def log_artifact(self, artifact: _FakeWandbArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finish_calls += 1


@pytest.mark.parametrize(
    ("mode", "configured_name", "expected_name"),
    [
        ("online", None, "factory-run"),
        ("offline", "explicit-name", "explicit-name"),
    ],
)
def test_tracker_factory_builds_local_first_with_resolved_wandb_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    configured_name: str | None,
    expected_name: str,
) -> None:
    init_calls: list[dict[str, Any]] = []
    run = _FakeWandbRun()
    fake_wandb = ModuleType("wandb")

    def init(**kwargs: Any) -> _FakeWandbRun:
        init_calls.append(kwargs)
        return run

    setattr(fake_wandb, "init", init)
    setattr(fake_wandb, "Artifact", _FakeWandbArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    wandb_dir = tmp_path / "wandb"
    config = ProjectConfig(
        run=RunConfig(
            name="factory-run",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(
            wandb=WandbConfig(
                enabled=True,
                project="factory-project",
                entity="factory-entity",
                group="factory-group",
                name=configured_name,
                tags=["configured"],
                mode=mode,  # type: ignore[arg-type]
                dir=str(wandb_dir),
            )
        ),
    )
    paths = prepare_run(config)

    tracker = build_tracker(config, paths, stage="pretrain")
    tracker.log({"train/loss": 0.5}, step=1)
    tracker.finish()

    metrics_path = paths.run_dir / config.tracking.jsonl.path
    assert isinstance(tracker, CompositeTracker)
    assert _read_jsonl(metrics_path) == [
        {
            "record_type": "metrics",
            "metrics": {"train/loss": 0.5},
            "step": 1,
        }
    ]
    assert run.logs == [({"train/loss": 0.5}, {"step": 1})]
    assert wandb_dir.is_dir()
    assert init_calls == [
        {
            "project": "factory-project",
            "entity": "factory-entity",
            "group": "factory-group",
            "name": expected_name,
            "tags": ["configured", "pipeline-stage:pretrain"],
            "mode": mode,
            "dir": str(wandb_dir),
        }
    ]


def test_tracker_factory_disabled_mode_stays_local_without_importing_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = __import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("disabled factory imported wandb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    config = ProjectConfig(
        run=RunConfig(
            name="local-only",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(wandb=WandbConfig(enabled=True, mode="disabled")),
    )
    paths = prepare_run(config)

    tracker = build_tracker(config, paths, stage="eval_base")
    tracker.log({"eval/val_bpb": 1.25})
    tracker.finish()

    assert _read_jsonl(paths.metrics_dir / "metrics.jsonl") == [
        {
            "record_type": "metrics",
            "metrics": {"eval/val_bpb": 1.25},
        }
    ]


def test_wandb_tracker_maps_the_complete_lifecycle_to_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls: list[dict[str, Any]] = []
    created_artifacts: list[_FakeWandbArtifact] = []
    run = _FakeWandbRun()
    fake_wandb = ModuleType("wandb")

    def init(**kwargs: Any) -> _FakeWandbRun:
        init_calls.append(kwargs)
        return run

    def artifact(*, name: str, type: str) -> _FakeWandbArtifact:
        created = _FakeWandbArtifact(name=name, type=type)
        created_artifacts.append(created)
        return created

    setattr(fake_wandb, "init", init)
    setattr(fake_wandb, "Artifact", artifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = WandbTracker(
        project="scratch-llm",
        entity="research",
        group="3090-pretrain",
        name="smoke",
        tags=["pretrain", "tiny"],
        mode="offline",
        dir=tmp_path,
    )

    tracker.log({"train/loss": 0.5}, step=4)
    tracker.log({"train/epoch": 1.0})
    tracker.log_config({"run": {"name": "smoke"}, "seed": 1337})
    tracker.log_artifact("runs/smoke/last.pt", name="last", type="model")
    tracker.finish()
    tracker.finish()

    assert init_calls == [
        {
            "project": "scratch-llm",
            "entity": "research",
            "group": "3090-pretrain",
            "name": "smoke",
            "tags": ["pretrain", "tiny"],
            "mode": "offline",
            "dir": str(tmp_path),
        }
    ]
    assert run.logs == [
        ({"train/loss": 0.5}, {"step": 4}),
        ({"train/epoch": 1.0}, {}),
    ]
    assert run.config.updates == [({"run": {"name": "smoke"}, "seed": 1337}, True)]
    assert created_artifacts == run.artifacts
    assert [
        (created.name, created.type, created.paths) for created in created_artifacts
    ] == [("last", "model", ["runs/smoke/last.pt"])]
    assert run.finish_calls == 1


def test_wandb_tracker_enabled_requires_the_tracking_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = __import__

    def missing_wandb(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "wandb":
            raise ModuleNotFoundError("No module named 'wandb'", name="wandb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", missing_wandb)

    with pytest.raises(ModuleNotFoundError, match="tracking.*extra"):
        WandbTracker(project="scratch-llm")
