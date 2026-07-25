"""Experiment-tracking contract and always-available local backends."""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TextIO

from scratch_llm.config import ProjectConfig
from scratch_llm.run import RunPaths
from scratch_llm.utils import load_json, save_json


RunStatus = Literal["running", "completed", "failed"]
_RUN_STATUSES = frozenset({"running", "completed", "failed"})
_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "run",
        "status",
        "latest_step",
        "latest_metrics",
    }
)
_RUN_IDENTITY_KEYS = frozenset({"name", "output_dir", "stage"})


class Tracker(ABC):
    """Record one run's JSON-compatible telemetry through a common lifecycle.

    Implementations record metrics, resolved configuration, and artifact
    metadata without mutating caller-owned values. ``finish`` releases any
    resources and must be safe to call more than once.
    """

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Record ``metrics``, associated with ``step`` when one is supplied."""

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Record a resolved run configuration."""

    @abstractmethod
    def log_artifact(self, path: str, name: str, type: str) -> None:
        """Record metadata for the artifact at ``path``."""

    @abstractmethod
    def finish(self) -> None:
        """Flush pending records and release resources idempotently."""


class NullTracker(Tracker):
    """Implement the tracking contract without producing any side effects."""

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Discard a metrics record."""

    def log_config(self, config: dict[str, Any]) -> None:
        """Discard a configuration record."""

    def log_artifact(self, path: str, name: str, type: str) -> None:
        """Discard an artifact record."""

    def finish(self) -> None:
        """Finish without allocating or releasing resources."""


class RunSummary:
    """Atomically maintain one run's compact lifecycle and latest metrics."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run: dict[str, str],
    ) -> None:
        self.path = Path(path)
        self._run = self._validate_run(run)
        self._latest_step: int | None = None
        self._latest_metrics: dict[str, Any] = {}

        if self.path.exists():
            state = self._load_existing()
            self._latest_step = state["latest_step"]
            self._latest_metrics = state["latest_metrics"]
        self._status: RunStatus = "running"
        self._write(
            status=self._status,
            latest_step=self._latest_step,
            latest_metrics=self._latest_metrics,
        )

    @staticmethod
    def _validate_run(run: dict[str, str]) -> dict[str, str]:
        if not isinstance(run, dict) or frozenset(run) != _RUN_IDENTITY_KEYS:
            raise ValueError(
                "run summary identity must contain name, output_dir, and stage"
            )
        copied: dict[str, str] = {}
        for key in ("name", "output_dir", "stage"):
            value = run[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"run summary {key} must be a non-empty string")
            copied[key] = value
        return copied

    @staticmethod
    def _validate_step(step: object) -> int | None:
        if step is None:
            return None
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("run summary latest_step must be a non-negative integer")
        return step

    @staticmethod
    def _validate_scalar_metrics(metrics: object) -> dict[str, Any]:
        if not isinstance(metrics, dict):
            raise ValueError("run summary latest_metrics must be an object")
        copied: dict[str, Any] = {}
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise ValueError("run summary metric names must be strings")
            if not (value is None or isinstance(value, (bool, int, float, str))):
                raise ValueError(f"run summary metric {key!r} must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"run summary metric {key!r} must be finite")
            copied[key] = value
        return copied

    def _load_existing(self) -> dict[str, Any]:
        try:
            state = load_json(self.path)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid run summary {self.path}: {error}") from error
        if not isinstance(state, dict) or frozenset(state) != _SUMMARY_KEYS:
            raise ValueError(f"invalid run summary schema in {self.path}")
        if state["schema_version"] != 1:
            raise ValueError(
                f"unsupported run summary schema version in {self.path}: "
                f"{state['schema_version']!r}"
            )
        existing_run = self._validate_run(state["run"])
        if existing_run != self._run:
            raise ValueError(f"{self.path} belongs to a different run identity")
        status = state["status"]
        if not isinstance(status, str) or status not in _RUN_STATUSES:
            raise ValueError(f"invalid run summary status in {self.path}: {status!r}")
        return {
            "latest_step": self._validate_step(state["latest_step"]),
            "latest_metrics": self._validate_scalar_metrics(state["latest_metrics"]),
        }

    @property
    def status(self) -> RunStatus:
        """Return the last lifecycle status written successfully."""

        return self._status

    def _write(
        self,
        *,
        status: RunStatus,
        latest_step: int | None,
        latest_metrics: dict[str, Any],
    ) -> None:
        save_json(
            {
                "schema_version": 1,
                "run": self._run,
                "status": status,
                "latest_step": latest_step,
                "latest_metrics": latest_metrics,
            },
            self.path,
        )

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Atomically merge the latest scalar metrics and optional step."""

        next_step = self._latest_step
        if step is not None:
            validated_step = self._validate_step(step)
            assert validated_step is not None
            if self._latest_step is not None and validated_step < self._latest_step:
                raise ValueError(
                    "run summary step cannot move backwards from "
                    f"{self._latest_step} to {validated_step}"
                )
            next_step = validated_step

        next_metrics = dict(self._latest_metrics)
        for key, value in metrics.items():
            if value is None or isinstance(value, (bool, int, float, str)):
                validated = self._validate_scalar_metrics({key: value})
                next_metrics.update(validated)

        self._write(
            status=self._status,
            latest_step=next_step,
            latest_metrics=next_metrics,
        )
        self._latest_step = next_step
        self._latest_metrics = next_metrics

    def set_status(self, status: RunStatus) -> None:
        """Atomically update the lifecycle status."""

        if not isinstance(status, str) or status not in _RUN_STATUSES:
            raise ValueError(f"unsupported run summary status: {status!r}")
        self._write(
            status=status,
            latest_step=self._latest_step,
            latest_metrics=self._latest_metrics,
        )
        self._status = status


class JsonlTracker(Tracker):
    """Append each tracking event as one UTF-8 JSON object.

    Records use ``record_type`` as their discriminator. Metric and
    configuration values remain nested under ``metrics`` and ``config`` so
    user keys cannot collide with envelope metadata. A metric ``step`` is
    omitted when callers leave it unspecified.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self._load_existing_records()
        self._stream: TextIO = self.path.open(
            mode="a",
            encoding="utf-8",
            newline="\n",
        )
        self._finished = False
        self._summary: RunSummary | None = None

    def _load_existing_records(self) -> list[dict[str, Any]]:
        try:
            contents = self.path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as error:
            raise ValueError(
                f"could not read tracking file {self.path}: {error}"
            ) from error
        if contents and not contents.endswith(b"\n"):
            raise ValueError(
                f"{self.path} must contain complete newline-delimited JSON records"
            )
        try:
            lines = contents.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ValueError(f"{self.path} is not valid UTF-8: {error}") from error

        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON record at {self.path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"tracking record at {self.path}:{line_number} must be an object"
                )
            records.append(record)
        return records

    def attach_summary(self, summary: RunSummary) -> None:
        """Attach the run-local summary sink before stage metrics are logged."""

        if not isinstance(summary, RunSummary):
            raise TypeError(
                f"summary must be a RunSummary, got {type(summary).__name__}"
            )
        if self._summary is not None:
            raise RuntimeError("a run summary is already attached")
        self._summary = summary

    def _append(self, record: dict[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log after tracker is finished")
        try:
            line = json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"tracking record is not valid JSON: {error}") from error

        self._stream.write(f"{line}\n")
        self._stream.flush()
        self._records.append(json.loads(line))

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append a metrics record and flush it for immediate local visibility."""

        record: dict[str, Any] = {
            "record_type": "metrics",
            "metrics": metrics,
        }
        if step is not None:
            record["step"] = step
        self._append(record)
        if self._summary is not None:
            self._summary.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Append a resolved-configuration record and flush it."""

        self._append(
            {
                "record_type": "config",
                "config": config,
            }
        )

    def log_config_once(self, config: dict[str, Any]) -> bool:
        """Append the first resolved config or validate the one already present."""

        config_records = [
            (index, record)
            for index, record in enumerate(self._records)
            if record.get("record_type") == "config"
        ]
        if config_records:
            if len(config_records) != 1 or config_records[0][0] != 0:
                raise ValueError(
                    f"{self.path} must contain exactly one config record first"
                )
            existing = config_records[0][1].get("config")
            if existing != config:
                raise ValueError(
                    f"{self.path} contains a conflicting resolved configuration"
                )
            return False
        if self._records:
            raise ValueError(
                f"{self.path} contains stage events before its resolved configuration"
            )
        self.log_config(config)
        return True

    def log_artifact(self, path: str, name: str, type: str) -> None:
        """Append an artifact-metadata record and flush it."""

        self._append(
            {
                "record_type": "artifact",
                "path": path,
                "name": name,
                "type": type,
            }
        )

    def finish(self) -> None:
        """Flush and close the JSONL stream, or do nothing when already finished."""

        if self._finished:
            return
        try:
            self._stream.flush()
        finally:
            self._stream.close()
            self._finished = True


class WandbTracker(Tracker):
    """Adapt one optional Weights & Biases run to the tracker contract.

    W&B is imported and initialized only for an enabled, non-disabled tracker,
    so local-only installations never need the optional dependency.
    """

    def __init__(
        self,
        project: str,
        *,
        enabled: bool = True,
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        tags: Sequence[str] = (),
        mode: str = "online",
        dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                f"wandb mode must be 'online', 'offline', or 'disabled', got {mode!r}"
            )

        self._active = enabled and mode != "disabled"
        self._finished = False
        self._wandb: Any = None
        self._run: Any = None
        if not self._active:
            return

        try:
            self._wandb = import_module("wandb")
        except ModuleNotFoundError as error:
            if error.name not in {None, "wandb"}:
                raise
            raise ModuleNotFoundError(
                "W&B tracking requires the optional 'tracking' extra; "
                "install it with `uv sync --extra tracking`"
            ) from error

        init_kwargs: dict[str, Any] = {
            "project": project,
            "tags": list(tags),
            "mode": mode,
        }
        optional_values: dict[str, Any] = {
            "entity": entity,
            "group": group,
            "name": name,
            "dir": str(dir) if dir is not None else None,
        }
        init_kwargs.update(
            {key: value for key, value in optional_values.items() if value is not None}
        )
        self._run = self._wandb.init(**init_kwargs)
        if self._run is None:
            raise RuntimeError("wandb.init did not return a run")

    def _run_for_logging(self) -> Any | None:
        if not self._active:
            return None
        if self._finished:
            raise RuntimeError("cannot log after tracker is finished")
        return self._run

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to the active W&B run, preserving an optional step."""

        run = self._run_for_logging()
        if run is None:
            return
        if step is None:
            run.log(metrics)
        else:
            run.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Update the active run with a resolved configuration."""

        run = self._run_for_logging()
        if run is None:
            return
        run.config.update(config, allow_val_change=True)

    def log_artifact(self, path: str, name: str, type: str) -> None:
        """Upload one file through a named, typed W&B artifact."""

        run = self._run_for_logging()
        if run is None:
            return
        artifact = self._wandb.Artifact(name=name, type=type)
        artifact.add_file(path)
        run.log_artifact(artifact)

    def finish(self) -> None:
        """Finish the active W&B run at most once."""

        if self._finished:
            return
        self._finished = True
        if self._active:
            self._run.finish()


class CompositeTracker(Tracker):
    """Fan out each lifecycle call to child trackers in stable order.

    Every child is attempted even when an earlier child raises. After fan-out,
    the first exception in child order is re-raised so failures remain visible
    while successful local writes are preserved.
    """

    def __init__(self, *trackers: Tracker) -> None:
        if not trackers:
            raise ValueError("CompositeTracker requires at least one child")
        for index, tracker in enumerate(trackers):
            if not isinstance(tracker, Tracker):
                raise TypeError(
                    "CompositeTracker children must be Tracker instances; "
                    f"child {index} is {type(tracker).__name__}"
                )
        self._trackers = trackers
        self._finished = False

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("cannot log after tracker is finished")

    def _fan_out(self, method: str, *args: Any, **kwargs: Any) -> None:
        first_error: Exception | None = None
        for tracker in self._trackers:
            try:
                getattr(tracker, method)(*args, **kwargs)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Forward metrics to every child."""

        self._ensure_open()
        self._fan_out("log", metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Forward a resolved configuration to every child."""

        self._ensure_open()
        self._fan_out("log_config", config)

    def log_artifact(self, path: str, name: str, type: str) -> None:
        """Forward artifact metadata to every child."""

        self._ensure_open()
        self._fan_out("log_artifact", path, name, type)

    def finish(self) -> None:
        """Finish every child once, preserving fan-out when one raises."""

        if self._finished:
            return
        self._finished = True
        self._fan_out("finish")


class RunTracker(CompositeTracker):
    """Own a composite tracker's success/failure lifecycle and run summary."""

    def __init__(self, summary: RunSummary, *trackers: Tracker) -> None:
        if not isinstance(summary, RunSummary):
            raise TypeError(
                f"summary must be a RunSummary, got {type(summary).__name__}"
            )
        super().__init__(*trackers)
        self.summary = summary
        self._lifecycle_finished = False

    def _finish_with_status(self, status: RunStatus) -> None:
        if self._lifecycle_finished:
            return
        self._lifecycle_finished = True
        first_error: Exception | None = None
        try:
            super().finish()
        except Exception as error:
            first_error = error
            status = "failed"
        try:
            self.summary.set_status(status)
        except Exception as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    def finish(self) -> None:
        """Finish every backend and mark the run completed."""

        self._finish_with_status("completed")

    def fail(self) -> None:
        """Finish every backend and mark the run failed."""

        self._finish_with_status("failed")

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del traceback
        if exception_type is None:
            self.finish()
        else:
            try:
                self.fail()
            except Exception as cleanup_error:
                if exception is None:
                    raise
                exception.add_note(f"tracking cleanup also failed: {cleanup_error}")
        return False


def build_tracker(
    config: ProjectConfig,
    paths: RunPaths,
    *,
    stage: str,
) -> RunTracker:
    """Build the run's always-local tracker and optional W&B fan-out.

    JSONL is always the first child. W&B is added only when it is enabled and
    its mode is not ``disabled``. A stable stage tag distinguishes pipeline
    commands while preserving configured tag order.
    """

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    if not isinstance(paths, RunPaths):
        raise TypeError(f"paths must be RunPaths, got {type(paths).__name__}")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    config.validate()

    local = JsonlTracker(paths.run_dir / config.tracking.jsonl.path)
    trackers: list[Tracker] = [local]
    summary: RunSummary | None = None
    try:
        resolved_config = config.to_dict()
        local.log_config_once(resolved_config)
        summary = RunSummary(
            paths.metrics_dir / "summary.json",
            run={
                "name": config.run.name,
                "output_dir": str(paths.run_dir),
                "stage": stage,
            },
        )
        local.attach_summary(summary)

        wandb_config = config.tracking.wandb
        if not wandb_config.enabled or wandb_config.mode == "disabled":
            return RunTracker(summary, *trackers)

        wandb_dir = Path(wandb_config.dir)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        stage_tag = f"pipeline-stage:{stage}"
        tags = list(dict.fromkeys([*wandb_config.tags, stage_tag]))
        remote = WandbTracker(
            project=wandb_config.project,
            entity=wandb_config.entity,
            group=wandb_config.group,
            name=wandb_config.name or config.run.name,
            tags=tags,
            mode=wandb_config.mode,
            dir=wandb_dir,
        )
        trackers.append(remote)
        remote.log_config(resolved_config)
        return RunTracker(summary, *trackers)
    except BaseException as error:
        for tracker in trackers:
            try:
                tracker.finish()
            except Exception as cleanup_error:
                error.add_note(f"tracking cleanup also failed: {cleanup_error}")
        if summary is not None:
            try:
                summary.set_status("failed")
            except Exception as cleanup_error:
                error.add_note(f"summary cleanup also failed: {cleanup_error}")
        raise


__all__ = [
    "CompositeTracker",
    "JsonlTracker",
    "NullTracker",
    "RunStatus",
    "RunSummary",
    "RunTracker",
    "Tracker",
    "WandbTracker",
    "build_tracker",
]
