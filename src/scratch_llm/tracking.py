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

from scratch_llm._validation import require_non_negative_integer
from scratch_llm.config import ProjectConfig
from scratch_llm.run import RunPaths
from scratch_llm.tracking_state import TrackingState
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

    def checkpoint_state(self) -> TrackingState | None:
        """Return small resumable remote state, or None for local-only trackers."""

        return None


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
        try:
            return require_non_negative_integer(
                step,
                name="run summary latest_step",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "run summary latest_step must be a non-negative integer"
            ) from error

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
        if any(
            existing_run[field] != self._run[field] for field in ("name", "output_dir")
        ):
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

    def _append_once(self, record: dict[str, Any], *, event_id: str) -> bool:
        if self._finished:
            raise RuntimeError("cannot log after tracker is finished")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        identified_record = {**record, "event_id": event_id}
        existing = [
            candidate
            for candidate in self._records
            if candidate.get("event_id") == event_id
        ]
        if existing:
            if len(existing) != 1 or existing[0] != identified_record:
                raise ValueError(
                    f"{self.path} contains a conflicting event for {event_id!r}"
                )
            return False
        self._append(identified_record)
        return True

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

    def log_once(
        self,
        metrics: dict[str, Any],
        *,
        event_id: str,
        step: int | None = None,
    ) -> bool:
        """Append one identified metrics event or validate its prior record."""

        record: dict[str, Any] = {
            "record_type": "metrics",
            "metrics": metrics,
        }
        if step is not None:
            record["step"] = step
        appended = self._append_once(record, event_id=event_id)
        if appended and self._summary is not None:
            self._summary.log(metrics, step=step)
        return appended

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

    def log_artifact_once(
        self,
        path: str,
        name: str,
        type: str,
        *,
        event_id: str,
    ) -> bool:
        """Append one identified artifact event or validate its prior record."""

        return self._append_once(
            {
                "record_type": "artifact",
                "path": path,
                "name": name,
                "type": type,
            },
            event_id=event_id,
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
        log_dataset_artifacts: bool = True,
        log_tokenizer_artifacts: bool = True,
        log_model_artifacts: bool = False,
        artifact_root: str | os.PathLike[str] | None = None,
        resume_state: TrackingState | None = None,
    ) -> None:
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                f"wandb mode must be 'online', 'offline', or 'disabled', got {mode!r}"
            )
        if not isinstance(log_dataset_artifacts, bool):
            raise TypeError("log_dataset_artifacts must be a boolean")
        if not isinstance(log_tokenizer_artifacts, bool):
            raise TypeError("log_tokenizer_artifacts must be a boolean")
        if not isinstance(log_model_artifacts, bool):
            raise TypeError("log_model_artifacts must be a boolean")
        if resume_state is not None and not isinstance(resume_state, TrackingState):
            raise TypeError("resume_state must be a TrackingState or None")
        if resume_state is not None and resume_state.backend != "wandb":
            raise ValueError(
                f"W&B cannot resume tracking backend {resume_state.backend!r}"
            )

        self._active = enabled and mode != "disabled"
        self._finished = False
        self._wandb: Any = None
        self._run: Any = None
        self._log_dataset_artifacts = log_dataset_artifacts
        self._log_tokenizer_artifacts = log_tokenizer_artifacts
        self._log_model_artifacts = log_model_artifacts
        self._artifact_root = Path(artifact_root) if artifact_root is not None else None
        self._checkpoint_state: TrackingState | None = None
        if resume_state is not None and not self._active:
            raise ValueError("cannot resume W&B state when W&B is disabled")
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
        if resume_state is not None:
            init_kwargs.update(
                {
                    "id": resume_state.run_id,
                    "resume": "must",
                }
            )
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
        run_id = getattr(self._run, "id", None)
        if not isinstance(run_id, str):
            raise RuntimeError(
                "wandb.init returned a run without a valid resumable run id"
            )
        try:
            self._checkpoint_state = TrackingState(
                backend="wandb",
                run_id=run_id,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "wandb.init returned a run without a valid resumable run id"
            ) from error
        if resume_state is not None and self._checkpoint_state != resume_state:
            raise RuntimeError(
                "W&B resumed a different run id than the checkpoint requested"
            )

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
        if type == "dataset" and not self._log_dataset_artifacts:
            return
        if type == "tokenizer" and not self._log_tokenizer_artifacts:
            return
        if type == "model" and not self._log_model_artifacts:
            return
        artifact_path = Path(path)
        if not artifact_path.is_absolute() and self._artifact_root is not None:
            artifact_path = self._artifact_root / artifact_path
        artifact = self._wandb.Artifact(name=name, type=type)
        artifact.add_file(str(artifact_path))
        run.log_artifact(artifact)

    def checkpoint_state(self) -> TrackingState | None:
        """Return the remote run ID needed by an exact checkpoint resume."""

        return self._checkpoint_state

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

    def checkpoint_state(self) -> TrackingState | None:
        """Return the sole remote child state, rejecting ambiguous fan-out."""

        states = [
            state
            for tracker in self._trackers
            if (state := tracker.checkpoint_state()) is not None
        ]
        if len(states) > 1:
            raise RuntimeError("multiple tracker children expose checkpoint state")
        return None if not states else states[0]

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

    def _local_jsonl_tracker(self) -> JsonlTracker:
        if not self._trackers or not isinstance(self._trackers[0], JsonlTracker):
            raise RuntimeError(
                "identified run events require JsonlTracker as the first child"
            )
        return self._trackers[0]

    def log_once(
        self,
        metrics: dict[str, Any],
        *,
        event_id: str,
        step: int | None = None,
    ) -> bool:
        """Record one durable metrics event without duplicating retries."""

        self._ensure_open()
        appended = self._local_jsonl_tracker().log_once(
            metrics,
            event_id=event_id,
            step=step,
        )
        if appended:
            first_error: Exception | None = None
            for tracker in self._trackers[1:]:
                try:
                    tracker.log(metrics, step=step)
                except Exception as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
        return appended

    def log_artifact_once(
        self,
        path: str,
        name: str,
        type: str,
        *,
        event_id: str,
    ) -> bool:
        """Record one durable artifact event without duplicating retries."""

        self._ensure_open()
        appended = self._local_jsonl_tracker().log_artifact_once(
            path,
            name,
            type,
            event_id=event_id,
        )
        if appended:
            first_error: Exception | None = None
            for tracker in self._trackers[1:]:
                try:
                    tracker.log_artifact(path, name, type)
                except Exception as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
        return appended

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
    wandb_resume_state: TrackingState | None = None,
    enable_remote: bool = True,
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
    if wandb_resume_state is not None and not isinstance(
        wandb_resume_state,
        TrackingState,
    ):
        raise TypeError("wandb_resume_state must be a TrackingState or None")
    if not isinstance(enable_remote, bool):
        raise TypeError("enable_remote must be a boolean")

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

        if not enable_remote:
            if wandb_resume_state is not None:
                raise ValueError("cannot resume W&B state when remote tracking is off")
            return RunTracker(summary, *trackers)

        wandb_config = config.tracking.wandb
        if not wandb_config.enabled or wandb_config.mode == "disabled":
            if wandb_resume_state is not None:
                raise ValueError("cannot resume W&B state while W&B is disabled")
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
            log_dataset_artifacts=wandb_config.log_dataset_artifacts,
            log_tokenizer_artifacts=wandb_config.log_tokenizer_artifacts,
            log_model_artifacts=wandb_config.log_model_artifacts,
            artifact_root=paths.run_dir,
            resume_state=wandb_resume_state,
        )
        trackers.append(remote)
        state = remote.checkpoint_state()
        if state is None:  # pragma: no cover - active W&B guarantees this.
            raise RuntimeError("active W&B tracker did not expose resume state")
        state_path = paths.metrics_dir / "tracking_state.json"
        if state_path.exists():
            try:
                existing_state = TrackingState.from_dict(load_json(state_path))
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"invalid tracking state {state_path}: {error}"
                ) from error
            if existing_state != state:
                raise ValueError(
                    f"{state_path} belongs to a different remote tracking run"
                )
        else:
            save_json(state.to_dict(), state_path)
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
