"""Experiment-tracking contract and always-available local backends."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, TextIO

from scratch_llm.config import ProjectConfig
from scratch_llm.run import RunPaths


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
        self._stream: TextIO = self.path.open(
            mode="a",
            encoding="utf-8",
            newline="\n",
        )
        self._finished = False

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

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append a metrics record and flush it for immediate local visibility."""

        record: dict[str, Any] = {
            "record_type": "metrics",
            "metrics": metrics,
        }
        if step is not None:
            record["step"] = step
        self._append(record)

    def log_config(self, config: dict[str, Any]) -> None:
        """Append a resolved-configuration record and flush it."""

        self._append(
            {
                "record_type": "config",
                "config": config,
            }
        )

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


def build_tracker(
    config: ProjectConfig,
    paths: RunPaths,
    *,
    stage: str,
) -> CompositeTracker:
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
    wandb_config = config.tracking.wandb
    if not wandb_config.enabled or wandb_config.mode == "disabled":
        return CompositeTracker(*trackers)

    wandb_dir = Path(wandb_config.dir)
    try:
        wandb_dir.mkdir(parents=True, exist_ok=True)
        stage_tag = f"pipeline-stage:{stage}"
        tags = list(dict.fromkeys([*wandb_config.tags, stage_tag]))
        trackers.append(
            WandbTracker(
                project=wandb_config.project,
                entity=wandb_config.entity,
                group=wandb_config.group,
                name=wandb_config.name or config.run.name,
                tags=tags,
                mode=wandb_config.mode,
                dir=wandb_dir,
            )
        )
    except BaseException:
        local.finish()
        raise
    return CompositeTracker(*trackers)


__all__ = [
    "CompositeTracker",
    "JsonlTracker",
    "NullTracker",
    "Tracker",
    "WandbTracker",
    "build_tracker",
]
