"""Experiment-tracking contract and always-available local backends."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TextIO


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


__all__ = ["JsonlTracker", "NullTracker", "Tracker"]
