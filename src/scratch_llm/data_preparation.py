"""Atomic data preparation with idempotent metrics and manifest artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Final

from scratch_llm.climbmix import CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX
from scratch_llm.data import write_tokenized_parquet_shards
from scratch_llm.data_stats import (
    RawDataStatistics,
    compute_raw_data_statistics,
    write_raw_data_statistics,
)
from scratch_llm.tokenized_data import (
    TokenizedDatasetManifest,
    TokenizedShardReader,
)
from scratch_llm.tokenizer import Tokenizer
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.utils import load_json, save_json


DATA_STATS_ARTIFACT: Final = Path("artifacts/data_stats.json")
TOKENIZED_MANIFEST_ARTIFACT: Final = Path("artifacts/tokenized_shard_manifest.json")
_STATE_PATH: Final = Path("artifacts/.data_preparation_state.json")
_STATE_FORMAT: Final = "scratch_llm_data_preparation_state"
_STATE_FORMAT_VERSION: Final = 1
_METRIC_NAMES: Final = frozenset(
    {
        "data/train_shards",
        "data/val_shards",
        "data/train_docs",
        "data/val_docs",
        "data/train_chars",
        "data/val_chars",
        "data/tokenized_train_tokens",
        "data/tokenized_val_tokens",
        "data/shard_write_seconds",
    }
)


class DataPreparationError(ValueError):
    """Tracked data preparation state is corrupt or contradicts current inputs."""


@dataclass(frozen=True)
class TrackedDataPreparationResult:
    """Durable data-preparation outputs and the one coherent metric record."""

    statistics: RawDataStatistics
    manifest: TokenizedDatasetManifest
    metrics: Mapping[str, int | float]
    data_stats_path: Path
    tokenized_manifest_path: Path
    reused_tokenized_data: bool


def prepare_tracked_tokenized_parquet_shards(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Tokenizer,
    tracker: Tracker,
    run_dir: str | Path,
    num_train_shards: int | None = None,
    validation_shard_index: int = CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    batch_size: int = 1024,
    text_column: str = "text",
    overwrite: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> TrackedDataPreparationResult:
    """Write validated shards, artifacts, and one retry-safe metrics record.

    Dataset payloads remain at ``output_dir`` and are never registered as
    artifacts. The two small JSON descriptions live below ``run_dir`` and are
    registered by run-relative path only after the tokenized dataset and both
    artifact files are durable. A private completion state preserves the
    original write duration and lets interrupted tracker fan-out resume without
    duplicate local records.
    """

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must implement Tracker, got {type(tracker).__name__}")
    if not isinstance(overwrite, bool):
        raise TypeError(f"overwrite must be a boolean, got {type(overwrite).__name__}")
    if not callable(clock):
        raise TypeError(f"clock must be callable, got {type(clock).__name__}")

    run_directory = _prepare_run_directory(run_dir)
    tokenized_output = Path(output_dir)
    data_stats_path = run_directory / DATA_STATS_ARTIFACT
    tokenized_manifest_path = run_directory / TOKENIZED_MANIFEST_ARTIFACT
    state_path = run_directory / _STATE_PATH

    statistics = compute_raw_data_statistics(
        data_dir,
        num_train_shards=num_train_shards,
        include_validation=True,
        validation_shard_index=validation_shard_index,
        batch_size=batch_size,
        text_column=text_column,
    )

    if state_path.exists() and overwrite:
        raise DataPreparationError(
            "cannot overwrite a completed tracked data preparation in the same "
            "run directory; use a new run identity"
        )
    if state_path.exists():
        manifest = _load_existing_manifest(tokenized_output, tokenizer=tokenizer)
        _validate_manifest_matches_statistics(manifest, statistics)
        metrics, completed_tracking_events = _load_completion_state(
            state_path,
            statistics=statistics,
            manifest=manifest,
            tokenized_output=tokenized_output,
        )
        _validate_artifact_file(data_stats_path, statistics.to_dict())
        _validate_artifact_file(tokenized_manifest_path, manifest.to_dict())
        reused_tokenized_data = True
    else:
        manifest, shard_write_seconds, reused_tokenized_data = _write_or_reuse_dataset(
            data_dir,
            tokenized_output,
            tokenizer=tokenizer,
            num_train_shards=num_train_shards,
            validation_shard_index=validation_shard_index,
            batch_size=batch_size,
            text_column=text_column,
            overwrite=overwrite,
            clock=clock,
        )
        _validate_manifest_matches_statistics(manifest, statistics)
        write_raw_data_statistics(statistics, data_stats_path)
        save_json(manifest.to_dict(), tokenized_manifest_path)
        _validate_artifact_file(data_stats_path, statistics.to_dict())
        _validate_artifact_file(tokenized_manifest_path, manifest.to_dict())
        metrics = _data_preparation_metrics(
            statistics,
            manifest,
            shard_write_seconds=shard_write_seconds,
        )
        completed_tracking_events = ()
        save_json(
            _completion_state(
                statistics=statistics,
                manifest=manifest,
                metrics=metrics,
                tokenized_output=tokenized_output,
                completed_tracking_events=completed_tracking_events,
            ),
            state_path,
        )

    _record_tracking_events(
        tracker,
        statistics=statistics,
        manifest=manifest,
        metrics=metrics,
        tokenized_output=tokenized_output,
        state_path=state_path,
        completed_tracking_events=completed_tracking_events,
    )

    return TrackedDataPreparationResult(
        statistics=statistics,
        manifest=manifest,
        metrics=MappingProxyType(dict(metrics)),
        data_stats_path=data_stats_path,
        tokenized_manifest_path=tokenized_manifest_path,
        reused_tokenized_data=reused_tokenized_data,
    )


def _prepare_run_directory(path: str | Path) -> Path:
    try:
        directory = Path(path)
    except TypeError as error:
        raise TypeError(
            f"run_dir must be path-like, got {type(path).__name__}"
        ) from error
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise NotADirectoryError(f"run_dir is not a regular directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "artifacts").mkdir(parents=True, exist_ok=True)
    return directory


def _write_or_reuse_dataset(
    data_dir: str | Path,
    output_dir: Path,
    *,
    tokenizer: Tokenizer,
    num_train_shards: int | None,
    validation_shard_index: int,
    batch_size: int,
    text_column: str,
    overwrite: bool,
    clock: Callable[[], float],
) -> tuple[TokenizedDatasetManifest, float, bool]:
    if output_dir.exists() and not overwrite:
        return _load_existing_manifest(output_dir, tokenizer=tokenizer), 0.0, True

    started_at = _clock_value(clock, label="start")
    manifest = write_tokenized_parquet_shards(
        data_dir,
        output_dir,
        tokenizer=tokenizer,
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
        batch_size=batch_size,
        text_column=text_column,
        overwrite=overwrite,
    )
    finished_at = _clock_value(clock, label="finish")
    elapsed = finished_at - started_at
    if elapsed < 0:
        raise ValueError(
            "clock must be monotonic; data shard finish preceded its start"
        )
    return manifest, elapsed, False


def _load_existing_manifest(
    output_dir: Path,
    *,
    tokenizer: Tokenizer,
) -> TokenizedDatasetManifest:
    with TokenizedShardReader(output_dir, tokenizer=tokenizer) as reader:
        return reader.manifest


def _validate_manifest_matches_statistics(
    manifest: TokenizedDatasetManifest,
    statistics: RawDataStatistics,
) -> None:
    expected_sources = {
        "train": statistics.train.selected_shards,
        "val": statistics.validation.selected_shards,
    }
    expected_documents = {
        "train": statistics.train.documents,
        "val": statistics.validation.documents,
    }
    for split in ("train", "val"):
        split_manifest = manifest.splits[split]
        sources = tuple(
            source for shard in split_manifest.shards for source in shard.source_shards
        )
        if sources != expected_sources[split]:
            raise DataPreparationError(
                f"tokenized {split} source shards do not match raw-data selection: "
                f"{sources!r} != {expected_sources[split]!r}"
            )
        if split_manifest.document_count != expected_documents[split]:
            raise DataPreparationError(
                f"tokenized {split} document count contradicts raw-data statistics: "
                f"{split_manifest.document_count} != {expected_documents[split]}"
            )


def _data_preparation_metrics(
    statistics: RawDataStatistics,
    manifest: TokenizedDatasetManifest,
    *,
    shard_write_seconds: float,
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
        "data/train_shards": len(statistics.train.selected_shards),
        "data/val_shards": len(statistics.validation.selected_shards),
        "data/train_docs": statistics.train.documents,
        "data/val_docs": statistics.validation.documents,
        "data/train_chars": statistics.train.characters,
        "data/val_chars": statistics.validation.characters,
        "data/tokenized_train_tokens": manifest.splits["train"].token_count,
        "data/tokenized_val_tokens": manifest.splits["val"].token_count,
        "data/shard_write_seconds": shard_write_seconds,
    }
    if frozenset(metrics) != _METRIC_NAMES:
        raise RuntimeError("data-preparation metrics drifted from the roadmap contract")
    return metrics


def _completion_state(
    *,
    statistics: RawDataStatistics,
    manifest: TokenizedDatasetManifest,
    metrics: Mapping[str, int | float],
    tokenized_output: Path,
    completed_tracking_events: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "artifacts": {
            "data_stats": DATA_STATS_ARTIFACT.as_posix(),
            "tokenized_manifest": TOKENIZED_MANIFEST_ARTIFACT.as_posix(),
        },
        "format": _STATE_FORMAT,
        "format_version": _STATE_FORMAT_VERSION,
        "completed_tracking_events": list(completed_tracking_events),
        "metrics": dict(metrics),
        "raw_data_statistics": statistics.to_dict(),
        "tokenized_manifest": manifest.to_dict(),
        "tokenized_output_dir": str(tokenized_output.resolve()),
    }


def _load_completion_state(
    path: Path,
    *,
    statistics: RawDataStatistics,
    manifest: TokenizedDatasetManifest,
    tokenized_output: Path,
) -> tuple[dict[str, int | float], tuple[str, ...]]:
    try:
        state = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataPreparationError(
            f"could not read data-preparation state {path}: {error}"
        ) from error
    metrics = _state_metrics(state)
    expected_metrics = _data_preparation_metrics(
        statistics,
        manifest,
        shard_write_seconds=float(metrics["data/shard_write_seconds"]),
    )
    if metrics != expected_metrics:
        raise DataPreparationError(
            "data-preparation state metrics contradict current raw-data or "
            "tokenized-manifest totals"
        )
    completed_tracking_events = _state_tracking_events(
        state,
        expected=_tracking_event_ids(statistics, manifest),
    )
    expected = _completion_state(
        statistics=statistics,
        manifest=manifest,
        metrics=metrics,
        tokenized_output=tokenized_output,
        completed_tracking_events=completed_tracking_events,
    )
    if state != expected:
        raise DataPreparationError(
            "data-preparation state contradicts current raw data, tokenized "
            "manifest, artifact paths, or output directory"
        )
    return metrics, completed_tracking_events


def _state_metrics(state: object) -> dict[str, int | float]:
    if not isinstance(state, dict):
        raise DataPreparationError("data-preparation state must be a JSON object")
    metrics = state.get("metrics")
    if not isinstance(metrics, dict) or frozenset(metrics) != _METRIC_NAMES:
        raise DataPreparationError(
            "data-preparation state contains invalid metric fields"
        )
    validated: dict[str, int | float] = {}
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise DataPreparationError(
                f"data-preparation state metric {name!r} is invalid"
            )
        validated[name] = value
    return validated


def _state_tracking_events(
    state: dict[str, Any],
    *,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    raw_events = state.get("completed_tracking_events")
    if not isinstance(raw_events, list) or not all(
        isinstance(event, str) for event in raw_events
    ):
        raise DataPreparationError(
            "data-preparation state completed_tracking_events must be a list of strings"
        )
    events = tuple(raw_events)
    if events != expected[: len(events)]:
        raise DataPreparationError(
            "data-preparation state contains unknown or out-of-order completed "
            "tracking events"
        )
    return events


def _validate_artifact_file(path: Path, expected: dict[str, Any]) -> None:
    try:
        actual = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataPreparationError(
            f"could not validate artifact {path}: {error}"
        ) from error
    if actual != expected:
        raise DataPreparationError(f"artifact {path} contradicts its durable data")


def _event_prefix(
    statistics: RawDataStatistics,
    manifest: TokenizedDatasetManifest,
) -> str:
    payload = json.dumps(
        {
            "raw_data_statistics": statistics.to_dict(),
            "tokenized_manifest": manifest.to_dict(),
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "data-preparation:sha256:" + hashlib.sha256(payload).hexdigest()


def _tracking_event_ids(
    statistics: RawDataStatistics,
    manifest: TokenizedDatasetManifest,
) -> tuple[str, ...]:
    prefix = _event_prefix(statistics, manifest)
    return (
        f"{prefix}:metrics",
        f"{prefix}:artifact:data_stats",
        f"{prefix}:artifact:tokenized_shard_manifest",
    )


def _record_tracking_events(
    tracker: Tracker,
    *,
    statistics: RawDataStatistics,
    manifest: TokenizedDatasetManifest,
    metrics: Mapping[str, int | float],
    tokenized_output: Path,
    state_path: Path,
    completed_tracking_events: tuple[str, ...],
) -> None:
    event_ids = _tracking_event_ids(statistics, manifest)
    completed = list(completed_tracking_events)
    operations: tuple[Callable[[], None], ...] = (
        lambda: _log_metrics_once(
            tracker,
            dict(metrics),
            event_id=event_ids[0],
        ),
        lambda: _log_artifact_once(
            tracker,
            DATA_STATS_ARTIFACT.as_posix(),
            name="data_stats",
            type="dataset",
            event_id=event_ids[1],
        ),
        lambda: _log_artifact_once(
            tracker,
            TOKENIZED_MANIFEST_ARTIFACT.as_posix(),
            name="tokenized_shard_manifest",
            type="dataset",
            event_id=event_ids[2],
        ),
    )
    for event_id, operation in zip(event_ids, operations, strict=True):
        if event_id in completed and not isinstance(tracker, RunTracker):
            continue
        operation()
        if event_id not in completed:
            completed.append(event_id)
            save_json(
                _completion_state(
                    statistics=statistics,
                    manifest=manifest,
                    metrics=metrics,
                    tokenized_output=tokenized_output,
                    completed_tracking_events=tuple(completed),
                ),
                state_path,
            )


def _log_metrics_once(
    tracker: Tracker,
    metrics: dict[str, Any],
    *,
    event_id: str,
) -> None:
    if isinstance(tracker, RunTracker):
        tracker.log_once(metrics, event_id=event_id)
    else:
        tracker.log(metrics)


def _log_artifact_once(
    tracker: Tracker,
    path: str,
    name: str,
    type: str,
    *,
    event_id: str,
) -> None:
    if isinstance(tracker, RunTracker):
        tracker.log_artifact_once(
            path,
            name,
            type,
            event_id=event_id,
        )
    else:
        tracker.log_artifact(path, name, type)


def _clock_value(clock: Callable[[], float], *, label: str) -> float:
    value = clock()
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"clock {label} value must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"clock {label} value must be finite")
    return converted


__all__ = [
    "DATA_STATS_ARTIFACT",
    "TOKENIZED_MANIFEST_ARTIFACT",
    "DataPreparationError",
    "TrackedDataPreparationResult",
    "prepare_tracked_tokenized_parquet_shards",
]
