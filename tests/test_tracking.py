"""Tests for the experiment-tracking contract and local backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scratch_llm.tracking import JsonlTracker, NullTracker, Tracker


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
