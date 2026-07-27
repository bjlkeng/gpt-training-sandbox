"""Tests for deterministic raw ClimbMix data statistics and its CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from scratch_llm.data_stats import (
    compute_raw_data_statistics,
    write_raw_data_statistics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"


def _write_parquet(path: Path, values: list[object], *, column: str = "text") -> None:
    pq.write_table(pa.table({column: values}), path, row_group_size=2)


def test_fixture_statistics_have_exact_split_and_total_counts() -> None:
    result = compute_raw_data_statistics(
        PARQUET_FIXTURE_DIR,
        num_train_shards=2,
        include_validation=True,
    )

    assert result.to_dict() == {
        "bounded": False,
        "format": "scratch_llm_raw_data_statistics",
        "format_version": 1,
        "limits": {
            "document_char_cap": None,
            "max_characters": None,
            "max_documents": None,
        },
        "selection": {
            "data_dir": str(PARQUET_FIXTURE_DIR),
            "include_validation": True,
            "num_train_shards": 2,
            "text_column": "text",
        },
        "splits": {
            "train": {
                "characters": 137,
                "documents": 6,
                "selected_shard_count": 2,
                "selected_shards": [
                    "shard_00000.parquet",
                    "shard_00001.parquet",
                ],
                "utf8_bytes": 147,
            },
            "validation": {
                "characters": 55,
                "documents": 3,
                "selected_shard_count": 1,
                "selected_shards": ["shard_06542.parquet"],
                "utf8_bytes": 63,
            },
        },
        "total": {
            "characters": 192,
            "documents": 9,
            "selected_shard_count": 3,
            "utf8_bytes": 210,
        },
    }


def test_statistics_caps_documents_characters_and_each_document(tmp_path: Path) -> None:
    _write_parquet(
        tmp_path / "shard_00000.parquet",
        ["éé", "", "abcdef", "not observed"],
    )

    result = compute_raw_data_statistics(
        tmp_path,
        num_train_shards=1,
        include_validation=False,
        max_documents=3,
        max_characters=3,
        document_char_cap=2,
        batch_size=1,
    )

    payload = result.to_dict()
    assert payload["bounded"] is True
    assert payload["limits"] == {
        "document_char_cap": 2,
        "max_characters": 3,
        "max_documents": 3,
    }
    assert payload["splits"]["train"] == {
        "characters": 3,
        "documents": 3,
        "selected_shard_count": 1,
        "selected_shards": ["shard_00000.parquet"],
        "utf8_bytes": 5,
    }
    assert payload["splits"]["validation"] == {
        "characters": 0,
        "documents": 0,
        "selected_shard_count": 0,
        "selected_shards": [],
        "utf8_bytes": 0,
    }
    assert payload["total"] == {
        "characters": 3,
        "documents": 3,
        "selected_shard_count": 1,
        "utf8_bytes": 5,
    }


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("max_documents", 0),
        ("max_documents", True),
        ("max_characters", -1),
        ("max_characters", 1.5),
        ("document_char_cap", ""),
        ("batch_size", 0),
    ],
)
def test_statistics_reject_invalid_limits(
    tmp_path: Path,
    argument: str,
    value: object,
) -> None:
    _write_parquet(tmp_path / "shard_00000.parquet", ["text"])

    with pytest.raises((TypeError, ValueError), match=argument):
        invalid_limit: dict[str, Any] = {argument: value}
        compute_raw_data_statistics(
            tmp_path,
            include_validation=False,
            **invalid_limit,
        )


def test_statistics_surface_invalid_schemas_and_unreadable_shards(
    tmp_path: Path,
) -> None:
    missing_column_dir = tmp_path / "missing-column"
    missing_column_dir.mkdir()
    _write_parquet(
        missing_column_dir / "shard_00000.parquet",
        ["value"],
        column="body",
    )

    with pytest.raises(ValueError, match=r"text column 'text'.*not found"):
        compute_raw_data_statistics(
            missing_column_dir,
            include_validation=False,
        )

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "shard_00000.parquet").write_bytes(b"not parquet")
    with pytest.raises(Exception, match=r"Parquet|parquet"):
        compute_raw_data_statistics(corrupt_dir, include_validation=False)


def test_json_report_is_deterministic_atomic_and_preserves_an_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = compute_raw_data_statistics(
        PARQUET_FIXTURE_DIR,
        num_train_shards=1,
        include_validation=True,
    )
    destination = tmp_path / "nested" / "data_stats.json"

    assert write_raw_data_statistics(result, destination) == destination
    first_bytes = destination.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == result.to_dict()
    assert write_raw_data_statistics(result, destination).read_bytes() == first_bytes

    def fail_replace(source: object, target: object) -> None:
        raise OSError("interrupted publication")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted publication"):
        write_raw_data_statistics(result, destination)

    assert destination.read_bytes() == first_bytes
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_data_stats_command_prints_human_summary_and_writes_exact_json(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "data_stats.json"
    command = [
        sys.executable,
        "-m",
        "scripts.data_stats",
        "--data-dir",
        str(PARQUET_FIXTURE_DIR),
        "--num-train-shards",
        "2",
        "--include-val",
        "--output",
        str(destination),
    ]

    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    first_report = destination.read_bytes()
    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == second.returncode == 0
    assert (
        first.stdout
        == second.stdout
        == (
            "train: 2 shards, 6 documents, 137 characters, 147 UTF-8 bytes\n"
            "validation: 1 shard, 3 documents, 55 characters, 63 UTF-8 bytes\n"
            "total: 3 shards, 9 documents, 192 characters, 210 UTF-8 bytes\n"
            "bounded: no\n"
            f"report: {destination}\n"
        )
    )
    assert first.stderr == second.stderr == ""
    assert destination.read_bytes() == first_report
    assert json.loads(first_report)["total"] == {
        "characters": 192,
        "documents": 9,
        "selected_shard_count": 3,
        "utf8_bytes": 210,
    }


def test_data_stats_command_fails_cleanly_without_partial_json(tmp_path: Path) -> None:
    (tmp_path / "shard_00000.parquet").write_bytes(b"broken")
    destination = tmp_path / "data_stats.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.data_stats",
            "--data-dir",
            str(tmp_path),
            "--no-val",
            "--output",
            str(destination),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not destination.exists()


def test_readme_documents_data_stats_selection_and_cap_semantics() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m scripts.data_stats" in readme
    assert "`--doc-cap` counts documents, including empty strings" in readme
    assert "`--max-chars` applies an exact per-split character budget" in readme
    assert "`--doc-cap-chars` truncates each document first" in readme
