"""Command-level local parquet smoke tests for SFT data preparation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["HF_TOKEN"] = "must-not-be-read"
    environment["WANDB_API_KEY"] = "must-not-be-read"
    return subprocess.run(
        [sys.executable, "-m", "scripts.prepare_sft_data", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_smoltalk(path: Path) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ]
        }
        for index in range(6)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)


def _field(output: str, label: str) -> str:
    prefix = f"{label}: "
    return next(
        line.removeprefix(prefix)
        for line in output.splitlines()
        if line.startswith(prefix)
    )


def test_local_parquet_command_is_repeatable_and_dependency_light(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "smoltalk.parquet"
    _write_smoltalk(parquet)
    common = (
        "--dataset",
        "smoltalk",
        "--split",
        "train",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--local-parquet",
        str(parquet),
        "--seed",
        "17",
        "--limit",
        "3",
        "--shuffle-buffer-size",
        "2",
    )

    first = _run(*common)
    second = _run(*common)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "Dataset: smoltalk/default/train" in first.stdout
    assert "Rows: 6" in first.stdout
    assert "Validated conversations: 3" in first.stdout
    assert _field(first.stdout, "Source identity") == _field(
        second.stdout,
        "Source identity",
    )
    assert _field(first.stdout, "Preview identity") == _field(
        second.stdout,
        "Preview identity",
    )
    assert "Traceback" not in first.stderr + second.stderr


def test_dry_run_does_not_create_cache_or_contact_network(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    result = _run(
        "--dataset",
        "gsm8k",
        "--split",
        "test",
        "--cache-dir",
        str(cache_dir),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "Dataset: gsm8k/main/test" in result.stdout
    assert "Dry run: no discovery, download, or cache mutation" in result.stdout
    assert not cache_dir.exists()


def test_command_rejects_dataset_split_mismatch_without_traceback(
    tmp_path: Path,
) -> None:
    result = _run(
        "--dataset",
        "mmlu",
        "--split",
        "train",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--dry-run",
    )

    assert result.returncode != 0
    assert "supported splits for mmlu" in result.stderr
    assert "Traceback" not in result.stderr
