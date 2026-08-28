"""Bounded subprocess coverage for the first-sprint train-to-sample workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from scratch_llm.attention_backends import (
    FLASH_CUDA_UNAVAILABLE,
    AttentionBackendError,
)
from scratch_llm.training.checkpoint import load_model_checkpoint
from scratch_llm.config import ProjectConfig, dump_config, load_config
from scratch_llm.run import prepare_run
from scratch_llm.tracking import NullTracker
from scratch_llm.training.pretraining import run_pretraining


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def _run_module(module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _integration_config(tmp_path: Path, *, run_name: str) -> ProjectConfig:
    config = load_config(SMOKE_CONFIG)
    config.run.name = run_name
    config.run.output_dir = str(tmp_path / "runs")
    config.data.profile = "tiny_text"
    config.data.base_dir = str(PROJECT_ROOT / "data")
    config.model.seq_len = 16
    config.model.n_layer = 1
    config.model.n_head = 1
    config.model.n_kv_head = 1
    config.model.n_embd = 16
    config.model.mlp_ratio = 2
    config.train.device_batch_size = 2
    config.train.total_batch_size_tokens = 32
    config.train.max_steps = 6
    config.train.learning_rate = 0.02
    config.train.weight_decay = 0.0
    config.train.warmup_steps = 0
    config.train.warmdown_ratio = 0.0
    config.train.save_every = 1
    config.train.log_every = 1
    config.generation.temperature = 0.0
    config.generation.top_k = 1
    config.generation.max_new_tokens = 8
    config.generation.seed = 17
    config.validate()
    return config


def _metric_records(path: Path) -> list[dict[str, object]]:
    return [
        record
        for line in path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line))["record_type"] == "metrics"
    ]


def test_strict_flash_preflight_fails_before_training_work(tmp_path: Path) -> None:
    config = _integration_config(tmp_path, run_name="strict-flash")
    config.model.attention_backend = "flash"
    config.model.attention_fallback_policy = "error"
    config.validate()
    paths = prepare_run(config)
    progress: list[str] = []

    with pytest.raises(AttentionBackendError, match=FLASH_CUDA_UNAVAILABLE):
        run_pretraining(
            config,
            paths=paths,
            tracker=NullTracker(),
            progress=progress.append,
        )

    assert progress == []
    assert not any(paths.checkpoints_dir.iterdir())


def _assert_nested_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, Mapping):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_nested_state_equal(actual[key], value)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_pretrain_checkpoint_sample_metrics_and_resume_workflow(
    tmp_path: Path,
) -> None:
    fresh_config = _integration_config(tmp_path, run_name="integration-fresh")
    fresh_config_path = dump_config(fresh_config, tmp_path / "fresh.yaml")

    trained = _run_module(
        "scripts.pretrain",
        "--config",
        str(fresh_config_path),
    )

    fresh_run = Path(fresh_config.run.output_dir) / fresh_config.run.name
    metrics_path = fresh_run / fresh_config.tracking.jsonl.path
    final_checkpoint = fresh_run / "checkpoints" / "last.pt"
    resume_checkpoint = fresh_run / "checkpoints" / "step_000005.pt"
    assert trained.returncode == 0, trained.stderr
    assert (fresh_run / "config.yaml").is_file()
    assert final_checkpoint.is_file()
    assert resume_checkpoint.is_file()
    records = _metric_records(metrics_path)
    all_records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == list(range(1, 7))
    assert [record["record_type"] for record in all_records].count("config") == 1
    assert all_records[0] == {
        "record_type": "config",
        "config": fresh_config.to_dict(),
    }
    losses = [float(record["metrics"]["train/loss"]) for record in records]  # type: ignore[index]
    assert losses[-1] < losses[0]
    assert load_model_checkpoint(final_checkpoint).step == 6
    assert json.loads((fresh_run / "metrics" / "summary.json").read_text()) == {
        "schema_version": 1,
        "run": {
            "name": fresh_config.run.name,
            "output_dir": str(fresh_run),
            "stage": "pretrain",
        },
        "status": "completed",
        "latest_step": 6,
        "latest_metrics": records[-1]["metrics"],
    }

    sampled = _run_module(
        "scripts.sample",
        "--checkpoint",
        str(final_checkpoint),
        "--prompt",
        "Byte by byte, ",
    )

    assert sampled.returncode == 0, sampled.stderr
    assert sampled.stdout.startswith("Byte by byte, ")
    assert len(sampled.stdout.strip()) > len("Byte by byte,")
    assert "Traceback" not in sampled.stderr

    resumed_config = _integration_config(tmp_path, run_name="integration-resumed")
    resumed_config_path = dump_config(resumed_config, tmp_path / "resumed.yaml")
    resumed = _run_module(
        "scripts.pretrain",
        "--config",
        str(resumed_config_path),
        "--resume",
        str(resume_checkpoint),
    )

    resumed_run = Path(resumed_config.run.output_dir) / resumed_config.run.name
    resumed_metrics = _metric_records(resumed_run / resumed_config.tracking.jsonl.path)
    resumed_checkpoint = resumed_run / "checkpoints" / "last.pt"
    assert resumed.returncode == 0, resumed.stderr
    assert [record["step"] for record in resumed_metrics] == [6]
    resumed_records = [
        json.loads(line)
        for line in (resumed_run / resumed_config.tracking.jsonl.path)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["record_type"] for record in resumed_records].count("config") == 1
    assert load_model_checkpoint(resumed_checkpoint).step == 6
    assert "Resumed from step 5" in resumed.stdout

    fresh_payload = torch.load(
        final_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    resumed_payload = torch.load(
        resumed_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    for field in ("model", "optimizer", "scheduler"):
        _assert_nested_state_equal(resumed_payload[field], fresh_payload[field])
    assert (
        resumed_payload["continuation"]["loader_state"]
        == fresh_payload["continuation"]["loader_state"]
    )
    assert (
        resumed_payload["continuation"]["rng_state"]
        == fresh_payload["continuation"]["rng_state"]
    )
