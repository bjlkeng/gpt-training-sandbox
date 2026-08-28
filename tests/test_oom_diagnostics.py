"""Tests for configuration-aware accelerator OOM diagnostics."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

import scratch_llm.training.pretraining as pretraining
import scripts.pretrain as pretrain_script
from scratch_llm.diagnostics.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
)
from scratch_llm.config import dump_config, load_config
from scratch_llm.diagnostics.oom import (
    PretrainingOOMError,
    diagnose_out_of_memory,
)
from scratch_llm.run import prepare_run
from scratch_llm.tracking import NullTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SMOKE_CONFIG = PROJECT_ROOT / "configs" / "base_smoke.yaml"
TINY_SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def _available_memory() -> AcceleratorMemorySnapshot:
    mib = 1024**2
    gib = 1024**3
    return AcceleratorMemorySnapshot(
        device=torch.device("cuda:0"),
        available=True,
        allocated_bytes=8_000 * mib,
        reserved_bytes=9_000 * mib,
        peak_allocated_bytes=10_000 * mib,
        peak_reserved_bytes=11_000 * mib,
        capacity_bytes=24 * gib,
    )


def test_oom_diagnostic_records_attempt_and_ordered_exact_overrides() -> None:
    config = load_config(BASE_SMOKE_CONFIG)
    original = deepcopy(config)
    error = torch.OutOfMemoryError("synthetic allocation failed")

    diagnostic = diagnose_out_of_memory(
        error,
        config=config,
        memory=_available_memory(),
    )

    assert diagnostic is not None
    assert config == original
    assert diagnostic.exception_type == "OutOfMemoryError"
    assert diagnostic.exception_message == "synthetic allocation failed"
    assert diagnostic.attempt.model_profile == config.model.profile
    assert diagnostic.attempt.vocab_size == config.model.vocab_size
    assert diagnostic.attempt.n_layer == config.model.n_layer
    assert diagnostic.attempt.n_head == config.model.n_head
    assert diagnostic.attempt.n_embd == config.model.n_embd
    assert diagnostic.attempt.seq_len == config.model.seq_len
    assert diagnostic.attempt.device_batch_size == config.train.device_batch_size
    assert (
        diagnostic.attempt.total_batch_size_tokens
        == config.train.total_batch_size_tokens
    )
    assert diagnostic.attempt.dtype == config.train.dtype
    assert diagnostic.memory == _available_memory()

    assert [advice.field for advice in diagnostic.recommendations] == [
        "train.device_batch_size",
        "model.seq_len",
        "model.n_embd",
        "model.n_layer",
    ]
    batch_advice = diagnostic.recommendations[0]
    assert batch_advice.current_value == 4
    assert batch_advice.proposed_value == 2
    assert batch_advice.preserves_total_batch_size_tokens is True
    assert batch_advice.resulting_total_batch_size_tokens == 8_192
    assert batch_advice.resulting_grad_accum_steps == 32
    assert batch_advice.cli_overrides == (
        "train.device_batch_size=2",
        "train.grad_accum_steps=32",
    )
    assert batch_advice.cli_example == (
        "--override train.device_batch_size=2 --override train.grad_accum_steps=32"
    )
    assert diagnostic.recommendations[1].cli_overrides == (
        "model.seq_len=64",
        "train.grad_accum_steps=32",
    )
    assert diagnostic.recommendations[2].cli_overrides == ("model.n_embd=64",)
    assert diagnostic.recommendations[3].cli_overrides == ("model.n_layer=1",)

    payload = json.loads(diagnostic.to_json())
    assert payload["schema_version"] == 3
    assert payload["memory"]["allocated_bytes"] == 8_000 * 1024**2
    assert payload["memory"]["capacity_bytes"] == 24 * 1024**3
    assert payload["attempt"]["dtype"] == "float32"
    assert payload["attempt"]["n_kv_head"] == 2
    assert payload["attempt"]["use_gqa"] is False
    assert payload["recommendations"][0]["priority"] == 1
    assert diagnostic.to_json() == diagnostic.to_json()


def test_batch_advice_uses_explicit_valid_budget_when_halving_is_not_divisible() -> (
    None
):
    config = load_config(BASE_SMOKE_CONFIG)
    config.model.seq_len = 8
    config.model.sliding_window_size = 8
    config.train.device_batch_size = 5
    config.train.total_batch_size_tokens = 120
    config.train.grad_accum_steps = "auto"
    config.validate()

    diagnostic = diagnose_out_of_memory(
        torch.OutOfMemoryError("synthetic"),
        config=config,
        memory=collect_accelerator_memory("cpu"),
    )

    assert diagnostic is not None
    batch_advice = diagnostic.recommendations[0]
    assert batch_advice.field == "train.device_batch_size"
    assert batch_advice.proposed_value == 2
    assert batch_advice.preserves_total_batch_size_tokens is False
    assert batch_advice.resulting_total_batch_size_tokens == 112
    assert batch_advice.resulting_grad_accum_steps == 7
    assert batch_advice.cli_overrides == (
        "train.device_batch_size=2",
        "train.total_batch_size_tokens=112",
        "train.grad_accum_steps=7",
    )
    assert "120" in batch_advice.reason
    assert "112" in batch_advice.reason


def test_non_oom_exceptions_are_not_diagnosed() -> None:
    config = load_config(BASE_SMOKE_CONFIG)
    memory = collect_accelerator_memory("cpu")

    assert (
        diagnose_out_of_memory(
            RuntimeError("CUDA out of memory in an untrusted message"),
            config=config,
            memory=memory,
        )
        is None
    )
    assert (
        diagnose_out_of_memory(
            ValueError("not an allocator failure"),
            config=config,
            memory=memory,
        )
        is None
    )


def test_unavailable_memory_is_explicit_and_human_output_is_machine_embedded() -> None:
    config = load_config(BASE_SMOKE_CONFIG)
    memory = collect_accelerator_memory("cpu")
    diagnostic = diagnose_out_of_memory(
        torch.OutOfMemoryError("synthetic"),
        config=config,
        memory=memory,
    )

    assert diagnostic is not None
    payload = diagnostic.to_dict()
    assert payload["memory"] == {
        "allocated_bytes": None,
        "available": False,
        "capacity_bytes": None,
        "device": "cpu",
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "reserved_bytes": None,
        "unavailable_reason": (
            "memory statistics are unavailable for device type 'cpu'"
        ),
    }
    rendered = diagnostic.render()
    assert rendered.count("OOM_DIAGNOSTIC_JSON=") == 1
    assert "Memory snapshot unavailable" in rendered
    assert "was not changed or retried" in rendered


def test_pretraining_oom_clears_partial_gradients_without_advancing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(TINY_SMOKE_CONFIG)
    config.run.name = "oom-boundary"
    config.run.output_dir = str(tmp_path / "runs")
    config.train.max_steps = 1
    config.train.warmup_steps = 0
    config.validate()
    paths = prepare_run(config)
    oom = torch.OutOfMemoryError("synthetic partial backward")
    captured: dict[str, Any] = {}

    def fail_training(
        model: torch.nn.Module,
        batches: object,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        **kwargs: object,
    ) -> None:
        del batches, kwargs
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        captured["optimizer"] = optimizer
        captured["scheduler"] = scheduler
        captured["parameter"] = parameter
        raise oom

    monkeypatch.setattr(pretraining, "run_training_steps", fail_training)

    with pytest.raises(PretrainingOOMError) as raised:
        pretraining.run_pretraining(
            config,
            paths=paths,
            tracker=NullTracker(),
        )

    assert raised.value.__cause__ is oom
    assert raised.value.diagnostic.memory.available is False
    assert captured["parameter"].grad is None
    assert captured["scheduler"].last_epoch == 0
    assert list(paths.checkpoints_dir.iterdir()) == []
    assert not (paths.run_dir / config.tracking.jsonl.path).exists()


def test_non_oom_pretraining_failure_propagates_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(TINY_SMOKE_CONFIG)
    config.run.name = "ordinary-failure"
    config.run.output_dir = str(tmp_path / "runs")
    paths = prepare_run(config)
    failure = RuntimeError("ordinary failure")

    def fail_training(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise failure

    monkeypatch.setattr(pretraining, "run_training_steps", fail_training)

    with pytest.raises(RuntimeError) as raised:
        pretraining.run_pretraining(
            config,
            paths=paths,
            tracker=NullTracker(),
        )

    assert raised.value is failure


def test_pretrain_cli_emits_one_diagnostic_and_does_not_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_config(TINY_SMOKE_CONFIG)
    config.run.name = "cli-oom"
    config.run.output_dir = str(tmp_path / "runs")
    config_path = dump_config(config, tmp_path / "oom.yaml")
    diagnostic = diagnose_out_of_memory(
        torch.OutOfMemoryError("synthetic CLI failure"),
        config=config,
        memory=collect_accelerator_memory("cpu"),
    )
    assert diagnostic is not None
    calls = 0

    def fail_once(*args: object, **kwargs: object) -> None:
        nonlocal calls
        del args, kwargs
        calls += 1
        raise PretrainingOOMError(diagnostic)

    monkeypatch.setattr(pretrain_script, "run_pretraining", fail_once)

    with pytest.raises(SystemExit) as raised:
        pretrain_script.main(
            [
                "--config",
                str(config_path),
                "--no-wandb",
                "--wandb-mode",
                "disabled",
            ]
        )

    captured = capsys.readouterr()
    run_dir = Path(config.run.output_dir) / config.run.name
    summary = json.loads(
        (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert raised.value.code == 2
    assert calls == 1
    assert captured.err.count("OOM_DIAGNOSTIC_JSON=") == 1
    assert "synthetic CLI failure" in captured.err
    assert summary["status"] == "failed"
    assert summary["latest_step"] is None
    assert list((run_dir / "checkpoints").iterdir()) == []


def test_readme_documents_oom_boundary_and_reduction_order() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "OOM_DIAGNOSTIC_JSON" in readme
    assert "does not retry or mutate" in normalized
    assert (
        "device batch size, sequence length, embedding width, then layer count"
        in normalized
    )
    assert "--override train.device_batch_size=2" in readme
    assert "--override train.grad_accum_steps=32" in readme
