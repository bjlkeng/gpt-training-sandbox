"""Production regex-BPE pretraining composition and offline integration tests."""

from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from scratch_llm import pretraining
from scratch_llm.bpe import RegexBPETokenizer, train_reference_bpe
from scratch_llm.checkpoint import CheckpointError, load_model_checkpoint
from scratch_llm.config import (
    DataConfig,
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
    dump_config,
)
from scratch_llm.data import write_tokenized_parquet_shards
from scratch_llm.generation import generate
from scratch_llm.pretraining import prepare_pretraining_batch, run_pretraining
from scratch_llm.run import prepare_run
from scratch_llm.tokenized_data import TokenizedDataError
from scratch_llm.tracking import NullTracker
from scratch_llm.training_telemetry import estimate_gpt_training_flops


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"
PRODUCTION_VOCAB_SIZE = 265


def _write_production_inputs(
    tmp_path: Path,
) -> tuple[RegexBPETokenizer, Path, Path]:
    tokenizer = RegexBPETokenizer(
        train_reference_bpe(
            ["Bounded regex BPE fixture."],
            vocab_size=PRODUCTION_VOCAB_SIZE,
        )
    )
    artifact_dir = tmp_path / "tokenizer"
    tokenizer.save(artifact_dir)
    tokenized_dir = tmp_path / "tokenized"
    write_tokenized_parquet_shards(
        PARQUET_FIXTURE_DIR,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=2,
        batch_size=2,
    )
    return tokenizer, artifact_dir, tokenized_dir


def _production_config(
    tmp_path: Path,
    *,
    artifact_dir: Path,
    tokenized_dir: Path,
    strategy: str = "packed",
    run_name: str = "production",
    max_steps: int = 2,
) -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(
            name=run_name,
            seed=17,
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        data=DataConfig(
            profile="nanochat_climbmix",
            tokenized_dir=str(tokenized_dir),
            loader_strategy=strategy,  # type: ignore[arg-type]
        ),
        tokenizer=TokenizerConfig(
            type="regex_byte_bpe",
            vocab_size=PRODUCTION_VOCAB_SIZE,
            artifact_dir=str(artifact_dir),
        ),
        model=GPTConfig(
            vocab_size=PRODUCTION_VOCAB_SIZE,
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=2,
            total_batch_size_tokens=16,
            grad_accum_steps=1,
            max_steps=max_steps,
            learning_rate=0.01,
            weight_decay=0.0,
            warmup_steps=0,
            warmdown_ratio=0.0,
            save_every=1,
            log_every=1,
        ),
    )


def _run_module(module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_nested_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_nested_state_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_packed_batches_apply_ignore_index_without_mutating_loader_values() -> None:
    inputs = torch.tensor([[4, 5, 6]])
    targets = torch.tensor([[5, 6, 7]])
    loss_mask = torch.tensor([[True, False, True]])

    prepared_inputs, prepared_targets = prepare_pretraining_batch(
        (inputs, targets, loss_mask),
        strategy="packed",
    )

    assert prepared_inputs is inputs
    assert prepared_targets.tolist() == [[5, -1, 7]]
    assert targets.tolist() == [[5, 6, 7]]

    flat_inputs, flat_targets = prepare_pretraining_batch(
        (inputs, targets),
        strategy="flat",
    )
    assert flat_inputs is inputs
    assert flat_targets is targets


def test_production_inputs_are_validated_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    manifest_path = tokenized_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer_identity"] = "sha256:" + "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
    )
    paths = prepare_run(config)

    def fail_model_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"model constructed with {args!r} {kwargs!r}")

    monkeypatch.setattr(pretraining, "GPT", fail_model_construction)

    with pytest.raises(TokenizedDataError, match="tokenizer identity"):
        run_pretraining(config, paths=paths, tracker=NullTracker())

    assert list(paths.checkpoints_dir.iterdir()) == []


def test_reader_is_closed_when_shared_training_logic_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        max_steps=1,
    )
    paths = prepare_run(config)
    close_calls: list[Path] = []
    original_close = pretraining.TokenizedShardReader.close

    def record_close(reader: pretraining.TokenizedShardReader) -> None:
        close_calls.append(reader.dataset_dir)
        original_close(reader)

    def fail_training(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(pretraining.TokenizedShardReader, "close", record_close)
    monkeypatch.setattr(pretraining, "run_training_steps", fail_training)

    with pytest.raises(RuntimeError, match="injected training failure"):
        run_pretraining(config, paths=paths, tracker=NullTracker())

    assert close_calls == [tokenized_dir]
    assert list(paths.checkpoints_dir.iterdir()) == []


@pytest.mark.parametrize("strategy", ["flat", "packed"])
def test_production_profile_supports_each_explicit_loader(
    tmp_path: Path,
    strategy: str,
) -> None:
    tokenizer, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        strategy=strategy,
        run_name=f"production-{strategy}",
        max_steps=1,
    )
    paths = prepare_run(config)

    result = run_pretraining(config, paths=paths, tracker=NullTracker())

    checkpoint = load_model_checkpoint(result.checkpoint_path)
    assert result.final_step == 1
    assert isinstance(checkpoint.tokenizer, RegexBPETokenizer)
    assert checkpoint.tokenizer.get_identity() == tokenizer.get_identity()
    telemetry = result.steps[0].telemetry
    assert telemetry is not None
    assert (
        telemetry.processed_model_tokens == config.train.total_batch_size_tokens == 16
    )
    assert 0 < telemetry.supervised_target_tokens <= 16
    assert telemetry.duration_seconds > 0
    assert telemetry.tokens_per_second == pytest.approx(
        telemetry.processed_model_tokens / telemetry.duration_seconds
    )
    expected_step_flops = estimate_gpt_training_flops(config.model).flops_for_tokens(16)
    assert telemetry.step_flops == expected_step_flops
    assert telemetry.total_training_flops == expected_step_flops
    assert telemetry.total_training_time_seconds == telemetry.duration_seconds
    assert telemetry.mfu is None
    assert telemetry.peak_flops_basis is None
    assert telemetry.peak_memory_mib is None


def test_scripts_pretrain_runs_offline_regex_bpe_to_sample(
    tmp_path: Path,
) -> None:
    tokenizer, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        max_steps=2,
    )
    config_path = dump_config(config, tmp_path / "production.yaml")

    trained = _run_module(
        "scripts.pretrain",
        "--config",
        str(config_path),
        "--no-wandb",
        "--wandb-mode",
        "disabled",
    )

    run_dir = Path(config.run.output_dir) / config.run.name
    metrics_path = run_dir / config.tracking.jsonl.path
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    assert trained.returncode == 0, trained.stderr
    assert metrics_path.is_file()
    assert checkpoint_path.is_file()
    metric_records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["record_type"] == "metrics"
    ]
    assert [record["step"] for record in metric_records] == [1, 2]

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 3
    assert payload["continuation"]["loader_format"] == (
        "scratch_llm_document_packing_loader_state"
    )
    assert payload["continuation"]["tracker_step"] == 2
    assert payload["continuation"]["total_training_flops"] == (
        estimate_gpt_training_flops(config.model).flops_for_tokens(
            config.train.total_batch_size_tokens
        )
        * config.train.max_steps
    )
    assert payload["continuation"]["total_training_time_seconds"] > 0
    assert payload["tokenizer"] == {
        "artifact_path": str(artifact_dir.resolve()),
        "identity": tokenizer.get_identity(),
        "special_tokens": list(config.tokenizer.special_tokens),
        "type": "regex_byte_bpe",
        "vocab_size": PRODUCTION_VOCAB_SIZE,
    }

    loaded = load_model_checkpoint(checkpoint_path)
    periodic = load_model_checkpoint(run_dir / "checkpoints" / "step_000001.pt")
    assert isinstance(loaded.tokenizer, RegexBPETokenizer)
    assert loaded.tokenizer.get_identity() == tokenizer.get_identity()
    assert isinstance(periodic.tokenizer, RegexBPETokenizer)
    assert periodic.tokenizer.get_identity() == tokenizer.get_identity()
    prompt = torch.tensor(
        [loaded.tokenizer.encode("Hi", prepend="<|bos|>")],
        dtype=torch.long,
    )
    generated = generate(
        loaded.model,
        prompt,
        max_new_tokens=3,
        temperature=0,
    )
    assert generated.shape == (1, prompt.shape[1] + 3)
    assert torch.all((0 <= generated) & (generated < PRODUCTION_VOCAB_SIZE))


@pytest.mark.parametrize("strategy", ["flat", "packed"])
def test_exact_resume_matches_uninterrupted_batches_losses_and_state(
    tmp_path: Path,
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    consumed_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_prepare = pretraining.prepare_pretraining_batch

    def record_batch(
        batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        strategy: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = original_prepare(
            batch,
            strategy=strategy,  # type: ignore[arg-type]
        )
        consumed_batches.append((prepared[0].clone(), prepared[1].clone()))
        return prepared

    monkeypatch.setattr(pretraining, "prepare_pretraining_batch", record_batch)
    uninterrupted_config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        strategy=strategy,
        run_name=f"uninterrupted-{strategy}",
        max_steps=4,
    )
    uninterrupted = run_pretraining(
        uninterrupted_config,
        paths=prepare_run(uninterrupted_config),
        tracker=NullTracker(),
    )
    uninterrupted_batches = tuple(consumed_batches)
    consumed_batches.clear()
    interruption = uninterrupted.paths.checkpoints_dir / "step_000002.pt"

    resumed_config = deepcopy(uninterrupted_config)
    resumed_config.run.name = f"resumed-{strategy}"
    resumed = run_pretraining(
        resumed_config,
        paths=prepare_run(resumed_config),
        tracker=NullTracker(),
        resume_from=interruption,
    )

    assert len(uninterrupted_batches) == 4
    assert len(consumed_batches) == 2
    for resumed_batch, uninterrupted_batch in zip(
        consumed_batches,
        uninterrupted_batches[2:],
        strict=True,
    ):
        _assert_nested_state_equal(resumed_batch, uninterrupted_batch)
    assert resumed.initial_step == 2
    assert resumed.final_step == uninterrupted.final_step == 4
    assert [result.loss for result in resumed.steps] == pytest.approx(
        [result.loss for result in uninterrupted.steps[2:]],
        rel=0,
        abs=0,
    )
    assert [result.grad_norm for result in resumed.steps] == pytest.approx(
        [result.grad_norm for result in uninterrupted.steps[2:]],
        rel=0,
        abs=0,
    )

    uninterrupted_payload = torch.load(
        uninterrupted.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    resumed_payload = torch.load(
        resumed.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert uninterrupted_payload["format_version"] == 3
    assert resumed_payload["format_version"] == 3
    for field in ("model", "optimizer", "scheduler"):
        _assert_nested_state_equal(
            resumed_payload[field],
            uninterrupted_payload[field],
        )
    assert (
        resumed_payload["continuation"]["loader_state"]
        == uninterrupted_payload["continuation"]["loader_state"]
    )
    assert (
        resumed_payload["continuation"]["rng_state"]
        == uninterrupted_payload["continuation"]["rng_state"]
    )
    assert resumed_payload["continuation"]["tracker_step"] == 4
    expected_total_flops = (
        estimate_gpt_training_flops(uninterrupted_config.model).flops_for_tokens(
            uninterrupted_config.train.total_batch_size_tokens
        )
        * uninterrupted_config.train.max_steps
    )
    assert (
        resumed_payload["continuation"]["total_training_flops"]
        == uninterrupted_payload["continuation"]["total_training_flops"]
        == expected_total_flops
    )
    assert resumed_payload["continuation"]["total_training_time_seconds"] >= 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("identity", "sha256:" + "0" * 64, "metadata conflicts"),
        (
            "artifact_path",
            "/different/tokenizer",
            "artifact path conflicts",
        ),
    ],
)
def test_regex_checkpoint_rejects_tokenizer_metadata_conflicts(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    _, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        max_steps=1,
    )
    result = run_pretraining(
        config,
        paths=prepare_run(config),
        tracker=NullTracker(),
    )
    payload = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    payload["tokenizer"][field] = value
    conflicting_path = tmp_path / f"conflicting-{field}.pt"
    torch.save(payload, conflicting_path)

    with pytest.raises(CheckpointError, match=message):
        load_model_checkpoint(conflicting_path)


def test_production_config_rejects_unsupported_loader_strategy(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"data\.loader_strategy"):
        DataConfig(loader_strategy="streaming")  # type: ignore[arg-type]


def test_missing_or_incomplete_production_artifacts_are_actionable(
    tmp_path: Path,
) -> None:
    missing_artifacts = tmp_path / "missing-tokenizer"
    missing_data = tmp_path / "missing-tokenized"
    config = _production_config(
        tmp_path,
        artifact_dir=missing_artifacts,
        tokenized_dir=missing_data,
    )
    paths = prepare_run(config)

    with pytest.raises(
        FileNotFoundError,
        match=rf"tokenizer artifact directory.*{missing_artifacts}",
    ):
        run_pretraining(config, paths=paths, tracker=NullTracker())

    missing_artifacts.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        run_pretraining(config, paths=paths, tracker=NullTracker())
