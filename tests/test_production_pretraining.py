"""Production regex-BPE pretraining composition and offline integration tests."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch

from scratch_llm.training import pretraining
from scratch_llm.evaluation.base_tracking import (
    FULL_DOCUMENT_MINIMUM_TRAIN_METRIC,
    NANOCHAT_MINIMUM_TRAIN_METRIC,
)
from scratch_llm.training.best_checkpoint import PeriodicValidationResult
from scratch_llm.evaluation.bpb import BPBAccumulation, BaseValidationResult
from scratch_llm.tokenization.bpe import RegexBPETokenizer, train_reference_bpe
from scratch_llm.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    load_model_checkpoint,
    load_training_checkpoint,
)
from scratch_llm.config import (
    DataConfig,
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
    dump_config,
)
from scratch_llm.data.loaders import write_tokenized_parquet_shards
from scratch_llm.generation import generate
from scratch_llm.evaluation.full_document_bpb import (
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_PROTOCOL_VERSION,
    FULL_DOCUMENT_TRAIN_METRIC,
)
from scratch_llm.evaluation.nanochat_bpb import (
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_PROTOCOL_VERSION,
    NANOCHAT_COMPAT_TRAIN_METRIC,
    NANOCHAT_REFERENCE_COMMIT,
)
from scratch_llm.training.pretraining import prepare_pretraining_batch, run_pretraining
from scratch_llm.run import prepare_run
from scratch_llm.data.tokenized import TokenizedDataError, TokenizedShardReader
from scratch_llm.tracking import NullTracker
from scratch_llm.tracking_state import TrackingState
from scratch_llm.training.telemetry import estimate_gpt_training_flops


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"
PRODUCTION_VOCAB_SIZE = 265


class _EventTracker(NullTracker):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        event = (
            "validation-tracker"
            if NANOCHAT_COMPAT_TRAIN_METRIC in metrics
            else "tracker"
        )
        self.events.append(f"{event}:{step}")

    def log_artifact(self, path: str, name: str, type: str) -> None:
        assert type == "model"
        self.events.append(f"artifact:{name}:{path}")

    def checkpoint_state(self) -> TrackingState:
        return TrackingState(backend="wandb", run_id="fixture-run")


class _MetricsTracker(NullTracker):
    def __init__(self) -> None:
        self.records: list[tuple[dict[str, Any], int | None]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.records.append((metrics, step))


def _fake_validation_result(
    *,
    step: int,
    compatibility_bpb: float,
    full_document_bpb: float | None,
) -> PeriodicValidationResult:
    def result(protocol_id: str, bpb: float) -> BaseValidationResult:
        compatibility = protocol_id == NANOCHAT_COMPAT_PROTOCOL_ID
        return BaseValidationResult.from_accumulation(
            BPBAccumulation(
                processed_model_tokens=4,
                counted_target_tokens=2,
                counted_target_bytes=2,
                total_nats=bpb * math.log(2) * 2,
            ),
            protocol_id=protocol_id,
            protocol_version=(
                NANOCHAT_COMPAT_PROTOCOL_VERSION
                if compatibility
                else FULL_DOCUMENT_PROTOCOL_VERSION
            ),
            reference_commit=NANOCHAT_REFERENCE_COMMIT if compatibility else None,
            reference_config={"fixture": protocol_id},
            checkpoint_identity=f"checkpoint:{step}",
            tokenizer_identity="tokenizer:fixture",
            validation_manifest_identity="manifest:fixture",
            source_documents=1,
            source_tokens=2,
            source_bytes=2,
            unique_source_tokens=2,
            unique_source_bytes=2,
        )

    return PeriodicValidationResult(
        compatibility=result(
            NANOCHAT_COMPAT_PROTOCOL_ID,
            compatibility_bpb,
        ),
        full_document=(
            None
            if full_document_bpb is None
            else result(FULL_DOCUMENT_PROTOCOL_ID, full_document_bpb)
        ),
    )


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


def test_periodic_validation_installs_best_before_independent_step_and_last(
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
    config.train.eval_every = 1
    events: list[str] = []
    real_save_checkpoint = pretraining.save_checkpoint

    def validation_runner(step: int) -> PeriodicValidationResult:
        events.append(f"validate:{step}")
        return _fake_validation_result(
            step=step,
            compatibility_bpb=1.5,
            full_document_bpb=1.75,
        )

    def record_checkpoint(path: str | Path, **kwargs: Any) -> Path:
        events.append(f"save:{Path(path).name}:{kwargs['step']}")
        return real_save_checkpoint(path, **kwargs)

    monkeypatch.setattr(pretraining, "save_checkpoint", record_checkpoint)
    result = run_pretraining(
        config,
        paths=prepare_run(config),
        tracker=_EventTracker(events),
        validation_runner=validation_runner,
    )

    assert events == [
        "tracker:1",
        "validate:1",
        "save:best.pt:1",
        "artifact:checkpoint_best:checkpoints/best.pt",
        "validation-tracker:1",
        "save:step_000001.pt:1",
        "artifact:checkpoint_step_000001:checkpoints/step_000001.pt",
        "save:last.pt:1",
        "artifact:checkpoint_latest:checkpoints/last.pt",
        "save:last.pt:1",
    ]
    assert result.validation_state is not None
    assert result.validation_state.validation_step == 1
    assert result.validation_state.current_compatibility_bpb == 1.5
    assert result.validation_state.minimum_compatibility_bpb == 1.5
    assert result.validation_state.current_full_document_bpb == 1.75
    assert result.validation_state.minimum_full_document_bpb == 1.75
    assert len(result.validation_results) == 1
    for name in ("best.pt", "step_000001.pt", "last.pt"):
        payload = torch.load(
            result.paths.checkpoints_dir / name,
            map_location="cpu",
            weights_only=True,
        )
        assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
        assert payload["validation"] == result.validation_state.to_dict()
        assert payload["tracking"] == {
            "backend": "wandb",
            "run_id": "fixture-run",
        }
    resumable_best = load_training_checkpoint(result.paths.checkpoints_dir / "best.pt")
    assert resumable_best.step == 1
    assert resumable_best.continuation is not None
    assert resumable_best.validation == result.validation_state


def test_periodic_validation_forwards_dual_protocol_metrics_on_training_step(
    tmp_path: Path,
) -> None:
    _, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        run_name="tracked-validation",
        max_steps=1,
    )
    config.train.eval_every = 1
    tracker = _MetricsTracker()

    result = run_pretraining(
        config,
        paths=prepare_run(config),
        tracker=tracker,
        validation_runner=lambda step: _fake_validation_result(
            step=step,
            compatibility_bpb=1.5,
            full_document_bpb=1.75,
        ),
    )

    assert result.validation_state is not None
    assert tracker.records[-1] == (
        {
            NANOCHAT_COMPAT_TRAIN_METRIC: 1.5,
            NANOCHAT_MINIMUM_TRAIN_METRIC: 1.5,
            FULL_DOCUMENT_TRAIN_METRIC: 1.75,
            FULL_DOCUMENT_MINIMUM_TRAIN_METRIC: 1.75,
        },
        1,
    )


def test_real_periodic_bpb_protocols_rank_one_bounded_cpu_step(
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "validation-parquet"
    parquet_dir.mkdir()
    pq.write_table(
        pa.table({"text": pa.array(["training fixture"], type=pa.string())}),
        parquet_dir / "shard_00000.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    pq.write_table(
        pa.table({"text": pa.array(["v"] * 1001, type=pa.string())}),
        parquet_dir / "shard_06542.parquet",
        row_group_size=128,
        compression="NONE",
        use_dictionary=False,
    )
    tokenizer = RegexBPETokenizer(
        train_reference_bpe(
            ["validation tokenizer fixture"],
            vocab_size=PRODUCTION_VOCAB_SIZE,
        )
    )
    artifact_dir = tmp_path / "validation-tokenizer"
    tokenizer.save(artifact_dir)
    tokenized_dir = tmp_path / "validation-tokenized"
    write_tokenized_parquet_shards(
        parquet_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=1,
        batch_size=128,
    )
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        run_name="real-validation",
        max_steps=1,
    )
    config.data.parquet_dir = str(parquet_dir)
    config.train.eval_every = 1
    config.train.eval_tokens = 4 * config.train.device_batch_size * config.model.seq_len
    progress: list[str] = []

    result = run_pretraining(
        config,
        paths=prepare_run(config),
        tracker=NullTracker(),
        progress=progress.append,
    )

    assert len(result.validation_results) == 1, progress
    validation = result.validation_results[0]
    assert validation.compatibility.protocol_id == NANOCHAT_COMPAT_PROTOCOL_ID
    assert validation.full_document is not None
    assert validation.full_document.protocol_id == FULL_DOCUMENT_PROTOCOL_ID
    assert validation.compatibility.checkpoint_identity == (
        validation.full_document.checkpoint_identity
    )
    assert validation.compatibility.tokenizer_identity == tokenizer.get_identity()
    assert math.isfinite(validation.compatibility.bpb)
    assert math.isfinite(validation.full_document.bpb)
    assert result.validation_state is not None
    assert result.validation_state.validation_identity == (
        validation.validation_identity
    )
    best = load_model_checkpoint(result.paths.checkpoints_dir / "best.pt")
    assert best.step == 1
    assert best.validation == result.validation_state


def test_best_checkpoint_uses_strict_compatibility_improvement_across_resume(
    tmp_path: Path,
) -> None:
    _, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        run_name="ranking-source",
        max_steps=4,
    )
    config.train.eval_every = 1
    curve = {
        1: (2.0, 3.0),
        2: (1.5, 2.8),
        3: (1.5, 2.4),
        4: (1.8, 2.2),
    }

    def source_runner(step: int) -> PeriodicValidationResult:
        compatibility_bpb, full_document_bpb = curve[step]
        return _fake_validation_result(
            step=step,
            compatibility_bpb=compatibility_bpb,
            full_document_bpb=full_document_bpb,
        )

    source = run_pretraining(
        config,
        paths=prepare_run(config),
        tracker=NullTracker(),
        validation_runner=source_runner,
    )
    source_best = load_model_checkpoint(source.paths.checkpoints_dir / "best.pt")
    assert source_best.step == 2
    assert source.validation_state is not None
    assert source.validation_state.validation_step == 4
    assert source.validation_state.current_compatibility_bpb == 1.8
    assert source.validation_state.minimum_compatibility_bpb == 1.5
    assert source.validation_state.minimum_full_document_bpb == 2.2

    resume_point = source.paths.checkpoints_dir / "step_000002.pt"
    no_improvement_config = deepcopy(config)
    no_improvement_config.run.name = "ranking-resume-no-improvement"
    no_improvement_events: list[int] = []

    def no_improvement_runner(step: int) -> PeriodicValidationResult:
        no_improvement_events.append(step)
        value = 1.5 if step == 3 else 1.6
        return _fake_validation_result(
            step=step,
            compatibility_bpb=value,
            full_document_bpb=2.0,
        )

    no_improvement = run_pretraining(
        no_improvement_config,
        paths=prepare_run(no_improvement_config),
        tracker=NullTracker(),
        resume_from=resume_point,
        validation_runner=no_improvement_runner,
    )
    assert no_improvement_events == [3, 4]
    assert not (no_improvement.paths.checkpoints_dir / "best.pt").exists()
    assert no_improvement.validation_state is not None
    assert no_improvement.validation_state.minimum_compatibility_bpb == 1.5

    improvement_config = deepcopy(config)
    improvement_config.run.name = "ranking-resume-improvement"
    best_save_steps: list[int] = []

    def improvement_runner(step: int) -> PeriodicValidationResult:
        value = 1.5 if step == 3 else 1.4
        return _fake_validation_result(
            step=step,
            compatibility_bpb=value,
            full_document_bpb=2.0,
        )

    improvement = run_pretraining(
        improvement_config,
        paths=prepare_run(improvement_config),
        tracker=NullTracker(),
        resume_from=resume_point,
        validation_runner=improvement_runner,
    )
    improved_best = load_model_checkpoint(improvement.paths.checkpoints_dir / "best.pt")
    best_save_steps.append(improved_best.step)
    assert best_save_steps == [4]
    assert improvement.validation_state is not None
    assert improvement.validation_state.minimum_compatibility_bpb == 1.4


def test_failed_and_partial_validation_preserve_best_and_periodic_checkpoints(
    tmp_path: Path,
) -> None:
    _, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
    config = _production_config(
        tmp_path,
        artifact_dir=artifact_dir,
        tokenized_dir=tokenized_dir,
        run_name="ranking-failures",
        max_steps=4,
    )
    config.train.eval_every = 1
    progress: list[str] = []

    def validation_runner(step: int) -> PeriodicValidationResult:
        if step == 2:
            raise RuntimeError("synthetic validation failure")
        validation = _fake_validation_result(
            step=step,
            compatibility_bpb=1.0 if step == 1 else 0.5,
            full_document_bpb=1.5 if step == 1 else None,
        )
        if step == 4:
            validation = _fake_validation_result(
                step=step,
                compatibility_bpb=0.25,
                full_document_bpb=1.0,
            )
            object.__setattr__(validation.compatibility, "bpb", float("inf"))
        return validation

    result = run_pretraining(
        config,
        paths=prepare_run(config),
        tracker=NullTracker(),
        validation_runner=validation_runner,
        progress=progress.append,
    )

    best = load_model_checkpoint(result.paths.checkpoints_dir / "best.pt")
    assert best.step == 1
    assert result.validation_state is not None
    assert result.validation_state.validation_step == 1
    assert len(result.validation_results) == 3
    assert sorted(
        path.name for path in result.paths.checkpoints_dir.glob("step_*")
    ) == [
        "step_000001.pt",
        "step_000002.pt",
        "step_000003.pt",
        "step_000004.pt",
    ]
    assert any("synthetic validation failure" in message for message in progress)
    assert any(
        "full_documents_v1 result is unavailable" in message for message in progress
    )
    assert any("finite" in message for message in progress)


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
    with TokenizedShardReader(tokenized_dir, tokenizer=tokenizer) as reader:
        training_tokens = reader.manifest.splits["train"].token_count
    step_flops = estimate_gpt_training_flops(config.model).flops_for_tokens(
        config.train.total_batch_size_tokens
    )
    prior_time = 0.0
    for record in metric_records:
        step = record["step"]
        metrics = record["metrics"]
        assert set(metrics) == {
            "train/loss",
            "train/lrm",
            "train/dt",
            "train/tok_per_sec",
            "train/mfu",
            "train/epoch",
            "train/grad_norm",
            "total_training_flops",
            "total_training_time",
        }
        assert metrics["train/dt"] > 0
        assert metrics["train/tok_per_sec"] == pytest.approx(
            config.train.total_batch_size_tokens / metrics["train/dt"]
        )
        assert metrics["train/mfu"] is None
        assert metrics["train/epoch"] == pytest.approx(
            step * config.train.total_batch_size_tokens / training_tokens
        )
        assert metrics["total_training_flops"] == step * step_flops
        assert metrics["total_training_time"] > prior_time
        prior_time = metrics["total_training_time"]
        assert "train/peak_memory_mib" not in metrics

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["validation"] is None
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
    tokenizer, artifact_dir, tokenized_dir = _write_production_inputs(tmp_path)
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
    resume_tracker = _MetricsTracker()
    resumed = run_pretraining(
        resumed_config,
        paths=prepare_run(resumed_config),
        tracker=resume_tracker,
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
    with TokenizedShardReader(
        tokenized_dir,
        tokenizer=tokenizer,
    ) as reader:
        training_token_count = reader.manifest.splits["train"].token_count
    tracked_steps = [
        (metrics, step)
        for metrics, step in resume_tracker.records
        if "train/loss" in metrics
    ]
    assert [step for _, step in tracked_steps] == [3, 4]
    assert [metrics["total_training_flops"] for metrics, _ in tracked_steps] == [
        estimate_gpt_training_flops(uninterrupted_config.model).flops_for_tokens(
            uninterrupted_config.train.total_batch_size_tokens
        )
        * step
        for step in (3, 4)
    ]
    assert [metrics["total_training_time"] for metrics, _ in tracked_steps] == sorted(
        metrics["total_training_time"] for metrics, _ in tracked_steps
    )
    assert [metrics["train/epoch"] for metrics, _ in tracked_steps] == pytest.approx(
        [
            step
            * uninterrupted_config.train.total_batch_size_tokens
            / training_token_count
            for step in (3, 4)
        ]
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
    assert uninterrupted_payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert resumed_payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert uninterrupted_payload["validation"] is None
    assert resumed_payload["validation"] is None
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
