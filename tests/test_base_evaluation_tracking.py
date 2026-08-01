"""Base-evaluation reporting and tracker fan-out tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from torch import nn

from scratch_llm.evaluation.base import BaseEvaluationContext, CompletedBaseEvaluation
from scratch_llm.evaluation.base_tracking import (
    BASE_EVALUATION_ARTIFACT_NAME,
    BASE_EVALUATION_ARTIFACT_TYPE,
    BASE_EVALUATION_REPORT_FORMAT,
    BASE_EVALUATION_REPORT_FORMAT_VERSION,
    BASE_SAMPLES_ARTIFACT_NAME,
    BaseEvaluationReportConflictError,
    FULL_DOCUMENT_MINIMUM_TRAIN_METRIC,
    FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC,
    NANOCHAT_MINIMUM_TRAIN_METRIC,
    NANOCHAT_SOURCE_BYTE_RETENTION_METRIC,
    report_base_samples,
    report_completed_base_evaluation,
    report_standalone_base_evaluation,
    track_periodic_base_validation,
)
from scratch_llm.evaluation.sampling import (
    FixedBaseSamplingConfig,
    generate_fixed_base_samples,
)
from scratch_llm.training.best_checkpoint import (
    PeriodicValidationResult,
    ValidationCheckpointState,
    advance_validation_state,
)
from scratch_llm.evaluation.bpb import BPBAccumulation, BaseValidationResult
from scratch_llm.config import (
    ProjectConfig,
    RunConfig,
    TrackingConfig,
    WandbConfig,
)
from scratch_llm.evaluation.core.results import (
    CoreEvaluationResult,
    CoreReferenceComparison,
    CoreTaskResult,
)
from scratch_llm.evaluation.core.tracking import (
    CORE_COMPARISON_ARTIFACT_NAME,
    CORE_EVAL_METRIC,
    CORE_MAX_PER_TASK_METRIC,
    CORE_RUN_KIND_METRIC,
    CORE_TASK_COUNT_METRIC,
)
from scratch_llm.data.loaders import write_tokenized_parquet_shards
from scratch_llm.evaluation.full_document_bpb import (
    FULL_DOCUMENT_EVAL_METRIC,
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_PROTOCOL_VERSION,
    FULL_DOCUMENT_TRAIN_METRIC,
    FullDocumentProtocolConfig,
    evaluate_full_document_bpb,
)
from scratch_llm.evaluation.nanochat_bpb import (
    NANOCHAT_COMPAT_EVAL_METRIC,
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_PROTOCOL_VERSION,
    NANOCHAT_COMPAT_TRAIN_METRIC,
    NANOCHAT_REFERENCE_COMMIT,
    NanochatCompatibilityConfig,
    evaluate_nanochat_compatible_bpb,
)
from scratch_llm.run import prepare_run
from scratch_llm.data.tokenized import TokenizedShardReader
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from scratch_llm.tokenization.artifacts import build_token_byte_lengths
from scratch_llm.tracking import Tracker, build_tracker
from tests.fixtures.bpb_conformance import BPB_CONFORMANCE_FIXTURE


class _SpyTracker(Tracker):
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append((metrics, step))

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.artifacts.append((path, name, type))

    def finish(self) -> None:
        pass


class _UnitLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        assert loss_reduction == "none"
        return torch.ones_like(inputs, dtype=torch.float64) + self.anchor * 0


class _ConstantCompletionModel(nn.Module):
    def __init__(self, token_id: int) -> None:
        super().__init__()
        self.max_seq_len = 128
        self.vocab_size = ByteTokenizer().get_vocab_size()
        self.token_id = token_id
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.vocab_size),
            -torch.inf,
            device=token_ids.device,
        )
        logits[:, -1, self.token_id] = self.anchor
        return logits


class _FakeWandbConfig:
    def update(
        self,
        config: dict[str, Any],
        *,
        allow_val_change: bool,
    ) -> None:
        del config, allow_val_change


class _FakeWandbArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.paths: list[str] = []

    def add_file(self, path: str) -> None:
        self.paths.append(path)


class _FakeWandbRun:
    def __init__(self) -> None:
        self.id = "base-eval-run"
        self.config = _FakeWandbConfig()
        self.logs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.artifacts: list[_FakeWandbArtifact] = []

    def log(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        self.logs.append((metrics, kwargs))

    def log_artifact(self, artifact: _FakeWandbArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        pass


def _protocol_result(
    *,
    protocol_id: str,
    bpb: float,
    unique_source_bytes: int,
) -> BaseValidationResult:
    compatibility = protocol_id == NANOCHAT_COMPAT_PROTOCOL_ID
    return BaseValidationResult.from_accumulation(
        BPBAccumulation(
            processed_model_tokens=16,
            counted_target_tokens=unique_source_bytes,
            counted_target_bytes=unique_source_bytes,
            total_nats=bpb * math.log(2) * unique_source_bytes,
        ),
        protocol_id=protocol_id,
        protocol_version=(
            NANOCHAT_COMPAT_PROTOCOL_VERSION
            if compatibility
            else FULL_DOCUMENT_PROTOCOL_VERSION
        ),
        reference_commit=NANOCHAT_REFERENCE_COMMIT if compatibility else None,
        reference_config={"fixture": protocol_id},
        checkpoint_identity="checkpoint:fixture",
        tokenizer_identity="tokenizer:fixture",
        validation_manifest_identity="manifest:fixture",
        source_documents=1,
        source_tokens=10,
        source_bytes=10,
        unique_source_tokens=unique_source_bytes,
        unique_source_bytes=unique_source_bytes,
    )


def _validation() -> PeriodicValidationResult:
    return PeriodicValidationResult(
        compatibility=_protocol_result(
            protocol_id=NANOCHAT_COMPAT_PROTOCOL_ID,
            bpb=1.75,
            unique_source_bytes=4,
        ),
        full_document=_protocol_result(
            protocol_id=FULL_DOCUMENT_PROTOCOL_ID,
            bpb=1.25,
            unique_source_bytes=10,
        ),
    )


def _state(
    validation: PeriodicValidationResult,
    *,
    step: int,
) -> ValidationCheckpointState:
    decision = advance_validation_state(None, validation, validation_step=step)
    assert decision.state is not None
    return decision.state


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _core_result(
    *,
    run_kind: str = "bounded",
    labels: tuple[str, str] = ("ARC Easy/v2", "BoolQ"),
) -> CoreEvaluationResult:
    bounded = run_kind == "bounded"
    evaluated = 4 if bounded else 8
    return CoreEvaluationResult(
        checkpoint_identity="checkpoint:fixture",
        tokenizer_identity="tokenizer:fixture",
        bundle_identity="sha256:" + "1" * 64,
        config_identity="sha256:" + "2" * 64,
        metadata_identity="sha256:" + "3" * 64,
        run_kind=run_kind,  # type: ignore[arg-type]
        max_per_task=4 if bounded else None,
        tasks=(
            CoreTaskResult(
                label=labels[0],
                task_type="multiple_choice",
                num_fewshot=1,
                random_baseline_percent=25.0,
                correct_examples=3 if bounded else 6,
                evaluated_examples=evaluated,
                available_examples=8,
                elapsed_seconds=1.0,
                data_identity="sha256:" + "4" * 64,
            ),
            CoreTaskResult(
                label=labels[1],
                task_type="multiple_choice",
                num_fewshot=0,
                random_baseline_percent=50.0,
                correct_examples=1 if bounded else 4,
                evaluated_examples=evaluated,
                available_examples=8,
                elapsed_seconds=2.0,
                data_identity="sha256:" + "5" * 64,
            ),
        ),
        references=(CoreReferenceComparison("reference", 0.25),),
        elapsed_seconds=3.0,
    )


def _completed_core(result: CoreEvaluationResult) -> CompletedBaseEvaluation:
    return CompletedBaseEvaluation(
        context=BaseEvaluationContext(
            checkpoint_identity=result.checkpoint_identity,
            checkpoint_step=12,
            config_identity="config:fixture",
            tokenizer_identity=result.tokenizer_identity,
            validation_manifest_identity=None,
            run_kind=result.run_kind,
            max_per_task=result.max_per_task,
        ),
        requested_modes=("core",),
        completed_modes=("core",),
        validation=None,
        samples=None,
        core_result=result,
    )


def test_periodic_validation_logs_both_current_and_minimum_protocol_values() -> None:
    validation = _validation()
    state = _state(validation, step=7)
    tracker = _SpyTracker()

    metrics = track_periodic_base_validation(
        validation,
        state,
        tracker=tracker,
        step=7,
    )

    assert metrics == {
        NANOCHAT_COMPAT_TRAIN_METRIC: 1.75,
        NANOCHAT_MINIMUM_TRAIN_METRIC: 1.75,
        FULL_DOCUMENT_TRAIN_METRIC: 1.25,
        FULL_DOCUMENT_MINIMUM_TRAIN_METRIC: 1.25,
    }
    assert tracker.metrics == [(metrics, 7)]
    assert tracker.artifacts == []


def test_core_reporting_logs_normalized_centered_and_raw_task_metrics(
    tmp_path: Path,
) -> None:
    tracker = _SpyTracker()
    result = _core_result()

    reported = report_completed_base_evaluation(
        _completed_core(result),
        tracker=tracker,
        run_dir=tmp_path / "run",
    )

    expected = {
        CORE_EVAL_METRIC: result.core_metric,
        CORE_RUN_KIND_METRIC: "bounded",
        CORE_MAX_PER_TASK_METRIC: 4,
        CORE_TASK_COUNT_METRIC: 2,
        "eval/core/arc_easy_v2": result.tasks[0].centered_score,
        "eval/core/boolq": result.tasks[1].centered_score,
        "eval/core_accuracy/arc_easy_v2": 0.75,
        "eval/core_accuracy/boolq": 0.25,
        "eval/core_random_baseline_percent/arc_easy_v2": 25.0,
        "eval/core_random_baseline_percent/boolq": 50.0,
        "eval/core_correct_examples/arc_easy_v2": 3,
        "eval/core_correct_examples/boolq": 1,
        "eval/core_evaluated_examples/arc_easy_v2": 4,
        "eval/core_evaluated_examples/boolq": 4,
        "eval/core_available_examples/arc_easy_v2": 8,
        "eval/core_available_examples/boolq": 8,
    }
    assert reported.metrics == expected
    assert tracker.metrics == [(expected, None)]
    assert tracker.artifacts == [
        (
            "metrics/base_eval.json",
            BASE_EVALUATION_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
        ),
        (
            "metrics/core_comparison.md",
            CORE_COMPARISON_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
        ),
    ]
    payload = json.loads(reported.report_path.read_text(encoding="utf-8"))
    assert payload["core"]["core_metric"] == result.core_metric
    assert payload["core"]["tasks"]["ARC Easy/v2"] == result.tasks[0].to_dict()
    assert payload["core"]["tasks"]["BoolQ"] == result.tasks[1].to_dict()


def test_core_metric_normalization_rejects_collisions_before_publication(
    tmp_path: Path,
) -> None:
    tracker = _SpyTracker()
    completed = _completed_core(_core_result(labels=("ARC Easy", "arc-easy")))

    with pytest.raises(ValueError, match="collide after normalization"):
        report_completed_base_evaluation(
            completed,
            tracker=tracker,
            run_dir=tmp_path / "run",
        )

    assert tracker.metrics == []
    assert tracker.artifacts == []
    assert not (tmp_path / "run/metrics/base_eval.json").exists()
    assert not (tmp_path / "run/metrics/core_comparison.md").exists()


def test_periodic_validation_rejects_state_from_a_different_step_or_result() -> None:
    validation = _validation()
    tracker = _SpyTracker()

    with pytest.raises(ValueError, match="validation step"):
        track_periodic_base_validation(
            validation,
            _state(validation, step=6),
            tracker=tracker,
            step=7,
        )

    mismatched = _state(validation, step=7)
    object.__setattr__(mismatched, "current_compatibility_bpb", 99.0)
    with pytest.raises(ValueError, match="compatibility BPB"):
        track_periodic_base_validation(
            validation,
            mismatched,
            tracker=tracker,
            step=7,
        )

    assert tracker.metrics == []


def test_standalone_evaluation_writes_exact_dual_protocol_report_and_metadata(
    tmp_path: Path,
) -> None:
    validation = _validation()
    tracker = _SpyTracker()
    run_dir = tmp_path / "run"

    result = report_standalone_base_evaluation(
        validation,
        tracker=tracker,
        run_dir=run_dir,
    )

    expected_metrics = {
        NANOCHAT_COMPAT_EVAL_METRIC: validation.compatibility.bpb,
        FULL_DOCUMENT_EVAL_METRIC: validation.full_document.bpb,  # type: ignore[union-attr]
        NANOCHAT_SOURCE_BYTE_RETENTION_METRIC: 0.4,
        FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC: 1.0,
    }
    assert result.metrics == expected_metrics
    assert result.report_path == run_dir / "metrics" / "base_eval.json"
    payload = json.loads(result.report_path.read_text())
    assert payload["format"] == BASE_EVALUATION_REPORT_FORMAT
    assert payload["format_version"] == BASE_EVALUATION_REPORT_FORMAT_VERSION
    assert payload["status"] == "completed"
    assert payload["requested_modes"] == payload["completed_modes"] == ["bpb"]
    assert payload["results"] == {
        NANOCHAT_COMPAT_PROTOCOL_ID: validation.compatibility.to_dict(),
        FULL_DOCUMENT_PROTOCOL_ID: validation.full_document.to_dict(),  # type: ignore[union-attr]
    }
    assert tracker.metrics == [(expected_metrics, None)]
    assert tracker.artifacts == [
        (
            "metrics/base_eval.json",
            BASE_EVALUATION_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
        )
    ]


def test_standalone_report_is_atomic_and_tracks_nothing_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    report_path = run_dir / "metrics" / "base_eval.json"
    report_path.parent.mkdir(parents=True)
    tracker = _SpyTracker()

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr("os.replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        report_standalone_base_evaluation(
            _validation(),
            tracker=tracker,
            run_dir=run_dir,
        )

    assert not report_path.exists()
    assert not tuple(report_path.parent.glob(".base_eval.json.*.tmp"))
    assert tracker.metrics == []
    assert tracker.artifacts == []


def test_fixed_samples_are_written_and_registered_without_model_artifacts(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    clock = iter(float(index) for index in range(14)).__next__
    samples = generate_fixed_base_samples(
        _ConstantCompletionModel(ord("A")),
        tokenizer,
        checkpoint_identity="checkpoint:fixture",
        config=FixedBaseSamplingConfig(
            max_new_tokens=2,
            temperature=0,
            top_k=None,
            seed=17,
        ),
        device="cpu",
        clock=clock,
    )
    tracker = _SpyTracker()
    run_dir = tmp_path / "run"

    result = report_base_samples(
        samples,
        tracker=tracker,
        run_dir=run_dir,
    )

    assert result.metrics == {"eval/sample_tokens_per_sec": 2.0}
    assert result.markdown_path == run_dir / "metrics" / "base_samples.md"
    markdown = result.markdown_path.read_text()
    assert markdown.count("## Sample ") == 7
    assert all(
        sample.completion_reason == "max_new_tokens" for sample in samples.samples
    )
    assert tracker.metrics == [(result.metrics, None)]
    assert tracker.artifacts == [
        (
            "metrics/base_samples.md",
            BASE_SAMPLES_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
        )
    ]
    assert all(artifact_type != "model" for _, _, artifact_type in tracker.artifacts)


def _oversized_validation(tmp_path: Path) -> PeriodicValidationResult:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "text": pa.array(
                    [BPB_CONFORMANCE_FIXTURE.documents[1]],
                    type=pa.string(),
                )
            }
        ),
        parquet_dir / "shard_06542.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    pq.write_table(
        pa.table({"text": pa.array(["train"], type=pa.string())}),
        parquet_dir / "shard_00000.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    tokenizer = ByteTokenizer()
    tokenized_dir = tmp_path / "tokenized"
    write_tokenized_parquet_shards(
        parquet_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=1,
        batch_size=128,
    )
    token_bytes = build_token_byte_lengths(tokenizer)
    model = _UnitLossModel()
    with TokenizedShardReader(tokenized_dir, tokenizer=tokenizer) as reader:
        compatibility = evaluate_nanochat_compatible_bpb(
            model,
            tokenizer,
            reader,
            token_bytes,
            parquet_dir=parquet_dir,
            checkpoint_identity="checkpoint:oversized",
            config=NanochatCompatibilityConfig(1, 8, 8),
            device="cpu",
        )
        complete = evaluate_full_document_bpb(
            model,
            tokenizer,
            reader,
            token_bytes,
            checkpoint_identity="checkpoint:oversized",
            config=FullDocumentProtocolConfig(1, 8),
            device="cpu",
        )
    return PeriodicValidationResult(
        compatibility=compatibility,
        full_document=complete,
    )


def test_oversized_results_preserve_distinct_values_in_jsonl_and_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _oversized_validation(tmp_path / "fixture")
    assert validation.full_document is not None
    assert validation.compatibility.source_byte_retention < 1
    assert validation.full_document.source_byte_retention == 1

    run = _FakeWandbRun()
    fake_wandb = ModuleType("wandb")
    setattr(fake_wandb, "init", lambda **kwargs: run)
    setattr(fake_wandb, "Artifact", _FakeWandbArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = ProjectConfig(
        run=RunConfig(
            name="base-eval-fan-out",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(
            wandb=WandbConfig(
                enabled=True,
                mode="offline",
                dir=str(tmp_path / "wandb"),
            )
        ),
    )
    paths = prepare_run(config)

    with build_tracker(config, paths, stage="eval_base") as tracker:
        reported = report_standalone_base_evaluation(
            validation,
            tracker=tracker,
            run_dir=paths.run_dir,
        )

    metric_records = [
        record
        for record in _read_jsonl(paths.metrics_dir / "metrics.jsonl")
        if record["record_type"] == "metrics"
    ]
    artifact_records = [
        record
        for record in _read_jsonl(paths.metrics_dir / "metrics.jsonl")
        if record["record_type"] == "artifact"
    ]
    assert len(metric_records) == 1
    assert metric_records[0]["metrics"] == reported.metrics
    assert (
        reported.metrics[NANOCHAT_SOURCE_BYTE_RETENTION_METRIC]
        != reported.metrics[FULL_DOCUMENT_SOURCE_BYTE_RETENTION_METRIC]
    )
    assert artifact_records == [
        {
            "event_id": artifact_records[0]["event_id"],
            "name": BASE_EVALUATION_ARTIFACT_NAME,
            "path": "metrics/base_eval.json",
            "record_type": "artifact",
            "type": BASE_EVALUATION_ARTIFACT_TYPE,
        }
    ]
    assert run.logs == [(reported.metrics, {})]
    assert [
        (artifact.name, artifact.type, artifact.paths) for artifact in run.artifacts
    ] == [
        (
            BASE_EVALUATION_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
            [str(reported.report_path)],
        )
    ]


def test_core_metrics_summary_and_artifacts_have_jsonl_wandb_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _FakeWandbRun()
    fake_wandb = ModuleType("wandb")
    setattr(fake_wandb, "init", lambda **kwargs: run)
    setattr(fake_wandb, "Artifact", _FakeWandbArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = ProjectConfig(
        run=RunConfig(
            name="core-eval-fan-out",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(
            wandb=WandbConfig(
                enabled=True,
                mode="offline",
                dir=str(tmp_path / "wandb"),
            )
        ),
    )
    paths = prepare_run(config)

    with build_tracker(config, paths, stage="eval_base") as tracker:
        reported = report_completed_base_evaluation(
            _completed_core(_core_result()),
            tracker=tracker,
            run_dir=paths.run_dir,
        )

    records = _read_jsonl(paths.metrics_dir / "metrics.jsonl")
    metric_records = [
        record for record in records if record["record_type"] == "metrics"
    ]
    artifact_records = [
        record for record in records if record["record_type"] == "artifact"
    ]
    summary = json.loads((paths.metrics_dir / "summary.json").read_text())
    assert [record["metrics"] for record in metric_records] == [reported.metrics]
    assert run.logs == [(reported.metrics, {})]
    assert summary["latest_metrics"] == reported.metrics
    assert summary["latest_metrics"][CORE_RUN_KIND_METRIC] == "bounded"
    assert summary["latest_metrics"][CORE_MAX_PER_TASK_METRIC] == 4
    assert [record["name"] for record in artifact_records] == [
        BASE_EVALUATION_ARTIFACT_NAME,
        CORE_COMPARISON_ARTIFACT_NAME,
    ]
    assert [
        (artifact.name, artifact.type, artifact.paths) for artifact in run.artifacts
    ] == [
        (
            BASE_EVALUATION_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
            [str(reported.report_path)],
        ),
        (
            CORE_COMPARISON_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
            [str(reported.core_comparison_path)],
        ),
    ]


def test_bounded_core_cannot_overwrite_a_full_core_summary(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(
            name="core-scope-conflict",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        )
    )
    paths = prepare_run(config)
    full = _core_result(run_kind="full")

    with build_tracker(config, paths, stage="eval_base") as tracker:
        report_completed_base_evaluation(
            _completed_core(full),
            tracker=tracker,
            run_dir=paths.run_dir,
        )
        before = json.loads((paths.metrics_dir / "summary.json").read_text())

        with pytest.raises(
            BaseEvaluationReportConflictError,
            match="different completed evaluation",
        ):
            report_completed_base_evaluation(
                _completed_core(_core_result()),
                tracker=tracker,
                run_dir=paths.run_dir,
            )

        after = json.loads((paths.metrics_dir / "summary.json").read_text())

    assert before["latest_metrics"] == after["latest_metrics"]
    assert after["latest_metrics"][CORE_RUN_KIND_METRIC] == "full"
    assert after["latest_metrics"][CORE_MAX_PER_TASK_METRIC] is None
    assert after["latest_metrics"][CORE_EVAL_METRIC] == full.core_metric


def test_disabled_wandb_reporting_never_imports_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = __import__

    def reject_wandb(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("disabled reporting imported wandb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", reject_wandb)
    config = ProjectConfig(
        run=RunConfig(
            name="base-eval-local",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(wandb=WandbConfig(enabled=True, mode="disabled")),
    )
    paths = prepare_run(config)

    with build_tracker(config, paths, stage="eval_base") as tracker:
        report_standalone_base_evaluation(
            _validation(),
            tracker=tracker,
            run_dir=paths.run_dir,
        )

    assert (paths.metrics_dir / "base_eval.json").is_file()


def test_core_reporting_works_with_wandb_disabled_and_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = __import__

    def reject_wandb(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("disabled CORE reporting imported wandb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", reject_wandb)
    config = ProjectConfig(
        run=RunConfig(
            name="core-eval-local",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(wandb=WandbConfig(enabled=True, mode="disabled")),
    )
    paths = prepare_run(config)

    with build_tracker(config, paths, stage="eval_base") as tracker:
        reported = report_completed_base_evaluation(
            _completed_core(_core_result()),
            tracker=tracker,
            run_dir=paths.run_dir,
        )

    records = _read_jsonl(paths.metrics_dir / "metrics.jsonl")
    assert [
        record["metrics"] for record in records if record["record_type"] == "metrics"
    ] == [reported.metrics]
    assert reported.core_comparison_path is not None
    assert reported.core_comparison_path.is_file()
