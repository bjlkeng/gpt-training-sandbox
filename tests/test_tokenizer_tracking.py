"""Tests for tokenizer metrics and artifact tracking."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from scratch_llm.bpe import RegexBPETokenizer, train_reference_bpe
from scratch_llm.config import (
    ProjectConfig,
    RunConfig,
    TrackingConfig,
    WandbConfig,
)
from scratch_llm.run import prepare_run
from scratch_llm.evaluation.tokenizer import (
    EvaluationCorpus,
    TokenizerEvaluationResult,
    evaluate_tokenizer,
    write_tokenizer_evaluation_reports,
)
from scratch_llm.evaluation.tokenizer_tracking import (
    track_tokenizer_evaluation,
    track_tokenizer_training,
)
from scratch_llm.tokenizer_training import TokenizerTrainingRunResult
from scratch_llm.tracking import Tracker, build_tracker
from scratch_llm.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        self.id = "tokenizer-run"
        self.config = _FakeWandbConfig()
        self.logs: list[dict[str, Any]] = []
        self.artifacts: list[_FakeWandbArtifact] = []

    def log(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        assert kwargs == {}
        self.logs.append(metrics)

    def log_artifact(self, artifact: _FakeWandbArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        pass


def _benchmark_clock() -> Any:
    return iter((0.0, 2.0, 3.0, 7.0)).__next__


def _completed_tokenizer_pipeline(
    tmp_path: Path,
) -> tuple[TokenizerTrainingRunResult, TokenizerEvaluationResult, Path]:
    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "artifacts" / "tokenizer"
    report_path = run_dir / "metrics" / "tokenizer_training.json"
    training_result = train_reference_bpe(
        ("aaaa bbbb", "tracking fixture"),
        vocab_size=267,
    )
    tokenizer = RegexBPETokenizer(training_result)
    tokenizer.save(artifact_dir)
    training = TokenizerTrainingRunResult(
        algorithm="reference",
        training_result=training_result,
        tokenizer=tokenizer,
        selected_shards=("shard_00000.parquet",),
        configured_max_documents=5,
        configured_max_characters=100,
        document_char_cap=40,
        elapsed_seconds=2.5,
        peak_memory_bytes=1234,
        artifact_dir=artifact_dir,
        report_path=report_path,
        run_dir=run_dir,
    )
    save_json(training.to_dict(), report_path)
    evaluation = evaluate_tokenizer(
        tokenizer,
        (
            EvaluationCorpus(
                name="fixture",
                kind="builtin",
                identifier="test:tracking",
                documents=("aaaa",),
            ),
        ),
        benchmark_warmup_iterations=0,
        benchmark_iterations=1,
        clock=_benchmark_clock(),
    )
    evaluation_json, _ = write_tokenizer_evaluation_reports(
        evaluation,
        run_dir / "metrics",
    )
    return training, evaluation, evaluation_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_training_forwards_exact_metrics_and_registers_canonical_artifacts(
    tmp_path: Path,
) -> None:
    training, evaluation, evaluation_json = _completed_tokenizer_pipeline(tmp_path)
    tracker = _SpyTracker()
    aggregate = evaluation.to_dict()["aggregate"]

    metrics = track_tokenizer_training(
        training,
        evaluation,
        evaluation_json,
        tracker=tracker,
    )

    assert metrics == {
        "tokenizer/vocab_size": 267,
        "tokenizer/max_chars": 100,
        "tokenizer/doc_cap": 5,
        "tokenizer/num_docs": 2,
        "tokenizer/num_chars": 25,
        "tokenizer/train_seconds": 2.5,
        "tokenizer/bytes_per_token": aggregate["bytes_per_token"],
        "tokenizer/encode_tokens_per_sec": aggregate["encode_tokens_per_second"],
        "tokenizer/decode_tokens_per_sec": aggregate["decode_tokens_per_second"],
    }
    assert tracker.metrics == [(metrics, None)]
    assert tracker.artifacts == [
        (
            "artifacts/tokenizer/tokenizer.json",
            "tokenizer",
            "tokenizer",
        ),
        (
            "artifacts/tokenizer/merges.json",
            "tokenizer_merges",
            "tokenizer",
        ),
        (
            "artifacts/tokenizer/vocab.json",
            "tokenizer_vocab",
            "tokenizer",
        ),
        (
            "artifacts/tokenizer/special_tokens.json",
            "tokenizer_special_tokens",
            "tokenizer",
        ),
        (
            "artifacts/tokenizer/token_bytes.pt",
            "tokenizer_token_bytes",
            "tokenizer",
        ),
        (
            "metrics/tokenizer_eval.json",
            "tokenizer_eval",
            "tokenizer",
        ),
    ]


def test_missing_completed_artifact_prevents_all_tracking(
    tmp_path: Path,
) -> None:
    training, evaluation, evaluation_json = _completed_tokenizer_pipeline(tmp_path)
    (training.artifact_dir / "vocab.json").unlink()
    tracker = _SpyTracker()

    with pytest.raises(FileNotFoundError, match="vocab.json"):
        track_tokenizer_training(
            training,
            evaluation,
            evaluation_json,
            tracker=tracker,
        )

    assert tracker.metrics == []
    assert tracker.artifacts == []


def test_standalone_evaluation_forwards_report_values_without_recomputing(
    tmp_path: Path,
) -> None:
    training, evaluation, evaluation_json = _completed_tokenizer_pipeline(tmp_path)
    del training
    tracker = _SpyTracker()
    aggregate = evaluation.to_dict()["aggregate"]

    metrics = track_tokenizer_evaluation(
        evaluation,
        evaluation_json,
        tracker=tracker,
        run_dir=tmp_path / "run",
    )

    assert metrics == {
        "tokenizer/vocab_size": 267,
        "tokenizer/bytes": aggregate["bytes"],
        "tokenizer/tokens": aggregate["tokens"],
        "tokenizer/bytes_per_token": aggregate["bytes_per_token"],
        "tokenizer/relative_diff_vs_gpt2": None,
        "tokenizer/relative_diff_vs_gpt4": None,
        "tokenizer/roundtrip_pass": aggregate["round_trip"],
        "tokenizer/encode_tokens_per_sec": aggregate["encode_tokens_per_second"],
        "tokenizer/decode_tokens_per_sec": aggregate["decode_tokens_per_second"],
    }
    assert tracker.metrics == [(metrics, None)]
    assert tracker.artifacts == [
        ("metrics/tokenizer_eval.json", "tokenizer_eval", "tokenizer")
    ]


@pytest.mark.parametrize("log_tokenizer_artifacts", [False, True])
def test_wandb_tokenizer_gate_never_changes_local_artifact_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log_tokenizer_artifacts: bool,
) -> None:
    run = _FakeWandbRun()
    fake_wandb = ModuleType("wandb")
    setattr(fake_wandb, "init", lambda **kwargs: run)
    setattr(fake_wandb, "Artifact", _FakeWandbArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = ProjectConfig(
        run=RunConfig(
            name=f"tokenizer-gate-{log_tokenizer_artifacts}",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(
            wandb=WandbConfig(
                enabled=True,
                mode="offline",
                dir=str(tmp_path / "wandb"),
                log_tokenizer_artifacts=log_tokenizer_artifacts,
            )
        ),
    )
    paths = prepare_run(config)
    training, evaluation, evaluation_json = _completed_tokenizer_pipeline(
        paths.run_dir.parent
    )
    training = replace(
        training,
        artifact_dir=paths.run_dir / "artifacts" / "tokenizer",
        report_path=paths.metrics_dir / "tokenizer_training.json",
        run_dir=paths.run_dir,
    )
    training.tokenizer.save(training.artifact_dir)
    save_json(training.to_dict(), training.report_path)
    evaluation_json, _ = write_tokenizer_evaluation_reports(
        evaluation,
        paths.metrics_dir,
    )
    tracker = build_tracker(config, paths, stage="train_tokenizer")

    with tracker:
        metrics = track_tokenizer_training(
            training,
            evaluation,
            evaluation_json,
            tracker=tracker,
        )
        retried_metrics = track_tokenizer_training(
            training,
            evaluation,
            evaluation_json,
            tracker=tracker,
        )

    local_records = _read_jsonl(paths.metrics_dir / "metrics.jsonl")
    local_artifacts = [
        record for record in local_records if record.get("record_type") == "artifact"
    ]
    assert len(local_artifacts) == 6
    assert all(record["type"] == "tokenizer" for record in local_artifacts)
    assert all((paths.run_dir / record["path"]).is_file() for record in local_artifacts)
    assert retried_metrics == metrics
    assert run.logs == [metrics]
    if log_tokenizer_artifacts:
        assert [
            (artifact.name, artifact.type, artifact.paths) for artifact in run.artifacts
        ] == [
            (
                record["name"],
                "tokenizer",
                [str(paths.run_dir / record["path"])],
            )
            for record in local_artifacts
        ]
    else:
        assert run.artifacts == []


def test_readme_documents_tokenizer_metrics_artifacts_and_wandb_gate() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for metric_name in (
        "tokenizer/vocab_size",
        "tokenizer/max_chars",
        "tokenizer/doc_cap",
        "tokenizer/num_docs",
        "tokenizer/num_chars",
        "tokenizer/train_seconds",
        "tokenizer/bytes_per_token",
        "tokenizer/encode_tokens_per_sec",
        "tokenizer/decode_tokens_per_sec",
    ):
        assert metric_name in readme
    for artifact_path in (
        "artifacts/tokenizer/tokenizer.json",
        "artifacts/tokenizer/merges.json",
        "artifacts/tokenizer/vocab.json",
        "artifacts/tokenizer/special_tokens.json",
        "artifacts/tokenizer/token_bytes.pt",
        "metrics/tokenizer_eval.json",
    ):
        assert artifact_path in readme
    assert "tracking.wandb.log_tokenizer_artifacts" in readme
