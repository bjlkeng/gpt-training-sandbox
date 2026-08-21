"""Tests for scope-safe ChatCORE tracker publication."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from scratch_llm.chat.rendering import CHAT_RENDERER_ID
from scratch_llm.config import (
    ProjectConfig,
    RunConfig,
    TrackingConfig,
    WandbMode,
    WandbConfig,
)
from scratch_llm.evaluation.chat.categorical import CategoricalTaskResult
from scratch_llm.evaluation.chat.diagnostics import (
    CodePromptDiagnostic,
    FixedSFTDiagnostics,
    JSONPromptDiagnostic,
)
from scratch_llm.evaluation.chat.generative import (
    GenerativeEvaluationConfig,
    GenerativeProblemResult,
    GenerativeSampleResult,
    GenerativeTaskResult,
)
from scratch_llm.evaluation.chat.reporting import (
    ChatEvaluationSettings,
    CompletedChatEvaluation,
    write_chat_evaluation_report,
)
from scratch_llm.evaluation.chat.tracking import (
    CHATCORE_CAT_METRIC,
    CHATCORE_METRIC,
    CHAT_EVALUATION_ARTIFACT_NAME,
    CHAT_EVALUATION_ARTIFACT_TYPE,
    ChatEvaluationTrackingError,
    chat_evaluation_metrics,
    track_completed_chat_evaluation,
)
from scratch_llm.evaluation.sft_sampling import (
    FIXED_SFT_PROMPT_SET_IDENTITY,
    FixedSFTSamplingConfig,
)
from scratch_llm.identity import file_identity
from scratch_llm.run import prepare_run
from scratch_llm.tracking import Tracker, build_tracker
from scratch_llm.tracking_state import TrackingState


_CHECKPOINT = "sha256:" + "1" * 64
_TOKENIZER = "sha256:" + "2" * 64
_ALL_TASKS = (
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
)


class _SpyTracker(Tracker):
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

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
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.config = _FakeWandbConfig()
        self.logs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.artifacts: list[_FakeWandbArtifact] = []

    def log(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        self.logs.append((dict(metrics), kwargs))

    def log_artifact(self, artifact: _FakeWandbArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        pass


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch) -> list[_FakeWandbRun]:
    runs: list[_FakeWandbRun] = []
    fake_wandb = ModuleType("wandb")

    def init(**kwargs: Any) -> _FakeWandbRun:
        run = _FakeWandbRun(kwargs.get("id", "chat-eval-run"))
        runs.append(run)
        return run

    setattr(fake_wandb, "init", init)
    setattr(fake_wandb, "Artifact", _FakeWandbArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return runs


def _generation() -> GenerativeEvaluationConfig:
    return GenerativeEvaluationConfig(
        num_samples=1,
        max_new_tokens=1,
        temperature=0,
        top_k=1,
        seed=7,
    )


def _categorical(task_name: str, *, bounded: bool) -> CategoricalTaskResult:
    return CategoricalTaskResult(
        task_name=task_name,
        checkpoint_identity=_CHECKPOINT,
        tokenizer_identity=_TOKENIZER,
        source_identity=f"source:{task_name}",
        dataset_identity=f"dataset:{task_name}",
        order_identity=f"order:{task_name}",
        run_kind="bounded" if bounded else "full",
        max_problems=1 if bounded else None,
        passed_count=1,
        evaluated_count=1,
        available_count=1,
        elapsed_seconds=1,
        model_max_seq_len=512,
    )


def _generative(task_name: str, *, bounded: bool) -> GenerativeTaskResult:
    passed = task_name == "GSM8K"
    sample = GenerativeSampleResult(
        problem_index=0,
        sample_index=0,
        seed=7,
        passed=passed,
        generated_token_count=1,
        sampled_token_count=1,
        completion_reason="max_new_tokens",
        stop_token_id=None,
        completion_identity=f"completion:{task_name}",
        score_outcome=("test_failure" if task_name == "HumanEval" else None),
    )
    return GenerativeTaskResult(
        task_name=task_name,
        checkpoint_identity=_CHECKPOINT,
        tokenizer_identity=_TOKENIZER,
        source_identity=f"source:{task_name}",
        dataset_identity=f"dataset:{task_name}",
        order_identity=f"order:{task_name}",
        run_kind="bounded" if bounded else "full",
        max_problems=1 if bounded else None,
        available_count=1,
        assistant_end_token_id=263,
        bos_token_id=264,
        config=_generation(),
        problems=(
            GenerativeProblemResult(
                problem_index=0,
                problem_identity=f"problem:{task_name}",
                source_row=0,
                passed=passed,
                samples=(sample,),
            ),
        ),
        scoring_identity=("executor:v1" if task_name == "HumanEval" else None),
    )


def _completed(
    task_names: tuple[str, ...] = _ALL_TASKS,
    *,
    bounded: bool = False,
) -> CompletedChatEvaluation:
    task_results = tuple(
        _categorical(task_name, bounded=bounded)
        if task_name in _ALL_TASKS[:3]
        else _generative(task_name, bounded=bounded)
        for task_name in task_names
    )
    has_humaneval = "HumanEval" in task_names
    return CompletedChatEvaluation(
        config_identity="config:v1",
        checkpoint_identity=_CHECKPOINT,
        checkpoint_step=11,
        tokenizer_identity=_TOKENIZER,
        settings=ChatEvaluationSettings(
            task_names=task_names,  # type: ignore[arg-type]
            batch_size=2,
            max_problems=1 if bounded else None,
            generation=_generation(),
            fixed_sampling=FixedSFTSamplingConfig(
                max_new_tokens=1,
                temperature=0,
                top_k=1,
                seed=7,
            ),
            allow_generated_code_execution=has_humaneval,
            executor_identity="executor:v1" if has_humaneval else None,
        ),
        task_results=task_results,
        diagnostics=FixedSFTDiagnostics(
            checkpoint_identity=_CHECKPOINT,
            tokenizer_identity=_TOKENIZER,
            renderer_identity=CHAT_RENDERER_ID,
            prompt_set_identity=FIXED_SFT_PROMPT_SET_IDENTITY,
            generation_identity="fixed-generation:v1",
            sample_count=5,
            assistant_end_stop_count=0,
            bos_safety_stop_count=0,
            max_token_count=5,
            visible_token_mean=1,
            visible_token_min=1,
            visible_token_max=1,
            empty_response_count=0,
            json_prompt=JSONPromptDiagnostic(
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            code_prompt=CodePromptDiagnostic("plain_code", 0),
        ),
    )


def _tracking_config(
    tmp_path: Path,
    *,
    wandb: bool,
    wandb_mode: WandbMode | None = None,
) -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(
            name="chat-eval-tracking",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tracking=TrackingConfig(
            wandb=WandbConfig(
                enabled=wandb,
                mode=(
                    wandb_mode
                    if wandb_mode is not None
                    else ("offline" if wandb else "disabled")
                ),
                dir=str(tmp_path / "wandb"),
            )
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_metric_names_are_exact_and_non_full_scopes_never_publish_aggregates() -> None:
    full = chat_evaluation_metrics(_completed())

    assert full == {
        CHATCORE_METRIC: 0.8,
        CHATCORE_CAT_METRIC: 1.0,
        "sft/chatcore/ARC-Easy": 1.0,
        "sft/chatcore/ARC-Challenge": 1.0,
        "sft/chatcore/MMLU": 1.0,
        "sft/chatcore/GSM8K": 1.0,
        "sft/chatcore/HumanEval": 0.0,
    }

    bounded = chat_evaluation_metrics(_completed(bounded=True))
    partial = chat_evaluation_metrics(_completed(("ARC-Easy", "GSM8K")))

    assert tuple(bounded) == tuple(
        f"sft/chatcore/bounded/{task_name}" for task_name in _ALL_TASKS
    )
    assert set(partial) == {
        "sft/chatcore/partial/ARC-Easy",
        "sft/chatcore/partial/GSM8K",
    }
    for metrics in (bounded, partial):
        assert CHATCORE_METRIC not in metrics
        assert CHATCORE_CAT_METRIC not in metrics


def test_adapter_verifies_exact_report_registers_it_without_rewriting(
    tmp_path: Path,
) -> None:
    completed = _completed()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = write_chat_evaluation_report(completed, run_dir=run_dir)
    original = report_path.read_bytes()
    tracker = _SpyTracker()

    tracked = track_completed_chat_evaluation(
        completed,
        report_path=report_path,
        tracker=tracker,
        run_dir=run_dir,
    )

    assert tracked.report_path == report_path
    assert tracked.artifact_identity == file_identity(report_path)
    assert tracked.metrics == chat_evaluation_metrics(completed)
    assert report_path.read_bytes() == original
    assert tracker.metrics == [(dict(tracked.metrics), completed.checkpoint_step)]
    assert tracker.artifacts == [
        (
            "metrics/chat_eval.json",
            CHAT_EVALUATION_ARTIFACT_NAME,
            CHAT_EVALUATION_ARTIFACT_TYPE,
        )
    ]

    payload = json.loads(report_path.read_text())
    assert [task["evaluation_type"] for task in payload["tasks"]] == [
        "categorical",
        "categorical",
        "categorical",
        "generative",
        "code_execution",
    ]
    payload["status"] = "failed"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ChatEvaluationTrackingError, match="immutable result"):
        track_completed_chat_evaluation(
            completed,
            report_path=report_path,
            tracker=tracker,
            run_dir=run_dir,
        )
    assert len(tracker.metrics) == len(tracker.artifacts) == 1


def test_run_tracker_has_jsonl_wandb_summary_and_artifact_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _install_fake_wandb(monkeypatch)
    config = _tracking_config(tmp_path, wandb=True)
    paths = prepare_run(config)
    completed = _completed()
    report_path = write_chat_evaluation_report(completed, run_dir=paths.run_dir)

    with build_tracker(config, paths, stage="eval_chat") as tracker:
        tracked = track_completed_chat_evaluation(
            completed,
            report_path=report_path,
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
    assert [record["metrics"] for record in metric_records] == [tracked.metrics]
    assert metric_records[0]["step"] == completed.checkpoint_step
    assert summary["latest_metrics"] == tracked.metrics
    assert summary["latest_step"] == completed.checkpoint_step
    assert runs[0].logs == [(dict(tracked.metrics), {"step": 11})]
    assert [
        (artifact.name, artifact.type, artifact.paths) for artifact in runs[0].artifacts
    ] == [
        (
            CHAT_EVALUATION_ARTIFACT_NAME,
            CHAT_EVALUATION_ARTIFACT_TYPE,
            [str(report_path)],
        )
    ]
    assert artifact_records[0]["path"] == "metrics/chat_eval.json"
    serialized_metrics = json.dumps(metric_records)
    assert "prompt" not in serialized_metrics.lower()
    assert "completion" not in serialized_metrics.lower()


def test_run_tracker_resume_is_idempotent_locally_and_remotely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _install_fake_wandb(monkeypatch)
    config = _tracking_config(tmp_path, wandb=True)
    paths = prepare_run(config)
    completed = _completed()
    report_path = write_chat_evaluation_report(completed, run_dir=paths.run_dir)

    with build_tracker(config, paths, stage="eval_chat") as tracker:
        track_completed_chat_evaluation(
            completed,
            report_path=report_path,
            tracker=tracker,
            run_dir=paths.run_dir,
        )
    with build_tracker(
        config,
        paths,
        stage="eval_chat",
        wandb_resume_state=TrackingState(backend="wandb", run_id="chat-eval-run"),
    ) as tracker:
        track_completed_chat_evaluation(
            completed,
            report_path=report_path,
            tracker=tracker,
            run_dir=paths.run_dir,
        )

    records = _read_jsonl(paths.metrics_dir / "metrics.jsonl")
    assert sum(record["record_type"] == "metrics" for record in records) == 1
    assert sum(record["record_type"] == "artifact" for record in records) == 1
    assert len(runs) == 2
    assert len(runs[0].logs) == len(runs[0].artifacts) == 1
    assert runs[1].logs == []
    assert runs[1].artifacts == []


@pytest.mark.parametrize(
    ("enabled", "mode"),
    ((False, "online"), (True, "disabled")),
)
def test_disabled_or_unrequested_wandb_never_imports_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    mode: WandbMode,
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
            raise AssertionError("disabled chat evaluation imported wandb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", reject_wandb)
    config = _tracking_config(
        tmp_path,
        wandb=enabled,
        wandb_mode=mode,
    )
    paths = prepare_run(config)
    completed = _completed(("ARC-Easy",))
    report_path = write_chat_evaluation_report(completed, run_dir=paths.run_dir)

    with build_tracker(config, paths, stage="eval_chat") as tracker:
        tracked = track_completed_chat_evaluation(
            completed,
            report_path=report_path,
            tracker=tracker,
            run_dir=paths.run_dir,
        )

    assert tracked.metrics == {"sft/chatcore/partial/ARC-Easy": 1.0}
    assert report_path.is_file()
