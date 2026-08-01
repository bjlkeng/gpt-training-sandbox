"""Base-evaluation mode orchestration tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from scratch_llm.base_evaluation import (
    BaseEvaluationContext,
    BaseEvaluationUnavailableError,
    execute_base_evaluation_modes,
    normalize_base_evaluation_modes,
)
from scratch_llm.base_evaluation_tracking import (
    BASE_EVALUATION_ARTIFACT_NAME,
    BASE_EVALUATION_ARTIFACT_TYPE,
    BASE_EVALUATION_REPORT_FORMAT,
    BASE_EVALUATION_REPORT_FORMAT_VERSION,
    BASE_SAMPLES_ARTIFACT_NAME,
    BaseEvaluationReportConflictError,
    report_completed_base_evaluation,
)
from scratch_llm.base_sampling import (
    BaseSample,
    BaseSamplesResult,
    FIXED_BASE_PROMPTS,
    FIXED_BASE_PROMPT_SET_IDENTITY,
    FixedBaseSamplingConfig,
)
from scratch_llm.core_evaluation import (
    CORE_PROTOCOL_ID,
    CoreEvaluationResult,
    CoreReferenceComparison,
    CoreTaskResult,
)
from scratch_llm.best_checkpoint import PeriodicValidationResult
from scratch_llm.bpb import BPBAccumulation, BaseValidationResult
from scratch_llm.full_document_bpb import (
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_PROTOCOL_VERSION,
)
from scratch_llm.nanochat_bpb import (
    NANOCHAT_COMPAT_EVAL_METRIC,
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_PROTOCOL_VERSION,
    NANOCHAT_REFERENCE_COMMIT,
)
from scratch_llm.full_document_bpb import FULL_DOCUMENT_EVAL_METRIC
from scratch_llm.tracking import Tracker


_CHECKPOINT_IDENTITY = "sha256:" + "1" * 64
_TOKENIZER_IDENTITY = "sha256:" + "2" * 64
_MANIFEST_IDENTITY = "sha256:" + "3" * 64


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


def _protocol_result(protocol_id: str, *, bpb: float) -> BaseValidationResult:
    compatibility = protocol_id == NANOCHAT_COMPAT_PROTOCOL_ID
    return BaseValidationResult.from_accumulation(
        BPBAccumulation(
            processed_model_tokens=8,
            counted_target_tokens=4,
            counted_target_bytes=4,
            total_nats=bpb * math.log(2) * 4,
        ),
        protocol_id=protocol_id,
        protocol_version=(
            NANOCHAT_COMPAT_PROTOCOL_VERSION
            if compatibility
            else FULL_DOCUMENT_PROTOCOL_VERSION
        ),
        reference_commit=NANOCHAT_REFERENCE_COMMIT if compatibility else None,
        reference_config={"fixture": protocol_id},
        checkpoint_identity=_CHECKPOINT_IDENTITY,
        tokenizer_identity=_TOKENIZER_IDENTITY,
        validation_manifest_identity=_MANIFEST_IDENTITY,
        source_documents=1,
        source_tokens=4,
        source_bytes=4,
        unique_source_tokens=4,
        unique_source_bytes=4,
    )


def _validation() -> PeriodicValidationResult:
    return PeriodicValidationResult(
        compatibility=_protocol_result(NANOCHAT_COMPAT_PROTOCOL_ID, bpb=1.5),
        full_document=_protocol_result(FULL_DOCUMENT_PROTOCOL_ID, bpb=1.25),
    )


def _samples() -> BaseSamplesResult:
    config = FixedBaseSamplingConfig(
        max_new_tokens=1,
        temperature=0,
        top_k=None,
        seed=10,
    )
    samples = tuple(
        BaseSample(
            prompt_index=index,
            prompt=prompt,
            prompt_token_count=1,
            seed=config.seed + index,
            generated_token_ids=(65,),
            sampled_token_count=1,
            elapsed_seconds=1.0,
            completion_reason="max_new_tokens",
            stop_token_id=None,
            text="A",
        )
        for index, prompt in enumerate(FIXED_BASE_PROMPTS)
    )
    generation_payload = json.dumps(
        {
            "config": config.to_dict(),
            "format": "scratch_llm_fixed_base_generation",
            "format_version": 1,
            "stop_token_ids": [256],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return BaseSamplesResult(
        checkpoint_identity=_CHECKPOINT_IDENTITY,
        tokenizer_identity=_TOKENIZER_IDENTITY,
        prompt_set_identity=FIXED_BASE_PROMPT_SET_IDENTITY,
        generation_identity=(
            "sha256:" + hashlib.sha256(generation_payload).hexdigest()
        ),
        bos_token_id=256,
        config=config,
        samples=samples,
    )


def _context() -> BaseEvaluationContext:
    return BaseEvaluationContext(
        checkpoint_identity=_CHECKPOINT_IDENTITY,
        checkpoint_step=12,
        config_identity="sha256:" + "5" * 64,
        tokenizer_identity=_TOKENIZER_IDENTITY,
        validation_manifest_identity=_MANIFEST_IDENTITY,
        run_kind="full",
        max_per_task=None,
    )


def _core_result() -> CoreEvaluationResult:
    return CoreEvaluationResult(
        checkpoint_identity=_CHECKPOINT_IDENTITY,
        tokenizer_identity=_TOKENIZER_IDENTITY,
        bundle_identity="sha256:" + "6" * 64,
        config_identity="sha256:" + "7" * 64,
        metadata_identity="sha256:" + "8" * 64,
        run_kind="bounded",
        max_per_task=2,
        tasks=(
            CoreTaskResult(
                label="fixture",
                task_type="language_modeling",
                num_fewshot=0,
                random_baseline_percent=0.0,
                correct_examples=1,
                evaluated_examples=2,
                available_examples=4,
                elapsed_seconds=1.0,
                data_identity="sha256:" + "9" * 64,
            ),
        ),
        references=(CoreReferenceComparison("reference", 0.25),),
        elapsed_seconds=1.0,
    )


def test_modes_are_case_normalized_deduplicated_and_order_preserving() -> None:
    assert normalize_base_evaluation_modes(" Sample, BPB,sample ") == (
        "sample",
        "bpb",
    )

    for invalid in ("", " , ", "bpb,unknown"):
        with pytest.raises(ValueError):
            normalize_base_evaluation_modes(invalid)


def test_execution_preflights_unavailable_core_before_any_mode_runs() -> None:
    calls: list[str] = []

    with pytest.raises(BaseEvaluationUnavailableError, match="Milestone 5"):
        execute_base_evaluation_modes(
            ("bpb", "core"),
            context=_context(),
            bpb_runner=lambda: calls.append("bpb") or _validation(),
            sample_runner=lambda: calls.append("sample") or _samples(),
            core_runner=None,
        )

    assert calls == []


def test_execution_runs_requested_modes_once_and_preserves_domain_results() -> None:
    calls: list[tuple[str, Any]] = []
    validation = _validation()
    samples = _samples()

    completed = execute_base_evaluation_modes(
        ("sample", "bpb"),
        context=_context(),
        bpb_runner=lambda: calls.append(("bpb", None)) or validation,
        sample_runner=lambda: calls.append(("sample", None)) or samples,
        core_runner=None,
    )

    assert calls == [("sample", None), ("bpb", None)]
    assert completed.requested_modes == ("sample", "bpb")
    assert completed.completed_modes == completed.requested_modes
    assert completed.validation is validation
    assert completed.samples is samples
    assert completed.core_result is None


def test_execution_runs_core_bpb_and_samples_in_requested_order(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Any]] = []
    core = _core_result()
    completed = execute_base_evaluation_modes(
        ("core", "bpb", "sample"),
        context=BaseEvaluationContext(
            checkpoint_identity=_CHECKPOINT_IDENTITY,
            checkpoint_step=12,
            config_identity="sha256:" + "5" * 64,
            tokenizer_identity=_TOKENIZER_IDENTITY,
            validation_manifest_identity=_MANIFEST_IDENTITY,
            run_kind="bounded",
            max_per_task=2,
        ),
        bpb_runner=lambda: calls.append(("bpb", None)) or _validation(),
        sample_runner=lambda: calls.append(("sample", None)) or _samples(),
        core_runner=lambda limit: calls.append(("core", limit)) or core,
    )

    assert calls == [("core", 2), ("bpb", None), ("sample", None)]
    assert completed.core_result is core

    reported = report_completed_base_evaluation(
        completed,
        tracker=_SpyTracker(),
        run_dir=tmp_path,
    )
    payload = json.loads(reported.report_path.read_text(encoding="utf-8"))
    assert payload["core"]["protocol_id"] == CORE_PROTOCOL_ID
    assert payload["requested_modes"] == ["core", "bpb", "sample"]
    assert reported.core_comparison_path == tmp_path / "metrics/core_comparison.md"
    assert reported.core_comparison_path.is_file()


def test_core_report_marker_failure_removes_new_comparison_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = execute_base_evaluation_modes(
        ("core",),
        context=BaseEvaluationContext(
            checkpoint_identity=_CHECKPOINT_IDENTITY,
            checkpoint_step=12,
            config_identity="sha256:" + "5" * 64,
            tokenizer_identity=_TOKENIZER_IDENTITY,
            validation_manifest_identity=None,
            run_kind="bounded",
            max_per_task=2,
        ),
        bpb_runner=None,
        sample_runner=None,
        core_runner=lambda _limit: _core_result(),
    )

    def fail_save(*_args: object, **_kwargs: object) -> Path:
        raise OSError("cannot install completion marker")

    monkeypatch.setattr(
        "scratch_llm.base_evaluation_tracking.save_json",
        fail_save,
    )
    with pytest.raises(OSError, match="completion marker"):
        report_completed_base_evaluation(
            completed,
            tracker=_SpyTracker(),
            run_dir=tmp_path,
        )

    assert not (tmp_path / "metrics/base_eval.json").exists()
    assert not (tmp_path / "metrics/core_comparison.md").exists()


def test_completed_execution_is_published_once_with_explicit_scope_and_modes(
    tmp_path: Path,
) -> None:
    completed = execute_base_evaluation_modes(
        ("bpb", "sample"),
        context=_context(),
        bpb_runner=_validation,
        sample_runner=_samples,
        core_runner=None,
    )
    tracker = _SpyTracker()

    reported = report_completed_base_evaluation(
        completed,
        tracker=tracker,
        run_dir=tmp_path,
    )

    payload = json.loads(reported.report_path.read_text(encoding="utf-8"))
    assert payload["format"] == BASE_EVALUATION_REPORT_FORMAT
    assert payload["format_version"] == BASE_EVALUATION_REPORT_FORMAT_VERSION
    assert payload["status"] == "completed"
    assert payload["run_kind"] == "full"
    assert payload["bounded"] is False
    assert payload["requested_modes"] == ["bpb", "sample"]
    assert payload["completed_modes"] == ["bpb", "sample"]
    assert payload["identities"] == {
        "checkpoint": {"identity": _CHECKPOINT_IDENTITY, "step": 12},
        "config": "sha256:" + "5" * 64,
        "tokenizer": _TOKENIZER_IDENTITY,
        "validation_manifest": _MANIFEST_IDENTITY,
    }
    assert set(payload["results"]) == {
        NANOCHAT_COMPAT_PROTOCOL_ID,
        FULL_DOCUMENT_PROTOCOL_ID,
    }
    assert payload["samples"]["artifact_path"] == "metrics/base_samples.md"
    assert payload["samples"]["sample_count"] == len(FIXED_BASE_PROMPTS)
    assert reported.sample_markdown_path == tmp_path / "metrics/base_samples.md"
    assert tracker.metrics == [(reported.metrics, None)]
    assert reported.metrics[NANOCHAT_COMPAT_EVAL_METRIC] == 1.5
    assert reported.metrics[FULL_DOCUMENT_EVAL_METRIC] == 1.25
    assert tracker.artifacts == [
        (
            "metrics/base_eval.json",
            BASE_EVALUATION_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
        ),
        (
            "metrics/base_samples.md",
            BASE_SAMPLES_ARTIFACT_NAME,
            BASE_EVALUATION_ARTIFACT_TYPE,
        ),
    ]


def test_conflicting_rerun_preserves_existing_report_and_writes_no_artifacts(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "metrics" / "base_eval.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"stable": true}\n', encoding="utf-8")
    tracker = _SpyTracker()
    completed = execute_base_evaluation_modes(
        ("sample",),
        context=BaseEvaluationContext(
            checkpoint_identity=_CHECKPOINT_IDENTITY,
            checkpoint_step=12,
            config_identity="sha256:" + "5" * 64,
            tokenizer_identity=_TOKENIZER_IDENTITY,
            validation_manifest_identity=None,
            run_kind="full",
            max_per_task=None,
        ),
        bpb_runner=None,
        sample_runner=_samples,
        core_runner=None,
    )

    with pytest.raises(BaseEvaluationReportConflictError, match="different"):
        report_completed_base_evaluation(
            completed,
            tracker=tracker,
            run_dir=tmp_path,
        )

    assert report_path.read_text(encoding="utf-8") == '{"stable": true}\n'
    assert not (tmp_path / "metrics" / "base_samples.md").exists()
    assert tracker.metrics == []
    assert tracker.artifacts == []
