"""SFT metric adapters and stable evaluation artifact publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scratch_llm._validation import (
    require_non_empty_string,
    require_non_negative_integer,
)
from scratch_llm.evaluation.sft_bpb import (
    SFTAssistantBPBResult,
    sft_validation_identity,
)
from scratch_llm.evaluation.sft_sampling import (
    FixedSFTSamplesResult,
    write_sft_samples_markdown,
)
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.utils import save_json


SFT_EVALUATION_REPORT_FORMAT: Final = "scratch_llm_sft_evaluation"
SFT_EVALUATION_REPORT_FORMAT_VERSION: Final = 1
SFT_EVALUATION_ARTIFACT_NAME: Final = "sft_eval"
SFT_SAMPLES_ARTIFACT_NAME: Final = "sft_samples"
SFT_EVALUATION_ARTIFACT_TYPE: Final = "evaluation"
SFT_TRAIN_LOSS_METRIC: Final = "sft/train_loss"
SFT_VALIDATION_BPB_METRIC: Final = "sft/val_bpb"
SFT_TOKENS_PER_SECOND_METRIC: Final = "sft/tok_per_sec"
SFT_MFU_METRIC: Final = "sft/mfu"
SFT_PEAK_MEMORY_METRIC: Final = "sft/peak_memory_mib"
_SFT_EVALUATION_RELATIVE_PATH = Path("metrics/sft_eval.json")
_SFT_SAMPLES_RELATIVE_PATH = Path("metrics/sft_samples.md")
_REQUIRED_BASE_TRAINING_METRICS = frozenset(
    {
        "train/loss",
        "train/lrm",
        "train/dt",
        "train/tok_per_sec",
        "train/mfu",
        "train/grad_norm",
        "total_training_flops",
        "total_training_time",
    }
)


@dataclass(frozen=True, slots=True)
class TrackedSFTEvaluation:
    """Completed SFT evaluation values and canonical artifact paths."""

    metrics: Mapping[str, float]
    report_path: Path
    samples_path: Path


def sft_training_metrics(
    base_metrics: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Translate shared-loop telemetry into the public SFT namespace."""

    if not isinstance(base_metrics, Mapping):
        raise TypeError(
            f"base_metrics must be a mapping, got {type(base_metrics).__name__}"
        )
    missing = sorted(_REQUIRED_BASE_TRAINING_METRICS - set(base_metrics))
    if missing:
        raise ValueError(
            f"shared training metrics are missing required keys: {missing}"
        )
    mapped: dict[str, float | None] = {
        SFT_TRAIN_LOSS_METRIC: base_metrics["train/loss"],
        "sft/lrm": base_metrics["train/lrm"],
        "sft/dt": base_metrics["train/dt"],
        SFT_TOKENS_PER_SECOND_METRIC: base_metrics["train/tok_per_sec"],
        SFT_MFU_METRIC: base_metrics["train/mfu"],
        "sft/grad_norm": base_metrics["train/grad_norm"],
        SFT_PEAK_MEMORY_METRIC: base_metrics.get("train/peak_memory_mib"),
        "total_training_flops": base_metrics["total_training_flops"],
        "total_training_time": base_metrics["total_training_time"],
    }
    if "train/epoch" in base_metrics:
        mapped["sft/epoch"] = base_metrics["train/epoch"]
    return mapped


def track_periodic_sft_validation(
    validation: SFTAssistantBPBResult,
    *,
    tracker: Tracker,
    step: int,
) -> dict[str, float]:
    """Log assistant-only validation BPB on its completed optimizer step."""

    if not isinstance(validation, SFTAssistantBPBResult):
        raise TypeError(
            "validation must be an SFTAssistantBPBResult, got "
            f"{type(validation).__name__}"
        )
    _require_tracker(tracker)
    step = require_non_negative_integer(step, name="step")
    metrics = {SFT_VALIDATION_BPB_METRIC: validation.bpb}
    if isinstance(tracker, RunTracker):
        identity = sft_validation_identity(
            tokenizer_identity=validation.tokenizer_identity,
            renderer_identity=validation.renderer_identity,
            validation_mixture_identity=validation.validation_mixture_identity,
            batch_budget=validation.batch_budget,
        )
        tracker.log_once(
            metrics,
            event_id=f"sft-validation:{identity}:step:{step}",
            step=step,
        )
    else:
        tracker.log(metrics, step=step)
    return metrics


def report_completed_sft_evaluation(
    validation: SFTAssistantBPBResult,
    samples: FixedSFTSamplesResult,
    *,
    tracker: Tracker,
    run_dir: str | Path,
    step: int,
    base_checkpoint_identity: str,
    checkpoint_identity: str,
) -> TrackedSFTEvaluation:
    """Atomically write and register the final BPB and fixed public samples."""

    if not isinstance(validation, SFTAssistantBPBResult):
        raise TypeError(
            "validation must be an SFTAssistantBPBResult, got "
            f"{type(validation).__name__}"
        )
    if not isinstance(samples, FixedSFTSamplesResult):
        raise TypeError(
            f"samples must be a FixedSFTSamplesResult, got {type(samples).__name__}"
        )
    _require_tracker(tracker)
    step = require_non_negative_integer(step, name="step")
    base_checkpoint_identity = require_non_empty_string(
        base_checkpoint_identity,
        name="base_checkpoint_identity",
    )
    checkpoint_identity = require_non_empty_string(
        checkpoint_identity,
        name="checkpoint_identity",
    )
    if samples.checkpoint_identity != checkpoint_identity:
        raise ValueError("fixed samples belong to a different checkpoint identity")
    if validation.tokenizer_identity != samples.tokenizer_identity:
        raise ValueError("validation and fixed samples use different tokenizers")
    if validation.renderer_identity != samples.renderer_identity:
        raise ValueError("validation and fixed samples use different chat renderers")

    resolved_run_dir = Path(run_dir).resolve()
    metrics = {SFT_VALIDATION_BPB_METRIC: validation.bpb}
    payload: dict[str, object] = {
        "format": SFT_EVALUATION_REPORT_FORMAT,
        "format_version": SFT_EVALUATION_REPORT_FORMAT_VERSION,
        "identities": {
            "base_checkpoint": base_checkpoint_identity,
            "checkpoint": checkpoint_identity,
            "renderer": validation.renderer_identity,
            "tokenizer": validation.tokenizer_identity,
            "validation_mixture": validation.validation_mixture_identity,
        },
        "metrics": metrics,
        "samples": {
            "artifact_path": _SFT_SAMPLES_RELATIVE_PATH.as_posix(),
            "generation_identity": samples.generation_identity,
            "prompt_set_identity": samples.prompt_set_identity,
            "sample_count": len(samples.samples),
        },
        "status": "completed",
        "step": step,
        "validation": validation.to_dict(),
    }
    samples_path = write_sft_samples_markdown(
        samples,
        resolved_run_dir / "metrics",
    )
    report_path = save_json(
        payload,
        resolved_run_dir / _SFT_EVALUATION_RELATIVE_PATH,
    )
    event_prefix = f"sft-evaluation:{_payload_identity(payload)}"
    _log_artifact_once(
        tracker,
        _SFT_EVALUATION_RELATIVE_PATH.as_posix(),
        name=SFT_EVALUATION_ARTIFACT_NAME,
        event_id=f"{event_prefix}:artifact:sft_eval.json",
    )
    _log_artifact_once(
        tracker,
        _SFT_SAMPLES_RELATIVE_PATH.as_posix(),
        name=SFT_SAMPLES_ARTIFACT_NAME,
        event_id=f"{event_prefix}:artifact:sft_samples.md",
    )
    return TrackedSFTEvaluation(
        metrics=MappingProxyType(metrics),
        report_path=report_path,
        samples_path=samples_path,
    )


def _payload_identity(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _log_artifact_once(
    tracker: Tracker,
    path: str,
    *,
    name: str,
    event_id: str,
) -> None:
    if isinstance(tracker, RunTracker):
        tracker.log_artifact_once(
            path,
            name,
            SFT_EVALUATION_ARTIFACT_TYPE,
            event_id=event_id,
        )
    else:
        tracker.log_artifact(path, name, SFT_EVALUATION_ARTIFACT_TYPE)


def _require_tracker(tracker: Tracker) -> None:
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")


__all__ = [
    "SFT_EVALUATION_ARTIFACT_NAME",
    "SFT_EVALUATION_ARTIFACT_TYPE",
    "SFT_EVALUATION_REPORT_FORMAT",
    "SFT_EVALUATION_REPORT_FORMAT_VERSION",
    "SFT_MFU_METRIC",
    "SFT_PEAK_MEMORY_METRIC",
    "SFT_SAMPLES_ARTIFACT_NAME",
    "SFT_TOKENS_PER_SECOND_METRIC",
    "SFT_TRAIN_LOSS_METRIC",
    "SFT_VALIDATION_BPB_METRIC",
    "TrackedSFTEvaluation",
    "report_completed_sft_evaluation",
    "sft_training_metrics",
    "track_periodic_sft_validation",
]
