"""Tracking adapters for completed tokenizer training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from scratch_llm.tokenizer_artifacts import TOKENIZER_ARTIFACT_FILENAMES
from scratch_llm.tokenizer_evaluation import TokenizerEvaluationResult
from scratch_llm.tokenizer_training import TokenizerTrainingRunResult
from scratch_llm.tracking import RunTracker, Tracker


TOKENIZER_ARTIFACT_TYPE: Final = "tokenizer"
_TOKENIZER_ARTIFACT_NAMES: Final = {
    "tokenizer.json": "tokenizer",
    "merges.json": "tokenizer_merges",
    "vocab.json": "tokenizer_vocab",
    "special_tokens.json": "tokenizer_special_tokens",
    "token_bytes.pt": "tokenizer_token_bytes",
    "tokenizer_eval.json": "tokenizer_eval",
}


def track_tokenizer_training(
    training: TokenizerTrainingRunResult,
    evaluation: TokenizerEvaluationResult,
    evaluation_json_path: str | Path,
    *,
    tracker: Tracker,
) -> dict[str, Any]:
    """Log one completed tokenizer training result and its canonical artifacts."""

    if not isinstance(training, TokenizerTrainingRunResult):
        raise TypeError(
            "training must be a TokenizerTrainingRunResult, "
            f"got {type(training).__name__}"
        )
    if not isinstance(evaluation, TokenizerEvaluationResult):
        raise TypeError(
            "evaluation must be a TokenizerEvaluationResult, "
            f"got {type(evaluation).__name__}"
        )
    _require_tracker(tracker)
    if training.tokenizer.get_identity() != evaluation.tokenizer_identity:
        raise ValueError("training and evaluation tokenizer identities do not match")
    if training.training_result.vocab_size != evaluation.vocab_size:
        raise ValueError("training and evaluation vocabulary sizes do not match")

    artifacts = _completed_training_artifacts(
        training,
        Path(evaluation_json_path),
    )
    aggregate = evaluation.to_dict()["aggregate"]
    metrics = {
        "tokenizer/vocab_size": training.training_result.vocab_size,
        "tokenizer/max_chars": training.configured_max_characters,
        "tokenizer/doc_cap": training.configured_max_documents,
        "tokenizer/num_docs": training.training_result.document_count,
        "tokenizer/num_chars": training.training_result.character_count,
        "tokenizer/train_seconds": training.elapsed_seconds,
        "tokenizer/bytes_per_token": aggregate["bytes_per_token"],
        "tokenizer/encode_tokens_per_sec": aggregate["encode_tokens_per_second"],
        "tokenizer/decode_tokens_per_sec": aggregate["decode_tokens_per_second"],
    }
    prefix = f"tokenizer:{training.tokenizer.get_identity()}:training"
    _log_metrics_once(
        tracker,
        metrics,
        event_id=f"{prefix}:metrics",
    )
    for relative_path, name in artifacts:
        _log_artifact_once(
            tracker,
            relative_path,
            name=name,
            event_id=f"{prefix}:artifact:{Path(relative_path).name}",
        )
    return metrics


def track_tokenizer_evaluation(
    evaluation: TokenizerEvaluationResult,
    evaluation_json_path: str | Path,
    *,
    tracker: Tracker,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Forward one completed evaluator result and its JSON report to a tracker."""

    if not isinstance(evaluation, TokenizerEvaluationResult):
        raise TypeError(
            "evaluation must be a TokenizerEvaluationResult, "
            f"got {type(evaluation).__name__}"
        )
    _require_tracker(tracker)
    relative_path = _completed_artifact_path(
        Path(evaluation_json_path),
        expected=Path(run_dir) / "metrics" / "tokenizer_eval.json",
        run_dir=Path(run_dir),
    )
    payload = evaluation.to_dict()
    aggregate = payload["aggregate"]
    comparisons = aggregate["comparisons"]
    metrics = {
        "tokenizer/vocab_size": evaluation.vocab_size,
        "tokenizer/bytes": aggregate["bytes"],
        "tokenizer/tokens": aggregate["tokens"],
        "tokenizer/bytes_per_token": aggregate["bytes_per_token"],
        "tokenizer/relative_diff_vs_gpt2": comparisons["gpt2"][
            "relative_token_count_difference"
        ],
        "tokenizer/relative_diff_vs_gpt4": comparisons["cl100k_base"][
            "relative_token_count_difference"
        ],
        "tokenizer/roundtrip_pass": aggregate["round_trip"],
        "tokenizer/encode_tokens_per_sec": aggregate["encode_tokens_per_second"],
        "tokenizer/decode_tokens_per_sec": aggregate["decode_tokens_per_second"],
    }
    prefix = f"tokenizer:{evaluation.tokenizer_identity}:evaluation"
    _log_metrics_once(
        tracker,
        metrics,
        event_id=f"{prefix}:metrics",
    )
    _log_artifact_once(
        tracker,
        relative_path,
        name=_TOKENIZER_ARTIFACT_NAMES["tokenizer_eval.json"],
        event_id=f"{prefix}:artifact:tokenizer_eval.json",
    )
    return metrics


def _completed_training_artifacts(
    training: TokenizerTrainingRunResult,
    evaluation_json_path: Path,
) -> tuple[tuple[str, str], ...]:
    expected_artifact_dir = training.run_dir / "artifacts" / "tokenizer"
    if training.artifact_dir != expected_artifact_dir:
        raise ValueError(
            "tokenizer artifact directory must be "
            f"{expected_artifact_dir}, got {training.artifact_dir}"
        )
    artifacts: list[tuple[str, str]] = []
    for filename in TOKENIZER_ARTIFACT_FILENAMES:
        path = training.artifact_dir / filename
        artifacts.append(
            (
                _completed_artifact_path(
                    path,
                    expected=expected_artifact_dir / filename,
                    run_dir=training.run_dir,
                ),
                _TOKENIZER_ARTIFACT_NAMES[filename],
            )
        )
    artifacts.append(
        (
            _completed_artifact_path(
                evaluation_json_path,
                expected=training.run_dir / "metrics" / "tokenizer_eval.json",
                run_dir=training.run_dir,
            ),
            _TOKENIZER_ARTIFACT_NAMES["tokenizer_eval.json"],
        )
    )
    return tuple(artifacts)


def _completed_artifact_path(
    path: Path,
    *,
    expected: Path,
    run_dir: Path,
) -> str:
    if path != expected:
        raise ValueError(f"tokenizer artifact path must be {expected}, got {path}")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"completed tokenizer artifact is not a regular file: {path}"
        )
    try:
        relative = path.relative_to(run_dir)
    except ValueError as error:
        raise ValueError(
            f"tokenizer artifact must be inside run directory {run_dir}: {path}"
        ) from error
    return relative.as_posix()


def _require_tracker(tracker: Tracker) -> None:
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")


def _log_metrics_once(
    tracker: Tracker,
    metrics: dict[str, Any],
    *,
    event_id: str,
) -> None:
    if isinstance(tracker, RunTracker):
        tracker.log_once(metrics, event_id=event_id)
    else:
        tracker.log(metrics)


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
            TOKENIZER_ARTIFACT_TYPE,
            event_id=event_id,
        )
    else:
        tracker.log_artifact(path, name, TOKENIZER_ARTIFACT_TYPE)


__all__ = [
    "TOKENIZER_ARTIFACT_TYPE",
    "track_tokenizer_evaluation",
    "track_tokenizer_training",
]
