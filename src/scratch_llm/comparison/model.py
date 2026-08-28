"""Shared immutable values for offline run-comparison modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from scratch_llm.evaluation.base_tracking import (
    FULL_DOCUMENT_MINIMUM_TRAIN_METRIC,
    NANOCHAT_MINIMUM_TRAIN_METRIC,
)
from scratch_llm.config import ProjectConfig
from scratch_llm.evaluation.full_document_bpb import FULL_DOCUMENT_TRAIN_METRIC
from scratch_llm.evaluation.nanochat_bpb import NANOCHAT_COMPAT_TRAIN_METRIC


RUN_COMPARISON_FORMAT: Final = "scratch_llm_run_comparison"
RUN_COMPARISON_FORMAT_VERSION: Final = 3
STEP_METRICS: Final = (
    NANOCHAT_COMPAT_TRAIN_METRIC,
    NANOCHAT_MINIMUM_TRAIN_METRIC,
    FULL_DOCUMENT_TRAIN_METRIC,
    FULL_DOCUMENT_MINIMUM_TRAIN_METRIC,
    "train/loss",
    "train/tok_per_sec",
    "train/mfu",
    "train/peak_memory_mib",
    "total_training_flops",
    "total_training_time",
)
IDENTITY_FIELDS: Final = (
    "checkpoint_identity",
    "tokenizer_identity",
    "validation_manifest_identity",
    "config_identity",
    "parameterization",
    "hardware",
    "precision",
    "code_identity",
)


class RunComparisonError(ValueError):
    """One or more local runs cannot satisfy the comparison contract."""


@dataclass(frozen=True)
class RunComparisonArtifacts:
    """Canonical JSON and Markdown paths installed by one comparison."""

    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class RunSnapshot:
    """Validated immutable inputs for one locally tracked training run."""

    path: Path
    name: str
    config: ProjectConfig
    summary: Mapping[str, Any]
    training_metrics: Mapping[int, Mapping[str, Any]]
    base_evaluation: Mapping[str, Any] | None


__all__ = [
    "IDENTITY_FIELDS",
    "RUN_COMPARISON_FORMAT",
    "RUN_COMPARISON_FORMAT_VERSION",
    "STEP_METRICS",
    "RunComparisonArtifacts",
    "RunComparisonError",
    "RunSnapshot",
]
