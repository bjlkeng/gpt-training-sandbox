"""Deterministic human-readable reporting for completed CORE results."""

from __future__ import annotations

from pathlib import Path

from scratch_llm.evaluation.core.results import CORE_PROTOCOL_ID, CoreEvaluationResult
from scratch_llm.utils import atomic_write


CORE_COMPARISON_FILENAME = "core_comparison.md"


def render_core_comparison_markdown(result: CoreEvaluationResult) -> str:
    """Render task details and rough pinned-reference aggregate comparisons."""

    if not isinstance(result, CoreEvaluationResult):
        raise TypeError("result must be a CoreEvaluationResult")
    scope = (
        f"Bounded estimate (max {result.max_per_task} examples per task)"
        if result.run_kind == "bounded"
        else "Full pinned 22-task run"
    )
    lines = [
        "# CORE Evaluation Comparison",
        "",
        f"- Protocol: `{CORE_PROTOCOL_ID}`",
        f"- Scope: {scope}",
        f"- CORE metric: `{result.core_metric:.6f}`",
        f"- Checkpoint: `{result.checkpoint_identity}`",
        f"- Bundle: `{result.bundle_identity}`",
        "",
    ]
    if result.run_kind == "bounded":
        lines.extend(
            (
                "> Bounded estimates are not comparable to the full reference runs; ",
                "> no ranking delta is reported.",
                "",
            )
        )
    lines.extend(
        (
            "## Tasks",
            "",
            "| Task | Evaluated | Accuracy | Random baseline (%) | Centered |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for task in result.tasks:
        lines.append(
            f"| {task.label} | {task.evaluated_examples} / "
            f"{task.available_examples} | {task.accuracy:.6f} | "
            f"{task.random_baseline_percent:.2f} | {task.centered_score:.6f} |"
        )
    lines.extend(
        (
            "",
            "## Pinned reference aggregates",
            "",
            "| Reference | CORE | Delta |",
            "|---|---:|---:|",
        )
    )
    for reference in result.references:
        delta = (
            f"{result.core_metric - reference.core_metric:+.6f}"
            if result.run_kind == "full"
            else "not comparable"
        )
        lines.append(
            f"| {reference.model_id} | {reference.core_metric:.6f} | {delta} |"
        )
    lines.extend(
        (
            "",
            "Reference values come from CSV files inside the pinned nanochat "
            "evaluation bundle and are a rough comparison, not a claim of "
            "identical training conditions.",
            "",
        )
    )
    return "\n".join(lines)


def write_core_comparison_markdown(
    result: CoreEvaluationResult,
    output_dir: str | Path,
) -> Path:
    """Atomically write the canonical CORE comparison Markdown artifact."""

    destination = Path(output_dir) / CORE_COMPARISON_FILENAME
    return atomic_write(destination, render_core_comparison_markdown(result))


__all__ = [
    "CORE_COMPARISON_FILENAME",
    "render_core_comparison_markdown",
    "write_core_comparison_markdown",
]
