"""Public orchestration boundary for deterministic offline run comparisons."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

from scratch_llm.comparison.loading import load_run_snapshots
from scratch_llm.comparison.model import (
    RUN_COMPARISON_FORMAT,
    RUN_COMPARISON_FORMAT_VERSION,
    RunComparisonArtifacts,
    RunComparisonError,
)
from scratch_llm.comparison.reporting import (
    build_comparison_payload,
    render_comparison_markdown,
)
from scratch_llm.utils import atomic_write, save_json


def compare_training_runs(
    run_dirs: Sequence[str | os.PathLike[str]],
    *,
    output_dir: str | os.PathLike[str],
) -> RunComparisonArtifacts:
    """Validate local runs and atomically replace deterministic reports."""

    snapshots = load_run_snapshots(run_dirs)
    payload = build_comparison_payload(snapshots)
    markdown = render_comparison_markdown(payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    markdown_path = atomic_write(destination / "comparison.md", markdown)
    json_path = save_json(payload, destination / "comparison.json")
    return RunComparisonArtifacts(
        json_path=json_path,
        markdown_path=markdown_path,
    )


__all__ = [
    "RUN_COMPARISON_FORMAT",
    "RUN_COMPARISON_FORMAT_VERSION",
    "RunComparisonArtifacts",
    "RunComparisonError",
    "compare_training_runs",
]
