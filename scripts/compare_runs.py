"""Compare completed local training runs without loading model checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.run_comparison import RunComparisonError, compare_training_runs


def build_parser() -> argparse.ArgumentParser:
    """Return the deterministic offline run-comparison parser."""

    parser = argparse.ArgumentParser(
        prog="python -m scripts.compare_runs",
        description="Compare two or more completed local training runs.",
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        help="Local run directory containing config and metrics artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for comparison.json and comparison.md.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate local artifacts and write both comparison reports."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        artifacts = compare_training_runs(
            arguments.run_dirs,
            output_dir=arguments.output_dir,
        )
    except (OSError, RunComparisonError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"JSON report: {artifacts.json_path}")
    print(f"Markdown report: {artifacts.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
