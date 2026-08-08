"""Compare identity-matched production training-throughput reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.diagnostics.throughput_comparison import (
    TRAINING_OPTIMIZATION_VARIANTS,
    TrainingOptimizationComparisonError,
    compare_training_benchmarks,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the deterministic offline benchmark-comparison parser."""

    parser = argparse.ArgumentParser(
        prog="python -m scripts.compare_training_benchmarks",
        description=(
            "Compare a float32/manual baseline with matched training optimization "
            "benchmarks."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Completed float32/manual throughput_benchmark.json report.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="OPTIMIZATION=REPORT",
        help=(
            "Completed variant report; repeat for amp, sdpa, flash, compile, "
            "activation_checkpointing, or combined."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for training_optimization_comparison.json and .md.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate local reports and write both comparison artifacts."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    variants: dict[str, Path] = {}
    for declaration in arguments.variant:
        if "=" not in declaration:
            parser.error(
                f"--variant must use OPTIMIZATION=REPORT syntax, got {declaration!r}"
            )
        name, raw_path = declaration.split("=", 1)
        if name not in TRAINING_OPTIMIZATION_VARIANTS:
            parser.error(
                f"unknown optimization {name!r}; choose from "
                + ", ".join(TRAINING_OPTIMIZATION_VARIANTS)
            )
        if not raw_path:
            parser.error(f"--variant {name!r} must include a report path")
        if name in variants:
            parser.error(f"--variant {name!r} was declared more than once")
        variants[name] = Path(raw_path)
    try:
        artifacts = compare_training_benchmarks(
            arguments.baseline,
            variants,
            output_dir=arguments.output_dir,
        )
    except (
        OSError,
        TrainingOptimizationComparisonError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(f"JSON report: {artifacts.json_path}")
    print(f"Markdown report: {artifacts.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
