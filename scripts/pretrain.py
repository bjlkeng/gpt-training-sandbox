"""Pretrain a decoder-only language model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.checkpoint import CheckpointError
from scratch_llm.pretraining import PretrainingError, run_tiny_pretraining
from scratch_llm.run import RunConflictError
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "pretrain"


def build_parser() -> argparse.ArgumentParser:
    """Return the pretraining command parser."""

    parser = config_parser(COMMAND, "Pretrain a decoder-only language model.")
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Resume a periodic checkpoint; the config may change only "
            "run.name and run.output_dir."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pretraining command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)

    if arguments.dry_run:
        if arguments.resume is not None:
            parser.error("--resume cannot be combined with --dry-run")
        paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
        return 0

    paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
    with tracker:
        try:
            result = run_tiny_pretraining(
                config,
                paths=paths,
                tracker=tracker,
                resume_from=arguments.resume,
            )
        except (
            CheckpointError,
            OSError,
            PretrainingError,
            RunConflictError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            parser.error(str(error))

    print(f"Run directory: {result.paths.run_dir}")
    print(f"Resolved config: {result.paths.config_path}")
    if result.initial_step:
        print(f"Resumed from step {result.initial_step}")
    print(f"Completed step {result.final_step}")
    if result.steps:
        print(f"Loss: {result.steps[0].loss:.6f} -> {result.steps[-1].loss:.6f}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
