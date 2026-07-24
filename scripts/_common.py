"""Shared, dependency-light conventions for command-module skeletons."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.config import ConfigValidationError, load_config
from scratch_llm.run import RunConflictError, prepare_run


def config_parser(command: str, description: str) -> argparse.ArgumentParser:
    """Build the common interface for config-driven pipeline commands."""

    parser = argparse.ArgumentParser(
        prog=f"python -m scripts.{command}",
        description=description,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration file to resolve.",
    )
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Dotted configuration override; repeat to apply in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and prepare run paths without doing work.",
    )
    return parser


def checkpoint_parser(command: str, description: str) -> argparse.ArgumentParser:
    """Build the common interface for checkpoint-driven commands."""

    parser = argparse.ArgumentParser(
        prog=f"python -m scripts.{command}",
        description=description,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to load.",
    )
    return parser


def run_config_stub(
    parser: argparse.ArgumentParser,
    *,
    command: str,
    argv: Sequence[str] | None = None,
) -> int:
    """Resolve a config dry-run or reject an unimplemented execution path."""

    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config, arguments.overrides)
    except ConfigValidationError as error:
        parser.error(str(error))

    if not arguments.dry_run:
        parser.error(
            f"scripts.{command} execution is not implemented yet; "
            "use --dry-run to validate its configuration"
        )

    try:
        paths = prepare_run(config)
    except (OSError, RunConflictError) as error:
        parser.error(str(error))

    print(f"Run directory: {paths.run_dir}")
    print(f"Resolved config: {paths.config_path}")
    print("Resolved values:")
    print(config.to_yaml(), end="")
    return 0


def run_checkpoint_stub(
    parser: argparse.ArgumentParser,
    *,
    command: str,
    argv: Sequence[str] | None = None,
) -> int:
    """Parse a checkpoint command and reject its unimplemented execution path."""

    parser.parse_args(argv)
    parser.error(f"scripts.{command} execution is not implemented yet")
