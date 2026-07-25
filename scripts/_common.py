"""Shared, dependency-light conventions for command-module skeletons."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.config import ConfigValidationError, ProjectConfig, load_config
from scratch_llm.run import RunConflictError, RunPaths, prepare_run
from scratch_llm.tracking import RunTracker, build_tracker


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
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action="store_true",
        default=None,
        help="Enable optional W&B tracking.",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb_enabled",
        action="store_false",
        help="Disable optional W&B tracking while keeping local JSONL enabled.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        help="Select the W&B operating mode.",
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
    config = resolve_config_arguments(parser, arguments)

    if not arguments.dry_run:
        parser.error(
            f"scripts.{command} execution is not implemented yet; "
            "use --dry-run to validate its configuration"
        )

    paths, tracker = prepare_tracked_run(
        parser,
        config,
        command=command,
    )
    with tracker:
        print(f"Run directory: {paths.run_dir}")
        print(f"Resolved config: {paths.config_path}")
        print("Resolved values:")
        print(config.to_yaml(), end="")
    return 0


def resolve_config_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> ProjectConfig:
    """Resolve one config command using the shared source precedence."""

    try:
        return load_config(
            arguments.config,
            arguments.overrides,
            environment=os.environ,
            wandb_enabled=arguments.wandb_enabled,
            wandb_mode=arguments.wandb_mode,
        )
    except ConfigValidationError as error:
        parser.error(str(error))


def prepare_tracked_run(
    parser: argparse.ArgumentParser,
    config: ProjectConfig,
    *,
    command: str,
) -> tuple[RunPaths, RunTracker]:
    """Prepare run paths and assemble the command's shared tracker."""

    try:
        paths = prepare_run(config)
        tracker = build_tracker(config, paths, stage=command)
    except (ModuleNotFoundError, OSError, RunConflictError, ValueError) as error:
        parser.error(str(error))
    return paths, tracker


def run_checkpoint_stub(
    parser: argparse.ArgumentParser,
    *,
    command: str,
    argv: Sequence[str] | None = None,
) -> int:
    """Parse a checkpoint command and reject its unimplemented execution path."""

    parser.parse_args(argv)
    parser.error(f"scripts.{command} execution is not implemented yet")
