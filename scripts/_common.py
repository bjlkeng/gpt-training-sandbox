"""Shared, dependency-light conventions for command-module skeletons."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from scratch_llm.config import (
    ConfigValidationError,
    GenerationConfig,
    ProjectConfig,
    apply_generation_overrides,
    load_config,
)
from scratch_llm.chat import ChatEventTracker
from scratch_llm.run import RunConflictError, RunPaths, prepare_run
from scratch_llm.tracking import RunTracker, build_tracker
from scratch_llm.tracking_state import TrackingState


def config_parser(command: str, description: str) -> argparse.ArgumentParser:
    """Build the common interface for config-driven pipeline commands."""

    parser = argparse.ArgumentParser(
        prog=f"python -m scripts.{command}",
        description=description,
    )
    add_config_arguments(parser, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve configuration and prepare run paths without doing work.",
    )
    return parser


def add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool,
) -> None:
    """Add the shared project-config and optional W&B source arguments."""

    parser.add_argument(
        "--config",
        type=Path,
        required=required,
        help=(
            "YAML configuration file to resolve."
            if required
            else "Optional YAML configuration enabling chat tracking."
        ),
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


def add_optional_chat_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """Add opt-in config resolution to a checkpoint-driven chat command."""

    add_config_arguments(parser, required=False)


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


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared checkpoint-generation device and sampling overrides."""

    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for checkpoint loading and generation (default: cpu).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the checkpoint's generation.max_new_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override the checkpoint's generation.temperature; zero is greedy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override the checkpoint's generation.top_k.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the checkpoint's generation.seed.",
    )


def resolve_generation_arguments(
    defaults: GenerationConfig,
    arguments: argparse.Namespace,
) -> GenerationConfig:
    """Apply explicit CLI values to detached canonical checkpoint defaults."""

    if not isinstance(defaults, GenerationConfig):
        raise TypeError(
            "checkpoint generation defaults must be a GenerationConfig, "
            f"got {type(defaults).__name__}"
        )
    return apply_generation_overrides(
        defaults,
        {
            field: getattr(arguments, field)
            for field in ("max_new_tokens", "temperature", "top_k", "seed")
        },
    )


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
    wandb_resume_state: TrackingState | None = None,
) -> tuple[RunPaths, RunTracker]:
    """Prepare run paths and assemble the command's shared tracker."""

    try:
        paths = prepare_run(config)
        tracker = build_tracker(
            config,
            paths,
            stage=command,
            wandb_resume_state=wandb_resume_state,
        )
    except (
        ModuleNotFoundError,
        OSError,
        RunConflictError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return paths, tracker


def _prepare_optional_chat_tracking(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    *,
    command: str,
) -> tuple[RunTracker | None, ChatEventTracker | None]:
    """Build chat tracking only when any project-config source was supplied."""

    requested = any(
        (
            arguments.config is not None,
            bool(arguments.overrides),
            arguments.wandb_enabled is not None,
            arguments.wandb_mode is not None,
        )
    )
    if not requested:
        return None, None
    config = resolve_config_arguments(parser, arguments)
    _paths, tracker = prepare_tracked_run(parser, config, command=command)
    wandb = config.tracking.wandb
    try:
        chat_tracking = ChatEventTracker(
            tracker,
            run_id=config.run.name,
            log_prompts=wandb.log_prompts,
            log_responses=wandb.log_responses,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        try:
            tracker.fail()
        finally:
            parser.error(str(error))
    return tracker, chat_tracking


@contextmanager
def optional_chat_tracking(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    *,
    command: str,
) -> Iterator[ChatEventTracker | None]:
    """Own optional tracker lifecycle and yield the shared chat boundary."""

    tracker, chat_tracking = _prepare_optional_chat_tracking(
        parser,
        arguments,
        command=command,
    )
    if tracker is None:
        yield None
        return
    with tracker:
        yield chat_tracking


def run_checkpoint_stub(
    parser: argparse.ArgumentParser,
    *,
    command: str,
    argv: Sequence[str] | None = None,
) -> int:
    """Parse a checkpoint command and reject its unimplemented execution path."""

    parser.parse_args(argv)
    parser.error(f"scripts.{command} execution is not implemented yet")
