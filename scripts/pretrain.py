"""Pretrain a decoder-only language model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from scratch_llm.checkpoint import (
    CheckpointError,
    load_checkpoint_metadata,
)
from scratch_llm.pretraining import PretrainingError, run_pretraining
from scratch_llm.resource_estimation import (
    estimate_training_resources,
    render_training_resource_estimate,
)
from scratch_llm.run import RunConflictError
from scratch_llm.tracking_state import resolve_wandb_resume_state
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
    parser.add_argument(
        "--allow-non-exact-resume",
        action="store_true",
        help=(
            "Explicitly migrate a legacy checkpoint without loader/RNG "
            "continuity. The resumed run is not bit-exact."
        ),
    )
    parser.add_argument(
        "--wandb-resume",
        choices=("same", "fork"),
        help=(
            "For W&B-enabled checkpoint resume, continue the saved remote run "
            "or explicitly create a new one."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pretraining command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)
    if arguments.allow_non_exact_resume and arguments.resume is None:
        parser.error("--allow-non-exact-resume requires --resume")
    if arguments.wandb_resume is not None and arguments.resume is None:
        parser.error("--wandb-resume requires --resume")
    try:
        resource_estimate = estimate_training_resources(config)
    except (OverflowError, TypeError, ValueError) as error:
        parser.error(str(error))

    if arguments.dry_run:
        if arguments.resume is not None:
            parser.error("--resume cannot be combined with --dry-run")
        paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
            print(f"Resource estimate JSON: {resource_estimate.to_json()}")
            print(render_training_resource_estimate(resource_estimate))
        return 0

    wandb_resume_state = None
    wandb_active = (
        config.tracking.wandb.enabled and config.tracking.wandb.mode != "disabled"
    )
    if arguments.resume is not None and wandb_active:
        try:
            metadata = load_checkpoint_metadata(arguments.resume)
            wandb_resume_state = resolve_wandb_resume_state(
                metadata.tracking,
                source_run_name=metadata.config.run.name,
                source_output_dir=metadata.config.run.output_dir,
                current_run_name=config.run.name,
                current_output_dir=config.run.output_dir,
                behavior=arguments.wandb_resume,
            )
        except (CheckpointError, OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    elif arguments.wandb_resume is not None:
        parser.error("--wandb-resume requires enabled, non-disabled W&B tracking")

    paths, tracker = prepare_tracked_run(
        parser,
        config,
        command=COMMAND,
        wandb_resume_state=wandb_resume_state,
    )
    with tracker:
        print(
            f"Resource estimate JSON: {resource_estimate.to_json()}",
            file=sys.stderr,
            flush=True,
        )
        print(
            render_training_resource_estimate(resource_estimate),
            file=sys.stderr,
            flush=True,
        )
        try:
            result = run_pretraining(
                config,
                paths=paths,
                tracker=tracker,
                resume_from=arguments.resume,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                allow_non_exact_resume=arguments.allow_non_exact_resume,
                allow_tracking_fork=arguments.wandb_resume == "fork",
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
