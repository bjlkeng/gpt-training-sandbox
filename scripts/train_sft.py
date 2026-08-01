"""Supervised-finetune a base model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.run import RunConflictError, prepare_run
from scratch_llm.tracking import build_tracker
from scratch_llm.tracking_state import resolve_wandb_resume_state
from scratch_llm.training.checkpoint import (
    CheckpointError,
    load_checkpoint_metadata,
)
from scratch_llm.training.sft import SFTTrainingError, run_sft_training
from scripts._common import config_parser, prepare_tracked_run, resolve_config_arguments


COMMAND = "train_sft"


def build_parser() -> argparse.ArgumentParser:
    """Return the supervised-finetuning command parser."""

    parser = config_parser(COMMAND, "Supervised-finetune a base model.")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--base-checkpoint",
        type=Path,
        help="Initialize fresh SFT weights and tokenizer from this base checkpoint.",
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        help="Exactly resume model, optimizer, loader, RNG, and ranking state.",
    )
    parser.add_argument(
        "--wandb-resume",
        choices=("same", "fork"),
        help="Continue the saved W&B run or explicitly fork its identity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the supervised-finetuning command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)
    if arguments.base_checkpoint is not None:
        config.sft.base_checkpoint = str(arguments.base_checkpoint)
        try:
            config.validate()
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    if arguments.wandb_resume is not None and arguments.resume is None:
        parser.error("--wandb-resume requires --resume")

    if arguments.dry_run:
        if arguments.resume is not None:
            parser.error("--resume cannot be combined with --dry-run")
        try:
            paths = prepare_run(config)
            tracker = build_tracker(
                config,
                paths,
                stage=COMMAND,
                enable_remote=False,
            )
        except (
            OSError,
            RunConflictError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            parser.error(str(error))
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
        return 0

    if arguments.resume is None and config.sft.base_checkpoint is None:
        parser.error("train_sft requires a base checkpoint or --resume")

    wandb_resume_state = None
    if arguments.resume is not None:
        try:
            metadata = load_checkpoint_metadata(arguments.resume)
            if metadata.training_stage != "sft":
                raise CheckpointError("SFT resume requires an SFT checkpoint")
            wandb_active = (
                config.tracking.wandb.enabled
                and config.tracking.wandb.mode != "disabled"
            )
            if wandb_active:
                wandb_resume_state = resolve_wandb_resume_state(
                    metadata.tracking,
                    source_run_name=metadata.config.run.name,
                    source_output_dir=metadata.config.run.output_dir,
                    current_run_name=config.run.name,
                    current_output_dir=config.run.output_dir,
                    behavior=arguments.wandb_resume,
                )
            elif arguments.wandb_resume is not None:
                raise ValueError(
                    "--wandb-resume requires enabled, non-disabled W&B tracking"
                )
        except (CheckpointError, OSError, TypeError, ValueError) as error:
            parser.error(str(error))

    paths, tracker = prepare_tracked_run(
        parser,
        config,
        command=COMMAND,
        wandb_resume_state=wandb_resume_state,
    )
    with tracker:
        try:
            result = run_sft_training(
                config,
                paths=paths,
                tracker=tracker,
                base_checkpoint=arguments.base_checkpoint,
                resume_from=arguments.resume,
                allow_tracking_fork=arguments.wandb_resume == "fork",
            )
        except (
            CheckpointError,
            OSError,
            RunConflictError,
            RuntimeError,
            SFTTrainingError,
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
