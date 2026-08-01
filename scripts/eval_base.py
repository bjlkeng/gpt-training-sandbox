"""Evaluate a pretrained base model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.evaluation.base import (
    BaseEvaluationError,
    BaseEvaluationUnavailableError,
    normalize_base_evaluation_modes,
)
from scratch_llm.evaluation.base_pipeline import evaluate_checkpoint_base_model
from scratch_llm.evaluation.base_tracking import BaseEvaluationReportConflictError
from scratch_llm.training.checkpoint import CheckpointError, load_checkpoint_metadata
from scratch_llm.config import ProjectConfig
from scratch_llm.evaluation.core.results import CoreEvaluationError, CoreTaskResult
from scratch_llm.run import RunConflictError
from scratch_llm.data.tokenized import TokenizedDataError
from scratch_llm.tracking_state import (
    TrackingState,
    resolve_wandb_resume_state,
)
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "eval_base"


def _resolve_wandb_evaluation_state(
    config: ProjectConfig,
    checkpoint_path: Path,
) -> TrackingState | None:
    """Continue the checkpoint's remote run only for the same local run."""

    wandb = config.tracking.wandb
    if not wandb.enabled or wandb.mode == "disabled":
        return None
    metadata = load_checkpoint_metadata(checkpoint_path)
    source = metadata.config.run
    current = config.run
    same_local_run = (
        source.name == current.name
        and Path(source.output_dir).resolve() == Path(current.output_dir).resolve()
    )
    return resolve_wandb_resume_state(
        metadata.tracking,
        source_run_name=source.name,
        source_output_dir=source.output_dir,
        current_run_name=current.name,
        current_output_dir=current.output_dir,
        behavior="same" if same_local_run else "fork",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the base-model evaluation command parser."""

    parser = config_parser(COMMAND, "Evaluate a pretrained base model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Model checkpoint to evaluate; required unless --dry-run is used.",
    )
    parser.add_argument(
        "--eval",
        default="bpb,sample",
        metavar="MODES",
        help="Comma-separated evaluation modes.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        help="Optional maximum examples per CORE task.",
    )
    parser.add_argument(
        "--core-bundle",
        type=Path,
        help="Pinned local eval_bundle.zip; required for CORE evaluation.",
    )
    return parser


def _print_core_progress(result: CoreTaskResult) -> None:
    """Print one compact progress line after a CORE task completes."""

    print(
        f"CORE {result.label}: accuracy={result.accuracy:.4f}, "
        f"centered={result.centered_score:.4f}, "
        f"examples={result.evaluated_examples}, "
        f"elapsed={result.elapsed_seconds:.1f}s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the base-model evaluation command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)
    try:
        modes = normalize_base_evaluation_modes(arguments.eval)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if arguments.max_per_task is not None and arguments.max_per_task <= 0:
        parser.error("--max-per-task must be a positive integer")
    if arguments.max_per_task is not None and "core" not in modes:
        parser.error("--max-per-task requires --eval core")
    if "core" in modes and arguments.core_bundle is None:
        parser.error("--core-bundle is required for --eval core")
    if "core" not in modes and arguments.core_bundle is not None:
        parser.error("--core-bundle requires --eval core")

    if arguments.dry_run:
        paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print(f"Evaluation modes: {','.join(modes)}")
            if arguments.core_bundle is not None:
                print(f"CORE bundle: {arguments.core_bundle}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
        return 0

    if arguments.checkpoint is None:
        parser.error("--checkpoint is required unless --dry-run is used")
    try:
        wandb_resume_state = _resolve_wandb_evaluation_state(
            config,
            arguments.checkpoint,
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
            result = evaluate_checkpoint_base_model(
                config,
                checkpoint_path=arguments.checkpoint,
                modes=modes,
                tracker=tracker,
                run_dir=paths.run_dir,
                max_per_task=arguments.max_per_task,
                core_bundle_path=arguments.core_bundle,
                core_progress=_print_core_progress,
            )
        except (
            BaseEvaluationError,
            BaseEvaluationReportConflictError,
            BaseEvaluationUnavailableError,
            CheckpointError,
            CoreEvaluationError,
            OSError,
            RunConflictError,
            RuntimeError,
            TokenizedDataError,
            TypeError,
            ValueError,
        ) as error:
            parser.error(str(error))

    print(f"Run directory: {paths.run_dir}")
    print(f"Evaluation modes: {','.join(modes)}")
    print(f"Report: {result.report_path}")
    if result.sample_markdown_path is not None:
        print(f"Samples: {result.sample_markdown_path}")
    if result.core_comparison_path is not None:
        print(f"CORE comparison: {result.core_comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
