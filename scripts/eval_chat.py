"""Evaluate a supervised-finetuned chat model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.evaluation.chat.chatcore import CHATCORE_TASK_ORDER
from scratch_llm.evaluation.chat.execution import LocalPythonExecutor
from scratch_llm.evaluation.chat.generative import GenerativeEvaluationConfig
from scratch_llm.evaluation.chat.pipeline import evaluate_checkpoint_chat_model
from scratch_llm.evaluation.chat.reporting import (
    ChatEvaluationError,
    ChatEvaluationReportConflictError,
    ChatEvaluationSettings,
    ChatTaskResult,
    normalize_chat_task_names,
)
from scratch_llm.evaluation.sft_sampling import FixedSFTSamplingConfig
from scratch_llm.run import RunConflictError
from scratch_llm.training.checkpoint import CheckpointError
from scratch_llm.utils import get_device
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "eval_chat"


def build_parser() -> argparse.ArgumentParser:
    """Return the chat-evaluation command parser."""

    parser = config_parser(COMMAND, "Evaluate a chat model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="SFT checkpoint to evaluate; required unless --dry-run is used.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help=(
            "Prepared offline Hub parquet cache root containing the requested "
            "tasks; required unless --dry-run is used."
        ),
    )
    parser.add_argument(
        "--tasks",
        default=",".join(CHATCORE_TASK_ORDER),
        metavar="TASKS",
        help=(
            "Comma-separated task filter (default: ARC-Easy, ARC-Challenge, "
            "MMLU, GSM8K, HumanEval)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Categorical evaluation batch size (default: 8).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        help="Optional deterministic per-task problem limit.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Generative samples per problem using pass-any scoring (default: 1).",
    )
    parser.add_argument(
        "--allow-generated-code-execution",
        action="store_true",
        help=(
            "Explicitly allow local HumanEval generated-code execution. The "
            "resource-limited subprocess is not safe for malicious or adversarial "
            "code."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chat-evaluation command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)
    try:
        task_names = normalize_chat_task_names(arguments.tasks)
        if config.generation.top_p is not None:
            raise ChatEvaluationError(
                "generation.top_p is not implemented for chat evaluation"
            )
        seed = (
            config.run.seed
            if config.generation.seed is None
            else config.generation.seed
        )
        generation = GenerativeEvaluationConfig(
            num_samples=arguments.num_samples,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            seed=seed,
        )
        fixed_sampling = FixedSFTSamplingConfig(
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            seed=seed,
        )
        code_execution_allowed = (
            "HumanEval" in task_names and arguments.allow_generated_code_execution
        )
        executor = LocalPythonExecutor() if code_execution_allowed else None
        settings = ChatEvaluationSettings(
            task_names=task_names,
            batch_size=arguments.batch_size,
            max_problems=arguments.max_problems,
            generation=generation,
            fixed_sampling=fixed_sampling,
            allow_generated_code_execution=code_execution_allowed,
            executor_identity=None if executor is None else executor.identity,
        )
        if not arguments.dry_run:
            get_device(config.run.device)
    except (ChatEvaluationError, TypeError, ValueError) as error:
        parser.error(str(error))

    if (
        not arguments.dry_run
        and "HumanEval" in task_names
        and not arguments.allow_generated_code_execution
    ):
        parser.error(
            "HumanEval requires the explicit generated-code execution opt-in; "
            "pass --allow-generated-code-execution only after accepting that the "
            "resource-limited subprocess is not safe for malicious or adversarial "
            "code"
        )

    if not arguments.dry_run:
        if arguments.checkpoint is None:
            parser.error("--checkpoint is required unless --dry-run is used")
        if arguments.cache_root is None:
            parser.error("--cache-root is required unless --dry-run is used")
        if not arguments.checkpoint.is_file():
            parser.error(f"--checkpoint must be a regular file: {arguments.checkpoint}")
        if not arguments.cache_root.is_dir():
            parser.error(
                f"--cache-root must be an existing directory: {arguments.cache_root}"
            )

    paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)

    if arguments.dry_run:
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print(f"Tasks: {','.join(task_names)}")
            print(f"Scope: {settings.kind}")
            print(f"Max problems: {settings.max_problems}")
            print(f"Generative samples per problem: {settings.generation.num_samples}")
            print(
                "Generated-code execution: "
                + ("allowed" if settings.allow_generated_code_execution else "disabled")
            )
            print("Resolved values:")
            print(config.to_yaml(), end="")
        return 0

    assert arguments.checkpoint is not None  # Validated before tracker setup.
    assert arguments.cache_root is not None  # Validated before tracker setup.
    try:
        with tracker:
            output = evaluate_checkpoint_chat_model(
                config,
                checkpoint_path=arguments.checkpoint,
                cache_root=arguments.cache_root,
                settings=settings,
                run_dir=paths.run_dir,
                executor=executor,
                progress=_print_task_progress,
            )
    except (
        ChatEvaluationError,
        ChatEvaluationReportConflictError,
        CheckpointError,
        OSError,
        RunConflictError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(f"Run directory: {paths.run_dir}")
    print(f"Scope: {settings.kind}")
    print(f"Report: {output.report_path}")
    return 0


def _print_task_progress(result: ChatTaskResult) -> None:
    """Print one content-free line after a requested task completes."""

    print(
        f"{result.task_name}: accuracy={result.accuracy:.4f}, "
        f"passed={result.passed_count}/{result.evaluated_count}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
