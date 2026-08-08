"""Benchmark shared naive and KV-cached checkpoint inference."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemoryError
from scratch_llm.diagnostics.inference import (
    INFERENCE_BENCHMARK_PROTOCOL_ID,
    InferenceBenchmarkConflictError,
    InferenceBenchmarkMismatchError,
    InferenceBenchmarkSettings,
    build_inference_benchmark,
    inference_benchmark_metrics,
    report_inference_benchmark,
)
from scratch_llm.diagnostics.inference_runtime import (
    execute_checkpoint_inference_benchmark,
)
from scratch_llm.training.checkpoint import CheckpointError
from scratch_llm.training.compilation import CompileRuntimeError
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "benchmark_inference"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Return the reproducible paired-inference benchmark parser."""

    parser = config_parser(
        COMMAND,
        "Benchmark naive and KV-cached shared generation from one checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Versioned model checkpoint to load.",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt used for every paired request; omitted from reports by default.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=2,
        help="Paired naive/cached requests excluded from aggregates.",
    )
    parser.add_argument(
        "--timed-iterations",
        type=int,
        default=10,
        help="Paired naive/cached requests included in aggregates.",
    )
    parser.add_argument(
        "--peak-memory-bandwidth-gbps",
        type=float,
        help="Optional decimal GB/s hardware peak used as the MBU denominator.",
    )
    parser.add_argument(
        "--peak-memory-bandwidth-basis",
        help="Required description when a peak memory bandwidth is supplied.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run or describe the bounded paired-inference protocol."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)
    bandwidth = arguments.peak_memory_bandwidth_gbps
    bandwidth_basis = arguments.peak_memory_bandwidth_basis
    if (bandwidth is None) != (bandwidth_basis is None):
        parser.error(
            "--peak-memory-bandwidth-gbps and "
            "--peak-memory-bandwidth-basis must be set together"
        )
    try:
        settings = InferenceBenchmarkSettings(
            warmup_iterations=arguments.warmup_iterations,
            timed_iterations=arguments.timed_iterations,
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            seed=config.generation.seed,
            peak_flops_per_second=config.train.mfu_peak_flops_per_second,
            peak_flops_basis=config.train.mfu_peak_flops_basis,
            peak_memory_bandwidth_bytes_per_second=(
                None if bandwidth is None else bandwidth * 1_000_000_000
            ),
            peak_memory_bandwidth_basis=bandwidth_basis,
        )
        if settings.top_p is not None:
            raise ValueError("generation.top_p is not implemented by shared generation")
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
    if arguments.dry_run:
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
            print(f"Benchmark protocol: {INFERENCE_BENCHMARK_PROTOCOL_ID}")
            print(f"Checkpoint: {arguments.checkpoint}")
            print(f"Warmup iterations: {settings.warmup_iterations}")
            print(f"Timed iterations: {settings.timed_iterations}")
            print("Compared cache modes: naive, cached")
            print(f"Requested torch.compile: {config.train.compile}")
            print(f"Prompt text reporting: {config.tracking.wandb.log_prompts}")
            print(f"Generated text reporting: {config.tracking.wandb.log_responses}")
        return 0

    with tracker:
        try:
            run = execute_checkpoint_inference_benchmark(
                arguments.checkpoint,
                config,
                prompt=arguments.prompt,
                settings=settings,
                repository_root=PROJECT_ROOT,
                include_prompt_text=config.tracking.wandb.log_prompts,
                include_generated_text=config.tracking.wandb.log_responses,
            )
            completed = build_inference_benchmark(
                run.model_config,
                settings=run.settings,
                execution=run.execution,
                prompt_text=run.prompt_text,
                generated_text=run.generated_text,
            )
            artifacts = report_inference_benchmark(
                completed,
                run_dir=paths.run_dir,
                tracker=tracker,
            )
        except (
            AcceleratorMemoryError,
            CheckpointError,
            CompileRuntimeError,
            InferenceBenchmarkConflictError,
            InferenceBenchmarkMismatchError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            parser.error(str(error))

    print(f"Run directory: {paths.run_dir}")
    print(f"Report: {artifacts.report_path}")
    for name, value in inference_benchmark_metrics(completed).items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
