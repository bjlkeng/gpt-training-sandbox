"""Run a bounded production-pretraining throughput benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from scratch_llm.attention_backends import preflight_attention_backend
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemoryError
from scratch_llm.training.pretraining import (
    PretrainingError,
    validate_production_pretraining_config,
)
from scratch_llm.diagnostics.resource_estimation import (
    estimate_training_resources,
    render_training_resource_estimate,
)
from scratch_llm.diagnostics.throughput import (
    THROUGHPUT_BENCHMARK_PROTOCOL_ID,
    ThroughputBenchmarkConflictError,
    build_throughput_benchmark,
    report_throughput_benchmark,
)
from scratch_llm.diagnostics.throughput_runtime import (
    execute_production_throughput_benchmark,
)
from scratch_llm.data.tokenized import TokenizedDataError
from scratch_llm.utils import get_device
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "benchmark_pretrain"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Return the bounded benchmark command parser."""

    parser = config_parser(
        COMMAND,
        "Benchmark production pretraining through shared optimizer-step telemetry.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Completed optimizer steps excluded from benchmark aggregates.",
    )
    parser.add_argument(
        "--timed-steps",
        type=int,
        default=10,
        help="Completed optimizer steps included in benchmark aggregates.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve one production preset and run or describe the protocol."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)
    if arguments.warmup_steps <= 0:
        parser.error("--warmup-steps must be a positive integer")
    if arguments.timed_steps <= 0:
        parser.error("--timed-steps must be a positive integer")
    if arguments.warmup_steps + arguments.timed_steps > config.train.max_steps:
        parser.error("--warmup-steps + --timed-steps cannot exceed train.max_steps")
    try:
        validate_production_pretraining_config(config)
        resource_estimate = estimate_training_resources(config)
        attention_preflight = preflight_attention_backend(
            config.model,
            device=get_device(config.run.device),
            dtype=config.train.dtype,
            training=True,
        )
    except (
        OverflowError,
        PretrainingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))

    paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
    if arguments.dry_run:
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
            print(f"Benchmark protocol: {THROUGHPUT_BENCHMARK_PROTOCOL_ID}")
            print(f"Warmup steps: {arguments.warmup_steps}")
            print(f"Timed steps: {arguments.timed_steps}")
            print(
                "Requested attention backend: "
                f"{attention_preflight.selection.requested_backend}"
            )
            print(
                "Effective attention backend: "
                f"{attention_preflight.selection.effective_backend}"
            )
            print(
                "Attention fallback reason: "
                f"{attention_preflight.selection.fallback_reason}"
            )
            print(f"Requested torch.compile: {config.train.compile}")
            print(
                "Effective torch.compile: "
                f"{'pending execution' if config.train.compile else False}"
            )
            print(f"Activation checkpointing: {config.train.activation_checkpointing}")
            print(f"Resource estimate JSON: {resource_estimate.to_json()}")
            print(render_training_resource_estimate(resource_estimate))
        return 0

    with tracker:
        try:
            execution = execute_production_throughput_benchmark(
                config,
                warmup_steps=arguments.warmup_steps,
                timed_steps=arguments.timed_steps,
                repository_root=PROJECT_ROOT,
                progress=lambda message: print(
                    message,
                    file=sys.stderr,
                    flush=True,
                ),
            )
            completed = build_throughput_benchmark(
                config,
                execution=execution,
                warmup_steps=arguments.warmup_steps,
                timed_steps=arguments.timed_steps,
            )
            artifacts = report_throughput_benchmark(
                completed,
                run_dir=paths.run_dir,
                tracker=tracker,
            )
        except (
            AcceleratorMemoryError,
            OSError,
            PretrainingError,
            RuntimeError,
            ThroughputBenchmarkConflictError,
            TokenizedDataError,
            TypeError,
            ValueError,
        ) as error:
            parser.error(str(error))

    measurements = completed.payload["measurements"]
    assert isinstance(measurements, dict)
    optimization_state = completed.payload["optimization_state"]
    assert isinstance(optimization_state, dict)
    attention = optimization_state["attention"]
    assert isinstance(attention, dict)
    compile_state = optimization_state["compile"]
    assert isinstance(compile_state, dict)
    activation_checkpointing = optimization_state["activation_checkpointing"]
    assert isinstance(activation_checkpointing, dict)
    print(f"Run directory: {paths.run_dir}")
    print(f"Report: {artifacts.report_path}")
    print(f"Tokens/sec: {measurements['tokens_per_second']}")
    print(f"MFU: {measurements['mfu']}")
    print(f"Peak allocated MiB: {measurements['peak_allocated_mib']}")
    print(f"Requested attention backend: {attention['requested_backend']}")
    print(f"Effective attention backend: {attention['effective_backend']}")
    print(f"Attention fallback reason: {attention['fallback_reason']}")
    print(f"Requested torch.compile: {compile_state['requested']}")
    print(f"Effective torch.compile: {compile_state['effective']}")
    print(f"Compile duration seconds: {compile_state['compile_duration_seconds']}")
    print(f"Compile fallback reason: {compile_state['fallback_reason']}")
    print(f"Activation checkpointing: {activation_checkpointing['effective']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
