"""Checkpoint loading and runtime identities for inference benchmarking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import torch

from scratch_llm.config import GPTConfig, ProjectConfig
from scratch_llm.diagnostics.inference import (
    InferenceBenchmarkExecution,
    InferenceBenchmarkSettings,
    run_shared_inference_benchmark,
)
from scratch_llm.diagnostics.throughput_runtime import (
    collect_code_identity,
    collect_runtime_identities,
)
from scratch_llm.identity import file_identity, project_config_identity
from scratch_llm.training.checkpoint import load_model_checkpoint
from scratch_llm.training.compilation import ModelCompiler, build_compile_runtime
from scratch_llm.utils import get_device


@dataclass(frozen=True)
class ExecutedInferenceBenchmark:
    """Completed runtime inputs ready for pure aggregation and reporting."""

    model_config: GPTConfig
    settings: InferenceBenchmarkSettings
    execution: InferenceBenchmarkExecution
    prompt_text: str | None
    generated_text: str | None


def execute_checkpoint_inference_benchmark(
    checkpoint_path: str | Path,
    benchmark_config: ProjectConfig,
    *,
    prompt: str,
    settings: InferenceBenchmarkSettings,
    repository_root: str | Path,
    include_prompt_text: bool = False,
    include_generated_text: bool = False,
    clock: Callable[[], float] = perf_counter,
    compiler: ModelCompiler | None = None,
) -> ExecutedInferenceBenchmark:
    """Load one checkpoint and benchmark paired shared-generation requests."""

    if not isinstance(benchmark_config, ProjectConfig):
        raise TypeError("benchmark_config must be a ProjectConfig")
    benchmark_config.validate()
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(settings, InferenceBenchmarkSettings):
        raise TypeError("settings must be InferenceBenchmarkSettings")
    if not isinstance(include_prompt_text, bool) or not isinstance(
        include_generated_text, bool
    ):
        raise TypeError("content inclusion switches must be booleans")
    if not callable(clock):
        raise TypeError("clock must be callable")
    device = get_device(benchmark_config.run.device)
    _synchronize(device)
    load_started = clock()
    checkpoint = load_model_checkpoint(checkpoint_path, device=device)
    _synchronize(device)
    checkpoint_load_seconds = clock() - load_started
    if checkpoint_load_seconds <= 0:
        raise ValueError("checkpoint load clock must advance")

    prompt_ids = checkpoint.tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [checkpoint.tokenizer.get_bos_token_id()]
    active_settings = replace(
        settings,
        stop_token_ids=(checkpoint.tokenizer.get_bos_token_id(),),
    )
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    canonical_model = checkpoint.model
    compile_runtime = build_compile_runtime(
        canonical_model,
        benchmark_config.train,
        compiler=compiler,
        clock=clock,
    )
    timing = run_shared_inference_benchmark(
        compile_runtime.execution_model,
        token_ids,
        settings=active_settings,
        clock=clock,
    )
    model_config = checkpoint.config.model
    generated = timing.cached_iterations[0].sequence.generated_token_ids
    hardware, cuda, pytorch = collect_runtime_identities(device)
    reference = next(canonical_model.parameters())
    execution = InferenceBenchmarkExecution(
        naive_iterations=timing.naive_iterations,
        cached_iterations=timing.cached_iterations,
        checkpoint_load_seconds=checkpoint_load_seconds,
        parameter_bytes=timing.parameter_bytes,
        cache_metadata=timing.cache_metadata,
        checkpoint_identity=file_identity(checkpoint_path),
        checkpoint_config_identity=project_config_identity(checkpoint.config),
        tokenizer_identity=checkpoint.tokenizer.get_identity(),
        hardware_identity=hardware,
        cuda_identity=cuda,
        pytorch_identity=pytorch,
        code_identity=collect_code_identity(repository_root),
        device=str(device),
        dtype=str(reference.dtype).removeprefix("torch."),
        attention_selection=canonical_model.attention_backend_selection(),
        compile_selection=compile_runtime.selection,
    )
    return ExecutedInferenceBenchmark(
        model_config=model_config,
        settings=active_settings,
        execution=execution,
        prompt_text=prompt if include_prompt_text else None,
        generated_text=(
            checkpoint.tokenizer.decode(generated) if include_generated_text else None
        ),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = [
    "ExecutedInferenceBenchmark",
    "execute_checkpoint_inference_benchmark",
]
