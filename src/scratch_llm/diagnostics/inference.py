"""Shared-generation inference timing, aggregation, and atomic reporting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Final

import torch
from torch import nn

from scratch_llm._validation import (
    JsonValueValidator,
    require_finite_non_negative_real,
    require_finite_positive_real,
    require_integer,
    require_non_empty_string,
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.attention_backends import AttentionBackendSelection
from scratch_llm.config import GPTConfig
from scratch_llm.diagnostics.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
    reset_accelerator_memory_peak,
)
from scratch_llm.generation import (
    GeneratedSequence,
    GeneratedToken,
    GenerationComplete,
    GenerationMode,
    stream_generate_sequence,
)
from scratch_llm.identity import canonical_json_identity
from scratch_llm.tracking import RunTracker, Tracker
from scratch_llm.training.compilation import CompileSelection
from scratch_llm.utils import load_json, save_json


INFERENCE_BENCHMARK_FORMAT: Final = "scratch_llm_inference_benchmark"
INFERENCE_BENCHMARK_FORMAT_VERSION: Final = 3
INFERENCE_BENCHMARK_PROTOCOL_ID: Final = "shared_generation_value_embedding_kv_cache_v3"
INFERENCE_FLOPS_FORMULA_ID: Final = "gpt_value_embedding_inference_v3"
INFERENCE_BYTES_FORMULA_ID: Final = "parameter_and_visible_kv_decode_bytes_v2"
_REPORT_RELATIVE_PATH = Path("metrics/inference_bench.json")
_SUMMARY_METHOD = "linear_interpolation_r7"
_QUANTILES = (0.5, 0.9, 0.95)
_JSON_VALUES: Final = JsonValueValidator(ValueError)


class InferenceBenchmarkMismatchError(RuntimeError):
    """Naive and cached shared generation did not produce the same result."""


class InferenceBenchmarkConflictError(RuntimeError):
    """An installed report belongs to another inference protocol identity."""


_EXISTING_JSON_VALUES: Final = JsonValueValidator(InferenceBenchmarkConflictError)


@dataclass(frozen=True)
class InferenceBenchmarkSettings:
    """Reproducible sampling, timing, and hardware-utilization inputs."""

    warmup_iterations: int
    timed_iterations: int
    max_new_tokens: int
    temperature: float
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    peak_flops_per_second: float | None = None
    peak_flops_basis: str | None = None
    peak_memory_bandwidth_bytes_per_second: float | None = None
    peak_memory_bandwidth_basis: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "warmup_iterations",
            require_positive_integer(
                self.warmup_iterations,
                name="warmup_iterations",
            ),
        )
        object.__setattr__(
            self,
            "timed_iterations",
            require_positive_integer(self.timed_iterations, name="timed_iterations"),
        )
        object.__setattr__(
            self,
            "max_new_tokens",
            require_positive_integer(self.max_new_tokens, name="max_new_tokens"),
        )
        object.__setattr__(
            self,
            "temperature",
            require_finite_non_negative_real(self.temperature, name="temperature"),
        )
        if self.top_k is not None:
            object.__setattr__(
                self,
                "top_k",
                require_positive_integer(self.top_k, name="top_k"),
            )
        if self.top_p is not None:
            top_p = require_finite_positive_real(self.top_p, name="top_p")
            if top_p > 1:
                raise ValueError("top_p must be at most 1")
            object.__setattr__(self, "top_p", top_p)
        if self.seed is not None:
            object.__setattr__(self, "seed", require_integer(self.seed, name="seed"))
        if not isinstance(self.stop_token_ids, tuple):
            raise TypeError("stop_token_ids must be a tuple")
        normalized_stop_ids = tuple(
            require_non_negative_integer(token_id, name=f"stop_token_ids[{index}]")
            for index, token_id in enumerate(self.stop_token_ids)
        )
        if len(set(normalized_stop_ids)) != len(normalized_stop_ids):
            raise ValueError("stop_token_ids must not contain duplicates")
        object.__setattr__(self, "stop_token_ids", normalized_stop_ids)
        _validate_peak_basis(
            self.peak_flops_per_second,
            self.peak_flops_basis,
            value_name="peak_flops_per_second",
            basis_name="peak_flops_basis",
        )
        _validate_peak_basis(
            self.peak_memory_bandwidth_bytes_per_second,
            self.peak_memory_bandwidth_basis,
            value_name="peak_memory_bandwidth_bytes_per_second",
            basis_name="peak_memory_bandwidth_basis",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "peak_flops_basis": self.peak_flops_basis,
            "peak_flops_per_second": self.peak_flops_per_second,
            "peak_memory_bandwidth_basis": self.peak_memory_bandwidth_basis,
            "peak_memory_bandwidth_bytes_per_second": (
                self.peak_memory_bandwidth_bytes_per_second
            ),
            "seed": self.seed,
            "stop_token_ids": list(self.stop_token_ids),
            "temperature": self.temperature,
            "timed_iterations": self.timed_iterations,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "warmup_iterations": self.warmup_iterations,
        }


@dataclass(frozen=True)
class InferenceIteration:
    """One timed shared-generation request with synchronized phase timings."""

    mode: GenerationMode
    sequence: GeneratedSequence
    prompt_context_tokens: int
    forward_query_lengths: tuple[int, ...]
    prefill_seconds: float
    time_to_first_token_seconds: float
    decode_seconds: float | None
    end_to_end_seconds: float
    memory: AcceleratorMemorySnapshot

    def __post_init__(self) -> None:
        if self.mode not in ("naive", "cached"):
            raise ValueError("mode must be 'naive' or 'cached'")
        if not isinstance(self.sequence, GeneratedSequence):
            raise TypeError("sequence must be a GeneratedSequence")
        prompt_context_tokens = require_positive_integer(
            self.prompt_context_tokens,
            name="prompt_context_tokens",
        )
        if prompt_context_tokens > len(self.sequence.prompt_token_ids):
            raise ValueError("prompt_context_tokens exceeds the prompt length")
        if not isinstance(self.forward_query_lengths, tuple):
            raise TypeError("forward_query_lengths must be a tuple")
        forward_lengths = tuple(
            require_positive_integer(length, name=f"forward_query_lengths[{index}]")
            for index, length in enumerate(self.forward_query_lengths)
        )
        if len(forward_lengths) != self.sequence.sampled_token_count:
            raise ValueError(
                "forward_query_lengths must contain one entry per sampled token"
            )
        if not forward_lengths or forward_lengths[0] != prompt_context_tokens:
            raise ValueError("the first forward must prefill the cropped prompt")
        if self.mode == "cached" and any(length != 1 for length in forward_lengths[1:]):
            raise ValueError("cached decode forwards must contain exactly one token")
        object.__setattr__(self, "forward_query_lengths", forward_lengths)
        prefill_seconds = require_finite_positive_real(
            self.prefill_seconds,
            name="prefill_seconds",
        )
        time_to_first = require_finite_positive_real(
            self.time_to_first_token_seconds,
            name="time_to_first_token_seconds",
        )
        end_to_end = require_finite_positive_real(
            self.end_to_end_seconds,
            name="end_to_end_seconds",
        )
        if prefill_seconds > time_to_first or time_to_first > end_to_end:
            raise ValueError(
                "phase timings must satisfy prefill <= first token <= end to end"
            )
        decode_token_count = self.sequence.sampled_token_count - 1
        if decode_token_count == 0:
            if self.decode_seconds is not None:
                raise ValueError("decode_seconds must be null without steady decode")
        else:
            decode_seconds = require_finite_positive_real(
                self.decode_seconds,
                name="decode_seconds",
            )
            if decode_seconds > end_to_end:
                raise ValueError("decode_seconds cannot exceed end_to_end_seconds")
            object.__setattr__(self, "decode_seconds", decode_seconds)
        if not isinstance(self.memory, AcceleratorMemorySnapshot):
            raise TypeError("memory must be an AcceleratorMemorySnapshot")
        object.__setattr__(self, "prefill_seconds", prefill_seconds)
        object.__setattr__(self, "time_to_first_token_seconds", time_to_first)
        object.__setattr__(self, "end_to_end_seconds", end_to_end)

    @property
    def decode_token_count(self) -> int:
        return self.sequence.sampled_token_count - 1

    @property
    def decode_ms_per_token(self) -> float | None:
        if self.decode_seconds is None:
            return None
        return self.decode_seconds * 1000 / self.decode_token_count

    @property
    def tokens_per_second(self) -> float | None:
        if self.decode_seconds is None:
            return None
        return self.decode_token_count / self.decode_seconds


@dataclass(frozen=True)
class InferenceTimingResult:
    """Timed mode pairs returned by the dependency-injectable runtime."""

    naive_iterations: tuple[InferenceIteration, ...]
    cached_iterations: tuple[InferenceIteration, ...]
    parameter_bytes: int
    cache_metadata: Mapping[str, object]


@dataclass(frozen=True)
class InferenceBenchmarkExecution:
    """Timed requests plus immutable checkpoint and runtime identities."""

    naive_iterations: tuple[InferenceIteration, ...]
    cached_iterations: tuple[InferenceIteration, ...]
    checkpoint_load_seconds: float
    parameter_bytes: int
    cache_metadata: Mapping[str, object]
    checkpoint_identity: str
    checkpoint_config_identity: str
    tokenizer_identity: str
    hardware_identity: Mapping[str, object]
    cuda_identity: Mapping[str, object]
    pytorch_identity: Mapping[str, object]
    code_identity: Mapping[str, object]
    device: str
    dtype: str
    attention_selection: AttentionBackendSelection
    compile_selection: CompileSelection

    def __post_init__(self) -> None:
        _validate_iteration_pairs(self.naive_iterations, self.cached_iterations)
        object.__setattr__(
            self,
            "checkpoint_load_seconds",
            require_finite_positive_real(
                self.checkpoint_load_seconds,
                name="checkpoint_load_seconds",
            ),
        )
        object.__setattr__(
            self,
            "parameter_bytes",
            require_positive_integer(self.parameter_bytes, name="parameter_bytes"),
        )
        for name in (
            "checkpoint_identity",
            "checkpoint_config_identity",
            "tokenizer_identity",
            "device",
            "dtype",
        ):
            object.__setattr__(
                self,
                name,
                require_non_empty_string(getattr(self, name), name=name),
            )
        for name in (
            "cache_metadata",
            "hardware_identity",
            "cuda_identity",
            "pytorch_identity",
            "code_identity",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    _json_object(
                        dict(value) if isinstance(value, Mapping) else value,
                        label=name,
                    )
                ),
            )
        if not isinstance(self.attention_selection, AttentionBackendSelection):
            raise TypeError("attention_selection must be an AttentionBackendSelection")
        if not isinstance(self.compile_selection, CompileSelection):
            raise TypeError("compile_selection must be a CompileSelection")


@dataclass(frozen=True)
class CompletedInferenceBenchmark:
    """One validated immutable inference benchmark payload."""

    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        canonical = _json_object(self.payload, label="inference benchmark payload")
        if canonical.get("format") != INFERENCE_BENCHMARK_FORMAT:
            raise ValueError("inference benchmark format is invalid")
        if canonical.get("format_version") != INFERENCE_BENCHMARK_FORMAT_VERSION:
            raise ValueError("inference benchmark version is invalid")
        if canonical.get("status") != "completed":
            raise ValueError("inference benchmark must be completed")
        object.__setattr__(self, "payload", MappingProxyType(canonical))

    @property
    def protocol_identity(self) -> str:
        identity = self.payload["protocol_identity"]
        assert isinstance(identity, str)
        return identity

    def to_dict(self) -> dict[str, object]:
        return _json_object(dict(self.payload), label="inference benchmark payload")


@dataclass(frozen=True)
class InferenceBenchmarkArtifacts:
    report_path: Path
    completed: CompletedInferenceBenchmark


def run_shared_inference_benchmark(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    settings: InferenceBenchmarkSettings,
    clock: Callable[[], float] = perf_counter,
    synchronize: Callable[[torch.device], None] | None = None,
    reset_memory_peak: Callable[[str | torch.device], bool] = (
        reset_accelerator_memory_peak
    ),
    collect_memory: Callable[[str | torch.device], AcceleratorMemorySnapshot] = (
        collect_accelerator_memory
    ),
) -> InferenceTimingResult:
    """Run interleaved naive/cached requests through the shared generator."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (1, sequence)")
    if token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise ValueError("token_ids must contain exactly one non-empty sequence")
    if not isinstance(settings, InferenceBenchmarkSettings):
        raise TypeError("settings must be InferenceBenchmarkSettings")
    if settings.top_p is not None:
        raise ValueError("top_p sampling is not implemented by shared generation")
    if not callable(clock):
        raise TypeError("clock must be callable")
    active_synchronize = _synchronize_device if synchronize is None else synchronize
    if not callable(active_synchronize):
        raise TypeError("synchronize must be callable")
    device = token_ids.device
    parameter_owner = getattr(model, "canonical_model", model)
    if not isinstance(parameter_owner, nn.Module):
        raise TypeError("model.canonical_model must be an nn.Module")
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameter_owner.parameters()
    )
    if parameter_bytes <= 0:
        raise ValueError("model must expose at least one parameter byte")
    cache_metadata = _probe_cache_metadata(model)

    for warmup_index in range(settings.warmup_iterations):
        naive = _run_generation_iteration(
            model,
            token_ids,
            mode="naive",
            settings=settings,
            device=device,
            clock=clock,
            synchronize=active_synchronize,
            measure_memory=False,
            reset_memory_peak=reset_memory_peak,
            collect_memory=collect_memory,
        )
        cached = _run_generation_iteration(
            model,
            token_ids,
            mode="cached",
            settings=settings,
            device=device,
            clock=clock,
            synchronize=active_synchronize,
            measure_memory=False,
            reset_memory_peak=reset_memory_peak,
            collect_memory=collect_memory,
        )
        _assert_sequence_parity(
            naive.sequence,
            cached.sequence,
            label=f"warmup iteration {warmup_index + 1}",
        )

    naive_iterations: list[InferenceIteration] = []
    cached_iterations: list[InferenceIteration] = []
    for timed_index in range(settings.timed_iterations):
        naive = _run_generation_iteration(
            model,
            token_ids,
            mode="naive",
            settings=settings,
            device=device,
            clock=clock,
            synchronize=active_synchronize,
            measure_memory=True,
            reset_memory_peak=reset_memory_peak,
            collect_memory=collect_memory,
        )
        cached = _run_generation_iteration(
            model,
            token_ids,
            mode="cached",
            settings=settings,
            device=device,
            clock=clock,
            synchronize=active_synchronize,
            measure_memory=True,
            reset_memory_peak=reset_memory_peak,
            collect_memory=collect_memory,
        )
        _assert_sequence_parity(
            naive.sequence,
            cached.sequence,
            label=f"timed iteration {timed_index + 1}",
        )
        naive_iterations.append(naive)
        cached_iterations.append(cached)
    return InferenceTimingResult(
        naive_iterations=tuple(naive_iterations),
        cached_iterations=tuple(cached_iterations),
        parameter_bytes=parameter_bytes,
        cache_metadata=MappingProxyType(cache_metadata),
    )


def build_inference_benchmark(
    model_config: GPTConfig,
    *,
    settings: InferenceBenchmarkSettings,
    execution: InferenceBenchmarkExecution,
    prompt_text: str | None = None,
    generated_text: str | None = None,
) -> CompletedInferenceBenchmark:
    """Validate parity and aggregate synchronized requests into one report."""

    if not isinstance(model_config, GPTConfig):
        raise TypeError("model_config must be a GPTConfig")
    model_config.validate()
    if not isinstance(settings, InferenceBenchmarkSettings):
        raise TypeError("settings must be InferenceBenchmarkSettings")
    if not isinstance(execution, InferenceBenchmarkExecution):
        raise TypeError("execution must be an InferenceBenchmarkExecution")
    if len(execution.naive_iterations) != settings.timed_iterations:
        raise ValueError("execution iteration count does not match timed_iterations")
    _validate_all_parity(execution)
    reference = execution.cached_iterations[0].sequence
    identities = {
        "checkpoint": execution.checkpoint_identity,
        "checkpoint_config": execution.checkpoint_config_identity,
        "code": dict(execution.code_identity),
        "cuda": dict(execution.cuda_identity),
        "generated_token_ids": canonical_json_identity(
            list(reference.generated_token_ids)
        ),
        "hardware": dict(execution.hardware_identity),
        "prompt_token_ids": canonical_json_identity(list(reference.prompt_token_ids)),
        "pytorch": dict(execution.pytorch_identity),
        "tokenizer": execution.tokenizer_identity,
    }
    attention_identity = execution.attention_selection.to_dict()
    attention_identity["window"] = model_config.attention_window_identity()
    optimization_state = {
        "attention": attention_identity,
        "cache": {
            "canonical_checkpoint_default": model_config.use_kv_cache,
            "compared_modes": ["naive", "cached"],
            "effective_by_mode": {"cached": True, "naive": False},
            "requested_by_mode": {"cached": True, "naive": False},
        },
        "compile": execution.compile_selection.to_dict(),
        "value_embeddings": model_config.value_embedding_identity(),
    }
    protocol = {
        "clock": "time.perf_counter",
        "context_policy": {
            "cached_overflow": "reject_before_runtime_mutation",
            "naive": "crop_to_last_model_max_seq_len_each_forward",
            "cached": "one_exact_cropped_prefill_then_single_token_decode",
            "model_max_seq_len": model_config.seq_len,
        },
        "excluded_work": [
            "checkpoint and tokenizer loading (reported separately)",
            "cold compiler startup (reported separately)",
            "warmup iterations",
            "report construction and Tracker fan-out",
        ],
        "id": INFERENCE_BENCHMARK_PROTOCOL_ID,
        "iteration_order": "interleaved_naive_then_cached",
        "summary_statistics": {
            "method": _SUMMARY_METHOD,
            "quantiles": list(_QUANTILES),
        },
        "synchronization": {
            "boundary": "before and after timed forward/event boundaries",
            "cuda": "torch.cuda.synchronize",
            "non_cuda": "no-op",
        },
        "timed_iterations": settings.timed_iterations,
        "version": 3,
        "warmup_iterations": settings.warmup_iterations,
    }
    payload: dict[str, object] = {
        "completion": {
            "completion_reason": reference.completion_reason,
            "generated_tokens": len(reference.generated_token_ids),
            "prompt_context_tokens": execution.cached_iterations[
                0
            ].prompt_context_tokens,
            "prompt_tokens": len(reference.prompt_token_ids),
            "sampled_tokens": reference.sampled_token_count,
            "stop_token_id": reference.stop_token_id,
        },
        "content_policy": {
            "generated_text_included": generated_text is not None,
            "prompt_text_included": prompt_text is not None,
            "token_ids_included": False,
        },
        "format": INFERENCE_BENCHMARK_FORMAT,
        "format_version": INFERENCE_BENCHMARK_FORMAT_VERSION,
        "identities": identities,
        "modes": {
            "cached": _aggregate_mode(
                "cached",
                execution.cached_iterations,
                model_config=model_config,
                settings=settings,
                parameter_bytes=execution.parameter_bytes,
                cache_metadata=execution.cache_metadata,
            ),
            "naive": _aggregate_mode(
                "naive",
                execution.naive_iterations,
                model_config=model_config,
                settings=settings,
                parameter_bytes=execution.parameter_bytes,
                cache_metadata=execution.cache_metadata,
            ),
        },
        "optimization_state": optimization_state,
        "protocol": protocol,
        "runtime": {
            "device": execution.device,
            "dtype": execution.dtype,
        },
        "sampling": settings.to_dict(),
        "startup": {
            "checkpoint_load_seconds": execution.checkpoint_load_seconds,
            "compile_seconds": execution.compile_selection.compile_duration_seconds,
            "compile_timing_includes": (
                "construction and first lazy execution"
                if execution.compile_selection.requested
                else "not requested"
            ),
        },
        "status": "completed",
    }
    if prompt_text is not None or generated_text is not None:
        payload["content"] = {
            "generated_text": generated_text,
            "prompt_text": prompt_text,
        }
    protocol_identity = _payload_identity(
        {
            "content_policy": payload["content_policy"],
            "identities": identities,
            "optimization_state": optimization_state,
            "protocol": protocol,
            "runtime": payload["runtime"],
            "sampling": payload["sampling"],
        }
    )
    payload["protocol_identity"] = protocol_identity
    return CompletedInferenceBenchmark(payload)


def report_inference_benchmark(
    completed: CompletedInferenceBenchmark,
    *,
    run_dir: str | Path,
    tracker: Tracker,
) -> InferenceBenchmarkArtifacts:
    """Atomically write one report and idempotently publish finalized metrics."""

    if not isinstance(completed, CompletedInferenceBenchmark):
        raise TypeError("completed must be a CompletedInferenceBenchmark")
    if not isinstance(tracker, Tracker):
        raise TypeError("tracker must be a Tracker")
    report_path = Path(run_dir) / _REPORT_RELATIVE_PATH
    payload = completed.to_dict()
    _validate_existing_protocol(report_path, completed.protocol_identity)
    report_path = save_json(payload, report_path)
    metrics = _tracking_metrics(payload)
    event_prefix = f"inference-benchmark:{_payload_identity(payload)}"
    if isinstance(tracker, RunTracker):
        tracker.log_once(metrics, event_id=f"{event_prefix}:metrics")
        tracker.log_artifact_once(
            _REPORT_RELATIVE_PATH.as_posix(),
            "inference_bench",
            "benchmark",
            event_id=f"{event_prefix}:artifact",
        )
    else:
        tracker.log(metrics)
        tracker.log_artifact(
            _REPORT_RELATIVE_PATH.as_posix(),
            "inference_bench",
            "benchmark",
        )
    return InferenceBenchmarkArtifacts(report_path=report_path, completed=completed)


def inference_benchmark_metrics(
    completed: CompletedInferenceBenchmark,
) -> dict[str, object]:
    """Return the exact finalized ``inference/*`` tracking namespace."""

    if not isinstance(completed, CompletedInferenceBenchmark):
        raise TypeError("completed must be a CompletedInferenceBenchmark")
    return _tracking_metrics(completed.to_dict())


def _run_generation_iteration(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    mode: GenerationMode,
    settings: InferenceBenchmarkSettings,
    device: torch.device,
    clock: Callable[[], float],
    synchronize: Callable[[torch.device], None],
    measure_memory: bool,
    reset_memory_peak: Callable[[str | torch.device], bool],
    collect_memory: Callable[[str | torch.device], AcceleratorMemorySnapshot],
) -> InferenceIteration:
    forward_starts: list[float] = []
    forward_durations: list[float] = []
    forward_lengths: list[int] = []

    def before_forward(
        _module: nn.Module,
        args: tuple[object, ...],
    ) -> None:
        inputs = args[0]
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("benchmark model forward input must be a Tensor")
        synchronize(device)
        forward_starts.append(clock())
        forward_lengths.append(inputs.shape[1])

    def after_forward(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
    ) -> None:
        synchronize(device)
        if len(forward_starts) != len(forward_durations) + 1:
            raise RuntimeError("model forward timing hooks became unbalanced")
        forward_durations.append(clock() - forward_starts[-1])

    memory_reset = False
    if measure_memory:
        memory_reset = reset_memory_peak(device)
        if not isinstance(memory_reset, bool):
            raise TypeError("reset_memory_peak must return a boolean")
    pre_handle = model.register_forward_pre_hook(before_forward)
    post_handle = model.register_forward_hook(after_forward)
    stream = stream_generate_sequence(
        model,
        token_ids,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_k=settings.top_k,
        seed=settings.seed,
        stop_token_ids=settings.stop_token_ids,
        mode=mode,
    )
    sample_times: list[float] = []
    completion: GeneratedSequence | None = None
    synchronize(device)
    request_started = clock()
    completed_at: float | None = None
    try:
        for event in stream:
            synchronize(device)
            event_time = clock()
            if isinstance(event, GeneratedToken):
                sample_times.append(event_time)
                continue
            if not isinstance(event, GenerationComplete):  # pragma: no cover
                raise RuntimeError("shared generator emitted an unknown event")
            completion = event.sequence
            if completion.completion_reason == "stop_token":
                sample_times.append(event_time)
            completed_at = event_time
    finally:
        stream.close()
        pre_handle.remove()
        post_handle.remove()
    if completion is None or completed_at is None:
        raise RuntimeError("shared generation ended without completion metadata")
    if len(sample_times) != completion.sampled_token_count:
        raise RuntimeError("sample timing count does not match shared generation")
    if len(forward_durations) != completion.sampled_token_count:
        raise RuntimeError("forward timing count does not match shared generation")
    max_seq_len = getattr(model, "max_seq_len", None)
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len <= 0
    ):
        raise ValueError("benchmark model.max_seq_len must be a positive integer")
    if measure_memory:
        memory = collect_memory(device)
        if not isinstance(memory, AcceleratorMemorySnapshot):
            raise TypeError("collect_memory must return AcceleratorMemorySnapshot")
        if not memory_reset and memory.available:
            raise RuntimeError(
                "peak reset was unavailable but memory collection reported available"
            )
    else:
        memory = AcceleratorMemorySnapshot(
            device=device,
            available=False,
            unavailable_reason="warmup memory is intentionally not retained",
        )
    decode_seconds = (
        None if len(sample_times) == 1 else sample_times[-1] - sample_times[0]
    )
    return InferenceIteration(
        mode=mode,
        sequence=completion,
        prompt_context_tokens=min(token_ids.shape[1], max_seq_len),
        forward_query_lengths=tuple(forward_lengths),
        prefill_seconds=forward_durations[0],
        time_to_first_token_seconds=sample_times[0] - request_started,
        decode_seconds=decode_seconds,
        end_to_end_seconds=completed_at - request_started,
        memory=memory,
    )


def _probe_cache_metadata(model: nn.Module) -> dict[str, object]:
    factory = getattr(model, "create_kv_cache", None)
    max_seq_len = getattr(model, "max_seq_len", None)
    if not callable(factory) or not isinstance(max_seq_len, int):
        raise TypeError("benchmark model must expose create_kv_cache and max_seq_len")
    cache = factory(batch_size=1, capacity=max_seq_len)
    try:
        metadata = getattr(cache, "metadata", None)
        to_dict = getattr(metadata, "to_dict", None)
        if callable(to_dict):
            return _json_object(to_dict(), label="cache metadata")
        return {
            "allocated_bytes": getattr(cache, "allocated_bytes", None),
            "bytes_per_token": getattr(cache, "bytes_per_token", None),
            "capacity": max_seq_len,
        }
    finally:
        reset = getattr(cache, "reset", None)
        if callable(reset):
            reset()


def _aggregate_mode(
    mode: GenerationMode,
    iterations: tuple[InferenceIteration, ...],
    *,
    model_config: GPTConfig,
    settings: InferenceBenchmarkSettings,
    parameter_bytes: int,
    cache_metadata: Mapping[str, object],
) -> dict[str, object]:
    prefill_ms = [item.prefill_seconds * 1000 for item in iterations]
    first_token_ms = [item.time_to_first_token_seconds * 1000 for item in iterations]
    end_to_end_ms = [item.end_to_end_seconds * 1000 for item in iterations]
    decode_ms = [
        value for item in iterations if (value := item.decode_ms_per_token) is not None
    ]
    throughputs = [
        value for item in iterations if (value := item.tokens_per_second) is not None
    ]
    memory = _aggregate_memory(tuple(item.memory for item in iterations))
    flops = [_estimate_decode_flops(model_config, item) for item in iterations]
    byte_estimates = [
        _estimate_decode_bytes(
            item,
            mode=mode,
            parameter_bytes=parameter_bytes,
            cache_metadata=cache_metadata,
        )
        for item in iterations
    ]
    cache_traffic = [
        _estimate_decode_cache_traffic(
            item,
            mode=mode,
            model_config=model_config,
            cache_metadata=cache_metadata,
        )
        for item in iterations
    ]
    mfu = _utilization_summary(
        numerators=flops,
        iterations=iterations,
        peak_value=settings.peak_flops_per_second,
        peak_description=settings.peak_flops_basis,
        peak_key="flops_per_second",
        unavailable_reason="peak FLOP/s basis was not configured",
    )
    mbu = _utilization_summary(
        numerators=byte_estimates,
        iterations=iterations,
        peak_value=settings.peak_memory_bandwidth_bytes_per_second,
        peak_description=settings.peak_memory_bandwidth_basis,
        peak_key="bytes_per_second",
        unavailable_reason="peak memory-bandwidth basis was not configured",
    )
    sequence = iterations[0].sequence
    return {
        "cache": {
            "allocated_bytes": (
                cache_metadata.get("allocated_bytes") if mode == "cached" else 0
            ),
            "bytes_per_token": (
                cache_metadata.get("bytes_per_token") if mode == "cached" else 0
            ),
            "capacity": cache_metadata.get("capacity") if mode == "cached" else 0,
            "enabled": mode == "cached",
            "layer_window_sizes": (
                cache_metadata.get("layer_window_sizes") if mode == "cached" else []
            ),
            "read_bytes_per_iteration": cache_traffic[0][0],
            "write_bytes_per_iteration": cache_traffic[0][1],
        },
        "counts": {
            "generated_tokens": len(sequence.generated_token_ids),
            "prompt_context_tokens": iterations[0].prompt_context_tokens,
            "prompt_tokens": len(sequence.prompt_token_ids),
            "sampled_tokens": sequence.sampled_token_count,
            "steady_decode_tokens": iterations[0].decode_token_count,
        },
        "formulae": {
            "decode_bytes": {
                "assumptions": [
                    "parameters are read once per steady decode forward",
                    "cached mode additionally reads visible external K/V and writes one K/V token",
                    "allocator traffic, activations, sampling, and framework overhead are excluded",
                ],
                "estimated_bytes_per_iteration": byte_estimates[0],
                "formula_id": INFERENCE_BYTES_FORMULA_ID,
                "parameter_bytes": parameter_bytes,
            },
            "decode_flops": {
                "assumptions": [
                    "one multiply-accumulate is two FLOPs",
                    "linear projections include QKV, attention output, MLP, and LM head",
                    "enabled value-gate projections are included",
                    "attention includes QK scores and weighted-value products",
                    "embedding lookup, sigmoid, elementwise mixing, normalization, activation, softmax, and sampling FLOPs are excluded",
                ],
                "estimated_flops_per_iteration": flops[0],
                "formula_id": INFERENCE_FLOPS_FORMULA_ID,
            },
        },
        "iterations": [
            {
                "decode_ms_per_token": item.decode_ms_per_token,
                "end_to_end_ms": item.end_to_end_seconds * 1000,
                "forward_query_lengths": list(item.forward_query_lengths),
                "peak_allocated_bytes": item.memory.peak_allocated_bytes,
                "peak_reserved_bytes": item.memory.peak_reserved_bytes,
                "prefill_ms": item.prefill_seconds * 1000,
                "time_to_first_token_ms": item.time_to_first_token_seconds * 1000,
                "tokens_per_second": item.tokens_per_second,
            }
            for item in iterations
        ],
        "latency": {
            "decode_ms_per_token": _summary(decode_ms),
            "end_to_end_ms": _summary(end_to_end_ms),
            "prefill_ms": _summary(prefill_ms),
            "time_to_first_token_ms": _summary(first_token_ms),
        },
        "memory": memory,
        "throughput": {"tokens_per_second": _summary(throughputs)},
        "utilization": {"mbu": mbu, "mfu": mfu},
    }


def _estimate_decode_flops(config: GPTConfig, iteration: InferenceIteration) -> int:
    channels = config.n_embd
    if config.n_kv_head is None:  # pragma: no cover - validated resolution.
        raise RuntimeError("validated config lost n_kv_head")
    kv_width = config.n_kv_head * (channels // config.n_head)
    per_layer_weights = (
        2 * channels**2 + 2 * channels * kv_width + 2 * config.mlp_ratio * channels**2
    )
    value_gate_weights = (
        len(config.value_embedding_layer_indices())
        * config.value_embedding_gate_channels
        * config.n_kv_head
    )
    executed_weights = (
        config.n_layer * per_layer_weights
        + value_gate_weights
        + channels * config.vocab_size
    )
    layer_windows = config.layer_attention_windows()
    total = 0
    for decode_index, query_length in enumerate(iteration.forward_query_lengths[1:], 1):
        if iteration.mode == "cached":
            key_length = iteration.prompt_context_tokens + decode_index
        else:
            key_length = query_length
        total += 2 * executed_weights * query_length
        for window in layer_windows:
            effective_key_length = (
                key_length if window is None else min(key_length, window + 1)
            )
            total += 4 * channels * query_length * effective_key_length
    return total


def _estimate_decode_bytes(
    iteration: InferenceIteration,
    *,
    mode: GenerationMode,
    parameter_bytes: int,
    cache_metadata: Mapping[str, object],
) -> int:
    decode_calls = iteration.decode_token_count
    total = parameter_bytes * decode_calls
    if mode == "naive" or decode_calls == 0:
        return total
    read_bytes, write_bytes = _estimate_decode_cache_traffic(
        iteration,
        mode=mode,
        model_config=None,
        cache_metadata=cache_metadata,
    )
    return total + read_bytes + write_bytes


def _estimate_decode_cache_traffic(
    iteration: InferenceIteration,
    *,
    mode: GenerationMode,
    model_config: GPTConfig | None,
    cache_metadata: Mapping[str, object],
) -> tuple[int, int]:
    """Return logical external K/V read bytes and physical append bytes."""

    decode_calls = iteration.decode_token_count
    if mode == "naive" or decode_calls == 0:
        return 0, 0
    bytes_per_token = cache_metadata.get("bytes_per_token")
    if isinstance(bytes_per_token, bool) or not isinstance(bytes_per_token, int):
        raise ValueError("cache metadata bytes_per_token must be an integer")
    layer_windows = cache_metadata.get("layer_window_sizes")
    if layer_windows is None:
        if model_config is None:
            layer_count = cache_metadata.get("layer_count")
            if isinstance(layer_count, bool) or not isinstance(layer_count, int):
                layer_count = 1
            active_windows: list[int | None] = [None] * layer_count
        else:
            active_windows = list(model_config.layer_attention_windows())
    elif not isinstance(layer_windows, list) or not layer_windows:
        raise ValueError("cache metadata layer_window_sizes must be a non-empty list")
    else:
        active_windows = []
        for index, window in enumerate(layer_windows):
            if window is not None and (
                isinstance(window, bool) or not isinstance(window, int) or window <= 0
            ):
                raise ValueError(
                    f"cache metadata layer_window_sizes[{index}] is invalid"
                )
            active_windows.append(window)
    if bytes_per_token % len(active_windows):
        raise ValueError("cache bytes_per_token must divide evenly across layers")
    bytes_per_layer_token = bytes_per_token // len(active_windows)
    read_bytes = 0
    for decode_index in range(1, decode_calls + 1):
        visible_tokens = iteration.prompt_context_tokens + decode_index
        read_bytes += bytes_per_layer_token * sum(
            visible_tokens if window is None else min(visible_tokens, window + 1)
            for window in active_windows
        )
    return read_bytes, bytes_per_token * decode_calls


def _utilization_summary(
    *,
    numerators: list[int],
    iterations: tuple[InferenceIteration, ...],
    peak_value: float | None,
    peak_description: str | None,
    peak_key: str,
    unavailable_reason: str,
) -> dict[str, object]:
    if peak_value is None:
        return {
            "basis": None,
            "unavailable_reason": unavailable_reason,
            "value": None,
        }
    assert peak_description is not None
    values = []
    for numerator, iteration in zip(numerators, iterations, strict=True):
        if iteration.decode_seconds is None:
            continue
        values.append(numerator / iteration.decode_seconds / peak_value)
    if not values:
        return {
            "basis": {
                peak_key: peak_value,
                "description": peak_description,
            },
            "unavailable_reason": "request sampled no steady decode tokens",
            "value": None,
        }
    return {
        "basis": {
            peak_key: float(peak_value),
            "description": peak_description,
        },
        "unavailable_reason": None,
        "value": _summary(values),
    }


def _aggregate_memory(
    snapshots: tuple[AcceleratorMemorySnapshot, ...],
) -> dict[str, object]:
    available = {snapshot.available for snapshot in snapshots}
    if len(available) != 1:
        raise ValueError("timed memory snapshots cannot mix availability")
    if not snapshots[0].available:
        reasons = sorted({str(snapshot.unavailable_reason) for snapshot in snapshots})
        return {
            "peak_allocated_bytes": None,
            "peak_allocated_mib": None,
            "peak_reserved_bytes": None,
            "peak_reserved_mib": None,
            "unavailable_reason": "; ".join(reasons),
        }
    peak_allocated = max(_present(item.peak_allocated_bytes) for item in snapshots)
    peak_reserved = max(_present(item.peak_reserved_bytes) for item in snapshots)
    return {
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_mib": peak_allocated / 1024**2,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_mib": peak_reserved / 1024**2,
        "unavailable_reason": None,
    }


def _summary(values: list[float]) -> dict[str, object] | None:
    if not values:
        return None
    validated = [
        require_finite_non_negative_real(value, name=f"sample[{index}]")
        for index, value in enumerate(values)
    ]
    ordered = sorted(validated)
    return {
        "count": len(ordered),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p50": _quantile(ordered, 0.5),
        "p90": _quantile(ordered, 0.9),
        "p95": _quantile(ordered, 0.95),
    }


def _quantile(ordered: list[float], quantile: float) -> float:
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _tracking_metrics(payload: dict[str, object]) -> dict[str, object]:
    modes = payload["modes"]
    assert isinstance(modes, dict)
    cached = modes["cached"]
    naive = modes["naive"]
    assert isinstance(cached, dict) and isinstance(naive, dict)
    completion = payload["completion"]
    sampling = payload["sampling"]
    assert isinstance(completion, dict) and isinstance(sampling, dict)

    def p50(mode_payload: dict[str, object], group: str, field: str) -> object:
        group_payload = mode_payload[group]
        assert isinstance(group_payload, dict)
        summary = group_payload[field]
        if summary is None:
            return None
        assert isinstance(summary, dict)
        return summary["p50"]

    cached_memory = cached["memory"]
    cached_cache = cached["cache"]
    cached_utilization = cached["utilization"]
    assert isinstance(cached_memory, dict)
    assert isinstance(cached_cache, dict)
    assert isinstance(cached_utilization, dict)

    def utilization(name: str) -> object:
        payload_value = cached_utilization[name]
        assert isinstance(payload_value, dict)
        summary = payload_value["value"]
        return None if summary is None else summary["p50"]

    return {
        "inference/decode_ms_per_token": p50(cached, "latency", "decode_ms_per_token"),
        "inference/end_to_end_ms": p50(cached, "latency", "end_to_end_ms"),
        "inference/generated_tokens": completion["generated_tokens"],
        "inference/kv_cache_bytes_per_token": cached_cache["bytes_per_token"],
        "inference/kv_cache_capacity": cached_cache["capacity"],
        "inference/kv_cache_enabled": True,
        "inference/kv_cache_read_bytes": cached_cache["read_bytes_per_iteration"],
        "inference/mbu": utilization("mbu"),
        "inference/mfu": utilization("mfu"),
        "inference/naive/decode_ms_per_token": p50(
            naive, "latency", "decode_ms_per_token"
        ),
        "inference/naive/prefill_ms": p50(naive, "latency", "prefill_ms"),
        "inference/naive/tokens_per_second": p50(
            naive, "throughput", "tokens_per_second"
        ),
        "inference/peak_memory_mib": cached_memory["peak_allocated_mib"],
        "inference/prefill_ms": p50(cached, "latency", "prefill_ms"),
        "inference/prompt_tokens": completion["prompt_tokens"],
        "inference/sampled_tokens": completion["sampled_tokens"],
        "inference/time_to_first_token_ms": p50(
            cached, "latency", "time_to_first_token_ms"
        ),
        "inference/temperature": sampling["temperature"],
        "inference/tokens_per_second": p50(cached, "throughput", "tokens_per_second"),
        "inference/top_k": sampling["top_k"],
        "inference/top_p": sampling["top_p"],
    }


def _validate_iteration_pairs(
    naive: tuple[InferenceIteration, ...],
    cached: tuple[InferenceIteration, ...],
) -> None:
    if not isinstance(naive, tuple) or not isinstance(cached, tuple):
        raise TypeError("iteration collections must be tuples")
    if not naive or len(naive) != len(cached):
        raise ValueError("naive and cached iterations must be non-empty and aligned")
    if any(not isinstance(item, InferenceIteration) for item in (*naive, *cached)):
        raise TypeError("iteration collections contain an invalid value")
    if any(item.mode != "naive" for item in naive):
        raise ValueError("naive_iterations contains a non-naive result")
    if any(item.mode != "cached" for item in cached):
        raise ValueError("cached_iterations contains a non-cached result")


def _validate_all_parity(execution: InferenceBenchmarkExecution) -> None:
    reference = execution.naive_iterations[0].sequence
    for index, (naive, cached) in enumerate(
        zip(
            execution.naive_iterations,
            execution.cached_iterations,
            strict=True,
        ),
        start=1,
    ):
        _assert_sequence_parity(
            naive.sequence, cached.sequence, label=f"iteration {index}"
        )
        _assert_sequence_parity(
            reference, naive.sequence, label=f"naive iteration {index}"
        )


def _assert_sequence_parity(
    naive: GeneratedSequence,
    cached: GeneratedSequence,
    *,
    label: str,
) -> None:
    if naive.prompt_token_ids != cached.prompt_token_ids:
        raise InferenceBenchmarkMismatchError(f"{label} prompt token IDs differ")
    if naive.generated_token_ids != cached.generated_token_ids:
        raise InferenceBenchmarkMismatchError(f"{label} generated token IDs differ")
    if (
        naive.completion_reason != cached.completion_reason
        or naive.stop_token_id != cached.stop_token_id
        or naive.sampled_token_count != cached.sampled_token_count
    ):
        raise InferenceBenchmarkMismatchError(f"{label} completion metadata differs")


def _validate_peak_basis(
    value: float | None,
    basis: str | None,
    *,
    value_name: str,
    basis_name: str,
) -> None:
    if (value is None) != (basis is None):
        raise ValueError(f"{value_name} and {basis_name} must be set together")
    if value is not None:
        require_finite_positive_real(value, name=value_name)
        assert basis is not None
        require_non_empty_string(basis, name=basis_name)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _present(value: int | None) -> int:
    if value is None:  # pragma: no cover - available snapshots guarantee values.
        raise ValueError("available memory snapshot omitted a peak counter")
    return value


def _validate_existing_protocol(path: Path, expected_identity: str) -> None:
    try:
        existing = load_json(path)
    except FileNotFoundError:
        return
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InferenceBenchmarkConflictError(
            f"existing inference benchmark cannot be validated: {error}"
        ) from error
    value = _EXISTING_JSON_VALUES.require_object(
        existing,
        label=f"existing inference benchmark {path}",
    )
    reconstructed_identity = _payload_identity(
        {
            "content_policy": value.get("content_policy"),
            "identities": value.get("identities"),
            "optimization_state": value.get("optimization_state"),
            "protocol": value.get("protocol"),
            "runtime": value.get("runtime"),
            "sampling": value.get("sampling"),
        }
    )
    if (
        value.get("format") != INFERENCE_BENCHMARK_FORMAT
        or value.get("format_version") != INFERENCE_BENCHMARK_FORMAT_VERSION
        or value.get("status") != "completed"
        or value.get("protocol_identity") != reconstructed_identity
        or value.get("protocol_identity") != expected_identity
    ):
        raise InferenceBenchmarkConflictError(
            f"{path} belongs to a different inference benchmark protocol identity"
        )


def _json_object(value: object, *, label: str) -> dict[str, object]:
    try:
        canonical = json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-compatible: {error}") from error
    return _JSON_VALUES.require_object(canonical, label=label)


def _payload_identity(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "INFERENCE_BENCHMARK_FORMAT",
    "INFERENCE_BENCHMARK_FORMAT_VERSION",
    "INFERENCE_BENCHMARK_PROTOCOL_ID",
    "CompletedInferenceBenchmark",
    "InferenceBenchmarkArtifacts",
    "InferenceBenchmarkConflictError",
    "InferenceBenchmarkExecution",
    "InferenceBenchmarkMismatchError",
    "InferenceBenchmarkSettings",
    "InferenceIteration",
    "InferenceTimingResult",
    "build_inference_benchmark",
    "inference_benchmark_metrics",
    "report_inference_benchmark",
    "run_shared_inference_benchmark",
]
