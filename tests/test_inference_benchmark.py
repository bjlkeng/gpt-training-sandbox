"""Inference benchmark timing, aggregation, and publication contracts."""

from __future__ import annotations

import json
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch

from scripts.benchmark_inference import main as benchmark_main
from scratch_llm.attention_backends import AttentionBackendSelection
from scratch_llm.config import (
    GPTConfig,
    GenerationConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.diagnostics.inference import (
    INFERENCE_BENCHMARK_FORMAT,
    INFERENCE_BENCHMARK_FORMAT_VERSION,
    INFERENCE_BENCHMARK_PROTOCOL_ID,
    InferenceBenchmarkExecution,
    InferenceBenchmarkMismatchError,
    InferenceBenchmarkSettings,
    InferenceIteration,
    build_inference_benchmark,
    report_inference_benchmark,
    run_shared_inference_benchmark,
)
from scratch_llm.diagnostics.inference_runtime import (
    execute_checkpoint_inference_benchmark,
)
from scratch_llm.generation import GeneratedSequence
from scratch_llm.model import GPT
from scratch_llm.tracking import (
    JsonlTracker,
    NullTracker,
    RunSummary,
    RunTracker,
    Tracker,
)
from scratch_llm.training.compilation import (
    CompileSelection,
    build_compile_runtime,
)
from scratch_llm.training.checkpoint import ExactTrainingState, save_checkpoint
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.rng_state import capture_training_rng_state
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_BENCHMARK_OPT_IN = "SCRATCH_LLM_RUN_INFERENCE_BENCHMARK"


class _RecordingTracker(Tracker):
    def __init__(self) -> None:
        self.metrics: list[dict[str, object]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log(self, metrics: dict[str, object], step: int | None = None) -> None:
        assert step is None
        self.metrics.append(dict(metrics))

    def log_config(self, config: dict[str, object]) -> None:
        del config

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.artifacts.append((path, name, type))

    def finish(self) -> None:
        pass


def _memory(
    peak_allocated: int | None,
    *,
    peak_reserved: int | None = None,
) -> AcceleratorMemorySnapshot:
    if peak_allocated is None:
        return AcceleratorMemorySnapshot(
            device=torch.device("cpu"),
            available=False,
            unavailable_reason="CPU allocator counters unavailable",
        )
    assert peak_reserved is not None
    return AcceleratorMemorySnapshot(
        device=torch.device("cuda", 0),
        available=True,
        allocated_bytes=peak_allocated // 2,
        reserved_bytes=peak_reserved // 2,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        capacity_bytes=24 * 1024**3,
    )


def _sequence(*, stop: bool = False) -> GeneratedSequence:
    return GeneratedSequence(
        prompt_token_ids=(1, 2),
        generated_token_ids=(3, 4),
        completion_reason="stop_token" if stop else "max_new_tokens",
        stop_token_id=7 if stop else None,
        sampled_token_count=3 if stop else 2,
    )


def _iteration(
    mode: str,
    *,
    prefill_seconds: float,
    decode_seconds: float,
    peak_allocated: int | None,
) -> InferenceIteration:
    sequence = _sequence()
    return InferenceIteration(
        mode=mode,  # type: ignore[arg-type]
        sequence=sequence,
        prompt_context_tokens=2,
        forward_query_lengths=(2, 3) if mode == "naive" else (2, 1),
        prefill_seconds=prefill_seconds,
        time_to_first_token_seconds=prefill_seconds + 0.001,
        decode_seconds=decode_seconds,
        end_to_end_seconds=prefill_seconds + decode_seconds + 0.002,
        memory=_memory(
            peak_allocated,
            peak_reserved=None if peak_allocated is None else peak_allocated + 1024,
        ),
    )


def _settings() -> InferenceBenchmarkSettings:
    return InferenceBenchmarkSettings(
        warmup_iterations=1,
        timed_iterations=2,
        max_new_tokens=2,
        temperature=0,
        top_k=None,
        top_p=None,
        seed=17,
        stop_token_ids=(7,),
        peak_flops_per_second=1_000_000,
        peak_flops_basis="fixture FP32 peak",
        peak_memory_bandwidth_bytes_per_second=2_000_000,
        peak_memory_bandwidth_basis="fixture memory peak",
    )


def _execution(*, unavailable_memory: bool = False) -> InferenceBenchmarkExecution:
    peaks = (None, None) if unavailable_memory else (10 * 1024**2, 12 * 1024**2)
    return InferenceBenchmarkExecution(
        naive_iterations=(
            _iteration(
                "naive",
                prefill_seconds=0.003,
                decode_seconds=0.008,
                peak_allocated=peaks[0],
            ),
            _iteration(
                "naive",
                prefill_seconds=0.005,
                decode_seconds=0.012,
                peak_allocated=peaks[1],
            ),
        ),
        cached_iterations=(
            _iteration(
                "cached",
                prefill_seconds=0.002,
                decode_seconds=0.006,
                peak_allocated=peaks[0],
            ),
            _iteration(
                "cached",
                prefill_seconds=0.004,
                decode_seconds=0.010,
                peak_allocated=peaks[1],
            ),
        ),
        checkpoint_load_seconds=0.25,
        parameter_bytes=4096,
        cache_metadata={
            "allocated_bytes": 1024,
            "bytes_per_token": 128,
            "capacity": 8,
        },
        checkpoint_identity="sha256:" + "a" * 64,
        checkpoint_config_identity="sha256:" + "b" * 64,
        tokenizer_identity="tokenizer:fixture",
        hardware_identity={"device": "cuda:0", "device_name": "Fake RTX"},
        cuda_identity={"available": True, "compiled_version": "fixture"},
        pytorch_identity={"version": "fixture"},
        code_identity={"commit": "abc123", "tracked_dirty": False},
        device="cuda:0",
        dtype="float32",
        attention_selection=AttentionBackendSelection(
            requested_backend="flash",
            effective_backend="sdpa",
            fallback_reason="flash_dependency_unavailable",
        ),
        compile_selection=CompileSelection(
            requested=True,
            effective=False,
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=False,
            compile_duration_seconds=0.125,
            fallback_reason="compile_execution_failed",
        ),
    )


def _model_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=16,
        seq_len=8,
        n_layer=1,
        n_head=1,
        n_embd=4,
        mlp_ratio=2,
        dropout=0,
    )


def test_aggregation_records_units_quantiles_formulas_and_fallbacks() -> None:
    completed = build_inference_benchmark(
        _model_config(),
        settings=_settings(),
        execution=_execution(),
    )
    payload = completed.to_dict()
    cached = payload["modes"]["cached"]

    assert payload["format"] == INFERENCE_BENCHMARK_FORMAT
    assert payload["format_version"] == INFERENCE_BENCHMARK_FORMAT_VERSION
    assert payload["protocol"]["summary_statistics"] == {
        "method": "linear_interpolation_r7",
        "quantiles": [0.5, 0.9, 0.95],
    }
    assert cached["latency"]["prefill_ms"] == {
        "count": 2,
        "max": 4.0,
        "mean": 3.0,
        "min": 2.0,
        "p50": 3.0,
        "p90": pytest.approx(3.8),
        "p95": pytest.approx(3.9),
    }
    assert cached["latency"]["decode_ms_per_token"]["p50"] == 8.0
    assert cached["throughput"]["tokens_per_second"]["p50"] == pytest.approx(
        (1 / 0.006 + 1 / 0.010) / 2
    )
    assert cached["memory"]["peak_allocated_mib"] == 12.0
    assert cached["cache"] == {
        "allocated_bytes": 1024,
        "bytes_per_token": 128,
        "capacity": 8,
        "enabled": True,
    }
    assert cached["utilization"]["mfu"]["value"]["p50"] is not None
    assert cached["utilization"]["mfu"]["unavailable_reason"] is None
    assert cached["utilization"]["mbu"]["basis"] == {
        "bytes_per_second": 2_000_000.0,
        "description": "fixture memory peak",
    }
    assert payload["optimization_state"]["attention"]["effective_backend"] == "sdpa"
    assert payload["optimization_state"]["compile"]["fallback_reason"] == (
        "compile_execution_failed"
    )
    assert payload["startup"]["checkpoint_load_seconds"] == 0.25
    assert payload["startup"]["compile_seconds"] == 0.125


def test_unavailable_memory_and_peak_bases_remain_null_with_reasons() -> None:
    settings = InferenceBenchmarkSettings(
        warmup_iterations=1,
        timed_iterations=2,
        max_new_tokens=2,
        temperature=0,
        seed=17,
        stop_token_ids=(7,),
    )
    completed = build_inference_benchmark(
        _model_config(),
        settings=settings,
        execution=_execution(unavailable_memory=True),
    )
    cached = completed.payload["modes"]["cached"]

    assert cached["memory"]["peak_allocated_bytes"] is None
    assert "CPU allocator" in cached["memory"]["unavailable_reason"]
    assert cached["utilization"]["mfu"] == {
        "basis": None,
        "unavailable_reason": "peak FLOP/s basis was not configured",
        "value": None,
    }
    assert cached["utilization"]["mbu"] == {
        "basis": None,
        "unavailable_reason": "peak memory-bandwidth basis was not configured",
        "value": None,
    }


def test_output_mismatch_refuses_to_build_performance_comparison() -> None:
    execution = _execution()
    mismatched = replace(
        execution,
        cached_iterations=(
            replace(
                execution.cached_iterations[0],
                sequence=GeneratedSequence(
                    prompt_token_ids=(1, 2),
                    generated_token_ids=(9, 9),
                    completion_reason="max_new_tokens",
                    stop_token_id=None,
                    sampled_token_count=2,
                ),
            ),
            execution.cached_iterations[1],
        ),
    )

    with pytest.raises(InferenceBenchmarkMismatchError, match="token IDs"):
        build_inference_benchmark(
            _model_config(),
            settings=_settings(),
            execution=mismatched,
        )


def test_report_is_atomic_idempotent_and_fans_out_exact_metrics(
    tmp_path: Path,
) -> None:
    completed = build_inference_benchmark(
        _model_config(),
        settings=_settings(),
        execution=_execution(),
    )
    local = JsonlTracker(tmp_path / "metrics.jsonl")
    remote = _RecordingTracker()
    tracker = RunTracker(
        RunSummary(
            tmp_path / "summary.json",
            run={
                "name": "inference",
                "output_dir": str(tmp_path),
                "stage": "benchmark_inference",
            },
        ),
        local,
        remote,
    )

    first = report_inference_benchmark(completed, run_dir=tmp_path, tracker=tracker)
    second = report_inference_benchmark(completed, run_dir=tmp_path, tracker=tracker)
    tracker.finish()

    assert (
        first.report_path
        == second.report_path
        == (tmp_path / "metrics/inference_bench.json")
    )
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert "content" not in report
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["record_type"] for record in records] == ["metrics", "artifact"]
    assert records[0]["metrics"] == remote.metrics[0]
    assert remote.metrics == [records[0]["metrics"]]
    assert remote.artifacts == [
        ("metrics/inference_bench.json", "inference_bench", "benchmark")
    ]
    assert set(records[0]["metrics"]) >= {
        "inference/prompt_tokens",
        "inference/generated_tokens",
        "inference/prefill_ms",
        "inference/decode_ms_per_token",
        "inference/tokens_per_second",
        "inference/peak_memory_mib",
        "inference/kv_cache_bytes_per_token",
        "inference/mbu",
        "inference/mfu",
        "inference/temperature",
        "inference/top_k",
        "inference/top_p",
    }


def test_report_works_without_wandb_or_any_remote_tracker(tmp_path: Path) -> None:
    completed = build_inference_benchmark(
        _model_config(),
        settings=_settings(),
        execution=_execution(unavailable_memory=True),
    )

    artifacts = report_inference_benchmark(
        completed,
        run_dir=tmp_path,
        tracker=NullTracker(),
    )

    assert artifacts.report_path.is_file()


def test_explicit_content_opt_ins_are_independent() -> None:
    prompt_only = build_inference_benchmark(
        _model_config(),
        settings=_settings(),
        execution=_execution(),
        prompt_text="explicit prompt",
    ).to_dict()

    assert prompt_only["content"] == {
        "generated_text": None,
        "prompt_text": "explicit prompt",
    }
    assert prompt_only["content_policy"] == {
        "generated_text_included": False,
        "prompt_text_included": True,
        "token_ids_included": False,
    }


class _Cache:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _RuntimeModel(torch.nn.Module):
    def __init__(self, *, mismatch: bool = False) -> None:
        super().__init__()
        self.max_seq_len = 8
        self.config = GPTConfig(
            vocab_size=8,
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=4,
            mlp_ratio=2,
        )
        self.mismatch = mismatch
        self.anchor = torch.nn.Parameter(torch.zeros(4))

    def create_kv_cache(self, *, batch_size: int, capacity: int) -> _Cache:
        assert (batch_size, capacity) == (1, 8)
        return _Cache()

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        kv_cache: object | None = None,
    ) -> torch.Tensor:
        next_id = 2 if self.mismatch and kv_cache is not None else 1
        logits = torch.full(
            (1, token_ids.shape[1], 8),
            -torch.inf,
            device=token_ids.device,
        )
        logits[:, -1, next_id] = 0
        return logits


def test_runtime_uses_shared_modes_warmup_sync_and_fake_clock() -> None:
    clock_value = 0.0
    synchronizations: list[str] = []

    def clock() -> float:
        nonlocal clock_value
        clock_value += 0.001
        return clock_value

    def synchronize(device: torch.device) -> None:
        synchronizations.append(str(device))

    settings = InferenceBenchmarkSettings(
        warmup_iterations=1,
        timed_iterations=2,
        max_new_tokens=3,
        temperature=0,
        seed=3,
    )
    timing = run_shared_inference_benchmark(
        _RuntimeModel(),
        torch.tensor([[3, 4]]),
        settings=settings,
        clock=clock,
        synchronize=synchronize,
        reset_memory_peak=lambda _device: False,
        collect_memory=lambda device: AcceleratorMemorySnapshot(
            device=torch.device(device),
            available=False,
            unavailable_reason="fixture counters unavailable",
        ),
    )

    assert len(timing.naive_iterations) == len(timing.cached_iterations) == 2
    assert all(
        item.forward_query_lengths == (2, 3, 4) for item in timing.naive_iterations
    )
    assert all(
        item.forward_query_lengths == (2, 1, 1) for item in timing.cached_iterations
    )
    assert all(
        item.prefill_seconds == pytest.approx(0.001)
        for item in timing.cached_iterations
    )
    assert all(item.decode_seconds is not None for item in timing.cached_iterations)
    assert synchronizations


def test_runtime_detects_warmup_output_mismatch_before_timed_results() -> None:
    settings = InferenceBenchmarkSettings(
        warmup_iterations=1,
        timed_iterations=1,
        max_new_tokens=2,
        temperature=0,
    )

    with pytest.raises(InferenceBenchmarkMismatchError, match="warmup"):
        run_shared_inference_benchmark(
            _RuntimeModel(mismatch=True),
            torch.tensor([[3]]),
            settings=settings,
        )


def test_tiny_gpt_runtime_supplies_real_cpu_measurements() -> None:
    torch.manual_seed(419)
    model = GPT(_model_config()).eval()
    timing = run_shared_inference_benchmark(
        model,
        torch.tensor([[1, 2]]),
        settings=InferenceBenchmarkSettings(
            warmup_iterations=1,
            timed_iterations=1,
            max_new_tokens=3,
            temperature=0,
        ),
    )

    naive = timing.naive_iterations[0]
    cached = timing.cached_iterations[0]
    assert naive.sequence == cached.sequence
    assert naive.forward_query_lengths == (2, 3, 4)
    assert cached.forward_query_lengths == (2, 1, 1)
    assert naive.prefill_seconds > 0
    assert cached.time_to_first_token_seconds > 0
    assert cached.tokens_per_second is not None


def test_compiled_execution_delegates_cache_contract_to_canonical_model() -> None:
    model = _RuntimeModel()
    runtime = build_compile_runtime(
        model,
        TrainConfig(
            compile=True,
            compile_backend="eager",
            compile_fallback_policy="error",
        ),
        compiler=lambda active_model, **_kwargs: active_model,
    )

    timing = run_shared_inference_benchmark(
        runtime.execution_model,
        torch.tensor([[3]]),
        settings=InferenceBenchmarkSettings(
            warmup_iterations=1,
            timed_iterations=1,
            max_new_tokens=2,
            temperature=0,
        ),
    )

    assert timing.cached_iterations[0].forward_query_lengths == (1, 1)
    assert runtime.selection.effective is True


def test_inference_cli_dry_run_is_bounded_and_does_not_load_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = benchmark_main(
        [
            "--config",
            str(PROJECT_ROOT / "configs/smoke.yaml"),
            "--checkpoint",
            str(tmp_path / "not-loaded.pt"),
            "--warmup-iterations",
            "1",
            "--timed-iterations",
            "2",
            "--dry-run",
            "-o",
            "run.name=inference-dry-run",
            "-o",
            f"run.output_dir={tmp_path.as_posix()}",
            "-o",
            "generation.max_new_tokens=2",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert f"Benchmark protocol: {INFERENCE_BENCHMARK_PROTOCOL_ID}" in output
    assert "Warmup iterations: 1" in output
    assert "Timed iterations: 2" in output
    assert "Compared cache modes: naive, cached" in output


def test_checkpoint_runtime_freezes_identities_and_default_content_privacy(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(name="checkpoint-runtime", device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
            dropout=0,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=8,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0,
        ),
        generation=GenerationConfig(
            temperature=0,
            top_k=None,
            max_new_tokens=2,
            seed=23,
        ),
    )
    model = GPT(config.model)
    for parameter in model.parameters():
        parameter.data.zero_()
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    checkpoint_path = save_checkpoint(
        tmp_path / "model.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=ByteTokenizer(),
        continuation=ExactTrainingState(
            loader_format="fixture_loader_v1",
            loader_state={"format": "fixture_loader_v1", "position": 0},
            rng_state=capture_training_rng_state("cpu"),
            tracker_step=0,
            total_training_time_seconds=0,
            total_training_flops=0,
        ),
        training_stage="pretrain",
    )
    settings = InferenceBenchmarkSettings(
        warmup_iterations=1,
        timed_iterations=1,
        max_new_tokens=2,
        temperature=0,
        seed=23,
    )

    run = execute_checkpoint_inference_benchmark(
        checkpoint_path,
        config,
        prompt="hi",
        settings=settings,
        repository_root=PROJECT_ROOT,
    )

    assert run.prompt_text is None
    assert run.generated_text is None
    assert run.settings.stop_token_ids == (ByteTokenizer().get_bos_token_id(),)
    assert run.execution.checkpoint_identity.startswith("sha256:")
    assert run.execution.checkpoint_config_identity.startswith("sha256:")
    assert run.execution.tokenizer_identity == ByteTokenizer().get_identity()
    assert run.execution.naive_iterations[0].sequence == (
        run.execution.cached_iterations[0].sequence
    )
    assert run.execution.cached_iterations[0].forward_query_lengths[-1] == 1


@pytest.mark.skipif(
    os.environ.get(DEVICE_BENCHMARK_OPT_IN) != "1",
    reason=f"set {DEVICE_BENCHMARK_OPT_IN}=1 for the bounded CUDA measurement",
)
def test_opt_in_available_device_supplies_real_measurements() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda", 0)
    torch.manual_seed(421)
    model = GPT(_model_config()).to(device).eval()
    timing = run_shared_inference_benchmark(
        model,
        torch.tensor([[1, 2]], device=device),
        settings=InferenceBenchmarkSettings(
            warmup_iterations=1,
            timed_iterations=2,
            max_new_tokens=3,
            temperature=0,
        ),
    )

    assert all(item.memory.available for item in timing.cached_iterations)
    assert all(item.prefill_seconds > 0 for item in timing.cached_iterations)
    assert all(item.tokens_per_second for item in timing.cached_iterations)
