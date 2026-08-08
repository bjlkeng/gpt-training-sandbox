"""Opt-in local CUDA timing and memory probe for SDPA and FlashAttention."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from scratch_llm.attention_backends import (
    AttentionBackendError,
    preflight_attention_backend,
    run_flash_attention,
)
from scratch_llm.config import GPTConfig
from scratch_llm.utils import save_json


OPT_IN_ENVIRONMENT_VARIABLE = "SCRATCH_LLM_RUN_FLASH_BENCHMARK"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record local long-context SDPA/FlashAttention time and CUDA peak memory."
        )
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--provider", choices=("auto", "fa2", "fa3"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if os.environ.get(OPT_IN_ENVIRONMENT_VARIABLE) != "1":
        parser.error(
            f"set {OPT_IN_ENVIRONMENT_VARIABLE}=1 to opt in to the local CUDA probe"
        )
    for name in (
        "sequence_length",
        "batch_size",
        "heads",
        "head_dim",
        "warmup_steps",
        "timed_steps",
    ):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    device = torch.device("cuda")
    dtype = torch.float16 if arguments.dtype == "float16" else torch.bfloat16
    config = GPTConfig(
        vocab_size=32,
        seq_len=arguments.sequence_length,
        n_layer=1,
        n_head=arguments.heads,
        n_embd=arguments.heads * arguments.head_dim,
        attention_backend="flash",
        attention_fallback_policy="error",
        flash_attention_provider=arguments.provider,
    )
    try:
        resolution = preflight_attention_backend(
            config,
            device=device,
            dtype=dtype,
            training=True,
        )
    except AttentionBackendError as error:
        parser.error(str(error))
    if resolution.provider is None:  # pragma: no cover - strict selection invariant.
        raise AssertionError("strict flash preflight did not return a provider")

    torch.manual_seed(97)
    shape = (
        arguments.batch_size,
        arguments.heads,
        arguments.sequence_length,
        arguments.head_dim,
    )
    q, k, v = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        for _ in range(3)
    )

    def sdpa() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

    def flash() -> torch.Tensor:
        assert resolution.provider is not None
        return run_flash_attention(
            resolution.provider,
            q,
            k,
            v,
            dropout_p=0.0,
            causal=True,
        )

    payload = {
        "config": {
            "batch_size": arguments.batch_size,
            "dtype": arguments.dtype,
            "head_dim": arguments.head_dim,
            "heads": arguments.heads,
            "sequence_length": arguments.sequence_length,
            "timed_steps": arguments.timed_steps,
            "warmup_steps": arguments.warmup_steps,
        },
        "cuda": {
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "device_name": torch.cuda.get_device_name(device),
            "torch_cuda_version": torch.version.cuda,
        },
        "flash_selection": resolution.selection.to_dict(),
        "measurements": {
            "flash": _measure(
                flash,
                tensors=(q, k, v),
                device=device,
                warmup_steps=arguments.warmup_steps,
                timed_steps=arguments.timed_steps,
            ),
            "sdpa": _measure(
                sdpa,
                tensors=(q, k, v),
                device=device,
                warmup_steps=arguments.warmup_steps,
                timed_steps=arguments.timed_steps,
            ),
        },
    }
    if arguments.output is not None:
        save_json(payload, arguments.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _measure(
    operation: Callable[[], torch.Tensor],
    *,
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, float | int]:
    for _ in range(warmup_steps):
        _forward_backward(operation, tensors)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    for _ in range(timed_steps):
        _forward_backward(operation, tensors)
    torch.cuda.synchronize(device)
    elapsed_seconds = perf_counter() - started
    return {
        "elapsed_seconds": elapsed_seconds,
        "milliseconds_per_step": elapsed_seconds * 1000 / timed_steps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _forward_backward(
    operation: Callable[[], torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
) -> None:
    operation().float().square().mean().backward()
    for tensor in tensors:
        tensor.grad = None


if __name__ == "__main__":
    raise SystemExit(main())
