"""Opt-in same-config CUDA comparison for activation checkpointing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import repeat
import json
import os
from pathlib import Path

import torch

from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.throughput_runtime import run_benchmark_training_steps
from scratch_llm.model import GPT
from scratch_llm.training.activation_checkpointing import (
    configure_activation_checkpointing,
)
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.utils import save_json


OPT_IN_ENVIRONMENT_VARIABLE = "SCRATCH_LLM_RUN_ACTIVATION_CHECKPOINT_BENCHMARK"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record local same-config activation-checkpoint tokens/sec and peak VRAM."
        )
    )
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
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
        "layers",
        "heads",
        "embedding_dim",
        "warmup_steps",
        "timed_steps",
    ):
        if getattr(arguments, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.embedding_dim % arguments.heads:
        parser.error("--embedding-dim must be divisible by --heads")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if arguments.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("CUDA device does not support bfloat16")

    device = torch.device("cuda")
    total_steps = arguments.warmup_steps + arguments.timed_steps
    config = ProjectConfig(
        run=RunConfig(name="activation-checkpoint-probe", device="cuda"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=265),
        model=GPTConfig(
            vocab_size=265,
            seq_len=arguments.sequence_length,
            n_layer=arguments.layers,
            n_head=arguments.heads,
            n_embd=arguments.embedding_dim,
            mlp_ratio=4,
            attention_backend="sdpa",
        ),
        train=TrainConfig(
            device_batch_size=arguments.batch_size,
            total_batch_size_tokens=arguments.batch_size * arguments.sequence_length,
            grad_accum_steps=1,
            max_steps=total_steps,
            warmup_steps=0,
            warmdown_ratio=0.0,
            dtype=arguments.dtype,
        ),
    )
    torch.manual_seed(127)
    eager = GPT(config.model).to(device)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in eager.state_dict().items()
    }
    del eager
    torch.cuda.empty_cache()
    inputs = torch.randint(
        0,
        config.model.vocab_size,
        (arguments.batch_size, arguments.sequence_length),
        device=device,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)

    measurements: dict[str, object] = {}
    for label, enabled in (("ordinary", False), ("checkpointed", True)):
        model = GPT(config.model).to(device)
        model.load_state_dict(initial_state, strict=True)
        selection = configure_activation_checkpointing(model, enabled=enabled)
        optimizer = build_optimizer(model, config.train)
        scheduler = build_lr_scheduler(optimizer, config.train)
        execution = run_benchmark_training_steps(
            config,
            model=model,
            batches=repeat((inputs, targets)),
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_steps=arguments.warmup_steps,
            timed_steps=arguments.timed_steps,
        )
        timed = execution.steps[arguments.warmup_steps :]
        telemetry = [step.telemetry for step in timed]
        assert all(item is not None for item in telemetry)
        elapsed = sum(item.duration_seconds for item in telemetry if item is not None)
        tokens = sum(
            item.processed_model_tokens for item in telemetry if item is not None
        )
        timed_snapshots = execution.memory_snapshots[arguments.warmup_steps :]
        measurements[label] = {
            "activation_checkpointing": selection.to_dict(),
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": max(
                snapshot.peak_allocated_bytes or 0 for snapshot in timed_snapshots
            ),
            "peak_reserved_bytes": max(
                snapshot.peak_reserved_bytes or 0 for snapshot in timed_snapshots
            ),
            "tokens_per_second": tokens / elapsed,
        }
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    payload = {
        "config": config.to_dict(),
        "cuda": {
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "device_name": torch.cuda.get_device_name(device),
            "torch_cuda_version": torch.version.cuda,
        },
        "measurements": measurements,
        "protocol": {
            "same_initial_state": True,
            "threshold": None,
            "timed_steps": arguments.timed_steps,
            "warmup_steps": arguments.warmup_steps,
        },
    }
    if arguments.output is not None:
        save_json(payload, arguments.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
