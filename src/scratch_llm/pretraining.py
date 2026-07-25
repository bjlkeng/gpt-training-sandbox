"""Command-level composition for the first-sprint tiny-text pretraining path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from scratch_llm.checkpoint import (
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from scratch_llm.config import ProjectConfig
from scratch_llm.data import NextTokenDataset
from scratch_llm.model import GPT
from scratch_llm.optim import build_lr_scheduler, build_optimizer
from scratch_llm.run import RunPaths
from scratch_llm.tokenizer import SPECIAL_TOKENS, ByteTokenizer
from scratch_llm.tracking import Tracker
from scratch_llm.training import (
    OptimizerStepResult,
    derive_grad_accum_steps,
    run_training_steps,
)
from scratch_llm.utils import get_device, set_seed


class PretrainingError(RuntimeError):
    """The requested tiny-text pretraining run is unsafe or unsupported."""


@dataclass(frozen=True)
class PretrainingResult:
    """Artifacts and bounded step history produced by one pretraining command."""

    paths: RunPaths
    metrics_path: Path
    checkpoint_path: Path
    initial_step: int
    final_step: int
    steps: tuple[OptimizerStepResult, ...]


def _read_tiny_text(config: ProjectConfig) -> str:
    if config.data.profile != "tiny_text":
        raise PretrainingError(
            "the first-sprint pretrain command requires data.profile='tiny_text'"
        )
    source = Path(config.data.base_dir) / "fixtures" / "tiny.txt"
    try:
        return source.read_text(encoding="utf-8")
    except OSError as error:
        raise PretrainingError(
            f"could not read tiny-text corpus {source}: {error}"
        ) from error


def _build_batches(
    text: str,
    config: ProjectConfig,
    tokenizer: ByteTokenizer,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(
        token_ids,
        config.model.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
    )
    if len(dataset) < config.train.device_batch_size:
        raise PretrainingError(
            "tiny text must produce at least one complete device batch; "
            f"found {len(dataset)} examples for batch size "
            f"{config.train.device_batch_size}"
        )
    generator = torch.Generator().manual_seed(config.run.seed)
    return DataLoader(
        dataset,
        batch_size=config.train.device_batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )


def _validate_tiny_text_config(config: ProjectConfig) -> None:
    config.validate()
    if config.tokenizer.type != "byte":
        raise PretrainingError("tiny-text pretraining requires tokenizer.type='byte'")
    tokenizer = ByteTokenizer()
    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise PretrainingError(
            "tiny-text pretraining requires tokenizer.vocab_size="
            f"{tokenizer.get_vocab_size()}"
        )
    if tuple(config.tokenizer.special_tokens) != SPECIAL_TOKENS:
        raise PretrainingError(
            "tiny-text pretraining requires the ByteTokenizer special-token order"
        )
    if config.train.dtype != "float32":
        raise PretrainingError(
            "the first-sprint pretrain command supports train.dtype='float32' only"
        )
    if config.train.compile:
        raise PretrainingError(
            "the first-sprint pretrain command does not support train.compile yet"
        )
    if config.train.activation_checkpointing:
        raise PretrainingError(
            "the first-sprint pretrain command does not support "
            "train.activation_checkpointing yet"
        )


def _resume_comparison(config: ProjectConfig) -> dict[str, Any]:
    values = config.to_dict()
    run = values["run"]
    if not isinstance(run, dict):  # pragma: no cover - ProjectConfig guarantees this.
        raise TypeError("resolved run configuration must be a dictionary")
    run.pop("name")
    run.pop("output_dir")
    return values


def _validate_resume_config(
    config: ProjectConfig,
    checkpoint_config: ProjectConfig,
) -> None:
    if _resume_comparison(config) != _resume_comparison(checkpoint_config):
        raise PretrainingError(
            "resume config must match the checkpoint config except for "
            "run.name and run.output_dir"
        )


def _last_metric_step(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last_step: int | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            if record.get("record_type") != "metrics":
                continue
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError("metric step must be an integer")
            if last_step is not None and step <= last_step:
                raise ValueError("metric steps must be strictly increasing")
            last_step = step
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PretrainingError(f"invalid metrics file {path}: {error}") from error
    return last_step


def _validate_existing_outputs(
    paths: RunPaths,
    metrics_path: Path,
    *,
    initial_step: int,
    is_resume: bool,
) -> None:
    checkpoints = sorted(paths.checkpoints_dir.glob("*.pt"))
    last_metric_step = _last_metric_step(metrics_path)
    if not is_resume:
        if checkpoints or last_metric_step is not None:
            raise PretrainingError(
                f"{paths.run_dir} already contains training outputs; "
                "use --resume or choose a new run.name"
            )
        return

    if last_metric_step is not None and last_metric_step > initial_step:
        raise PretrainingError(
            f"{metrics_path} already records step {last_metric_step}, "
            f"which is newer than resume step {initial_step}"
        )
    last_checkpoint_path = paths.checkpoints_dir / "last.pt"
    if last_checkpoint_path.exists():
        last_checkpoint = load_model_checkpoint(
            last_checkpoint_path,
            device="cpu",
        )
        if last_checkpoint.step > initial_step:
            raise PretrainingError(
                f"{last_checkpoint_path} is at step {last_checkpoint.step}, "
                f"which is newer than resume step {initial_step}"
            )


def run_tiny_pretraining(
    config: ProjectConfig,
    *,
    paths: RunPaths,
    tracker: Tracker,
    resume_from: str | Path | None = None,
) -> PretrainingResult:
    """Train or resume the deterministic first-sprint tiny-text workflow."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    if not isinstance(paths, RunPaths):
        raise TypeError(f"paths must be RunPaths, got {type(paths).__name__}")
    if not isinstance(tracker, Tracker):
        raise TypeError(f"tracker must be a Tracker, got {type(tracker).__name__}")
    _validate_tiny_text_config(config)
    text = _read_tiny_text(config)
    set_seed(config.run.seed)
    device = get_device(config.run.device)

    model: GPT
    tokenizer: ByteTokenizer
    optimizer: Optimizer
    scheduler: LRScheduler
    if resume_from is None:
        tokenizer = ByteTokenizer()
        model = GPT(config.model).to(device)
        optimizer = build_optimizer(model, config.train)
        scheduler = build_lr_scheduler(optimizer, config.train)
        initial_step = 0
    else:
        checkpoint = load_training_checkpoint(resume_from, device=device)
        _validate_resume_config(config, checkpoint.config)
        model = checkpoint.model
        tokenizer = checkpoint.tokenizer
        optimizer = checkpoint.optimizer
        scheduler = checkpoint.scheduler
        initial_step = checkpoint.step
        if initial_step >= config.train.max_steps:
            raise PretrainingError(
                f"checkpoint step {initial_step} has already reached "
                f"train.max_steps={config.train.max_steps}"
            )

    batches = _build_batches(text, config, tokenizer)
    metrics_path = paths.run_dir / config.tracking.jsonl.path
    _validate_existing_outputs(
        paths,
        metrics_path,
        initial_step=initial_step,
        is_resume=resume_from is not None,
    )
    metrics_were_empty = not metrics_path.exists() or metrics_path.stat().st_size == 0
    if metrics_were_empty:
        tracker.log_config(config.to_dict())

    checkpoint_path = paths.checkpoints_dir / "last.pt"

    def save_periodic_checkpoint(
        step: int,
        _result: OptimizerStepResult,
    ) -> None:
        if step % config.train.save_every != 0:
            return
        save_checkpoint(
            paths.checkpoints_dir / f"step_{step:06d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
            tokenizer=tokenizer,
        )
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
            tokenizer=tokenizer,
        )

    grad_accum_steps = derive_grad_accum_steps(
        device_batch_size=config.train.device_batch_size,
        seq_len=config.model.seq_len,
        total_batch_size_tokens=config.train.total_batch_size_tokens,
    )
    step_results = run_training_steps(
        model,
        batches,
        optimizer,
        scheduler,
        max_steps=config.train.max_steps,
        grad_accum_steps=grad_accum_steps,
        grad_clip=config.train.grad_clip,
        device=device,
        tracker=tracker,
        log_every=config.train.log_every,
        on_step=save_periodic_checkpoint,
    )
    final_step = scheduler.last_epoch
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=final_step,
        tokenizer=tokenizer,
    )

    return PretrainingResult(
        paths=paths,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        initial_step=initial_step,
        final_step=final_step,
        steps=tuple(step_results),
    )


__all__ = [
    "PretrainingError",
    "PretrainingResult",
    "run_tiny_pretraining",
]
