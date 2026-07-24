"""Versioned, atomic checkpoints for base-model training and sampling."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scratch_llm.config import ProjectConfig
from scratch_llm.model import GPT
from scratch_llm.optim import (
    WarmupConstantWarmdownLR,
    build_lr_scheduler,
    build_optimizer,
)
from scratch_llm.tokenizer import (
    BYTE_VOCAB_SIZE,
    SPECIAL_TOKENS,
    ByteTokenizer,
)
from scratch_llm.utils import get_device


CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_KEYS = frozenset(
    {
        "format_version",
        "model",
        "optimizer",
        "scheduler",
        "config",
        "step",
        "tokenizer",
    }
)


class CheckpointError(RuntimeError):
    """A checkpoint does not satisfy the supported base-model contract."""


@dataclass(frozen=True)
class ModelCheckpoint:
    """Model, tokenizer, and metadata reconstructed for sampling."""

    model: GPT
    tokenizer: ByteTokenizer
    config: ProjectConfig
    step: int


@dataclass(frozen=True)
class TrainingCheckpoint(ModelCheckpoint):
    """Full optimizer and scheduler state reconstructed for training resume."""

    optimizer: Optimizer
    scheduler: LRScheduler


@dataclass(frozen=True)
class _DecodedCheckpoint:
    payload: dict[str, Any]
    config: ProjectConfig
    tokenizer: ByteTokenizer
    step: int
    device: torch.device


def _validate_save_state(
    *,
    model: GPT,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: ProjectConfig,
    step: int,
    tokenizer: ByteTokenizer,
) -> None:
    if not isinstance(model, GPT):
        raise TypeError(f"model must be a GPT, got {type(model).__name__}")
    if not isinstance(optimizer, AdamW):
        raise TypeError(f"optimizer must be an AdamW, got {type(optimizer).__name__}")
    if not isinstance(scheduler, WarmupConstantWarmdownLR):
        raise TypeError(
            "scheduler must be a WarmupConstantWarmdownLR, "
            f"got {type(scheduler).__name__}"
        )
    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError(f"step must be an integer, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError(
            f"tokenizer must be a ByteTokenizer, got {type(tokenizer).__name__}"
        )

    config.validate()
    if model.config != config.model:
        raise ValueError("model configuration does not match the resolved config")
    if scheduler.optimizer is not optimizer:
        raise ValueError("scheduler must be attached to the saved optimizer")
    if scheduler.last_epoch != step:
        raise ValueError(
            f"step {step} does not match scheduler step {scheduler.last_epoch}"
        )
    if step > config.train.max_steps:
        raise ValueError(
            f"step {step} exceeds configured max_steps {config.train.max_steps}"
        )
    if config.tokenizer.type != "byte":
        raise ValueError("base smoke checkpoints require tokenizer.type='byte'")
    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise ValueError(
            "resolved tokenizer vocabulary size does not match ByteTokenizer"
        )
    if tuple(config.tokenizer.special_tokens) != SPECIAL_TOKENS:
        raise ValueError("resolved tokenizer special tokens do not match ByteTokenizer")


def _atomic_torch_save(value: object, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        binary_file = os.fdopen(file_descriptor, mode="wb")
        file_descriptor = -1
        with binary_file:
            torch.save(value, binary_file)
            binary_file.flush()
            os.fsync(binary_file.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if file_descriptor >= 0:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: GPT,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: ProjectConfig,
    step: int,
    tokenizer: ByteTokenizer,
) -> Path:
    """Atomically save all state needed for sampling and basic training resume."""

    _validate_save_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=step,
        tokenizer=tokenizer,
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "tokenizer": {
            "type": "byte",
            "byte_vocab_size": BYTE_VOCAB_SIZE,
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": list(SPECIAL_TOKENS),
        },
    }
    return _atomic_torch_save(payload, Path(path))


def _restore_config(value: object) -> ProjectConfig:
    if not isinstance(value, dict):
        raise CheckpointError("checkpoint config must be a dictionary")
    try:
        structured = OmegaConf.structured(ProjectConfig)
        OmegaConf.set_struct(structured, True)
        resolved = OmegaConf.merge(structured, value)
        config = OmegaConf.to_object(resolved)
    except Exception as error:
        raise CheckpointError(
            f"checkpoint contains an invalid resolved config: {error}"
        ) from error
    if not isinstance(config, ProjectConfig):
        raise CheckpointError("checkpoint config did not reconstruct ProjectConfig")
    return config


def _restore_tokenizer(value: object, config: ProjectConfig) -> ByteTokenizer:
    expected_metadata = {
        "type": "byte",
        "byte_vocab_size": BYTE_VOCAB_SIZE,
        "vocab_size": BYTE_VOCAB_SIZE + len(SPECIAL_TOKENS),
        "special_tokens": list(SPECIAL_TOKENS),
    }
    if value != expected_metadata:
        raise CheckpointError(
            "checkpoint byte-tokenizer metadata does not match this runtime"
        )
    if config.tokenizer.type != "byte":
        raise CheckpointError("checkpoint config does not select the byte tokenizer")
    tokenizer = ByteTokenizer()
    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise CheckpointError(
            "checkpoint config vocabulary size does not match ByteTokenizer"
        )
    if tuple(config.tokenizer.special_tokens) != SPECIAL_TOKENS:
        raise CheckpointError(
            "checkpoint config special tokens do not match ByteTokenizer"
        )
    return tokenizer


def _load_checkpoint(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device,
) -> _DecodedCheckpoint:
    destination = Path(path)
    resolved_device = get_device(device)
    try:
        payload = torch.load(
            destination,
            map_location=resolved_device,
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointError(
            f"could not load checkpoint {destination}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be a dictionary")
    if set(payload) != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - set(payload))
        unexpected = sorted(set(payload) - _CHECKPOINT_KEYS)
        raise CheckpointError(
            "checkpoint fields do not match format version 1; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            "unsupported checkpoint format version "
            f"{payload['format_version']!r}; expected {CHECKPOINT_FORMAT_VERSION}"
        )

    step = payload["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise CheckpointError(
            f"checkpoint step must be a non-negative integer, got {step!r}"
        )
    config = _restore_config(payload["config"])
    if step > config.train.max_steps:
        raise CheckpointError(
            f"checkpoint step {step} exceeds configured max_steps "
            f"{config.train.max_steps}"
        )
    tokenizer = _restore_tokenizer(payload["tokenizer"], config)
    return _DecodedCheckpoint(
        payload=payload,
        config=config,
        tokenizer=tokenizer,
        step=step,
        device=resolved_device,
    )


def _restore_model(checkpoint: _DecodedCheckpoint) -> GPT:
    model = GPT(checkpoint.config.model).to(checkpoint.device)
    try:
        model.load_state_dict(checkpoint.payload["model"])
    except Exception as error:
        raise CheckpointError(f"could not restore model state: {error}") from error
    return model


def load_model_checkpoint(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> ModelCheckpoint:
    """Reconstruct an evaluation-mode GPT and byte tokenizer for sampling."""

    checkpoint = _load_checkpoint(path, device=device)
    model = _restore_model(checkpoint)
    model.eval()
    return ModelCheckpoint(
        model=model,
        tokenizer=checkpoint.tokenizer,
        config=checkpoint.config,
        step=checkpoint.step,
    )


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    """Reconstruct train-mode model, optimizer, and scheduler state for resume."""

    checkpoint = _load_checkpoint(path, device=device)
    model = _restore_model(checkpoint)
    optimizer = build_optimizer(model, checkpoint.config.train)
    scheduler = build_lr_scheduler(optimizer, checkpoint.config.train)
    try:
        optimizer.load_state_dict(checkpoint.payload["optimizer"])
        scheduler.load_state_dict(checkpoint.payload["scheduler"])
    except Exception as error:
        raise CheckpointError(f"could not restore training state: {error}") from error
    if scheduler.last_epoch != checkpoint.step:
        raise CheckpointError(
            f"checkpoint step {checkpoint.step} does not match restored "
            f"scheduler step {scheduler.last_epoch}"
        )
    model.train()
    return TrainingCheckpoint(
        model=model,
        tokenizer=checkpoint.tokenizer,
        config=checkpoint.config,
        step=checkpoint.step,
        optimizer=optimizer,
        scheduler=scheduler,
    )


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointError",
    "ModelCheckpoint",
    "TrainingCheckpoint",
    "load_model_checkpoint",
    "load_training_checkpoint",
    "save_checkpoint",
]
