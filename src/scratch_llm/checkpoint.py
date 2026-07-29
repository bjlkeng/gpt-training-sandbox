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
from scratch_llm.bpe import RegexBPETokenizer
from scratch_llm.model import GPT
from scratch_llm.optim import (
    WarmupConstantWarmdownLR,
    build_lr_scheduler,
    build_optimizer,
)
from scratch_llm.tokenizer import (
    BYTE_VOCAB_SIZE,
    NANOCHAT_SPECIAL_TOKENS,
    ByteTokenizer,
    Tokenizer,
)
from scratch_llm.utils import get_device


CHECKPOINT_FORMAT_VERSION = 2
_SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = frozenset({1, CHECKPOINT_FORMAT_VERSION})
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
    tokenizer: Tokenizer
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
    tokenizer: Tokenizer
    step: int
    device: torch.device


def _validate_save_state(
    *,
    model: GPT,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: ProjectConfig,
    step: int,
    tokenizer: Tokenizer,
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
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
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
    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise ValueError(
            "resolved tokenizer vocabulary size does not match the tokenizer"
        )
    if tuple(config.tokenizer.special_tokens) != NANOCHAT_SPECIAL_TOKENS:
        raise ValueError(
            "resolved tokenizer special tokens do not match the nanochat vocabulary"
        )
    if tokenizer.get_special_tokens() != set(NANOCHAT_SPECIAL_TOKENS):
        raise ValueError(
            "tokenizer special tokens do not match the nanochat vocabulary"
        )
    expected_special_ids = list(
        range(
            config.tokenizer.vocab_size - len(NANOCHAT_SPECIAL_TOKENS),
            config.tokenizer.vocab_size,
        )
    )
    actual_special_ids = [
        tokenizer.encode_special(token) for token in NANOCHAT_SPECIAL_TOKENS
    ]
    if actual_special_ids != expected_special_ids:
        raise ValueError(
            "tokenizer special-token IDs do not match the resolved vocabulary"
        )
    if tokenizer.get_bos_token_id() != expected_special_ids[0]:
        raise ValueError(
            "tokenizer BOS token ID does not match the resolved vocabulary"
        )
    if config.tokenizer.type == "byte":
        expected_identity = ByteTokenizer().get_identity()
        if tokenizer.get_identity() != expected_identity:
            raise ValueError("byte-tokenizer identity does not match this runtime")


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


def _regex_artifact_metadata(
    config: ProjectConfig,
    tokenizer: Tokenizer,
) -> dict[str, object]:
    artifact_dir = config.tokenizer.artifact_dir
    if artifact_dir is None:
        raise ValueError("regex-BPE checkpoints require tokenizer.artifact_dir")
    try:
        canonical_path = Path(artifact_dir).resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"could not resolve tokenizer artifact directory {artifact_dir}: {error}"
        ) from error
    try:
        artifact_tokenizer = RegexBPETokenizer.load(canonical_path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid regex-BPE tokenizer artifact directory {canonical_path}: {error}"
        ) from error
    if artifact_tokenizer.get_identity() != tokenizer.get_identity():
        raise ValueError(
            "resolved tokenizer does not match tokenizer.artifact_dir identity"
        )
    if artifact_tokenizer.get_vocab_size() != tokenizer.get_vocab_size():
        raise ValueError(
            "resolved tokenizer does not match tokenizer.artifact_dir vocabulary"
        )
    if artifact_tokenizer.get_special_tokens() != tokenizer.get_special_tokens():
        raise ValueError(
            "resolved tokenizer does not match tokenizer.artifact_dir special tokens"
        )
    return {
        "artifact_path": str(canonical_path),
        "identity": tokenizer.get_identity(),
        "special_tokens": list(NANOCHAT_SPECIAL_TOKENS),
        "type": "regex_byte_bpe",
        "vocab_size": tokenizer.get_vocab_size(),
    }


def _tokenizer_metadata(
    config: ProjectConfig,
    tokenizer: Tokenizer,
) -> dict[str, object]:
    if config.tokenizer.type == "byte":
        return {
            "type": "byte",
            "identity": tokenizer.get_identity(),
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": list(NANOCHAT_SPECIAL_TOKENS),
        }
    return _regex_artifact_metadata(config, tokenizer)


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: GPT,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: ProjectConfig,
    step: int,
    tokenizer: Tokenizer,
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
        "tokenizer": _tokenizer_metadata(config, tokenizer),
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


def _restore_version_one_tokenizer(
    value: object,
    config: ProjectConfig,
) -> Tokenizer:
    expected_metadata = {
        "type": "byte",
        "byte_vocab_size": BYTE_VOCAB_SIZE,
        "vocab_size": BYTE_VOCAB_SIZE + len(NANOCHAT_SPECIAL_TOKENS),
        "special_tokens": list(NANOCHAT_SPECIAL_TOKENS),
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
    if tuple(config.tokenizer.special_tokens) != NANOCHAT_SPECIAL_TOKENS:
        raise CheckpointError(
            "checkpoint config special tokens do not match ByteTokenizer"
        )
    return tokenizer


def _restore_version_two_tokenizer(
    value: object,
    config: ProjectConfig,
) -> Tokenizer:
    if config.tokenizer.type == "byte":
        tokenizer: Tokenizer = ByteTokenizer()
        expected_metadata = {
            "type": "byte",
            "identity": tokenizer.get_identity(),
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": list(NANOCHAT_SPECIAL_TOKENS),
        }
        if value != expected_metadata:
            raise CheckpointError(
                "checkpoint byte-tokenizer metadata does not match this runtime"
            )
    elif config.tokenizer.type == "regex_byte_bpe":
        if not isinstance(value, dict):
            raise CheckpointError(
                "checkpoint regex-BPE tokenizer metadata must be a dictionary"
            )
        expected_keys = {
            "artifact_path",
            "identity",
            "special_tokens",
            "type",
            "vocab_size",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            unexpected = sorted(set(value) - expected_keys)
            raise CheckpointError(
                "checkpoint regex-BPE tokenizer metadata fields do not match "
                f"format version 2; missing={missing}, unexpected={unexpected}"
            )
        artifact_path = value["artifact_path"]
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            raise CheckpointError(
                "checkpoint regex-BPE artifact_path must be a non-empty string"
            )
        stored_artifact_path = Path(artifact_path)
        if not stored_artifact_path.is_absolute():
            raise CheckpointError(
                "checkpoint regex-BPE artifact_path must be canonical and absolute"
            )
        if config.tokenizer.artifact_dir is None:
            raise CheckpointError(
                "checkpoint config must define tokenizer.artifact_dir for regex-BPE"
            )
        configured_artifact_path = Path(config.tokenizer.artifact_dir)
        if (
            configured_artifact_path.is_absolute()
            and configured_artifact_path.resolve() != stored_artifact_path
        ):
            raise CheckpointError(
                "checkpoint tokenizer artifact path conflicts with the resolved config: "
                f"metadata={stored_artifact_path}, "
                f"config={configured_artifact_path.resolve()}"
            )
        try:
            tokenizer = RegexBPETokenizer.load(stored_artifact_path)
        except (OSError, TypeError, ValueError) as error:
            raise CheckpointError(
                "could not restore regex-BPE tokenizer artifacts "
                f"{stored_artifact_path}: {error}"
            ) from error
        expected_metadata = {
            "artifact_path": str(stored_artifact_path),
            "identity": tokenizer.get_identity(),
            "special_tokens": list(NANOCHAT_SPECIAL_TOKENS),
            "type": "regex_byte_bpe",
            "vocab_size": tokenizer.get_vocab_size(),
        }
        if value != expected_metadata:
            raise CheckpointError(
                "checkpoint regex-BPE tokenizer metadata conflicts with "
                "the loaded artifact"
            )
    else:  # pragma: no cover - ProjectConfig validates the supported choices.
        raise CheckpointError(
            f"unsupported checkpoint tokenizer type {config.tokenizer.type!r}"
        )

    if config.tokenizer.vocab_size != tokenizer.get_vocab_size():
        raise CheckpointError(
            "checkpoint config vocabulary size does not match the restored tokenizer"
        )
    if tuple(config.tokenizer.special_tokens) != NANOCHAT_SPECIAL_TOKENS:
        raise CheckpointError(
            "checkpoint config special tokens do not match the restored tokenizer"
        )
    return tokenizer


def _restore_tokenizer(
    value: object,
    config: ProjectConfig,
    *,
    format_version: int,
) -> Tokenizer:
    if format_version == 1:
        return _restore_version_one_tokenizer(value, config)
    return _restore_version_two_tokenizer(value, config)


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
            "checkpoint fields do not match the supported schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    format_version = payload["format_version"]
    if (
        not isinstance(format_version, int)
        or isinstance(format_version, bool)
        or format_version not in _SUPPORTED_CHECKPOINT_FORMAT_VERSIONS
    ):
        raise CheckpointError(
            "unsupported checkpoint format version "
            f"{format_version!r}; expected one of "
            f"{sorted(_SUPPORTED_CHECKPOINT_FORMAT_VERSIONS)}"
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
    tokenizer = _restore_tokenizer(
        payload["tokenizer"],
        config,
        format_version=format_version,
    )
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
    """Reconstruct an evaluation-mode GPT and its tokenizer for sampling."""

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
