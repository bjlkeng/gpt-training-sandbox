"""Versioned, atomic checkpoints for base-model training and sampling."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from scratch_llm._validation import (
    require_finite_non_negative_real,
    require_non_empty_string,
    require_non_negative_integer,
    require_real,
)
from scratch_llm.training.best_checkpoint import (
    BestCheckpointError,
    ValidationCheckpointState,
)
from scratch_llm.config import ProjectConfig, TrainConfig
from scratch_llm.evaluation.sft_bpb import (
    SFTValidationCheckpointState,
    SFTValidationError,
)
from scratch_llm.tokenization.bpe import RegexBPETokenizer
from scratch_llm.model import GPT
from scratch_llm.training.optim import (
    WarmupConstantWarmdownLR,
    build_lr_scheduler,
    build_optimizer,
)
from scratch_llm.training.rng_state import (
    RNGStateError,
    TrainingRNGState,
    preserve_global_rng_state,
)
from scratch_llm.tokenization.tokenizer import (
    BYTE_VOCAB_SIZE,
    NANOCHAT_SPECIAL_TOKENS,
    ByteTokenizer,
    Tokenizer,
)
from scratch_llm.tracking_state import (
    TrackingState,
    TrackingStateError,
)
from scratch_llm.utils import get_device


TrainingStage = Literal["pretrain", "sft"]
ValidationCheckpointMetadata: TypeAlias = (
    ValidationCheckpointState | SFTValidationCheckpointState
)

CHECKPOINT_FORMAT_VERSION = 6
_TRACKING_CHECKPOINT_FORMAT_VERSION = 5
_VALIDATION_CHECKPOINT_FORMAT_VERSION = 4
_EXACT_CHECKPOINT_FORMAT_VERSION = 3
_LEGACY_CHECKPOINT_FORMAT_VERSION = 2
_SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = frozenset(
    {
        1,
        _LEGACY_CHECKPOINT_FORMAT_VERSION,
        _EXACT_CHECKPOINT_FORMAT_VERSION,
        _VALIDATION_CHECKPOINT_FORMAT_VERSION,
        _TRACKING_CHECKPOINT_FORMAT_VERSION,
        CHECKPOINT_FORMAT_VERSION,
    }
)
_BASE_CHECKPOINT_KEYS = frozenset(
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
_EXACT_CHECKPOINT_KEYS = _BASE_CHECKPOINT_KEYS | {"continuation"}
_VALIDATION_CHECKPOINT_KEYS = _EXACT_CHECKPOINT_KEYS | {"validation"}
_TRACKING_CHECKPOINT_KEYS = _VALIDATION_CHECKPOINT_KEYS | {"tracking"}
_CURRENT_CHECKPOINT_KEYS = _TRACKING_CHECKPOINT_KEYS | {
    "base_checkpoint_identity",
    "training_stage",
}
_CONTINUATION_KEYS = frozenset(
    {
        "loader_format",
        "loader_state",
        "rng_state",
        "total_training_flops",
        "total_training_time_seconds",
        "tracker_step",
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
    validation: ValidationCheckpointMetadata | None
    tracking: TrackingState | None
    training_stage: TrainingStage
    base_checkpoint_identity: str | None


@dataclass(frozen=True)
class TrainingCheckpoint(ModelCheckpoint):
    """Full optimizer and scheduler state reconstructed for training resume."""

    optimizer: Optimizer
    scheduler: LRScheduler
    continuation: ExactTrainingState | None


@dataclass(frozen=True)
class CheckpointMetadata:
    """Configuration and small state read without constructing the model."""

    config: ProjectConfig
    step: int
    validation: ValidationCheckpointMetadata | None
    tracking: TrackingState | None
    training_stage: TrainingStage
    base_checkpoint_identity: str | None


@dataclass(frozen=True)
class ExactTrainingState:
    """State installed only after all resume-time objects are reconstructed."""

    loader_format: str
    loader_state: dict[str, object]
    rng_state: TrainingRNGState
    tracker_step: int
    total_training_time_seconds: float
    total_training_flops: float

    def __post_init__(self) -> None:
        if not isinstance(self.loader_format, str) or not self.loader_format.strip():
            raise ValueError("loader_format must be a non-empty string")
        canonical_loader_state = _canonical_json_object(
            self.loader_state,
            label="loader_state",
        )
        if canonical_loader_state.get("format") != self.loader_format:
            raise ValueError(
                "loader_state format must match continuation loader_format"
            )
        if not isinstance(self.rng_state, TrainingRNGState):
            raise TypeError(
                "rng_state must be a TrainingRNGState, got "
                f"{type(self.rng_state).__name__}"
            )
        require_non_negative_integer(self.tracker_step, name="tracker_step")
        require_finite_non_negative_real(
            self.total_training_time_seconds,
            name="total_training_time_seconds",
        )
        require_finite_non_negative_real(
            self.total_training_flops,
            name="total_training_flops",
        )
        object.__setattr__(self, "loader_state", canonical_loader_state)
        object.__setattr__(
            self,
            "total_training_time_seconds",
            float(self.total_training_time_seconds),
        )
        object.__setattr__(
            self,
            "total_training_flops",
            float(self.total_training_flops),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the serialized format-v3 continuation payload."""

        return {
            "loader_format": self.loader_format,
            "loader_state": _canonical_json_object(
                self.loader_state,
                label="loader_state",
            ),
            "rng_state": self.rng_state.to_dict(),
            "total_training_flops": self.total_training_flops,
            "total_training_time_seconds": self.total_training_time_seconds,
            "tracker_step": self.tracker_step,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExactTrainingState:
        """Validate a serialized continuation without mutating runtime state."""

        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise CheckpointError(
                "checkpoint continuation must be an object with string keys"
            )
        if set(value) != _CONTINUATION_KEYS:
            missing = sorted(_CONTINUATION_KEYS - set(value))
            unexpected = sorted(set(value) - _CONTINUATION_KEYS)
            raise CheckpointError(
                "checkpoint continuation fields do not match format version 3; "
                f"missing={missing}, unexpected={unexpected}"
            )
        try:
            loader_format = value["loader_format"]
            if not isinstance(loader_format, str):
                raise TypeError("loader_format must be a string")
            tracker_step = require_non_negative_integer(
                value["tracker_step"],
                name="tracker_step",
            )
            return cls(
                loader_format=loader_format,
                loader_state=_canonical_json_object(
                    value["loader_state"],
                    label="loader_state",
                ),
                rng_state=TrainingRNGState.from_dict(value["rng_state"]),
                tracker_step=tracker_step,
                total_training_time_seconds=require_real(
                    value["total_training_time_seconds"],
                    name="total_training_time_seconds",
                ),
                total_training_flops=require_real(
                    value["total_training_flops"],
                    name="total_training_flops",
                ),
            )
        except CheckpointError:
            raise
        except (RNGStateError, TypeError, ValueError) as error:
            raise CheckpointError(
                f"checkpoint contains invalid exact continuation state: {error}"
            ) from error


@dataclass(frozen=True)
class _DecodedCheckpoint:
    payload: dict[str, Any]
    config: ProjectConfig
    tokenizer: Tokenizer
    step: int
    device: torch.device
    continuation: ExactTrainingState | None
    validation: ValidationCheckpointMetadata | None
    tracking: TrackingState | None
    training_stage: TrainingStage
    base_checkpoint_identity: str | None


def _validate_training_stage(value: object) -> TrainingStage:
    if value not in {"pretrain", "sft"}:
        raise ValueError("training_stage must be 'pretrain' or 'sft'")
    return value  # type: ignore[return-value]


def _validate_stage_provenance(
    training_stage: TrainingStage,
    base_checkpoint_identity: object,
) -> str | None:
    if training_stage == "pretrain":
        if base_checkpoint_identity is not None:
            raise ValueError(
                "pretrain checkpoints must not record base_checkpoint_identity"
            )
        return None
    if base_checkpoint_identity is None:
        raise ValueError("SFT checkpoint requires base_checkpoint_identity")
    identity = require_non_empty_string(
        base_checkpoint_identity,
        name="base_checkpoint_identity",
    )
    if re.fullmatch(r"sha256:[0-9a-f]{64}", identity) is None:
        raise ValueError(
            "base_checkpoint_identity must be a lowercase SHA-256 identity"
        )
    return identity


def _training_config_for_stage(
    config: ProjectConfig,
    training_stage: TrainingStage,
) -> TrainConfig:
    if training_stage == "pretrain":
        return config.train
    return config.sft.to_train_config(config.model.seq_len)


def _validate_save_state(
    *,
    model: GPT,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: ProjectConfig,
    step: int,
    tokenizer: Tokenizer,
    training_stage: TrainingStage,
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
    step = require_non_negative_integer(step, name="step")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )

    config.validate()
    active_train = _training_config_for_stage(config, training_stage)
    if model.config != config.model:
        raise ValueError("model configuration does not match the resolved config")
    if scheduler.optimizer is not optimizer:
        raise ValueError("scheduler must be attached to the saved optimizer")
    if any(
        float(base_lr) != float(active_train.learning_rate)
        for base_lr in scheduler.base_lrs
    ):
        raise ValueError(
            f"optimizer learning rate does not match {training_stage} config"
        )
    if any(
        group.get("betas") != (active_train.beta1, active_train.beta2)
        or float(group.get("weight_decay", -1.0)) != active_train.weight_decay
        for group in optimizer.param_groups
    ):
        raise ValueError(
            f"optimizer hyperparameters do not match {training_stage} config"
        )
    if (
        scheduler.max_steps != active_train.max_steps
        or scheduler.warmup_steps != active_train.warmup_steps
        or scheduler.warmdown_ratio != active_train.warmdown_ratio
        or scheduler.final_lr_frac != active_train.final_lr_frac
    ):
        raise ValueError(
            f"scheduler hyperparameters do not match {training_stage} config"
        )
    if scheduler.last_epoch != step:
        raise ValueError(
            f"step {step} does not match scheduler step {scheduler.last_epoch}"
        )
    if step > active_train.max_steps:
        raise ValueError(
            f"step {step} exceeds configured max_steps {active_train.max_steps}"
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
    continuation: ExactTrainingState | None = None,
    validation: ValidationCheckpointMetadata | None = None,
    tracking: TrackingState | None = None,
    training_stage: TrainingStage = "pretrain",
    base_checkpoint_identity: str | None = None,
) -> Path:
    """Atomically save legacy state or an exact current continuation."""

    training_stage = _validate_training_stage(training_stage)
    base_checkpoint_identity = _validate_stage_provenance(
        training_stage,
        base_checkpoint_identity,
    )
    _validate_save_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=step,
        tokenizer=tokenizer,
        training_stage=training_stage,
    )
    if continuation is not None and not isinstance(continuation, ExactTrainingState):
        raise TypeError(
            "continuation must be an ExactTrainingState or None, got "
            f"{type(continuation).__name__}"
        )
    if continuation is not None and continuation.tracker_step != step:
        raise ValueError(
            f"continuation tracker_step {continuation.tracker_step} "
            f"does not match checkpoint step {step}"
        )
    expected_validation_type: type[
        ValidationCheckpointState | SFTValidationCheckpointState
    ] = (
        ValidationCheckpointState
        if training_stage == "pretrain"
        else SFTValidationCheckpointState
    )
    if validation is not None and not isinstance(validation, expected_validation_type):
        label = "pretraining" if training_stage == "pretrain" else "SFT"
        raise TypeError(
            f"{label} validation must be a {expected_validation_type.__name__} "
            f"or None, got {type(validation).__name__}"
        )
    if validation is not None and continuation is None:
        raise ValueError("validation metadata requires an exact continuation")
    if validation is not None and validation.validation_step > step:
        raise ValueError(
            f"validation step {validation.validation_step} exceeds "
            f"checkpoint step {step}"
        )
    if tracking is not None and not isinstance(tracking, TrackingState):
        raise TypeError(
            f"tracking must be a TrackingState or None, got {type(tracking).__name__}"
        )
    if tracking is not None and continuation is None:
        raise ValueError("tracking metadata requires an exact continuation")
    if training_stage == "sft" and continuation is None:
        raise ValueError("SFT checkpoints require an exact continuation")
    format_version = (
        CHECKPOINT_FORMAT_VERSION
        if continuation is not None
        else _LEGACY_CHECKPOINT_FORMAT_VERSION
    )
    payload: dict[str, object] = {
        "format_version": format_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "tokenizer": _tokenizer_metadata(config, tokenizer),
    }
    if continuation is not None:
        payload["continuation"] = continuation.to_dict()
        payload["validation"] = None if validation is None else validation.to_dict()
        payload["tracking"] = None if tracking is None else tracking.to_dict()
        payload["training_stage"] = training_stage
        payload["base_checkpoint_identity"] = base_checkpoint_identity
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
    format_version = payload.get("format_version")
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
    if format_version == CHECKPOINT_FORMAT_VERSION:
        expected_keys = _CURRENT_CHECKPOINT_KEYS
    elif format_version == _TRACKING_CHECKPOINT_FORMAT_VERSION:
        expected_keys = _TRACKING_CHECKPOINT_KEYS
    elif format_version == _VALIDATION_CHECKPOINT_FORMAT_VERSION:
        expected_keys = _VALIDATION_CHECKPOINT_KEYS
    elif format_version == _EXACT_CHECKPOINT_FORMAT_VERSION:
        expected_keys = _EXACT_CHECKPOINT_KEYS
    else:
        expected_keys = _BASE_CHECKPOINT_KEYS
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        unexpected = sorted(set(payload) - expected_keys)
        raise CheckpointError(
            "checkpoint fields do not match format version "
            f"{format_version}; missing={missing}, unexpected={unexpected}"
        )

    try:
        training_stage = (
            _validate_training_stage(payload["training_stage"])
            if format_version == CHECKPOINT_FORMAT_VERSION
            else "pretrain"
        )
        base_checkpoint_identity = _validate_stage_provenance(
            training_stage,
            (
                payload["base_checkpoint_identity"]
                if format_version == CHECKPOINT_FORMAT_VERSION
                else None
            ),
        )
    except (TypeError, ValueError) as error:
        raise CheckpointError(
            f"checkpoint contains invalid stage metadata: {error}"
        ) from error

    step = payload["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise CheckpointError(
            f"checkpoint step must be a non-negative integer, got {step!r}"
        )
    config = _restore_config(payload["config"])
    active_train = _training_config_for_stage(config, training_stage)
    if step > active_train.max_steps:
        raise CheckpointError(
            f"checkpoint step {step} exceeds configured max_steps "
            f"{active_train.max_steps}"
        )
    tokenizer = _restore_tokenizer(
        payload["tokenizer"],
        config,
        format_version=format_version,
    )
    continuation = (
        ExactTrainingState.from_dict(payload["continuation"])
        if format_version
        in {
            _EXACT_CHECKPOINT_FORMAT_VERSION,
            _VALIDATION_CHECKPOINT_FORMAT_VERSION,
            _TRACKING_CHECKPOINT_FORMAT_VERSION,
            CHECKPOINT_FORMAT_VERSION,
        }
        else None
    )
    if continuation is not None and continuation.tracker_step != step:
        raise CheckpointError(
            "checkpoint continuation tracker_step does not match checkpoint step"
        )
    try:
        has_validation = format_version in {
            _VALIDATION_CHECKPOINT_FORMAT_VERSION,
            _TRACKING_CHECKPOINT_FORMAT_VERSION,
            CHECKPOINT_FORMAT_VERSION,
        }
        if not has_validation or payload["validation"] is None:
            validation: ValidationCheckpointMetadata | None = None
        elif training_stage == "sft":
            validation = SFTValidationCheckpointState.from_dict(payload["validation"])
        else:
            validation = ValidationCheckpointState.from_dict(payload["validation"])
    except (BestCheckpointError, SFTValidationError) as error:
        raise CheckpointError(str(error)) from error
    if validation is not None and validation.validation_step > step:
        raise CheckpointError("checkpoint validation_step exceeds checkpoint step")
    try:
        tracking = (
            None
            if format_version
            not in {
                _TRACKING_CHECKPOINT_FORMAT_VERSION,
                CHECKPOINT_FORMAT_VERSION,
            }
            or payload["tracking"] is None
            else TrackingState.from_dict(payload["tracking"])
        )
    except TrackingStateError as error:
        raise CheckpointError(str(error)) from error
    return _DecodedCheckpoint(
        payload=payload,
        config=config,
        tokenizer=tokenizer,
        step=step,
        device=resolved_device,
        continuation=continuation,
        validation=validation,
        tracking=tracking,
        training_stage=training_stage,
        base_checkpoint_identity=base_checkpoint_identity,
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

    with preserve_global_rng_state(device):
        checkpoint = _load_checkpoint(path, device=device)
        model = _restore_model(checkpoint)
        model.eval()
        return ModelCheckpoint(
            model=model,
            tokenizer=checkpoint.tokenizer,
            config=checkpoint.config,
            step=checkpoint.step,
            validation=checkpoint.validation,
            tracking=checkpoint.tracking,
            training_stage=checkpoint.training_stage,
            base_checkpoint_identity=checkpoint.base_checkpoint_identity,
        )


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    allow_non_exact_resume: bool = False,
    expected_stage: TrainingStage | None = None,
) -> TrainingCheckpoint:
    """Reconstruct train-mode model, optimizer, and scheduler state for resume."""

    if not isinstance(allow_non_exact_resume, bool):
        raise TypeError("allow_non_exact_resume must be a boolean")
    if expected_stage is not None:
        expected_stage = _validate_training_stage(expected_stage)
    checkpoint = _load_checkpoint(path, device=device)
    if expected_stage is not None and checkpoint.training_stage != expected_stage:
        raise CheckpointError(
            "checkpoint training stage does not match requested resume: "
            f"expected {expected_stage!r}, got {checkpoint.training_stage!r}"
        )
    if checkpoint.continuation is None and not allow_non_exact_resume:
        raise CheckpointError(
            "exact training resume requires checkpoint format version 3 or newer; "
            "legacy format versions remain valid for model-only loading, or "
            "choose the documented --allow-non-exact-resume migration"
        )
    with preserve_global_rng_state(device):
        model = _restore_model(checkpoint)
        active_train = _training_config_for_stage(
            checkpoint.config,
            checkpoint.training_stage,
        )
        optimizer = build_optimizer(model, active_train)
        scheduler = build_lr_scheduler(optimizer, active_train)
        try:
            optimizer.load_state_dict(checkpoint.payload["optimizer"])
            scheduler.load_state_dict(checkpoint.payload["scheduler"])
        except Exception as error:
            raise CheckpointError(
                f"could not restore training state: {error}"
            ) from error
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
            validation=checkpoint.validation,
            tracking=checkpoint.tracking,
            optimizer=optimizer,
            scheduler=scheduler,
            continuation=checkpoint.continuation,
            training_stage=checkpoint.training_stage,
            base_checkpoint_identity=checkpoint.base_checkpoint_identity,
        )


def load_checkpoint_metadata(
    path: str | os.PathLike[str],
) -> CheckpointMetadata:
    """Read config and small resume state without constructing model objects."""

    checkpoint = _load_checkpoint(path, device="cpu")
    return CheckpointMetadata(
        config=checkpoint.config,
        step=checkpoint.step,
        validation=checkpoint.validation,
        tracking=checkpoint.tracking,
        training_stage=checkpoint.training_stage,
        base_checkpoint_identity=checkpoint.base_checkpoint_identity,
    )


def _canonical_json_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a dictionary")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must contain only finite JSON values: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must encode a JSON object")
    return decoded


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointMetadata",
    "CheckpointError",
    "ExactTrainingState",
    "ModelCheckpoint",
    "TrainingCheckpoint",
    "TrainingStage",
    "ValidationCheckpointMetadata",
    "load_checkpoint_metadata",
    "load_model_checkpoint",
    "load_training_checkpoint",
    "save_checkpoint",
]
