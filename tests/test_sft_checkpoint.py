"""Stage-aware base initialization and exact SFT checkpoint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.evaluation.sft_bpb import (
    SFT_ASSISTANT_BPB_PROTOCOL_ID,
    SFTValidationCheckpointState,
)
from scratch_llm.model import GPT
from scratch_llm.training.best_checkpoint import ValidationCheckpointState
from scratch_llm.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    ExactTrainingState,
    load_checkpoint_metadata,
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.rng_state import capture_training_rng_state
from scratch_llm.tokenization.tokenizer import ByteTokenizer, VOCAB_SIZE


BASE_IDENTITY = "sha256:" + "a" * 64


def _config() -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=2,
            total_batch_size_tokens=8,
            max_steps=4,
            learning_rate=0.1,
            weight_decay=0.1,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
        sft=SFTConfig(
            device_batch_size=2,
            total_batch_size_tokens=8,
            max_steps=4,
            learning_rate=0.001,
            weight_decay=0.0,
            warmup_steps=0,
            warmdown_ratio=0.0,
            eval_every=1,
            eval_batches=1,
            save_every=1,
            log_every=1,
        ),
    )


def _continuation() -> ExactTrainingState:
    return ExactTrainingState(
        loader_format="fixture_loader_v1",
        loader_state={"format": "fixture_loader_v1", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )


def _sft_validation() -> SFTValidationCheckpointState:
    return SFTValidationCheckpointState(
        ranking_protocol_id=SFT_ASSISTANT_BPB_PROTOCOL_ID,
        validation_identity="sha256:" + "b" * 64,
        validation_step=0,
        current_bpb=1.5,
        minimum_bpb=1.5,
    )


def _save(
    path: Path,
    *,
    training_stage: str,
    base_checkpoint_identity: str | None,
    validation: object = None,
) -> Path:
    config = _config()
    tokenizer = ByteTokenizer()
    model = GPT(config.model)
    active_train = (
        config.sft.to_train_config(config.model.seq_len)
        if training_stage == "sft"
        else config.train
    )
    optimizer = build_optimizer(model, active_train)
    scheduler = build_lr_scheduler(optimizer, active_train)
    return save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=_continuation(),
        validation=validation,  # type: ignore[arg-type]
        training_stage=training_stage,  # type: ignore[arg-type]
        base_checkpoint_identity=base_checkpoint_identity,
    )


def test_sft_checkpoint_round_trips_stage_provenance_and_sft_optimizer(
    tmp_path: Path,
) -> None:
    path = _save(
        tmp_path / "sft.pt",
        training_stage="sft",
        base_checkpoint_identity=BASE_IDENTITY,
        validation=_sft_validation(),
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model_only = load_model_checkpoint(path)
    training = load_training_checkpoint(path, expected_stage="sft")
    metadata = load_checkpoint_metadata(path)

    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["training_stage"] == "sft"
    assert payload["base_checkpoint_identity"] == BASE_IDENTITY
    assert model_only.training_stage == "sft"
    assert model_only.base_checkpoint_identity == BASE_IDENTITY
    assert model_only.model.training is False
    assert isinstance(model_only.validation, SFTValidationCheckpointState)
    assert training.training_stage == "sft"
    assert training.model.training is True
    assert training.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert training.optimizer.param_groups[0]["weight_decay"] == 0.0
    assert metadata.training_stage == "sft"
    assert metadata.base_checkpoint_identity == BASE_IDENTITY


def test_exact_pretrain_checkpoint_has_explicit_stage_and_no_base_identity(
    tmp_path: Path,
) -> None:
    path = _save(
        tmp_path / "pretrain.pt",
        training_stage="pretrain",
        base_checkpoint_identity=None,
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    loaded = load_training_checkpoint(path, expected_stage="pretrain")

    assert payload["training_stage"] == "pretrain"
    assert payload["base_checkpoint_identity"] is None
    assert loaded.training_stage == "pretrain"
    assert loaded.base_checkpoint_identity is None
    assert loaded.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_training_resume_rejects_the_other_stage_but_model_loading_accepts_both(
    tmp_path: Path,
) -> None:
    pretrain = _save(
        tmp_path / "pretrain.pt",
        training_stage="pretrain",
        base_checkpoint_identity=None,
    )
    sft = _save(
        tmp_path / "sft.pt",
        training_stage="sft",
        base_checkpoint_identity=BASE_IDENTITY,
        validation=_sft_validation(),
    )

    with pytest.raises(CheckpointError, match="training stage.*pretrain.*sft"):
        load_training_checkpoint(sft, expected_stage="pretrain")
    with pytest.raises(CheckpointError, match="training stage.*sft.*pretrain"):
        load_training_checkpoint(pretrain, expected_stage="sft")

    assert load_model_checkpoint(pretrain).training_stage == "pretrain"
    assert load_model_checkpoint(sft).training_stage == "sft"


@pytest.mark.parametrize(
    ("training_stage", "base_identity", "message"),
    [
        ("sft", None, "requires base_checkpoint_identity"),
        ("pretrain", BASE_IDENTITY, "must not record base_checkpoint_identity"),
        ("unknown", None, "training_stage"),
    ],
)
def test_save_rejects_invalid_stage_provenance_pairs(
    tmp_path: Path,
    training_stage: str,
    base_identity: str | None,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _save(
            tmp_path / "invalid.pt",
            training_stage=training_stage,
            base_checkpoint_identity=base_identity,
        )


def test_stage_rejects_the_other_validation_state_type(tmp_path: Path) -> None:
    base_validation = ValidationCheckpointState(
        ranking_protocol_id="nanochat_compat_v1",
        validation_identity="sha256:" + "c" * 64,
        validation_step=0,
        current_compatibility_bpb=2.0,
        minimum_compatibility_bpb=2.0,
        current_full_document_bpb=2.0,
        minimum_full_document_bpb=2.0,
    )

    with pytest.raises(TypeError, match="SFT validation"):
        _save(
            tmp_path / "sft-base-validation.pt",
            training_stage="sft",
            base_checkpoint_identity=BASE_IDENTITY,
            validation=base_validation,
        )
    with pytest.raises(TypeError, match="pretraining validation"):
        _save(
            tmp_path / "pretrain-sft-validation.pt",
            training_stage="pretrain",
            base_checkpoint_identity=None,
            validation=_sft_validation(),
        )
