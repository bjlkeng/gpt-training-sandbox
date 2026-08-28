"""Bounded CPU coverage for the base-model checkpoint contract."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from scratch_llm.training import checkpoint
from scratch_llm.training.best_checkpoint import ValidationCheckpointState
from scratch_llm.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    ExactTrainingState,
    load_model_checkpoint,
    load_checkpoint_metadata,
    load_training_checkpoint,
    save_checkpoint,
)
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.data.loaders import NextTokenDataset
from scratch_llm.model import GPT
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.rng_state import capture_training_rng_state
from scratch_llm.training.precision import PrecisionCheckpointState
from scratch_llm.tokenization.tokenizer import (
    BYTE_VOCAB_SIZE,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    ByteTokenizer,
    Tokenizer,
)
from scratch_llm.tracking import NullTracker
from scratch_llm.training.loop import run_training_steps
from scratch_llm.tracking_state import TrackingState


def _assert_nested_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected)
    elif isinstance(expected, Mapping):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_nested_state_equal(actual[key], value)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


class _StepTracker(NullTracker):
    def __init__(self) -> None:
        self.steps: list[int | None] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.steps.append(step)


class _IndependentByteCompatibleTokenizer(Tokenizer):
    """Stand in for a future implementation at shared consumer boundaries."""

    def __init__(self) -> None:
        self._delegate = ByteTokenizer()

    def encode(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]:
        return self._delegate.encode(text, prepend=prepend, append=append)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self._delegate.decode(token_ids)

    def encode_special(self, token: str) -> int:
        return self._delegate.encode_special(token)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self._delegate.decode_single_token_bytes(token_id)

    def get_vocab_size(self) -> int:
        return self._delegate.get_vocab_size()

    def get_bos_token_id(self) -> int:
        return self._delegate.get_bos_token_id()

    def get_special_tokens(self) -> set[str]:
        return self._delegate.get_special_tokens()

    def get_identity(self) -> str:
        return self._delegate.get_identity()


def _checkpoint_state() -> tuple[
    ProjectConfig,
    ByteTokenizer,
    GPT,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    config = ProjectConfig(
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
            grad_accum_steps=1,
            max_steps=4,
            learning_rate=0.01,
            weight_decay=0.0,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
    )
    tokenizer = ByteTokenizer()
    dataset = NextTokenDataset(
        tokenizer.encode("abcd abcd abcd"),
        seq_len=config.model.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
    )
    batches = DataLoader(
        dataset,
        batch_size=config.train.device_batch_size,
        shuffle=False,
    )
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    return config, tokenizer, model, optimizer, scheduler, batches


def test_last_checkpoint_records_complete_resumable_state(tmp_path: Path) -> None:
    config, tokenizer, model, optimizer, scheduler, batches = _checkpoint_state()
    run_training_steps(
        model,
        batches,
        optimizer,
        scheduler,
        max_steps=2,
        grad_accum_steps=1,
        grad_clip=config.train.grad_clip,
        device="cpu",
    )
    checkpoint_path = tmp_path / "checkpoints" / "last.pt"

    saved_path = save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=2,
        tokenizer=tokenizer,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved_path == checkpoint_path
    assert set(payload) == {
        "format_version",
        "model",
        "optimizer",
        "scheduler",
        "config",
        "step",
        "tokenizer",
    }
    assert payload["format_version"] == 2
    _assert_nested_state_equal(payload["model"], model.state_dict())
    _assert_nested_state_equal(payload["optimizer"], optimizer.state_dict())
    _assert_nested_state_equal(payload["scheduler"], scheduler.state_dict())
    assert payload["config"] == config.to_dict()
    assert payload["step"] == scheduler.last_epoch == 2
    assert payload["tokenizer"] == {
        "type": "byte",
        "identity": tokenizer.get_identity(),
        "vocab_size": VOCAB_SIZE,
        "special_tokens": list(SPECIAL_TOKENS),
    }
    assert not list(checkpoint_path.parent.glob(".last.pt.*.tmp"))


def test_version_one_byte_checkpoint_remains_readable(tmp_path: Path) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    current_path = save_checkpoint(
        tmp_path / "current.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )
    payload = torch.load(current_path, map_location="cpu", weights_only=True)
    payload["format_version"] = 1
    payload["tokenizer"] = {
        "type": "byte",
        "byte_vocab_size": BYTE_VOCAB_SIZE,
        "vocab_size": VOCAB_SIZE,
        "special_tokens": list(SPECIAL_TOKENS),
    }
    del payload["config"]["data"]["loader_strategy"]
    del payload["config"]["tokenizer"]["artifact_dir"]
    del payload["config"]["model"]["attention_backend"]
    legacy_path = tmp_path / "legacy-v1-byte.pt"
    torch.save(payload, legacy_path)

    loaded = load_model_checkpoint(legacy_path)

    assert loaded.config == config
    assert loaded.config.model.attention_backend == "manual"
    assert loaded.step == 0
    assert isinstance(loaded.tokenizer, ByteTokenizer)


def test_save_rejects_model_whose_weight_sharing_disagrees_with_config(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())

    with pytest.raises(
        ValueError,
        match=r"model weight sharing.*tie_weights=true.*independent",
    ):
        save_checkpoint(
            tmp_path / "invalid-topology.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=0,
            tokenizer=tokenizer,
        )


def test_load_rejects_checkpoint_whose_weight_sharing_disagrees_with_config(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    checkpoint_path = save_checkpoint(
        tmp_path / "valid.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["config"]["model"]["tie_weights"] = False
    tampered_path = tmp_path / "mismatched-topology.pt"
    torch.save(payload, tampered_path)

    with pytest.raises(
        CheckpointError,
        match=r"checkpoint model weight sharing.*tie_weights=false.*shared",
    ):
        load_model_checkpoint(tampered_path)


def test_model_only_load_preserves_python_numpy_and_torch_rng(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    checkpoint_path = save_checkpoint(
        tmp_path / "model-only.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_state = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    torch_state = torch.get_rng_state().clone()

    load_model_checkpoint(checkpoint_path, device="cpu")

    assert random.getstate() == python_state
    restored_numpy = np.random.get_state(legacy=True)
    assert restored_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_legacy_training_resume_requires_explicit_non_exact_opt_in(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    checkpoint_path = save_checkpoint(
        tmp_path / "legacy-v2.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )

    with pytest.raises(CheckpointError, match="exact training resume.*version 3"):
        load_training_checkpoint(checkpoint_path, device="cpu")

    migrated = load_training_checkpoint(
        checkpoint_path,
        device="cpu",
        allow_non_exact_resume=True,
    )
    assert migrated.continuation is None


def test_malformed_exact_state_fails_before_model_or_rng_mutation(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    continuation = ExactTrainingState(
        loader_format="test_loader",
        loader_state={"format": "test_loader", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )
    valid_path = save_checkpoint(
        tmp_path / "valid-v3.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=continuation,
    )
    payload = torch.load(valid_path, map_location="cpu", weights_only=True)
    del payload["continuation"]["rng_state"]["torch_cpu_state"]
    malformed_path = tmp_path / "malformed-v3.pt"
    torch.save(payload, malformed_path)
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    python_state = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    torch_state = torch.get_rng_state().clone()

    with pytest.raises(CheckpointError, match="invalid exact continuation"):
        load_training_checkpoint(malformed_path, device="cpu")

    assert random.getstate() == python_state
    restored_numpy = np.random.get_state(legacy=True)
    assert restored_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_exact_checkpoint_round_trips_versioned_precision_state(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    continuation = ExactTrainingState(
        loader_format="test_loader",
        loader_state={"format": "test_loader", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )
    precision = PrecisionCheckpointState(
        dtype="float32",
        device_type="cpu",
        scaler_enabled=False,
        scaler_state=None,
    )

    checkpoint_path = save_checkpoint(
        tmp_path / "precision.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=continuation,
        precision=precision,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    loaded = load_training_checkpoint(
        checkpoint_path,
        expected_precision=precision,
    )
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["precision"] == precision.to_dict()
    assert loaded.precision == precision


def test_resume_rejects_incompatible_precision_before_model_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    continuation = ExactTrainingState(
        loader_format="test_loader",
        loader_state={"format": "test_loader", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )
    checkpoint_path = save_checkpoint(
        tmp_path / "float32.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=continuation,
        precision=PrecisionCheckpointState(
            dtype="float32",
            device_type="cpu",
            scaler_enabled=False,
            scaler_state=None,
        ),
    )
    model_restored = False

    def record_model_restore(*args: Any, **kwargs: Any) -> None:
        nonlocal model_restored
        model_restored = True
        raise AssertionError("model restoration must not run")

    monkeypatch.setattr(checkpoint, "_restore_model", record_model_restore)

    with pytest.raises(CheckpointError, match="precision policy is incompatible"):
        load_training_checkpoint(
            checkpoint_path,
            expected_precision=PrecisionCheckpointState(
                dtype="bfloat16",
                device_type="cpu",
                scaler_enabled=False,
                scaler_state=None,
            ),
        )

    assert model_restored is False


def test_current_format_round_trips_stage_validation_tracking_and_exact_state(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    continuation = ExactTrainingState(
        loader_format="test_loader",
        loader_state={"format": "test_loader", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=1.5,
        total_training_flops=2.5,
    )
    validation = ValidationCheckpointState(
        ranking_protocol_id="nanochat_compat_v1",
        validation_identity="sha256:" + "a" * 64,
        validation_step=0,
        current_compatibility_bpb=1.5,
        minimum_compatibility_bpb=1.25,
        current_full_document_bpb=1.75,
        minimum_full_document_bpb=1.5,
    )
    tracking = TrackingState(backend="wandb", run_id="run-id")

    checkpoint_path = save_checkpoint(
        tmp_path / "current.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=continuation,
        validation=validation,
        tracking=tracking,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == checkpoint.CHECKPOINT_FORMAT_VERSION
    assert payload["training_stage"] == "pretrain"
    assert payload["base_checkpoint_identity"] is None
    assert payload["validation"] == validation.to_dict()
    assert payload["tracking"] == tracking.to_dict()
    model_only = load_model_checkpoint(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_path)
    resumed = load_training_checkpoint(checkpoint_path)
    assert metadata.tracking == tracking
    assert metadata.validation == validation
    assert model_only.validation == validation
    assert model_only.tracking == tracking
    assert resumed.validation == validation
    assert resumed.tracking == tracking
    assert resumed.continuation == continuation

    payload["format_version"] = 5
    del payload["training_stage"]
    del payload["base_checkpoint_identity"]
    del payload["precision"]
    format_five = tmp_path / "format-five.pt"
    torch.save(payload, format_five)
    legacy = load_training_checkpoint(format_five, expected_stage="pretrain")
    assert legacy.validation == validation
    assert legacy.tracking == tracking
    assert legacy.training_stage == "pretrain"


def test_format_four_checkpoint_remains_exact_without_tracking_state(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    continuation = ExactTrainingState(
        loader_format="test_loader",
        loader_state={"format": "test_loader", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )
    current = save_checkpoint(
        tmp_path / "format-five.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=continuation,
    )
    payload = torch.load(current, map_location="cpu", weights_only=True)
    payload["format_version"] = 4
    del payload["tracking"]
    del payload["training_stage"]
    del payload["base_checkpoint_identity"]
    del payload["precision"]
    previous = tmp_path / "format-four.pt"
    torch.save(payload, previous)

    resumed = load_training_checkpoint(previous)

    assert resumed.continuation == continuation
    assert resumed.tracking is None


def test_format_three_exact_checkpoint_remains_resumable_without_validation(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    continuation = ExactTrainingState(
        loader_format="test_loader",
        loader_state={"format": "test_loader", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )
    current = save_checkpoint(
        tmp_path / "format-five.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=continuation,
    )
    payload = torch.load(current, map_location="cpu", weights_only=True)
    payload["format_version"] = 3
    del payload["validation"]
    del payload["tracking"]
    del payload["training_stage"]
    del payload["base_checkpoint_identity"]
    del payload["precision"]
    previous = tmp_path / "format-three.pt"
    torch.save(payload, previous)

    resumed = load_training_checkpoint(previous)

    assert resumed.continuation == continuation
    assert resumed.validation is None


def test_cpu_training_load_never_checks_or_initializes_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    continuation = ExactTrainingState(
        loader_format="test_loader",
        loader_state={"format": "test_loader", "position": 0},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )
    checkpoint_path = save_checkpoint(
        tmp_path / "cpu-v3.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
        continuation=continuation,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("CPU checkpoint load touched CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)

    loaded = load_training_checkpoint(checkpoint_path, device="cpu")

    assert loaded.continuation == continuation


def test_shared_loaders_reconstruct_sampling_and_next_step_training_state(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, batches = _checkpoint_state()
    run_training_steps(
        model,
        batches,
        optimizer,
        scheduler,
        max_steps=2,
        grad_accum_steps=1,
        grad_clip=config.train.grad_clip,
        device="cpu",
    )
    checkpoint_path = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=2,
        tokenizer=tokenizer,
    )
    inputs, _ = next(iter(batches))
    model.eval()
    with torch.inference_mode():
        expected_logits = model(inputs)

    sampling = load_model_checkpoint(checkpoint_path, device="cpu")

    assert sampling.config == config
    assert sampling.step == 2
    assert isinstance(sampling.tokenizer, ByteTokenizer)
    assert sampling.model.training is False
    assert all(
        parameter.device.type == "cpu" for parameter in sampling.model.parameters()
    )
    with torch.inference_mode():
        torch.testing.assert_close(sampling.model(inputs), expected_logits)

    resumed = load_training_checkpoint(
        checkpoint_path,
        device="cpu",
        allow_non_exact_resume=True,
    )

    assert resumed.config == config
    assert resumed.step == 2
    assert resumed.scheduler.last_epoch == resumed.step
    assert resumed.model.training is True
    _assert_nested_state_equal(resumed.model.state_dict(), model.state_dict())
    _assert_nested_state_equal(resumed.optimizer.state_dict(), optimizer.state_dict())
    _assert_nested_state_equal(resumed.scheduler.state_dict(), scheduler.state_dict())

    tracker = _StepTracker()
    results = run_training_steps(
        resumed.model,
        batches,
        resumed.optimizer,
        resumed.scheduler,
        max_steps=resumed.config.train.max_steps,
        grad_accum_steps=1,
        grad_clip=resumed.config.train.grad_clip,
        device="cpu",
        tracker=tracker,
    )

    assert len(results) == 2
    assert tracker.steps == [3, 4]
    assert resumed.scheduler.last_epoch == resumed.config.train.max_steps


def test_checkpoint_replacement_is_atomic_and_cleans_a_failed_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    checkpoint_path = tmp_path / "last.pt"
    original_contents = b"previous complete checkpoint"
    checkpoint_path.write_bytes(original_contents)
    observed_payload: dict[str, Any] = {}

    def fail_install(source: object, destination: object) -> None:
        assert Path(destination) == checkpoint_path  # type: ignore[arg-type]
        assert checkpoint_path.read_bytes() == original_contents
        observed_payload.update(
            torch.load(Path(source), map_location="cpu", weights_only=True)  # type: ignore[arg-type]
        )
        raise OSError("checkpoint install failed")

    monkeypatch.setattr(checkpoint.os, "replace", fail_install)

    with pytest.raises(OSError, match="checkpoint install failed"):
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=0,
            tokenizer=tokenizer,
        )

    assert observed_payload["step"] == 0
    assert checkpoint_path.read_bytes() == original_contents
    assert not list(tmp_path.glob(".last.pt.*.tmp"))


def test_checkpoint_save_accepts_an_independent_tokenizer_contract(
    tmp_path: Path,
) -> None:
    config, _, model, optimizer, scheduler, _ = _checkpoint_state()
    tokenizer = _IndependentByteCompatibleTokenizer()

    checkpoint_path = save_checkpoint(
        tmp_path / "contract.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["tokenizer"]["vocab_size"] == tokenizer.get_vocab_size()


def test_save_rejects_training_state_the_shared_loader_cannot_reconstruct(
    tmp_path: Path,
) -> None:
    config, tokenizer, model, optimizer, scheduler, _ = _checkpoint_state()
    incompatible_optimizer = SGD(model.parameters(), lr=0.01)
    incompatible_optimizer_scheduler = build_lr_scheduler(
        incompatible_optimizer,
        config.train,
    )

    with pytest.raises(TypeError, match="optimizer must be an AdamW"):
        save_checkpoint(
            tmp_path / "sgd.pt",
            model=model,
            optimizer=incompatible_optimizer,
            scheduler=incompatible_optimizer_scheduler,
            config=config,
            step=0,
            tokenizer=tokenizer,
        )

    incompatible_scheduler = StepLR(optimizer, step_size=1)
    with pytest.raises(
        TypeError,
        match="scheduler must be a WarmupConstantWarmdownLR",
    ):
        save_checkpoint(
            tmp_path / "step-lr.pt",
            model=model,
            optimizer=optimizer,
            scheduler=incompatible_scheduler,
            config=config,
            step=0,
            tokenizer=tokenizer,
        )

    assert not list(tmp_path.glob("*.pt"))
