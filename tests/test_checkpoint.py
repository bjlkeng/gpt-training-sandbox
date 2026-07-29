"""Bounded CPU coverage for the base-model checkpoint contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from scratch_llm import checkpoint
from scratch_llm.checkpoint import (
    load_model_checkpoint,
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
from scratch_llm.data import NextTokenDataset
from scratch_llm.model import GPT
from scratch_llm.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tokenizer import (
    BYTE_VOCAB_SIZE,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    ByteTokenizer,
    Tokenizer,
)
from scratch_llm.tracking import NullTracker
from scratch_llm.training import run_training_steps


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
    legacy_path = tmp_path / "legacy-v1-byte.pt"
    torch.save(payload, legacy_path)

    loaded = load_model_checkpoint(legacy_path)

    assert loaded.config == config
    assert loaded.step == 0
    assert isinstance(loaded.tokenizer, ByteTokenizer)


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

    resumed = load_training_checkpoint(checkpoint_path, device="cpu")

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
