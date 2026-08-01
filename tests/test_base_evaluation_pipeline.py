"""Checkpoint-backed base-evaluation pipeline tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import scratch_llm.base_evaluation_pipeline as pipeline
from scratch_llm.base_evaluation import BaseEvaluationError
from scratch_llm.base_evaluation_pipeline import evaluate_checkpoint_base_model
from scratch_llm.bpb import BPBAccumulation, BaseValidationResult
from scratch_llm.config import (
    GPTConfig,
    GenerationConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.core_evaluation import (
    CORE_PROTOCOL_ID,
    CoreEvaluationResult,
    CoreReferenceComparison,
    CoreTaskResult,
)
from scratch_llm.full_document_bpb import (
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_PROTOCOL_VERSION,
)
from scratch_llm.nanochat_bpb import (
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_PROTOCOL_VERSION,
    NANOCHAT_REFERENCE_COMMIT,
)
from scratch_llm.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.tracking import Tracker


_CHECKPOINT_IDENTITY = "sha256:" + "a" * 64
_MANIFEST_IDENTITY = "sha256:" + "b" * 64


class _SpyTracker(Tracker):
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append((metrics, step))

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.artifacts.append((path, name, type))

    def finish(self) -> None:
        pass


class _ConstantCompletionModel(nn.Module):
    def __init__(self, token_id: int) -> None:
        super().__init__()
        self.max_seq_len = 16
        self.vocab_size = VOCAB_SIZE
        self.token_id = token_id
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.vocab_size),
            -torch.inf,
            device=token_ids.device,
        )
        logits[:, -1, self.token_id] = self.anchor
        return logits


def _config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(
            name="base-eval",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=8,
            grad_accum_steps=1,
            max_steps=12,
            warmup_steps=0,
            warmdown_ratio=0.0,
            eval_tokens=8,
        ),
        generation=GenerationConfig(
            temperature=0,
            top_k=1,
            max_new_tokens=1,
            seed=7,
        ),
    )


def _protocol_result(
    protocol_id: str,
    *,
    tokenizer_identity: str,
) -> BaseValidationResult:
    compatibility = protocol_id == NANOCHAT_COMPAT_PROTOCOL_ID
    return BaseValidationResult.from_accumulation(
        BPBAccumulation(
            processed_model_tokens=8,
            counted_target_tokens=4,
            counted_target_bytes=4,
            total_nats=math.log(2) * 4,
        ),
        protocol_id=protocol_id,
        protocol_version=(
            NANOCHAT_COMPAT_PROTOCOL_VERSION
            if compatibility
            else FULL_DOCUMENT_PROTOCOL_VERSION
        ),
        reference_commit=NANOCHAT_REFERENCE_COMMIT if compatibility else None,
        reference_config={"fixture": protocol_id},
        checkpoint_identity=_CHECKPOINT_IDENTITY,
        tokenizer_identity=tokenizer_identity,
        validation_manifest_identity=_MANIFEST_IDENTITY,
        source_documents=1,
        source_tokens=4,
        source_bytes=4,
        unique_source_tokens=4,
        unique_source_bytes=4,
    )


def _core_result(*, tokenizer_identity: str) -> CoreEvaluationResult:
    return CoreEvaluationResult(
        checkpoint_identity=_CHECKPOINT_IDENTITY,
        tokenizer_identity=tokenizer_identity,
        bundle_identity="sha256:" + "c" * 64,
        config_identity="sha256:" + "d" * 64,
        metadata_identity="sha256:" + "e" * 64,
        run_kind="bounded",
        max_per_task=1,
        tasks=(
            CoreTaskResult(
                label="fixture",
                task_type="language_modeling",
                num_fewshot=0,
                random_baseline_percent=0.0,
                correct_examples=1,
                evaluated_examples=1,
                available_examples=2,
                elapsed_seconds=1.0,
                data_identity="sha256:" + "f" * 64,
            ),
        ),
        references=(CoreReferenceComparison("reference", 0.25),),
        elapsed_seconds=1.0,
    )


def test_sample_mode_does_not_open_validation_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    tokenizer = ByteTokenizer()
    checkpoint = SimpleNamespace(
        config=config,
        model=_ConstantCompletionModel(ord("A")),
        step=12,
        tokenizer=tokenizer,
    )
    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(pipeline, "load_model_checkpoint", lambda *_a, **_k: checkpoint)

    class _UnexpectedReader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("sample-only evaluation opened validation data")

    monkeypatch.setattr(pipeline, "TokenizedShardReader", _UnexpectedReader)
    tracker = _SpyTracker()

    result = evaluate_checkpoint_base_model(
        config,
        checkpoint_path=checkpoint_path,
        modes=("sample",),
        tracker=tracker,
        run_dir=tmp_path / "run",
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["requested_modes"] == ["sample"]
    assert payload["identities"]["validation_manifest"] is None
    assert payload["results"] == {}
    assert result.sample_markdown_path is not None
    assert result.sample_markdown_path.is_file()


def test_core_mode_loads_the_explicit_bundle_without_validation_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    tokenizer = ByteTokenizer()
    checkpoint = SimpleNamespace(
        config=config,
        model=_ConstantCompletionModel(ord("A")),
        step=12,
        tokenizer=tokenizer,
    )
    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    bundle_path = tmp_path / "eval_bundle.zip"
    bundle_path.write_bytes(b"bundle")
    bundle = object()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(pipeline, "load_model_checkpoint", lambda *_a, **_k: checkpoint)
    monkeypatch.setattr(pipeline, "file_identity", lambda _path: _CHECKPOINT_IDENTITY)
    monkeypatch.setattr(
        pipeline,
        "load_core_bundle",
        lambda path: calls.append(("load", path)) or bundle,
    )

    def evaluate(*args: object, **kwargs: object) -> CoreEvaluationResult:
        calls.append(("evaluate", args[2]))
        assert kwargs["max_per_task"] == 1
        return _core_result(tokenizer_identity=tokenizer.get_identity())

    monkeypatch.setattr(pipeline, "evaluate_core_bundle", evaluate)

    class _UnexpectedReader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("core-only evaluation opened validation data")

    monkeypatch.setattr(pipeline, "TokenizedShardReader", _UnexpectedReader)
    tracker = _SpyTracker()

    result = evaluate_checkpoint_base_model(
        config,
        checkpoint_path=checkpoint_path,
        modes=("core",),
        tracker=tracker,
        run_dir=tmp_path / "run",
        max_per_task=1,
        core_bundle_path=bundle_path,
    )

    assert calls == [("load", bundle_path), ("evaluate", bundle)]
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["core"]["protocol_id"] == CORE_PROTOCOL_ID
    assert payload["core"]["scope"]["bounded"] is True
    assert result.core_comparison_path == tmp_path / "run/metrics/core_comparison.md"
    assert result.core_comparison_path.is_file()
    assert tracker.metrics == []


def test_bpb_mode_runs_both_protocols_with_one_frozen_identity_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    tokenizer = ByteTokenizer()
    checkpoint = SimpleNamespace(
        config=config,
        model=object(),
        step=12,
        tokenizer=tokenizer,
    )
    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(pipeline, "load_model_checkpoint", lambda *_a, **_k: checkpoint)
    monkeypatch.setattr(pipeline, "file_identity", lambda _path: _CHECKPOINT_IDENTITY)
    monkeypatch.setattr(
        pipeline,
        "load_evaluation_token_bytes",
        lambda *_a, **_k: torch.ones(VOCAB_SIZE),
    )
    monkeypatch.setattr(
        pipeline,
        "tokenized_manifest_identity",
        lambda _manifest: _MANIFEST_IDENTITY,
    )

    class _Reader:
        manifest = object()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> _Reader:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(pipeline, "TokenizedShardReader", _Reader)
    calls: list[tuple[str, str]] = []

    def compatibility(*_args: object, **kwargs: Any) -> BaseValidationResult:
        calls.append(("compatibility", kwargs["checkpoint_identity"]))
        return _protocol_result(
            NANOCHAT_COMPAT_PROTOCOL_ID,
            tokenizer_identity=tokenizer.get_identity(),
        )

    def full_document(*_args: object, **kwargs: Any) -> BaseValidationResult:
        calls.append(("full_document", kwargs["checkpoint_identity"]))
        return _protocol_result(
            FULL_DOCUMENT_PROTOCOL_ID,
            tokenizer_identity=tokenizer.get_identity(),
        )

    monkeypatch.setattr(pipeline, "evaluate_nanochat_compatible_bpb", compatibility)
    monkeypatch.setattr(pipeline, "evaluate_full_document_bpb", full_document)

    result = evaluate_checkpoint_base_model(
        config,
        checkpoint_path=checkpoint_path,
        modes=("bpb",),
        tracker=_SpyTracker(),
        run_dir=tmp_path / "run",
    )

    assert calls == [
        ("compatibility", _CHECKPOINT_IDENTITY),
        ("full_document", _CHECKPOINT_IDENTITY),
    ]
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert set(payload["results"]) == {
        NANOCHAT_COMPAT_PROTOCOL_ID,
        FULL_DOCUMENT_PROTOCOL_ID,
    }
    assert payload["identities"]["validation_manifest"] == _MANIFEST_IDENTITY


def test_checkpoint_config_identity_mismatch_fails_before_artifact_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _config(tmp_path)
    checkpoint_config = _config(tmp_path)
    requested.model.n_embd = 16
    checkpoint = SimpleNamespace(
        config=checkpoint_config,
        model=_ConstantCompletionModel(ord("A")),
        step=12,
        tokenizer=ByteTokenizer(),
    )
    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(pipeline, "load_model_checkpoint", lambda *_a, **_k: checkpoint)
    tracker = _SpyTracker()
    run_dir = tmp_path / "run"

    with pytest.raises(BaseEvaluationError, match="model"):
        evaluate_checkpoint_base_model(
            requested,
            checkpoint_path=checkpoint_path,
            modes=("sample",),
            tracker=tracker,
            run_dir=run_dir,
        )

    assert not (run_dir / "metrics" / "base_eval.json").exists()
    assert tracker.metrics == []
    assert tracker.artifacts == []
