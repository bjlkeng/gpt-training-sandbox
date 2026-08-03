"""SFT metric names and fixed public evaluation artifact contracts."""

from __future__ import annotations

from itertools import count
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import scratch_llm.evaluation.sft_sampling as sft_sampling
from scratch_llm.chat.rendering import CHAT_RENDERER_ID, render_completion_prompt
from scratch_llm.evaluation.sft_bpb import (
    SFT_ASSISTANT_BPB_PROTOCOL_ID,
    SFT_ASSISTANT_BPB_PROTOCOL_VERSION,
    SFTAssistantBPBResult,
)
from scratch_llm.evaluation.sft_sampling import (
    FIXED_SFT_PROMPTS,
    FixedSFTSamplingConfig,
    FixedSFTSamplesResult,
    generate_fixed_sft_samples,
)
from scratch_llm.evaluation.sft_tracking import (
    report_completed_sft_evaluation,
    sft_training_metrics,
    track_periodic_sft_validation,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from scratch_llm.tracking import Tracker


class _SpyTracker(Tracker):
    def __init__(self) -> None:
        self.metrics: list[tuple[dict[str, Any], int | None]] = []
        self.artifacts: list[tuple[str, str, str]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    def log_config(self, config: dict[str, Any]) -> None:
        pass

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.artifacts.append((path, name, type))

    def finish(self) -> None:
        pass


class _ScriptedChatModel(nn.Module):
    def __init__(self, tokenizer: ByteTokenizer) -> None:
        super().__init__()
        self.max_seq_len = 128
        self.vocab_size = tokenizer.get_vocab_size()
        self.assistant_start = tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = tokenizer.encode_special("<|assistant_end|>")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -100.0,
            dtype=torch.float32,
            device=token_ids.device,
        )
        for row, last_token in enumerate(token_ids[:, -1].tolist()):
            if last_token == self.assistant_start:
                next_token = ord("O")
            elif last_token == ord("O"):
                next_token = ord("K")
            else:
                next_token = self.assistant_end
            logits[row, -1, next_token] = 100.0
        return logits


def _validation(*, bpb: float = 1.25) -> SFTAssistantBPBResult:
    target_bytes = 8
    return SFTAssistantBPBResult(
        protocol_id=SFT_ASSISTANT_BPB_PROTOCOL_ID,
        protocol_version=SFT_ASSISTANT_BPB_PROTOCOL_VERSION,
        checkpoint_identity="sft:fixture#step:4",
        tokenizer_identity=ByteTokenizer().get_identity(),
        renderer_identity=CHAT_RENDERER_ID,
        validation_mixture_identity="sha256:" + "1" * 64,
        batch_budget=1,
        evaluated_batches=1,
        source_conversations=2,
        processed_model_tokens=64,
        supervised_target_tokens=10,
        supervised_target_bytes=target_bytes,
        total_nats=bpb * math.log(2) * target_bytes,
        bpb=bpb,
    )


def _samples(
    *, checkpoint_identity: str = "sha256:" + "2" * 64
) -> FixedSFTSamplesResult:
    tokenizer = ByteTokenizer()
    ticks = count()
    return generate_fixed_sft_samples(
        _ScriptedChatModel(tokenizer),
        tokenizer,
        checkpoint_identity=checkpoint_identity,
        config=FixedSFTSamplingConfig(
            max_new_tokens=4,
            temperature=0.0,
            top_k=1,
            seed=23,
        ),
        device="cpu",
        clock=lambda: float(next(ticks)),
    )


def test_fixed_sft_samples_use_the_shared_renderer_and_chat_stop_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def render_spy(conversation: object, tokenizer: ByteTokenizer):
        calls.append(conversation.messages[-1].content)  # type: ignore[union-attr]
        return render_completion_prompt(conversation, tokenizer)

    monkeypatch.setattr(sft_sampling, "render_completion_prompt", render_spy)
    tokenizer = ByteTokenizer()
    result = _samples()
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    assert FIXED_SFT_PROMPTS == (
        "Explain gradient descent in simple terms.",
        "Write a Python function to reverse a string.",
        "Give me three project ideas for learning PyTorch.",
        "What is 17 * 23? Show your work.",
        "Return a JSON object with keys name, age, and city.",
    )
    assert calls == list(FIXED_SFT_PROMPTS)
    assert result.renderer_identity == CHAT_RENDERER_ID
    assert result.assistant_end_token_id == assistant_end
    assert result.bos_token_id == tokenizer.get_bos_token_id()
    assert all(
        sample.prompt_token_ids[-1] == assistant_start for sample in result.samples
    )
    assert all(sample.text == "OK" for sample in result.samples)
    assert all(sample.completion_reason == "stop_token" for sample in result.samples)
    assert all(sample.stop_token_id == assistant_end for sample in result.samples)
    assert all(sample.sampled_token_count == 3 for sample in result.samples)
    assert all("<|assistant_end|>" not in sample.text for sample in result.samples)


def test_sft_training_metric_adapter_uses_only_the_stage_namespace() -> None:
    mapped = sft_training_metrics(
        {
            "train/loss": 0.75,
            "train/lrm": 0.5,
            "train/dt": 2.0,
            "train/tok_per_sec": 64.0,
            "train/mfu": 0.25,
            "train/grad_norm": 1.5,
            "train/peak_memory_mib": 321.0,
            "total_training_flops": 1000.0,
            "total_training_time": 4.0,
        }
    )

    assert mapped == {
        "sft/train_loss": 0.75,
        "sft/lrm": 0.5,
        "sft/dt": 2.0,
        "sft/tok_per_sec": 64.0,
        "sft/mfu": 0.25,
        "sft/grad_norm": 1.5,
        "sft/peak_memory_mib": 321.0,
        "total_training_flops": 1000.0,
        "total_training_time": 4.0,
    }
    assert not any(key.startswith("train/") for key in mapped)


def test_periodic_validation_and_completed_artifacts_use_exact_names_and_paths(
    tmp_path: Path,
) -> None:
    tracker = _SpyTracker()
    validation = _validation()
    samples = _samples()

    metrics = track_periodic_sft_validation(
        validation,
        tracker=tracker,
        step=4,
    )
    completed = report_completed_sft_evaluation(
        validation,
        samples,
        tracker=tracker,
        run_dir=tmp_path / "run",
        step=4,
        base_checkpoint_identity="sha256:" + "3" * 64,
        checkpoint_identity=samples.checkpoint_identity,
    )

    assert metrics == {"sft/val_bpb": 1.25}
    assert tracker.metrics == [({"sft/val_bpb": 1.25}, 4)]
    assert completed.report_path == tmp_path / "run/metrics/sft_eval.json"
    assert completed.samples_path == tmp_path / "run/metrics/sft_samples.md"
    assert tracker.artifacts == [
        ("metrics/sft_eval.json", "sft_eval", "evaluation"),
        ("metrics/sft_samples.md", "sft_samples", "evaluation"),
    ]
    report = json.loads(completed.report_path.read_text(encoding="utf-8"))
    markdown = completed.samples_path.read_text(encoding="utf-8")
    assert report["format"] == "scratch_llm_sft_evaluation"
    assert report["step"] == 4
    assert report["metrics"] == {"sft/val_bpb": 1.25}
    assert report["validation"] == validation.to_dict()
    assert report["samples"]["artifact_path"] == "metrics/sft_samples.md"
    assert all(prompt in markdown for prompt in FIXED_SFT_PROMPTS)
    assert markdown.count("OK") == len(FIXED_SFT_PROMPTS)
    assert "chatcore" not in completed.report_path.read_text(encoding="utf-8").lower()
    assert "chatcore" not in markdown.lower()
    assert "PRIVATE_TRAINING_CONVERSATION" not in markdown
    assert not tuple((tmp_path / "run/metrics").glob(".*.tmp"))
