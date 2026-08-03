"""Bounded CPU proof of tiny SFT overfit and chat-stage stop semantics."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scratch_llm.chat.conversation import Conversation, read_conversations
from scratch_llm.chat.rendering import render_completion_prompt, render_conversation
from scratch_llm.config import ProjectConfig, dump_config, load_config
from scratch_llm.generation import generate_sequences
from scratch_llm.model import GPT
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from scratch_llm.training.checkpoint import load_model_checkpoint, save_checkpoint
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_SMOKE_CONFIG = PROJECT_ROOT / "configs" / "sft_smoke.yaml"
TRAIN_FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "chat" / "train.jsonl"
VALIDATION_FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "chat" / "validation.jsonl"
OVERFIT_MAX_STEPS = 200
OVERFIT_MAX_MEAN_LOSS = 0.35
OVERFIT_MIN_TOKEN_ACCURACY = 0.95


def _overfit_config(tmp_path: Path, *, base_path: Path) -> ProjectConfig:
    config = load_config(SFT_SMOKE_CONFIG)
    config.run.name = "sft-overfit"
    config.run.seed = 2027
    config.run.device = "cpu"
    config.run.output_dir = str(tmp_path / "runs")
    config.model.n_layer = 2
    config.model.n_head = 2
    config.model.n_embd = 64
    config.model.mlp_ratio = 2
    config.model.dropout = 0.0
    config.sft.base_checkpoint = str(base_path)
    config.sft.packing_buffer_size = 1
    config.sft.device_batch_size = 1
    config.sft.total_batch_size_tokens = config.model.seq_len
    config.sft.max_steps = OVERFIT_MAX_STEPS
    config.sft.learning_rate = 0.01
    config.sft.min_lr = 0.0
    config.sft.weight_decay = 0.0
    config.sft.warmup_steps = 0
    config.sft.warmdown_ratio = 0.5
    config.sft.eval_every = OVERFIT_MAX_STEPS
    config.sft.eval_batches = 2
    config.sft.save_every = OVERFIT_MAX_STEPS
    config.sft.log_every = 20
    config.generation.temperature = 0.0
    config.generation.top_k = 1
    config.generation.max_new_tokens = 16
    config.generation.seed = 2027
    config.validate()
    return config


def _write_deterministic_base_checkpoint(
    path: Path,
    config: ProjectConfig,
) -> Path:
    torch.manual_seed(2027)
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    return save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=ByteTokenizer(),
    )


def _masked_loss_and_accuracy(
    model: nn.Module,
    tokenizer: ByteTokenizer,
    conversations: tuple[Conversation, ...],
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    supervised = 0
    with torch.inference_mode():
        for conversation in conversations:
            rendered = render_conversation(conversation, tokenizer)
            inputs = torch.tensor([rendered.token_ids[:-1]], dtype=torch.long)
            labels = torch.tensor(
                [
                    [
                        token_id if selected else -1
                        for token_id, selected in zip(
                            rendered.token_ids[1:],
                            rendered.loss_mask[1:],
                            strict=True,
                        )
                    ]
                ],
                dtype=torch.long,
            )
            logits = model(inputs)
            mask = labels.ne(-1)
            assert bool(mask.any().item())
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-1,
                reduction="sum",
            )
            perturbed = logits.clone()
            perturbed[~mask] = 1_000.0
            perturbed_loss = F.cross_entropy(
                perturbed.reshape(-1, perturbed.shape[-1]),
                labels.reshape(-1),
                ignore_index=-1,
                reduction="sum",
            )
            torch.testing.assert_close(loss, perturbed_loss, rtol=0, atol=0)
            total_loss += float(loss.item())
            correct += int((logits.argmax(dim=-1)[mask] == labels[mask]).sum().item())
            supervised += int(mask.sum().item())
    return total_loss / supervised, correct / supervised


def test_real_train_sft_command_overfits_and_generates_from_its_chat_checkpoint(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "deterministic-base.pt"
    config = _overfit_config(tmp_path, base_path=base_path)
    _write_deterministic_base_checkpoint(base_path, config)
    config_path = dump_config(config, tmp_path / "sft-overfit.yaml")
    tokenizer = ByteTokenizer()
    training_conversations = read_conversations(TRAIN_FIXTURE)
    assert any(
        conversation.messages[0].role == "system"
        for conversation in training_conversations
    )
    assert any(
        len(conversation.messages) > 2 for conversation in training_conversations
    )
    base = load_model_checkpoint(base_path)
    base_loss, base_accuracy = _masked_loss_and_accuracy(
        base.model,
        tokenizer,
        training_conversations,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_sft",
            "--config",
            str(config_path),
            "--base-checkpoint",
            str(base_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    run_dir = Path(config.run.output_dir) / config.run.name
    best_path = run_dir / "checkpoints/best.pt"
    last_path = run_dir / "checkpoints/last.pt"
    assert completed.returncode == 0, completed.stderr
    assert f"Run directory: {run_dir}" in completed.stdout
    assert "Base checkpoint identity: sha256:" in completed.stdout
    assert f"Completed step {OVERFIT_MAX_STEPS}" in completed.stdout
    assert "Assistant validation BPB:" in completed.stdout
    assert f"Best checkpoint: {best_path}" in completed.stdout
    assert f"Last checkpoint: {last_path}" in completed.stdout
    assert best_path.is_file()
    assert last_path.is_file()
    assert load_model_checkpoint(best_path).training_stage == "sft"
    trained = load_model_checkpoint(last_path)
    assert trained.training_stage == "sft"
    final_loss, final_accuracy = _masked_loss_and_accuracy(
        trained.model,
        tokenizer,
        training_conversations,
    )
    observed = {
        "base_loss": base_loss,
        "base_accuracy": base_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
    }
    assert final_loss < OVERFIT_MAX_MEAN_LOSS, observed
    assert final_loss < base_loss * 0.1, observed
    assert final_accuracy >= OVERFIT_MIN_TOKEN_ACCURACY, observed
    assert final_accuracy > base_accuracy

    evaluation = json.loads((run_dir / "metrics/sft_eval.json").read_text())
    assert math.isfinite(evaluation["metrics"]["sft/val_bpb"])
    assert evaluation["metrics"]["sft/val_bpb"] >= 0
    assert (run_dir / "metrics/sft_samples.md").is_file()
    assert (run_dir / "metrics/metrics.jsonl").is_file()

    held_conversation = read_conversations(VALIDATION_FIXTURE)[0]
    held_prompt = Conversation(messages=held_conversation.messages[:-1])
    rendered_prompt = render_completion_prompt(held_prompt, tokenizer)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    generated = generate_sequences(
        trained.model,
        torch.tensor([rendered_prompt.token_ids], dtype=torch.long),
        max_new_tokens=16,
        temperature=0.0,
        top_k=1,
        seed=2027,
        stop_token_ids={assistant_end, tokenizer.get_bos_token_id()},
    ).sequences[0]
    assistant_text = tokenizer.decode(generated.generated_token_ids)
    assert generated.generated_token_ids
    assert assistant_text
    assert generated.completion_reason == "stop_token"
    assert generated.stop_token_id == assistant_end
    assert "<|assistant_end|>" not in assistant_text


class _AssistantEndModel(nn.Module):
    def __init__(self, tokenizer: ByteTokenizer) -> None:
        super().__init__()
        self.max_seq_len = 64
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
        last_token = int(token_ids[0, -1].item())
        next_token = (
            ord("A") if last_token == self.assistant_start else self.assistant_end
        )
        logits[0, -1, next_token] = 100.0
        return logits


def test_assistant_end_is_consumed_as_stop_metadata_without_visible_control_text() -> (
    None
):
    tokenizer = ByteTokenizer()
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    prompt = render_completion_prompt(
        Conversation(messages=read_conversations(VALIDATION_FIXTURE)[0].messages[:-1]),
        tokenizer,
    )

    sequence = generate_sequences(
        _AssistantEndModel(tokenizer),
        torch.tensor([prompt.token_ids], dtype=torch.long),
        max_new_tokens=8,
        temperature=0.0,
        top_k=1,
        seed=9,
        stop_token_ids={assistant_end, tokenizer.get_bos_token_id()},
    ).sequences[0]

    assert sequence.completion_reason == "stop_token"
    assert sequence.stop_token_id == assistant_end
    assert sequence.sampled_token_count == 2
    assert sequence.generated_token_ids == (ord("A"),)
    assert assistant_end not in sequence.generated_token_ids
    assert tokenizer.decode(sequence.generated_token_ids) == "A"
    assert "<|assistant_end|>" not in tokenizer.decode(sequence.generated_token_ids)


def test_readme_documents_bounded_thresholds_and_optional_3090_smoke() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "assistant-only mean training loss below `0.35`" in readme
    assert "at least `95%` supervised-token accuracy" in readme
    assert "tests/test_sft_overfit_integration.py" in readme
    assert "--config configs/sft_20m_3090.yaml" in readme
    assert "--base-checkpoint runs/base-20m/checkpoints/best.pt" in readme
    assert "outside CI" in readme
