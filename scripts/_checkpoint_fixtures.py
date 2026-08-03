"""Small deterministic checkpoints shared by command-level smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch

from scratch_llm.config import (
    GPTConfig,
    GenerationConfig,
    ProjectConfig,
    RunConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.model import GPT
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.training.checkpoint import ExactTrainingState, save_checkpoint
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.rng_state import capture_training_rng_state


def create_tiny_sft_checkpoint(
    path: Path,
    *,
    preferred_token_id: int = 0,
    seq_len: int = 16,
    n_layer: int = 1,
    n_head: int = 1,
    n_embd: int = 8,
    max_new_tokens: int = 2,
) -> Path:
    """Save a CPU checkpoint whose greedy output repeats one byte token."""

    if not 0 <= preferred_token_id < VOCAB_SIZE:
        raise ValueError("preferred_token_id must be in the byte vocabulary")
    if n_embd < 2:
        raise ValueError("n_embd must be at least 2")
    config = ProjectConfig(
        run=RunConfig(device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            mlp_ratio=2,
            tie_weights=False,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=seq_len,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0,
        ),
        sft=SFTConfig(
            device_batch_size=1,
            total_batch_size_tokens=seq_len,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0,
            eval_every=1,
            eval_batches=1,
            save_every=1,
            log_every=1,
        ),
        generation=GenerationConfig(
            temperature=0,
            top_k=1,
            max_new_tokens=max_new_tokens,
        ),
    )
    model = GPT(config.model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.token_embedding.weight[:, 0] = 1
        model.token_embedding.weight[:, 1] = -1
        model.ln_f.weight.fill_(1)
        model.lm_head.weight[preferred_token_id, 0] = 1
        model.lm_head.weight[preferred_token_id, 1] = -1
    active_train = config.sft.to_train_config(config.model.seq_len)
    optimizer = build_optimizer(model, active_train)
    scheduler = build_lr_scheduler(optimizer, active_train)
    return save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=ByteTokenizer(),
        continuation=ExactTrainingState(
            loader_format="fixture_loader_v1",
            loader_state={"format": "fixture_loader_v1", "position": 0},
            rng_state=capture_training_rng_state("cpu"),
            tracker_step=0,
            total_training_time_seconds=0,
            total_training_flops=0,
        ),
        training_stage="sft",
        base_checkpoint_identity="sha256:" + "a" * 64,
    )


__all__ = ["create_tiny_sft_checkpoint"]
