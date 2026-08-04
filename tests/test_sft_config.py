"""Typed SFT source, optimization, and preset configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scratch_llm.config import (
    ConfigValidationError,
    SFTConfig,
    SFTSourceConfig,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sft_defaults_match_the_fresh_optimizer_contract() -> None:
    config = SFTConfig()

    assert config.base_checkpoint is None
    assert config.learning_rate == pytest.approx(2e-5)
    assert config.weight_decay == 0.0
    assert config.device_batch_size == 2
    assert config.total_batch_size_tokens == 32_768
    assert config.max_steps == 5_000
    assert config.eval_every == 250
    assert config.eval_batches == 8
    assert config.train_sources == [
        SFTSourceConfig(
            kind="jsonl",
            path="data/fixtures/chat/train.jsonl",
            repeat_weight=1,
            shuffle=True,
        )
    ]
    assert config.validation_sources == [
        SFTSourceConfig(
            kind="jsonl",
            path="data/fixtures/chat/validation.jsonl",
            repeat_weight=1,
            shuffle=False,
        )
    ]


def test_sft_training_adapter_preserves_every_optimizer_and_budget_value() -> None:
    config = SFTConfig(
        device_batch_size=3,
        total_batch_size_tokens=96,
        grad_accum_steps=2,
        max_steps=12,
        learning_rate=3e-5,
        min_lr=1e-6,
        weight_decay=0.0,
        beta1=0.8,
        beta2=0.9,
        grad_clip=0.5,
        warmup_steps=2,
        warmdown_ratio=0.25,
        final_lr_frac=0.1,
        eval_every=3,
        eval_batches=2,
        save_every=4,
        log_every=1,
    )

    train = config.to_train_config(seq_len=16)

    assert train.device_batch_size == 3
    assert train.total_batch_size_tokens == 96
    assert train.grad_accum_steps == 2
    assert train.max_steps == 12
    assert train.learning_rate == pytest.approx(3e-5)
    assert train.min_lr == pytest.approx(1e-6)
    assert train.weight_decay == 0.0
    assert (train.beta1, train.beta2) == pytest.approx((0.8, 0.9))
    assert train.grad_clip == pytest.approx(0.5)
    assert train.warmup_steps == 2
    assert train.warmdown_ratio == pytest.approx(0.25)
    assert train.final_lr_frac == pytest.approx(0.1)
    assert train.eval_every == 3
    assert train.eval_tokens == 96
    assert train.save_every == 4
    assert train.log_every == 1


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"base_checkpoint": "   "}, "sft.base_checkpoint"),
        ({"base_checkpoint": "base.bin"}, "sft.base_checkpoint"),
        ({"train_sources": []}, "sft.train_sources"),
        ({"validation_sources": []}, "sft.validation_sources"),
        ({"device_batch_size": 0}, "sft.device_batch_size"),
        ({"total_batch_size_tokens": 0}, "sft.total_batch_size_tokens"),
        ({"eval_batches": 0}, "sft.eval_batches"),
        ({"eval_every": 11, "max_steps": 10}, "sft.eval_every"),
        (
            {"save_every": 11, "eval_every": 1, "max_steps": 10},
            "sft.save_every",
        ),
    ],
)
def test_sft_config_rejects_invalid_paths_sources_budgets_and_cadence(
    kwargs: dict[str, object],
    path: str,
) -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        SFTConfig(**kwargs)  # type: ignore[arg-type]

    assert exc_info.value.path == path


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"kind": "unknown"}, "sft.source.kind"),
        ({"path": ""}, "sft.source.path"),
        ({"repeat_weight": 0}, "sft.source.repeat_weight"),
        (
            {"kind": "jsonl", "path": "train.txt"},
            "sft.source.path",
        ),
        (
            {"kind": "hub_cache", "path": "data/parquet/sft"},
            "sft.source.dataset",
        ),
        (
            {
                "kind": "jsonl",
                "path": "train.jsonl",
                "dataset": "smoltalk",
                "split": "train",
            },
            "sft.source.dataset",
        ),
    ],
)
def test_sft_source_config_is_an_exact_local_or_cache_choice(
    kwargs: dict[str, object],
    path: str,
) -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        SFTSourceConfig(**kwargs)  # type: ignore[arg-type]

    assert exc_info.value.path == path


def test_project_validation_checks_sft_token_budget_and_validation_source_policy() -> (
    None
):
    config = load_config(PROJECT_ROOT / "configs" / "sft_smoke.yaml")
    config.sft.total_batch_size_tokens += 1

    with pytest.raises(
        ConfigValidationError,
        match=r"^sft\.total_batch_size_tokens:.*divisible",
    ):
        config.validate()

    config = load_config(PROJECT_ROOT / "configs" / "sft_smoke.yaml")
    config.sft.validation_sources[0].shuffle = True
    with pytest.raises(
        ConfigValidationError,
        match=r"^sft\.validation_sources\.0\.shuffle:.*false",
    ):
        config.validate()


def test_sft_smoke_preset_is_bounded_cpu_local_and_deterministic() -> None:
    path = PROJECT_ROOT / "configs" / "sft_smoke.yaml"

    first = load_config(path)
    second = load_config(path)

    assert first == second
    assert first.run.device == "cpu"
    assert first.tokenizer.type == "byte"
    assert first.model.seq_len == 64
    assert first.sft.device_batch_size == 2
    assert first.sft.total_batch_size_tokens == 128
    assert first.sft.max_steps == 20
    assert first.sft.weight_decay == 0.0
    assert all(source.kind == "jsonl" for source in first.sft.train_sources)
    assert first.to_yaml() == second.to_yaml()


def test_sft_20m_3090_preset_encodes_single_device_baseline() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "sft_20m_3090.yaml")

    assert config.run.device == "cuda"
    assert config.model.profile == "simple_gpt"
    assert config.model.seq_len == 1024
    assert config.sft.device_batch_size == 2
    assert config.sft.total_batch_size_tokens == 32_768
    assert config.sft.learning_rate == pytest.approx(2e-5)
    assert config.sft.weight_decay == 0.0
    assert {source.dataset for source in config.sft.train_sources} == {
        "gsm8k",
        "mmlu",
        "smoltalk",
    }
    assert config.sft.to_train_config(config.model.seq_len).grad_accum_steps == 16


def test_sft_111m_3090_experiment_preset_matches_the_base_and_weighted_mix() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "sft_111m_3090.yaml")

    assert config.run.name == "sft-111m-base30k-3090"
    assert config.run.device == "cuda"
    assert config.tracking.wandb.enabled is True
    assert config.tracking.wandb.project == "gpt-training-sandbox"
    assert config.tracking.wandb.entity is None
    assert config.tracking.wandb.group == "111m-3090-base-to-sft"
    assert config.tracking.wandb.mode == "online"
    assert config.tracking.wandb.log_model_artifacts is False
    assert config.tracking.wandb.log_dataset_artifacts is False
    assert config.tracking.wandb.log_tokenizer_artifacts is False
    assert config.data.loader_strategy == "packed"
    assert config.data.tokenized_dir == "data/tokenized_37"
    assert config.data.num_pretrain_train_shards == 37
    assert config.data.always_use_final_shard_for_val is True
    assert config.model.seq_len == 1_024
    assert config.model.n_layer == 12
    assert config.model.n_head == 12
    assert config.model.n_embd == 768
    assert config.sft.device_batch_size == 8
    assert config.sft.total_batch_size_tokens == 32_768
    assert config.sft.max_steps == 2_000
    assert config.sft.learning_rate == pytest.approx(1e-5)
    assert config.sft.warmup_steps == 50
    assert config.sft.warmdown_ratio == pytest.approx(0.5)
    assert config.sft.save_every == 250
    assert {
        source.dataset: source.repeat_weight for source in config.sft.train_sources
    } == {"smoltalk": 1, "mmlu": 3, "gsm8k": 4}
    assert config.sft.to_train_config(config.model.seq_len).grad_accum_steps == 4
