"""Tests for the typed project configuration schema."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Callable, cast

import pytest

from scratch_llm.config import (
    ActivationType,
    AttentionBackend,
    AttentionFallbackPolicy,
    CompileFallbackPolicy,
    CompileMode,
    ConfigValidationError,
    DEFAULT_SPECIAL_TOKENS,
    GPTConfig,
    GenerationConfig,
    FlashAttentionProvider,
    NormType,
    ProjectConfig,
    RunConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
    TrainDType,
    WandbConfig,
    WandbMode,
    WebConfig,
    apply_generation_overrides,
)


def test_defaults_cover_the_roadmap_sections_and_are_deterministic() -> None:
    first = ProjectConfig()
    second = ProjectConfig()

    assert is_dataclass(first)
    assert [field.name for field in fields(first)] == [
        "run",
        "tracking",
        "data",
        "tokenizer",
        "model",
        "train",
        "sft",
        "generation",
        "web",
    ]
    assert first == second
    assert first.run == RunConfig(
        name="smoke", seed=1337, device="cuda", output_dir="runs/out"
    )
    assert first.model == GPTConfig(
        vocab_size=32768,
        seq_len=512,
        n_layer=6,
        n_head=6,
        n_embd=384,
        mlp_ratio=4,
        dropout=0.0,
        bias=False,
        tie_weights=True,
    )
    assert first.train.grad_accum_steps == "auto"
    assert first.tracking.jsonl.enabled is True
    assert first.tracking.wandb.tags == []
    assert first.to_dict() == {
        "run": {
            "name": "smoke",
            "seed": 1337,
            "device": "cuda",
            "output_dir": "runs/out",
        },
        "tracking": {
            "jsonl": {"enabled": True, "path": "metrics/metrics.jsonl"},
            "wandb": {
                "enabled": False,
                "project": "scratch-llm",
                "entity": None,
                "group": None,
                "name": None,
                "tags": [],
                "mode": "online",
                "dir": "runs/wandb",
                "log_code": False,
                "log_model_artifacts": False,
                "log_dataset_artifacts": False,
                "log_tokenizer_artifacts": True,
                "log_prompts": False,
                "log_responses": False,
            },
        },
        "data": {
            "profile": "nanochat_climbmix",
            "base_dir": "data",
            "parquet_dir": "data/parquet/base_data_climbmix",
            "tokenized_dir": "data/tokenized",
            "loader_strategy": "packed",
            "text_column": "text",
            "num_tokenizer_train_shards": 8,
            "num_pretrain_train_shards": 16,
            "always_use_final_shard_for_val": True,
            "max_shard": 6542,
            "doc_cap_chars": 10_000,
        },
        "tokenizer": {
            "type": "regex_byte_bpe",
            "vocab_size": 32_768,
            "artifact_dir": None,
            "max_chars": 2_000_000_000,
            "doc_cap": 10_000,
            "special_tokens": list(DEFAULT_SPECIAL_TOKENS),
        },
        "model": {
            "profile": "simple_gpt",
            "depth": None,
            "aspect_ratio": None,
            "head_dim": None,
            "vocab_size": 32_768,
            "seq_len": 512,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "mlp_ratio": 4,
            "dropout": 0.0,
            "bias": False,
            "tie_weights": True,
            "norm": "layernorm",
            "activation": "gelu",
            "use_rope": False,
            "rope_theta": 10_000.0,
            "use_rmsnorm": False,
            "use_qk_norm": False,
            "use_gqa": False,
            "attention_backend": "manual",
            "attention_fallback_policy": "allow",
            "flash_attention_provider": "auto",
            "use_flash_attention": False,
            "use_kv_cache": False,
        },
        "train": {
            "device_batch_size": 4,
            "total_batch_size_tokens": 65_536,
            "grad_accum_steps": "auto",
            "max_steps": 20_000,
            "learning_rate": 0.0003,
            "min_lr": 0.000015,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "warmup_steps": 40,
            "warmdown_ratio": 0.65,
            "final_lr_frac": 0.05,
            "eval_every": 250,
            "eval_tokens": 1_048_576,
            "sample_every": 1_000,
            "save_every": 1_000,
            "log_every": 10,
            "mfu_peak_flops_per_second": None,
            "mfu_peak_flops_basis": None,
            "dtype": "float32",
            "compile": False,
            "compile_backend": "inductor",
            "compile_mode": "default",
            "compile_fallback_policy": "eager",
            "compile_fullgraph": False,
            "compile_dynamic": False,
            "activation_checkpointing": False,
        },
        "sft": SFTConfig().to_dict(),
        "generation": {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": None,
            "max_new_tokens": 256,
            "seed": None,
        },
        "web": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8000,
            "checkpoint_dir": "runs/out",
            "allow_remote_bind": False,
        },
    }
    assert first.tokenizer.special_tokens is not second.tokenizer.special_tokens
    assert first.tracking.wandb.tags is not second.tracking.wandb.tags
    assert first.sft.train_sources is not second.sft.train_sources


@pytest.mark.parametrize("field_name", ["log_prompts", "log_responses"])
@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_wandb_chat_content_gates_require_real_booleans(
    field_name: str,
    invalid: object,
) -> None:
    config = WandbConfig()
    setattr(config, field_name, invalid)

    with pytest.raises(
        ConfigValidationError,
        match=rf"tracking\.wandb\.{field_name}.*boolean",
    ):
        config.validate()


def test_generation_overrides_are_shared_validated_and_detached() -> None:
    defaults = GenerationConfig(
        temperature=0.7,
        top_k=11,
        max_new_tokens=9,
        seed=3,
    )

    updated = apply_generation_overrides(
        defaults,
        {"temperature": 0.25, "top_k": None, "max_new_tokens": 4},
    )

    assert updated == GenerationConfig(
        temperature=0.25,
        top_k=11,
        max_new_tokens=4,
        seed=3,
    )
    assert defaults.temperature == 0.7
    with pytest.raises(ValueError, match="unsupported generation overrides"):
        apply_generation_overrides(defaults, {"unknown": 1})


def test_gpt_config_exposes_the_planned_dimensions_and_architecture() -> None:
    config = GPTConfig(
        vocab_size=265,
        seq_len=128,
        n_layer=2,
        n_head=2,
        n_embd=128,
        mlp_ratio=3,
        tie_weights=False,
    )

    assert (
        config.vocab_size,
        config.seq_len,
        config.n_layer,
        config.n_head,
        config.n_embd,
        config.mlp_ratio,
        config.tie_weights,
    ) == (265, 128, 2, 2, 128, 3, False)


@pytest.mark.parametrize(
    "special_tokens",
    [
        [*DEFAULT_SPECIAL_TOKENS, "<|pad|>"],
        list(reversed(DEFAULT_SPECIAL_TOKENS)),
        list(DEFAULT_SPECIAL_TOKENS[:-1]),
        ["<|bos|>", *DEFAULT_SPECIAL_TOKENS[1:-1], "<|unknown|>"],
    ],
)
def test_tokenizer_config_requires_the_locked_nanochat_special_token_order(
    special_tokens: list[str],
) -> None:
    with pytest.raises(
        ConfigValidationError,
        match=r"^tokenizer\.special_tokens:.*nine ordered nanochat special tokens",
    ):
        TokenizerConfig(special_tokens=special_tokens)


@pytest.mark.parametrize("vocab_size", [264, 266, 32_768])
def test_byte_tokenizer_config_requires_its_stable_vocabulary_size(
    vocab_size: int,
) -> None:
    with pytest.raises(
        ConfigValidationError,
        match=r"^tokenizer\.vocab_size:.*exactly 265.*byte tokenizer",
    ):
        TokenizerConfig(type="byte", vocab_size=vocab_size)


def test_to_dict_preserves_nested_configured_values_without_aliasing() -> None:
    config = ProjectConfig(
        run=RunConfig(name="experiment", seed=7, device="cpu", output_dir="tmp/run"),
        train=TrainConfig(
            device_batch_size=2,
            total_batch_size_tokens=1024,
            grad_accum_steps=1,
        ),
        generation=GenerationConfig(
            temperature=0.25,
            top_k=None,
            top_p=0.9,
            max_new_tokens=12,
            seed=99,
        ),
        web=WebConfig(enabled=False, port=9000),
    )

    serialized = config.to_dict()

    assert serialized["run"] == {
        "name": "experiment",
        "seed": 7,
        "device": "cpu",
        "output_dir": "tmp/run",
    }
    assert serialized["model"]["tie_weights"] is True
    assert serialized["generation"] == {
        "temperature": 0.25,
        "top_k": None,
        "top_p": 0.9,
        "max_new_tokens": 12,
        "seed": 99,
    }
    assert serialized["web"]["port"] == 9000
    assert set(serialized) == {
        "run",
        "tracking",
        "data",
        "tokenizer",
        "model",
        "train",
        "sft",
        "generation",
        "web",
    }

    serialized["tracking"]["wandb"]["tags"].append("changed")
    serialized["tokenizer"]["special_tokens"].append("<|changed|>")
    assert config.tracking.wandb.tags == []
    assert "<|changed|>" not in config.tokenizer.special_tokens


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"vocab_size": 0}, "model.vocab_size"),
        ({"seq_len": 0}, "model.seq_len"),
        ({"n_layer": 0}, "model.n_layer"),
        ({"n_head": 0}, "model.n_head"),
        ({"n_embd": 0}, "model.n_embd"),
        ({"mlp_ratio": 0}, "model.mlp_ratio"),
        ({"n_head": 3, "n_embd": 128}, "model.n_embd"),
    ],
)
def test_gpt_validation_rejects_invalid_dimensions(
    kwargs: dict[str, object], path: str
) -> None:
    with pytest.raises(ConfigValidationError, match=rf"^{path}:") as exc_info:
        GPTConfig(**kwargs)  # type: ignore[arg-type]

    assert exc_info.value.path == path


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"device_batch_size": 0}, "train.device_batch_size"),
        ({"total_batch_size_tokens": 0}, "train.total_batch_size_tokens"),
        ({"grad_accum_steps": 0}, "train.grad_accum_steps"),
    ],
)
def test_train_validation_rejects_non_positive_batches(
    kwargs: dict[str, object], path: str
) -> None:
    with pytest.raises(ConfigValidationError, match=rf"^{path}:"):
        TrainConfig(**kwargs)  # type: ignore[arg-type]


def test_train_validation_rejects_overlapping_schedule_ranges() -> None:
    with pytest.raises(
        ConfigValidationError,
        match=r"^train\.warmup_steps:.*warmdown",
    ):
        TrainConfig(max_steps=10, warmup_steps=7, warmdown_ratio=0.4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mfu_peak_flops_per_second": 35.58e12},
        {"mfu_peak_flops_basis": "advertised FP32 peak"},
        {
            "mfu_peak_flops_per_second": float("inf"),
            "mfu_peak_flops_basis": "invalid infinite peak",
        },
        {
            "mfu_peak_flops_per_second": 0.0,
            "mfu_peak_flops_basis": "invalid zero peak",
        },
        {
            "mfu_peak_flops_per_second": 35.58e12,
            "mfu_peak_flops_basis": " ",
        },
    ],
)
def test_train_mfu_basis_requires_a_complete_finite_explicit_pair(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ConfigValidationError, match=r"^train\.mfu_peak_flops"):
        TrainConfig(**kwargs)  # type: ignore[arg-type]


def test_train_mfu_basis_preserves_the_explicit_hardware_assumption() -> None:
    config = TrainConfig(
        mfu_peak_flops_per_second=35.58e12,
        mfu_peak_flops_basis="RTX 3090 advertised FP32 peak",
    )

    assert config.mfu_peak_flops_per_second == 35.58e12
    assert config.mfu_peak_flops_basis == "RTX 3090 advertised FP32 peak"


@pytest.mark.parametrize("field_name", ["beta1", "beta2"])
def test_train_validation_rejects_adamw_beta_of_one(field_name: str) -> None:
    with pytest.raises(
        ConfigValidationError,
        match=rf"^train\.{field_name}:.*\[0, 1\)",
    ):
        TrainConfig(**{field_name: 1.0})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "path"),
    [
        (
            lambda: WandbConfig(mode=cast(WandbMode, "sometimes")),
            "tracking.wandb.mode",
        ),
        (lambda: GPTConfig(norm=cast(NormType, "batchnorm")), "model.norm"),
        (
            lambda: GPTConfig(activation=cast(ActivationType, "relu")),
            "model.activation",
        ),
        (
            lambda: GPTConfig(attention_backend=cast(AttentionBackend, "automatic")),
            "model.attention_backend",
        ),
        (
            lambda: GPTConfig(
                attention_fallback_policy=cast(AttentionFallbackPolicy, "silent")
            ),
            "model.attention_fallback_policy",
        ),
        (
            lambda: GPTConfig(
                flash_attention_provider=cast(FlashAttentionProvider, "fa4")
            ),
            "model.flash_attention_provider",
        ),
        (
            lambda: TrainConfig(compile_mode=cast(CompileMode, "fastest")),
            "train.compile_mode",
        ),
        (
            lambda: SFTConfig(
                compile_fallback_policy=cast(CompileFallbackPolicy, "silent")
            ),
            "sft.compile_fallback_policy",
        ),
        (lambda: TrainConfig(dtype=cast(TrainDType, "float64")), "train.dtype"),
    ],
)
def test_validation_rejects_invalid_modes(
    factory: Callable[[], object], path: str
) -> None:
    with pytest.raises(ConfigValidationError, match=rf"^{path}:"):
        factory()


def test_legacy_flash_boolean_cannot_contradict_the_canonical_backend() -> None:
    with pytest.raises(
        ConfigValidationError,
        match="model.use_flash_attention:.*canonical model.attention_backend",
    ):
        GPTConfig(use_flash_attention=True)


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_tracking_schema_accepts_every_wandb_mode_with_jsonl_always_on(
    mode: WandbMode,
) -> None:
    config = ProjectConfig()
    config.tracking.wandb.enabled = True
    config.tracking.wandb.mode = mode

    config.validate()

    assert config.tracking.jsonl.enabled is True


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (
            lambda config: setattr(config.tracking.jsonl, "enabled", False),
            "tracking.jsonl.enabled",
        ),
        (
            lambda config: setattr(config.model, "vocab_size", 265),
            "model.vocab_size",
        ),
        (
            lambda config: setattr(config.train, "total_batch_size_tokens", 65535),
            "train.total_batch_size_tokens",
        ),
        (
            lambda config: setattr(config.train, "grad_accum_steps", 2),
            "train.grad_accum_steps",
        ),
        (
            lambda config: setattr(config.web, "host", "0.0.0.0"),
            "web.host",
        ),
        (
            lambda config: setattr(config.model, "use_rmsnorm", True),
            "model.use_rmsnorm",
        ),
    ],
)
def test_project_validation_rejects_documented_contradictions(
    mutate: Callable[[ProjectConfig], None], path: str
) -> None:
    config = ProjectConfig()
    mutate(config)

    with pytest.raises(ConfigValidationError, match=rf"^{path}:") as exc_info:
        config.validate()

    assert exc_info.value.path == path
