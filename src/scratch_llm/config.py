"""Typed, serializable configuration for the training pipeline.

The defaults in this module mirror the example configuration in the project
roadmap.  Validation deliberately happens both at construction time and when
``validate`` is called so that loaders can re-check a configuration after
applying overrides.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, NoReturn, get_args

from omegaconf import DictConfig, OmegaConf

from scratch_llm._validation import (
    ConfigValidationError,
    _fail,
    _require_choice,
    _require_half_open_unit_interval,
    _require_int,
    _require_non_empty,
    _require_non_negative_int,
    _require_non_negative_real,
    _require_positive_int,
    _require_positive_real,
    _require_real,
    _require_unit_interval,
)
from scratch_llm.tokenization.tokenizer import (
    NANOCHAT_SPECIAL_TOKENS,
    VOCAB_SIZE as BYTE_TOKENIZER_VOCAB_SIZE,
)
from scratch_llm.utils import atomic_write


WandbMode = Literal["online", "offline", "disabled"]
TokenizerType = Literal["byte", "regex_byte_bpe"]
TokenLoaderStrategy = Literal["flat", "packed"]
NormType = Literal["layernorm", "rmsnorm"]
ActivationType = Literal["gelu", "relu_squared"]
AttentionBackend = Literal["manual", "sdpa", "flash"]
AttentionFallbackPolicy = Literal["allow", "error"]
FlashAttentionProvider = Literal["auto", "fa2", "fa3"]
TrainDType = Literal["float32", "float16", "bfloat16"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune"]
CompileFallbackPolicy = Literal["eager", "error"]
SFTSourceKind = Literal["jsonl", "hub_cache"]
# OmegaConf does not yet support combining ``Literal`` with another type in a
# union. Runtime validation below still restricts string values to ``"auto"``.
GradAccumSteps = int | str

DEFAULT_SPECIAL_TOKENS = NANOCHAT_SPECIAL_TOKENS

_WANDB_MODES: frozenset[str] = frozenset(get_args(WandbMode))
_TOKENIZER_TYPES: frozenset[str] = frozenset(get_args(TokenizerType))
_TOKEN_LOADER_STRATEGIES: frozenset[str] = frozenset(get_args(TokenLoaderStrategy))
_NORM_TYPES: frozenset[str] = frozenset(get_args(NormType))
_ACTIVATION_TYPES: frozenset[str] = frozenset(get_args(ActivationType))
_ATTENTION_BACKENDS: frozenset[str] = frozenset(get_args(AttentionBackend))
_ATTENTION_FALLBACK_POLICIES: frozenset[str] = frozenset(
    get_args(AttentionFallbackPolicy)
)
_FLASH_ATTENTION_PROVIDERS: frozenset[str] = frozenset(get_args(FlashAttentionProvider))
_TRAIN_DTYPES: frozenset[str] = frozenset(get_args(TrainDType))
_COMPILE_MODES: frozenset[str] = frozenset(get_args(CompileMode))
_COMPILE_FALLBACK_POLICIES: frozenset[str] = frozenset(get_args(CompileFallbackPolicy))
_SFT_SOURCE_KINDS: frozenset[str] = frozenset(get_args(SFTSourceKind))
_SFT_DATASET_SPLITS = {
    "gsm8k": frozenset({"train", "test"}),
    "mmlu": frozenset({"auxiliary_train", "test"}),
    "smoltalk": frozenset({"train", "test"}),
}
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_WANDB_ENVIRONMENT_FIELDS = {
    "WANDB_MODE": "mode",
    "WANDB_PROJECT": "project",
    "WANDB_ENTITY": "entity",
    "WANDB_RUN_GROUP": "group",
}


def _error_summary(error: Exception) -> str:
    summary = str(error).splitlines()[0].strip()
    return summary or type(error).__name__


def _require_bool(value: object, path: str) -> None:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")


def _omegaconf_error_path(error: Exception, fallback: str) -> str:
    full_key = getattr(error, "full_key", None)
    return str(full_key) if full_key else fallback


def _fail_from_omegaconf(error: Exception, *, path: str, context: str) -> NoReturn:
    _fail(
        _omegaconf_error_path(error, path),
        f"{context}: {_error_summary(error)}",
    )


def _load_yaml_config(path: Path) -> DictConfig:
    try:
        loaded = OmegaConf.load(path)
    except (OSError, UnicodeError) as error:
        _fail("config", f"could not read {path}: {error}")
    except Exception as error:
        _fail("config", f"invalid YAML in {path}: {_error_summary(error)}")
    if not isinstance(loaded, DictConfig):
        _fail("config", "YAML document root must be a mapping")
    return loaded


def _parse_dotted_override(override: object) -> DictConfig:
    if not isinstance(override, str):
        _fail("override", "must be a PATH=VALUE string")
    raw_path, separator, _ = override.partition("=")
    path = raw_path.strip()
    if not separator:
        _fail(path or "override", "must use PATH=VALUE syntax")

    parts = path.split(".")
    if not path or any(not part.isidentifier() for part in parts):
        _fail(path or "override", "must be a dotted configuration field path")
    try:
        return OmegaConf.from_dotlist([override])
    except Exception as error:
        _fail_from_omegaconf(
            error,
            path=path,
            context="invalid configuration override",
        )


@dataclass
class _SerializableConfig:
    """Shared lossless conversion for dataclass-backed configuration."""

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively copied dictionary of primitive config values."""

        return asdict(self)

    def to_yaml(self) -> str:
        """Return resolved deterministic YAML for the complete config value."""

        return OmegaConf.to_yaml(
            OmegaConf.create(self.to_dict()),
            resolve=True,
        )


@dataclass
class RunConfig(_SerializableConfig):
    """Run identity, reproducibility, device, and output settings."""

    name: str = "smoke"
    seed: int = 1337
    device: str = "cuda"
    output_dir: str = "runs/out"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_non_empty(self.name, "run.name")
        if _RUN_NAME_PATTERN.fullmatch(self.name) is None:
            _fail(
                "run.name",
                "must be a safe portable path component containing only "
                "letters, numbers, dots, hyphens, and underscores",
            )
        _require_int(self.seed, "run.seed")
        _require_non_empty(self.device, "run.device")
        _require_non_empty(self.output_dir, "run.output_dir")


@dataclass
class JsonlTrackingConfig(_SerializableConfig):
    """Always-available local JSONL tracking settings."""

    enabled: bool = True
    path: str = "metrics/metrics.jsonl"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.enabled is not True:
            _fail(
                "tracking.jsonl.enabled",
                "must be true because local JSONL metrics are always enabled",
            )
        _require_non_empty(self.path, "tracking.jsonl.path")


@dataclass
class WandbConfig(_SerializableConfig):
    """Optional Weights & Biases settings."""

    enabled: bool = False
    project: str = "scratch-llm"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: WandbMode = "online"
    dir: str = "runs/wandb"
    log_code: bool = False
    log_model_artifacts: bool = False
    log_dataset_artifacts: bool = False
    log_tokenizer_artifacts: bool = True
    log_prompts: bool = False
    log_responses: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_choice(self.mode, "tracking.wandb.mode", _WANDB_MODES)
        _require_non_empty(self.project, "tracking.wandb.project")
        _require_non_empty(self.dir, "tracking.wandb.dir")
        _require_bool(self.log_prompts, "tracking.wandb.log_prompts")
        _require_bool(self.log_responses, "tracking.wandb.log_responses")
        for index, tag in enumerate(self.tags):
            _require_non_empty(tag, f"tracking.wandb.tags.{index}")


@dataclass
class TrackingConfig(_SerializableConfig):
    """Local and optional remote experiment tracking settings."""

    jsonl: JsonlTrackingConfig = field(default_factory=JsonlTrackingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.jsonl.validate()
        self.wandb.validate()


@dataclass
class DataConfig(_SerializableConfig):
    """Raw and tokenized dataset locations and shard selection."""

    # This profile follows nanochat's ClimbMix-400B pretraining data layout:
    # https://github.com/karpathy/nanochat/blob/master/nanochat/dataset.py
    profile: str = "nanochat_climbmix"
    base_dir: str = "data"
    parquet_dir: str = "data/parquet/base_data_climbmix"
    tokenized_dir: str = "data/tokenized"
    loader_strategy: TokenLoaderStrategy = "packed"
    text_column: str = "text"
    num_tokenizer_train_shards: int = 8
    num_pretrain_train_shards: int = 16
    always_use_final_shard_for_val: bool = True
    max_shard: int = 6542
    doc_cap_chars: int = 10_000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for field_name in (
            "profile",
            "base_dir",
            "parquet_dir",
            "tokenized_dir",
            "text_column",
        ):
            _require_non_empty(getattr(self, field_name), f"data.{field_name}")
        _require_choice(
            self.loader_strategy,
            "data.loader_strategy",
            _TOKEN_LOADER_STRATEGIES,
        )
        _require_positive_int(
            self.num_tokenizer_train_shards, "data.num_tokenizer_train_shards"
        )
        _require_positive_int(
            self.num_pretrain_train_shards, "data.num_pretrain_train_shards"
        )
        _require_non_negative_int(self.max_shard, "data.max_shard")
        _require_positive_int(self.doc_cap_chars, "data.doc_cap_chars")


@dataclass
class TokenizerConfig(_SerializableConfig):
    """Byte or regex byte-BPE tokenizer settings."""

    type: TokenizerType = "regex_byte_bpe"
    vocab_size: int = 32_768
    artifact_dir: str | None = None
    max_chars: int = 2_000_000_000
    doc_cap: int = 10_000
    special_tokens: list[str] = field(
        default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS)
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_choice(self.type, "tokenizer.type", _TOKENIZER_TYPES)
        _require_positive_int(self.vocab_size, "tokenizer.vocab_size")
        if self.artifact_dir is not None:
            _require_non_empty(self.artifact_dir, "tokenizer.artifact_dir")
        _require_positive_int(self.max_chars, "tokenizer.max_chars")
        _require_positive_int(self.doc_cap, "tokenizer.doc_cap")
        if not isinstance(self.special_tokens, list):
            _fail(
                "tokenizer.special_tokens",
                "must be a list containing the nine ordered nanochat special tokens",
            )
        for index, token in enumerate(self.special_tokens):
            _require_non_empty(token, f"tokenizer.special_tokens.{index}")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            _fail("tokenizer.special_tokens", "must not contain duplicates")
        if tuple(self.special_tokens) != NANOCHAT_SPECIAL_TOKENS:
            _fail(
                "tokenizer.special_tokens",
                "must exactly match the nine ordered nanochat special tokens; "
                "<|pad|> is not supported and BOS is the categorical-padding "
                "fallback",
            )
        if self.type == "byte" and self.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE:
            _fail(
                "tokenizer.vocab_size",
                f"must be exactly {BYTE_TOKENIZER_VOCAB_SIZE} for the byte tokenizer",
            )
        if (
            self.type == "regex_byte_bpe"
            and self.vocab_size < BYTE_TOKENIZER_VOCAB_SIZE
        ):
            _fail(
                "tokenizer.vocab_size",
                f"must be at least {BYTE_TOKENIZER_VOCAB_SIZE} "
                "for bytes and special tokens",
            )


@dataclass
class GPTConfig(_SerializableConfig):
    """Dimensions and architecture switches for the decoder-only GPT."""

    profile: str = "simple_gpt"
    vocab_size: int = 32_768
    seq_len: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    norm: NormType = "layernorm"
    activation: ActivationType = "gelu"
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_qk_norm: bool = False
    use_gqa: bool = False
    attention_backend: AttentionBackend = "manual"
    attention_fallback_policy: AttentionFallbackPolicy = "allow"
    flash_attention_provider: FlashAttentionProvider = "auto"
    use_flash_attention: bool = False
    use_kv_cache: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def parameter_compatibility_dict(self) -> dict[str, Any]:
        """Return config fields that must agree when loading model weights.

        Attention execution and inference-cache choices do not alter parameter
        names or shapes, so a checkpoint may safely select different values for
        those fields when constructing its destination model.
        """

        values = asdict(self)
        for field_name in (
            "attention_backend",
            "attention_fallback_policy",
            "flash_attention_provider",
            "use_flash_attention",
            "use_kv_cache",
        ):
            values.pop(field_name)
        return values

    def validate(self) -> None:
        _require_non_empty(self.profile, "model.profile")
        for field_name in (
            "vocab_size",
            "seq_len",
            "n_layer",
            "n_head",
            "n_embd",
            "mlp_ratio",
        ):
            _require_positive_int(getattr(self, field_name), f"model.{field_name}")
        if self.n_embd % self.n_head != 0:
            _fail("model.n_embd", "must be divisible by model.n_head")
        dropout = _require_real(self.dropout, "model.dropout")
        if dropout < 0 or dropout >= 1:
            _fail("model.dropout", "must be in [0, 1)")
        _require_choice(self.norm, "model.norm", _NORM_TYPES)
        _require_choice(self.activation, "model.activation", _ACTIVATION_TYPES)
        _require_choice(
            self.attention_backend,
            "model.attention_backend",
            _ATTENTION_BACKENDS,
        )
        _require_choice(
            self.attention_fallback_policy,
            "model.attention_fallback_policy",
            _ATTENTION_FALLBACK_POLICIES,
        )
        _require_choice(
            self.flash_attention_provider,
            "model.flash_attention_provider",
            _FLASH_ATTENTION_PROVIDERS,
        )
        _require_bool(self.use_flash_attention, "model.use_flash_attention")
        if self.use_flash_attention:
            _fail(
                "model.use_flash_attention",
                "must remain false; select the canonical model.attention_backend "
                "setting instead",
            )
        _require_bool(self.use_kv_cache, "model.use_kv_cache")
        expected_rmsnorm = self.norm == "rmsnorm"
        if self.use_rmsnorm is not expected_rmsnorm:
            _fail(
                "model.use_rmsnorm",
                "must agree with whether model.norm is 'rmsnorm'",
            )


@dataclass
class TrainConfig(_SerializableConfig):
    """Single-device optimization, scheduling, and cadence settings."""

    device_batch_size: int = 4
    total_batch_size_tokens: int = 65_536
    grad_accum_steps: GradAccumSteps = "auto"
    max_steps: int = 20_000
    learning_rate: float = 0.0003
    min_lr: float = 0.000015
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    eval_every: int = 250
    eval_tokens: int = 1_048_576
    sample_every: int = 1_000
    save_every: int = 1_000
    log_every: int = 10
    mfu_peak_flops_per_second: float | None = None
    mfu_peak_flops_basis: str | None = None
    dtype: TrainDType = "float32"
    compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: CompileMode = "default"
    compile_fallback_policy: CompileFallbackPolicy = "eager"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_positive_int(self.device_batch_size, "train.device_batch_size")
        _require_positive_int(
            self.total_batch_size_tokens, "train.total_batch_size_tokens"
        )
        if self.grad_accum_steps != "auto":
            _require_positive_int(self.grad_accum_steps, "train.grad_accum_steps")
        _require_positive_int(self.max_steps, "train.max_steps")
        _require_positive_real(self.learning_rate, "train.learning_rate")
        _require_non_negative_real(self.min_lr, "train.min_lr")
        if self.min_lr > self.learning_rate:
            _fail("train.min_lr", "must not exceed train.learning_rate")
        _require_non_negative_real(self.weight_decay, "train.weight_decay")
        _require_half_open_unit_interval(self.beta1, "train.beta1")
        _require_half_open_unit_interval(self.beta2, "train.beta2")
        _require_positive_real(self.grad_clip, "train.grad_clip")
        _require_non_negative_int(self.warmup_steps, "train.warmup_steps")
        _require_unit_interval(self.warmdown_ratio, "train.warmdown_ratio")
        _require_unit_interval(
            self.final_lr_frac, "train.final_lr_frac", include_zero=False
        )
        warmdown_steps = round(self.warmdown_ratio * self.max_steps)
        warmdown_start_step = self.max_steps - warmdown_steps
        if self.warmup_steps > warmdown_start_step:
            _fail(
                "train.warmup_steps",
                "must not extend past the warmdown start at step "
                f"{warmdown_start_step}",
            )
        for field_name in (
            "eval_every",
            "eval_tokens",
            "sample_every",
            "save_every",
            "log_every",
        ):
            _require_positive_int(getattr(self, field_name), f"train.{field_name}")
        has_peak_flops = self.mfu_peak_flops_per_second is not None
        has_peak_basis = self.mfu_peak_flops_basis is not None
        if has_peak_flops != has_peak_basis:
            _fail(
                "train.mfu_peak_flops",
                "per-second value and basis must either both be set or both be null",
            )
        if has_peak_flops:
            peak_flops = _require_real(
                self.mfu_peak_flops_per_second,
                "train.mfu_peak_flops_per_second",
            )
            if not math.isfinite(peak_flops) or peak_flops <= 0:
                _fail(
                    "train.mfu_peak_flops_per_second",
                    "must be finite and greater than zero",
                )
            _require_non_empty(
                self.mfu_peak_flops_basis,
                "train.mfu_peak_flops_basis",
            )
        _require_choice(self.dtype, "train.dtype", _TRAIN_DTYPES)
        _require_bool(self.compile, "train.compile")
        _require_non_empty(self.compile_backend, "train.compile_backend")
        _require_choice(self.compile_mode, "train.compile_mode", _COMPILE_MODES)
        _require_choice(
            self.compile_fallback_policy,
            "train.compile_fallback_policy",
            _COMPILE_FALLBACK_POLICIES,
        )
        _require_bool(self.compile_fullgraph, "train.compile_fullgraph")
        _require_bool(self.compile_dynamic, "train.compile_dynamic")
        _require_bool(
            self.activation_checkpointing,
            "train.activation_checkpointing",
        )


@dataclass
class SFTSourceConfig(_SerializableConfig):
    """One strict local JSONL or verified Hub-cache conversation source."""

    kind: SFTSourceKind = "jsonl"
    path: str = "data/fixtures/chat/train.jsonl"
    dataset: str | None = None
    split: str | None = None
    repeat_weight: int = 1
    shuffle: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, path: str = "sft.source") -> None:
        _require_choice(self.kind, f"{path}.kind", _SFT_SOURCE_KINDS)
        _require_non_empty(self.path, f"{path}.path")
        _require_positive_int(self.repeat_weight, f"{path}.repeat_weight")
        if not isinstance(self.shuffle, bool):
            _fail(f"{path}.shuffle", "must be a boolean")
        if self.kind == "jsonl":
            if Path(self.path).suffix != ".jsonl":
                _fail(f"{path}.path", "must end in .jsonl for a JSONL source")
            if self.dataset is not None:
                _fail(f"{path}.dataset", "must be null for a JSONL source")
            if self.split is not None:
                _fail(f"{path}.split", "must be null for a JSONL source")
            return

        if self.dataset is None:
            _fail(f"{path}.dataset", "is required for a Hub cache source")
        if self.dataset not in _SFT_DATASET_SPLITS:
            _fail(
                f"{path}.dataset",
                "must be one of gsm8k, mmlu, or smoltalk",
            )
        if self.split is None:
            _fail(f"{path}.split", "is required for a Hub cache source")
        if self.split not in _SFT_DATASET_SPLITS[self.dataset]:
            supported = ", ".join(sorted(_SFT_DATASET_SPLITS[self.dataset]))
            _fail(
                f"{path}.split",
                f"must be one of {supported} for {self.dataset}",
            )


def _default_sft_train_sources() -> list[SFTSourceConfig]:
    return [
        SFTSourceConfig(
            kind="jsonl",
            path="data/fixtures/chat/train.jsonl",
            repeat_weight=1,
            shuffle=True,
        )
    ]


def _default_sft_validation_sources() -> list[SFTSourceConfig]:
    return [
        SFTSourceConfig(
            kind="jsonl",
            path="data/fixtures/chat/validation.jsonl",
            repeat_weight=1,
            shuffle=False,
        )
    ]


@dataclass
class SFTConfig(_SerializableConfig):
    """SFT-only data, optimization, validation, and checkpoint settings."""

    base_checkpoint: str | None = None
    train_sources: list[SFTSourceConfig] = field(
        default_factory=_default_sft_train_sources
    )
    validation_sources: list[SFTSourceConfig] = field(
        default_factory=_default_sft_validation_sources
    )
    packing_buffer_size: int = 100
    shuffle_buffer_size: int = 1024
    row_batch_size: int = 1024
    device_batch_size: int = 2
    total_batch_size_tokens: int = 32_768
    grad_accum_steps: GradAccumSteps = "auto"
    max_steps: int = 5_000
    learning_rate: float = 0.00002
    min_lr: float = 0.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 0
    warmdown_ratio: float = 0.0
    final_lr_frac: float = 0.05
    eval_every: int = 250
    eval_batches: int = 8
    save_every: int = 1_000
    log_every: int = 10
    mfu_peak_flops_per_second: float | None = None
    mfu_peak_flops_basis: str | None = None
    dtype: TrainDType = "float32"
    compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: CompileMode = "default"
    compile_fallback_policy: CompileFallbackPolicy = "eager"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.base_checkpoint is not None:
            _require_non_empty(self.base_checkpoint, "sft.base_checkpoint")
            if Path(self.base_checkpoint).suffix != ".pt":
                _fail("sft.base_checkpoint", "must end in .pt")
        for field_name in (
            "train_sources",
            "validation_sources",
        ):
            sources = getattr(self, field_name)
            if not isinstance(sources, list) or not sources:
                _fail(f"sft.{field_name}", "must be a non-empty list")
            identities: set[tuple[object, ...]] = set()
            for index, source in enumerate(sources):
                if not isinstance(source, SFTSourceConfig):
                    _fail(
                        f"sft.{field_name}.{index}",
                        "must be an SFTSourceConfig",
                    )
                source.validate(f"sft.{field_name}.{index}")
                identity = (
                    source.kind,
                    source.path,
                    source.dataset,
                    source.split,
                )
                if identity in identities:
                    _fail(
                        f"sft.{field_name}.{index}",
                        "duplicates an earlier logical source",
                    )
                identities.add(identity)
                if field_name == "validation_sources":
                    if source.repeat_weight != 1:
                        _fail(
                            f"sft.{field_name}.{index}.repeat_weight",
                            "must be 1 for finite validation",
                        )
                    if source.shuffle:
                        _fail(
                            f"sft.{field_name}.{index}.shuffle",
                            "must be false for deterministic validation",
                        )

        for field_name in (
            "packing_buffer_size",
            "shuffle_buffer_size",
            "row_batch_size",
            "device_batch_size",
            "total_batch_size_tokens",
            "max_steps",
            "eval_every",
            "eval_batches",
            "save_every",
            "log_every",
        ):
            _require_positive_int(getattr(self, field_name), f"sft.{field_name}")
        if self.grad_accum_steps != "auto":
            _require_positive_int(self.grad_accum_steps, "sft.grad_accum_steps")
        _require_positive_real(self.learning_rate, "sft.learning_rate")
        _require_non_negative_real(self.min_lr, "sft.min_lr")
        if self.min_lr > self.learning_rate:
            _fail("sft.min_lr", "must not exceed sft.learning_rate")
        _require_non_negative_real(self.weight_decay, "sft.weight_decay")
        _require_half_open_unit_interval(self.beta1, "sft.beta1")
        _require_half_open_unit_interval(self.beta2, "sft.beta2")
        _require_positive_real(self.grad_clip, "sft.grad_clip")
        _require_non_negative_int(self.warmup_steps, "sft.warmup_steps")
        _require_unit_interval(self.warmdown_ratio, "sft.warmdown_ratio")
        _require_unit_interval(
            self.final_lr_frac,
            "sft.final_lr_frac",
            include_zero=False,
        )
        warmdown_steps = round(self.warmdown_ratio * self.max_steps)
        warmdown_start_step = self.max_steps - warmdown_steps
        if self.warmup_steps > warmdown_start_step:
            _fail(
                "sft.warmup_steps",
                "must not extend past the warmdown start at step "
                f"{warmdown_start_step}",
            )
        if self.eval_every > self.max_steps:
            _fail("sft.eval_every", "must not exceed sft.max_steps")
        if self.save_every > self.max_steps:
            _fail("sft.save_every", "must not exceed sft.max_steps")
        has_peak_flops = self.mfu_peak_flops_per_second is not None
        has_peak_basis = self.mfu_peak_flops_basis is not None
        if has_peak_flops != has_peak_basis:
            _fail(
                "sft.mfu_peak_flops",
                "per-second value and basis must either both be set or both be null",
            )
        if has_peak_flops:
            peak_flops = _require_real(
                self.mfu_peak_flops_per_second,
                "sft.mfu_peak_flops_per_second",
            )
            if not math.isfinite(peak_flops) or peak_flops <= 0:
                _fail(
                    "sft.mfu_peak_flops_per_second",
                    "must be finite and greater than zero",
                )
            _require_non_empty(
                self.mfu_peak_flops_basis,
                "sft.mfu_peak_flops_basis",
            )
        _require_choice(self.dtype, "sft.dtype", _TRAIN_DTYPES)
        _require_bool(self.compile, "sft.compile")
        _require_non_empty(self.compile_backend, "sft.compile_backend")
        _require_choice(self.compile_mode, "sft.compile_mode", _COMPILE_MODES)
        _require_choice(
            self.compile_fallback_policy,
            "sft.compile_fallback_policy",
            _COMPILE_FALLBACK_POLICIES,
        )
        _require_bool(self.compile_fullgraph, "sft.compile_fullgraph")
        _require_bool(self.compile_dynamic, "sft.compile_dynamic")
        _require_bool(
            self.activation_checkpointing,
            "sft.activation_checkpointing",
        )

    def to_train_config(self, seq_len: int) -> TrainConfig:
        """Return the shared optimizer/scheduler view for this SFT contract."""

        self.validate()
        _require_positive_int(seq_len, "model.seq_len")
        tokens_per_microbatch = self.device_batch_size * seq_len
        if self.total_batch_size_tokens % tokens_per_microbatch != 0:
            _fail(
                "sft.total_batch_size_tokens",
                "must be divisible by sft.device_batch_size * model.seq_len",
            )
        derived_grad_accum_steps = self.total_batch_size_tokens // tokens_per_microbatch
        if (
            self.grad_accum_steps != "auto"
            and self.grad_accum_steps != derived_grad_accum_steps
        ):
            _fail(
                "sft.grad_accum_steps",
                "must match total_batch_size_tokens / "
                "(device_batch_size * model.seq_len)",
            )
        return TrainConfig(
            device_batch_size=self.device_batch_size,
            total_batch_size_tokens=self.total_batch_size_tokens,
            grad_accum_steps=(
                derived_grad_accum_steps
                if self.grad_accum_steps == "auto"
                else self.grad_accum_steps
            ),
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            min_lr=self.min_lr,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            grad_clip=self.grad_clip,
            warmup_steps=self.warmup_steps,
            warmdown_ratio=self.warmdown_ratio,
            final_lr_frac=self.final_lr_frac,
            eval_every=self.eval_every,
            eval_tokens=(self.eval_batches * self.device_batch_size * seq_len),
            sample_every=self.max_steps,
            save_every=self.save_every,
            log_every=self.log_every,
            mfu_peak_flops_per_second=self.mfu_peak_flops_per_second,
            mfu_peak_flops_basis=self.mfu_peak_flops_basis,
            dtype=self.dtype,
            compile=self.compile,
            compile_backend=self.compile_backend,
            compile_mode=self.compile_mode,
            compile_fallback_policy=self.compile_fallback_policy,
            compile_fullgraph=self.compile_fullgraph,
            compile_dynamic=self.compile_dynamic,
            activation_checkpointing=self.activation_checkpointing,
        )


@dataclass
class GenerationConfig(_SerializableConfig):
    """Autoregressive sampling settings."""

    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = None
    max_new_tokens: int = 256
    seed: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_non_negative_real(self.temperature, "generation.temperature")
        if self.top_k is not None:
            _require_positive_int(self.top_k, "generation.top_k")
        if self.top_p is not None:
            _require_unit_interval(self.top_p, "generation.top_p", include_zero=False)
        _require_positive_int(self.max_new_tokens, "generation.max_new_tokens")
        if self.seed is not None:
            _require_int(self.seed, "generation.seed")


_GENERATION_OVERRIDE_FIELDS = frozenset(
    {"max_new_tokens", "seed", "temperature", "top_k"}
)


def apply_generation_overrides(
    defaults: GenerationConfig,
    overrides: Mapping[str, object],
) -> GenerationConfig:
    """Return validated generation defaults with explicit non-None overrides."""

    if not isinstance(defaults, GenerationConfig):
        raise TypeError(
            f"defaults must be a GenerationConfig, got {type(defaults).__name__}"
        )
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")
    unexpected = set(overrides) - _GENERATION_OVERRIDE_FIELDS
    if unexpected:
        raise ValueError(f"unsupported generation overrides: {sorted(unexpected)}")
    values = defaults.to_dict()
    for field_name, value in overrides.items():
        if value is not None:
            values[field_name] = value
    return GenerationConfig(**values)


@dataclass
class WebConfig(_SerializableConfig):
    """Local web testing harness settings."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    checkpoint_dir: str = "runs/out"
    allow_remote_bind: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_non_empty(self.host, "web.host")
        _require_positive_int(self.port, "web.port")
        if self.port > 65_535:
            _fail("web.port", "must be at most 65535")
        _require_non_empty(self.checkpoint_dir, "web.checkpoint_dir")
        if (
            self.enabled
            and not self.allow_remote_bind
            and self.host not in _LOOPBACK_HOSTS
        ):
            _fail(
                "web.host",
                "must be a loopback host unless web.allow_remote_bind is true",
            )


@dataclass
class ProjectConfig(_SerializableConfig):
    """Complete resolved configuration for the end-to-end project."""

    run: RunConfig = field(default_factory=RunConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: GPTConfig = field(default_factory=GPTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate fields plus invariants that span configuration sections."""

        self.run.validate()
        self.tracking.validate()
        self.data.validate()
        self.tokenizer.validate()
        self.model.validate()
        self.train.validate()
        self.sft.validate()
        self.generation.validate()
        self.web.validate()

        if self.model.vocab_size != self.tokenizer.vocab_size:
            _fail(
                "model.vocab_size",
                "must equal tokenizer.vocab_size",
            )

        tokens_per_microbatch = self.train.device_batch_size * self.model.seq_len
        if self.train.total_batch_size_tokens % tokens_per_microbatch != 0:
            _fail(
                "train.total_batch_size_tokens",
                "must be divisible by train.device_batch_size * model.seq_len",
            )

        derived_grad_accum_steps = (
            self.train.total_batch_size_tokens // tokens_per_microbatch
        )
        if (
            self.train.grad_accum_steps != "auto"
            and self.train.grad_accum_steps != derived_grad_accum_steps
        ):
            _fail(
                "train.grad_accum_steps",
                "must match total_batch_size_tokens / "
                "(device_batch_size * model.seq_len)",
            )

        self.sft.to_train_config(self.model.seq_len)


# A concise alias for callers that prefer ``Config`` at integration boundaries.
Config = ProjectConfig


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | str = (),
    *,
    environment: Mapping[str, str] | None = None,
    wandb_enabled: bool | None = None,
    wandb_mode: WandbMode | None = None,
) -> ProjectConfig:
    """Resolve defaults, YAML, W&B environment, and ordered CLI overrides.

    Source precedence is defaults, YAML, supported ``WANDB_*`` variables,
    dotted CLI overrides, then dedicated W&B CLI options. The supplied
    environment mapping is read without being mutated.
    """

    defaults = OmegaConf.structured(ProjectConfig)
    OmegaConf.set_struct(defaults, True)
    sources = [defaults]
    if path is not None:
        sources.append(_load_yaml_config(Path(path)))

    if environment is not None:
        environment_values: dict[str, str | None] = {}
        for variable, field_name in _WANDB_ENVIRONMENT_FIELDS.items():
            if variable not in environment:
                continue
            value: str | None = environment[variable]
            if field_name in {"entity", "group"} and value == "":
                value = None
            environment_values[field_name] = value
        if environment_values:
            sources.append(
                OmegaConf.create(
                    {"tracking": {"wandb": environment_values}},
                )
            )

    if isinstance(overrides, str):
        overrides = (overrides,)
    sources.extend(_parse_dotted_override(override) for override in overrides)

    dedicated_wandb_values: dict[str, object] = {}
    if wandb_enabled is not None:
        dedicated_wandb_values["enabled"] = wandb_enabled
    if wandb_mode is not None:
        dedicated_wandb_values["mode"] = wandb_mode
    if dedicated_wandb_values:
        sources.append(
            OmegaConf.create(
                {"tracking": {"wandb": dedicated_wandb_values}},
            )
        )

    try:
        resolved = OmegaConf.merge(*sources)
    except Exception as error:
        _fail_from_omegaconf(
            error,
            path="config",
            context="could not merge configuration sources",
        )
    if not isinstance(resolved, DictConfig):
        _fail("config", "resolved configuration must be a mapping")

    try:
        config = OmegaConf.to_object(resolved)
    except ConfigValidationError:
        raise
    except Exception as error:
        _fail_from_omegaconf(
            error,
            path="config",
            context="could not construct configuration",
        )
    if not isinstance(config, ProjectConfig):
        _fail("config", "resolved configuration must be a ProjectConfig")
    return config


def dump_config(config: ProjectConfig, path: str | Path) -> Path:
    """Write a complete, validated configuration through OmegaConf."""

    config.validate()
    destination = Path(path)
    try:
        atomic_write(destination, config.to_yaml())
    except OSError as error:
        _fail("config", f"could not write {destination}: {error}")
    return destination


def save_config(config: ProjectConfig, path: str | Path) -> Path:
    """Alias for :func:`dump_config` at file-oriented call sites."""

    return dump_config(config, path)


__all__ = [
    "ActivationType",
    "Config",
    "ConfigValidationError",
    "DEFAULT_SPECIAL_TOKENS",
    "DataConfig",
    "GPTConfig",
    "GenerationConfig",
    "GradAccumSteps",
    "JsonlTrackingConfig",
    "NormType",
    "ProjectConfig",
    "RunConfig",
    "SFTConfig",
    "SFTSourceConfig",
    "SFTSourceKind",
    "TokenLoaderStrategy",
    "TokenizerConfig",
    "TokenizerType",
    "TrackingConfig",
    "TrainConfig",
    "TrainDType",
    "WandbConfig",
    "WandbMode",
    "WebConfig",
    "dump_config",
    "load_config",
    "save_config",
]
