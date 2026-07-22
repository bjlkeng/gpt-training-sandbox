"""Typed, serializable configuration for the training pipeline.

The defaults in this module mirror the example configuration in the project
roadmap.  Validation deliberately happens both at construction time and when
``validate`` is called so that loaders can re-check a configuration after
applying overrides.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, NoReturn, get_args

from omegaconf import DictConfig, OmegaConf

from scratch_llm.tokenizer import SPECIAL_TOKENS


WandbMode = Literal["online", "offline", "disabled"]
TokenizerType = Literal["byte", "regex_byte_bpe"]
NormType = Literal["layernorm", "rmsnorm"]
ActivationType = Literal["gelu", "relu_squared"]
TrainDType = Literal["float32", "float16", "bfloat16"]
# OmegaConf does not yet support combining ``Literal`` with another type in a
# union. Runtime validation below still restricts string values to ``"auto"``.
GradAccumSteps = int | str

DEFAULT_SPECIAL_TOKENS = SPECIAL_TOKENS

_WANDB_MODES: frozenset[str] = frozenset(get_args(WandbMode))
_TOKENIZER_TYPES: frozenset[str] = frozenset(get_args(TokenizerType))
_NORM_TYPES: frozenset[str] = frozenset(get_args(NormType))
_ACTIVATION_TYPES: frozenset[str] = frozenset(get_args(ActivationType))
_TRAIN_DTYPES: frozenset[str] = frozenset(get_args(TrainDType))
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class ConfigValidationError(ValueError):
    """A configuration error tied to a dotted field path."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _fail(path: str, message: str) -> NoReturn:
    raise ConfigValidationError(path, message)


def _require_non_empty(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")


def _require_int(value: object, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "must be an integer")


def _require_positive_int(value: object, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(path, "must be a positive integer")


def _require_non_negative_int(value: object, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(path, "must be a non-negative integer")


def _require_real(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, "must be a number")
    return float(value)


def _require_positive_real(value: object, path: str) -> None:
    if _require_real(value, path) <= 0:
        _fail(path, "must be greater than zero")


def _require_non_negative_real(value: object, path: str) -> None:
    if _require_real(value, path) < 0:
        _fail(path, "must be non-negative")


def _require_unit_interval(
    value: object, path: str, *, include_zero: bool = True
) -> None:
    numeric = _require_real(value, path)
    lower_bound_satisfied = numeric >= 0 if include_zero else numeric > 0
    if not lower_bound_satisfied or numeric > 1:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        _fail(path, f"must be in {interval}")


def _require_choice(value: object, path: str, choices: frozenset[str]) -> None:
    if value not in choices:
        options = ", ".join(sorted(choices))
        _fail(path, f"must be one of: {options}")


def _error_summary(error: Exception) -> str:
    summary = str(error).splitlines()[0].strip()
    return summary or type(error).__name__


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
        if self.enabled and self.mode == "disabled":
            _fail(
                "tracking.wandb.mode",
                "cannot be 'disabled' when tracking.wandb.enabled is true",
            )
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
        _require_positive_int(self.max_chars, "tokenizer.max_chars")
        _require_positive_int(self.doc_cap, "tokenizer.doc_cap")
        for index, token in enumerate(self.special_tokens):
            _require_non_empty(token, f"tokenizer.special_tokens.{index}")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            _fail("tokenizer.special_tokens", "must not contain duplicates")
        minimum_vocab_size = 256 + len(self.special_tokens)
        if self.vocab_size < minimum_vocab_size:
            _fail(
                "tokenizer.vocab_size",
                f"must be at least {minimum_vocab_size} for bytes and special tokens",
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
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    norm: NormType = "layernorm"
    activation: ActivationType = "gelu"
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_qk_norm: bool = False
    use_gqa: bool = False
    use_flash_attention: bool = False
    use_kv_cache: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_non_empty(self.profile, "model.profile")
        for field_name in ("vocab_size", "seq_len", "n_layer", "n_head", "n_embd"):
            _require_positive_int(getattr(self, field_name), f"model.{field_name}")
        if self.n_embd % self.n_head != 0:
            _fail("model.n_embd", "must be divisible by model.n_head")
        dropout = _require_real(self.dropout, "model.dropout")
        if dropout < 0 or dropout >= 1:
            _fail("model.dropout", "must be in [0, 1)")
        _require_choice(self.norm, "model.norm", _NORM_TYPES)
        _require_choice(self.activation, "model.activation", _ACTIVATION_TYPES)
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
    dtype: TrainDType = "float32"
    compile: bool = False
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
        _require_unit_interval(self.beta1, "train.beta1")
        _require_unit_interval(self.beta2, "train.beta2")
        _require_positive_real(self.grad_clip, "train.grad_clip")
        _require_non_negative_int(self.warmup_steps, "train.warmup_steps")
        _require_unit_interval(self.warmdown_ratio, "train.warmdown_ratio")
        _require_unit_interval(
            self.final_lr_frac, "train.final_lr_frac", include_zero=False
        )
        for field_name in (
            "eval_every",
            "eval_tokens",
            "sample_every",
            "save_every",
            "log_every",
        ):
            _require_positive_int(getattr(self, field_name), f"train.{field_name}")
        _require_choice(self.dtype, "train.dtype", _TRAIN_DTYPES)


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


# A concise alias for callers that prefer ``Config`` at integration boundaries.
Config = ProjectConfig


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] | str = (),
) -> ProjectConfig:
    """Resolve defaults, partial YAML, and ordered OmegaConf overrides."""

    defaults = OmegaConf.structured(ProjectConfig)
    OmegaConf.set_struct(defaults, True)
    sources = [defaults]
    if path is not None:
        sources.append(_load_yaml_config(Path(path)))
    if isinstance(overrides, str):
        overrides = (overrides,)
    sources.extend(_parse_dotted_override(override) for override in overrides)

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
        destination.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            config=OmegaConf.create(config.to_dict()),
            f=destination,
            resolve=True,
        )
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
