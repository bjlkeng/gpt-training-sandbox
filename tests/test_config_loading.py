"""Tests for loading and resolving project configuration files."""

from __future__ import annotations

from pathlib import Path

import pytest

from scratch_llm.config import (
    ConfigValidationError,
    ProjectConfig,
    dump_config,
    load_config,
)
from scratch_llm.tokenizer import VOCAB_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_merges_partial_yaml_onto_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "partial.yaml"
    config_path.write_text(
        """
run:
  seed: 7
model:
  dropout: 0.25
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.run.seed == 7
    assert config.model.dropout == 0.25
    assert config.run.name == ProjectConfig().run.name
    assert config.model.n_layer == ProjectConfig().model.n_layer


def test_dotted_overrides_win_in_order_and_parse_scalars_and_lists(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
run:
  seed: 7
tracking:
  wandb:
    tags: [from-yaml]
model:
  n_layer: 4
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(
        config_path,
        overrides=[
            "run.seed=11",
            "model.n_layer=3",
            "model.n_layer=2",
            "tracking.wandb.tags=[smoke, cli]",
            "generation.top_k=null",
            "train.compile=true",
        ],
    )

    assert config.run.seed == 11
    assert config.model.n_layer == 2
    assert config.tracking.wandb.tags == ["smoke", "cli"]
    assert config.generation.top_k is None
    assert config.train.compile is True


def test_load_config_accepts_one_override_without_a_wrapper_list() -> None:
    config = load_config(overrides="run.device=cpu")

    assert config.run.device == "cpu"


def test_load_config_accepts_omegaconf_scalar_coercion(tmp_path: Path) -> None:
    config_path = tmp_path / "coercible.yaml"
    config_path.write_text(
        """
run:
  seed: "7"
model:
  dropout: "0.25"
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.run.seed == 7
    assert config.model.dropout == 0.25


@pytest.mark.parametrize(
    ("yaml_text", "path"),
    [
        ("model:\n  layers: 2\n", "model.layers"),
        ("model:\n  n_layer: two\n", "model.n_layer"),
        ("tracking:\n  wandb:\n    tags: invalid\n", "tracking.wandb.tags"),
        ("model:\n  n_embd: 127\n", "model.n_embd"),
        ("- not\n- a\n- mapping\n", "config"),
    ],
)
def test_yaml_errors_report_the_actionable_field_path(
    tmp_path: Path, yaml_text: str, path: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ConfigValidationError) as exc_info:
        load_config(config_path)

    assert exc_info.value.path == path


@pytest.mark.parametrize(
    ("override", "path"),
    [
        ("model..n_layer=2", "model..n_layer"),
        ("model.n_layer", "model.n_layer"),
        ("model.layers=2", "model.layers"),
        ("model.n_layer.extra=2", "model.n_layer"),
        ("model.n_layer=two", "model.n_layer"),
    ],
)
def test_override_errors_report_the_actionable_field_path(
    override: str, path: str
) -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        load_config(overrides=[override])

    assert exc_info.value.path == path


def test_yaml_loader_rejects_unsafe_python_tags(tmp_path: Path) -> None:
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(
        "run: !!python/object/apply:builtins.str [unsafe]\n", encoding="utf-8"
    )

    with pytest.raises(ConfigValidationError, match="^config: invalid YAML"):
        load_config(config_path)


def test_resolved_config_dumps_and_reloads_identically(tmp_path: Path) -> None:
    config = load_config(
        overrides=[
            "run.device=cpu",
            "tracking.wandb.tags=[resolved, smoke]",
            "generation.top_p=0.9",
        ]
    )
    resolved_path = tmp_path / "run" / "resolved.yaml"

    dump_config(config, resolved_path)

    assert resolved_path.is_file()
    assert load_config(resolved_path) == config


def test_smoke_config_is_a_cpu_safe_tiny_byte_gpt() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "smoke.yaml")

    assert config.run.device == "cpu"
    assert config.tracking.wandb.enabled is False
    assert config.tracking.wandb.mode == "disabled"
    assert config.tokenizer.type == "byte"
    assert config.tokenizer.vocab_size == VOCAB_SIZE
    assert config.model.vocab_size == VOCAB_SIZE
    assert config.model.seq_len == 128
    assert (config.model.n_layer, config.model.n_head, config.model.n_embd) == (
        2,
        2,
        128,
    )
    assert config.model.mlp_ratio == 4
    assert config.model.tie_weights is True
    assert config.model.use_flash_attention is False
    assert config.train.device_batch_size == 2
    assert config.train.max_steps == 100
    assert config.train.dtype == "float32"
    assert config.train.compile is False
    assert config.web.enabled is False
