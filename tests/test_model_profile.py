"""Nanochat-depth geometry resolution and bounded profile evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scratch_llm.config import (
    ConfigValidationError,
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
    dump_config,
    load_config,
)
from scratch_llm.diagnostics.resource_estimation import estimate_training_resources
from scratch_llm.identity import project_config_identity
from scratch_llm.model import GPT
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.training.checkpoint import load_model_checkpoint, save_checkpoint
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scripts.pretrain import main as pretrain_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("depth", "aspect_ratio", "head_dim", "expected_width", "expected_heads"),
    [
        (1, 63, 64, 64, 1),
        (2, 63, 64, 128, 2),
        (3, 63, 64, 192, 3),
        (4, 64, 128, 256, 2),
    ],
)
def test_nanochat_depth_profile_resolves_exact_geometry(
    depth: int,
    aspect_ratio: int,
    head_dim: int,
    expected_width: int,
    expected_heads: int,
) -> None:
    config = GPTConfig(
        profile="nanochat_depth",
        depth=depth,
        aspect_ratio=aspect_ratio,
        head_dim=head_dim,
    )

    assert config.n_layer == depth
    assert config.n_embd == expected_width
    assert config.n_head == expected_heads
    assert config.seq_len == 512


def test_config_loader_resolves_profile_before_cross_section_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        """
model:
  profile: nanochat_depth
  depth: 4
  aspect_ratio: 64
  head_dim: 128
  seq_len: 256
train:
  device_batch_size: 1
  total_batch_size_tokens: 256
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert (config.model.n_layer, config.model.n_embd, config.model.n_head) == (
        4,
        256,
        2,
    )
    assert config.model.to_dict() == {
        **GPTConfig().to_dict(),
        "profile": "nanochat_depth",
        "depth": 4,
        "aspect_ratio": 64,
        "head_dim": 128,
        "seq_len": 256,
        "n_layer": 4,
        "n_embd": 256,
        "n_head": 2,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_layer", 6), ("n_embd", 384), ("n_head", 6)],
)
def test_explicit_dimensions_cannot_contradict_depth_profile(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    path = tmp_path / "contradiction.yaml"
    path.write_text(
        f"""
model:
  profile: nanochat_depth
  depth: 4
  aspect_ratio: 64
  head_dim: 128
  {field}: {value}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigValidationError,
        match=rf"^model\.{field}:.*derived nanochat_depth value",
    ):
        load_config(path)


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"profile": "unknown"}, "model.profile"),
        ({"profile": "nanochat_depth"}, "model.depth"),
        (
            {
                "profile": "nanochat_depth",
                "depth": 0,
                "aspect_ratio": 64,
                "head_dim": 128,
            },
            "model.depth",
        ),
        (
            {
                "profile": "nanochat_depth",
                "depth": 2**62,
                "aspect_ratio": 4,
                "head_dim": 128,
            },
            "model.aspect_ratio",
        ),
        ({"depth": 4}, "model.depth"),
    ],
)
def test_profile_validation_rejects_invalid_and_overflowing_inputs(
    kwargs: dict[str, object],
    path: str,
) -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        GPTConfig(**kwargs)  # type: ignore[arg-type]

    assert exc_info.value.path == path


def _project_config(model: GPTConfig, *, output_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(name="depth-profile", device="cpu", output_dir=str(output_dir)),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=model,
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=model.seq_len,
            grad_accum_steps=1,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
    )


def test_resolved_profile_feeds_model_resources_and_identity(tmp_path: Path) -> None:
    profile_model = GPTConfig(
        profile="nanochat_depth",
        depth=2,
        aspect_ratio=4,
        head_dim=4,
        vocab_size=VOCAB_SIZE,
        seq_len=8,
        mlp_ratio=2,
    )
    explicit_model = GPTConfig(
        vocab_size=VOCAB_SIZE,
        seq_len=8,
        n_layer=2,
        n_embd=8,
        n_head=2,
        mlp_ratio=2,
    )
    profiled = _project_config(profile_model, output_dir=tmp_path / "profiled")
    explicit = _project_config(explicit_model, output_dir=tmp_path / "explicit")

    profiled_model = GPT(profiled.model)
    explicit_gpt = GPT(explicit.model)
    assert [tuple(value.shape) for value in profiled_model.state_dict().values()] == [
        tuple(value.shape) for value in explicit_gpt.state_dict().values()
    ]
    profiled_estimate = estimate_training_resources(profiled)
    explicit_estimate = estimate_training_resources(explicit)
    assert profiled_estimate.model.unique_parameters == (
        explicit_estimate.model.unique_parameters
    )
    geometry = profiled_estimate.to_dict()["model"]["geometry"]
    assert geometry == {
        "profile": "nanochat_depth",
        "requested": {"aspect_ratio": 4, "depth": 2, "head_dim": 4},
        "resolved": {
            "n_embd": 8,
            "n_head": 2,
            "n_layer": 2,
            "seq_len": 8,
        },
    }
    assert project_config_identity(profiled) != project_config_identity(explicit)
    assert profile_model.norm == "layernorm"
    assert profile_model.activation == "gelu"
    assert not any(
        (
            profile_model.use_rope,
            profile_model.use_rmsnorm,
            profile_model.use_qk_norm,
            profile_model.use_gqa,
            profile_model.use_flash_attention,
            profile_model.use_kv_cache,
        )
    )


def test_profile_round_trips_config_checkpoint_and_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_config = GPTConfig(
        profile="nanochat_depth",
        depth=2,
        aspect_ratio=4,
        head_dim=4,
        vocab_size=VOCAB_SIZE,
        seq_len=8,
        mlp_ratio=2,
    )
    config = _project_config(model_config, output_dir=tmp_path / "runs")
    config_path = dump_config(config, tmp_path / "profile.yaml")

    reloaded = load_config(config_path)
    assert reloaded == config
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    checkpoint_path = save_checkpoint(
        tmp_path / "profile.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=ByteTokenizer(),
    )
    loaded = load_model_checkpoint(checkpoint_path, device="cpu")
    assert loaded.config.model == config.model
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    model.eval()
    loaded.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(token_ids), loaded.model(token_ids))

    dry_run_config = copy.deepcopy(config)
    dry_run_config.run.name = "depth-profile-dry-run"
    dry_run_path = dump_config(dry_run_config, tmp_path / "dry-run.yaml")
    assert pretrain_main(["--config", str(dry_run_path), "--dry-run"]) == 0
    stdout = capsys.readouterr().out
    assert "profile: nanochat_depth" in stdout
    assert "depth: 2" in stdout
    resource_line = next(
        line.removeprefix("Resource estimate JSON: ")
        for line in stdout.splitlines()
        if line.startswith("Resource estimate JSON: ")
    )
    payload = json.loads(resource_line)
    assert payload["model"]["geometry"]["resolved"]["n_embd"] == 8


def test_documented_depth_matrix_matches_deterministic_resource_estimates() -> None:
    expected_rows = [
        (2, 128, 1, 4_653_696, "1,144.533 MiB"),
        (4, 256, 2, 11_667_712, "1,397.059 MiB"),
        (6, 384, 3, 23_401_344, "1,817.600 MiB"),
    ]
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "### Nanochat depth profile" in readme
    assert "n_embd = ceil_to_multiple(depth * aspect_ratio, head_dim)" in readme
    assert "does not claim compute-optimality" in readme
    for depth, width, heads, parameters, memory in expected_rows:
        config = load_config(
            overrides=[
                "model.profile=nanochat_depth",
                f"model.depth={depth}",
                "model.aspect_ratio=64",
                "model.head_dim=128",
            ]
        )
        estimate = estimate_training_resources(config)
        assert (config.model.n_embd, config.model.n_head) == (width, heads)
        assert estimate.model.unique_parameters == parameters
        assert f"| {depth} | {width} | {heads} | {parameters:,} | {memory} |" in readme
