"""Bounded configuration and construction checks for base-model presets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scratch_llm.config import ProjectConfig, load_config
from scratch_llm.model import GPT
from scratch_llm.training.loop import derive_grad_accum_steps
from scratch_llm.utils import count_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
TOKENIZER_ARTIFACTS = "runs/tokenizer-32k/artifacts/tokenizer"
TOKENIZED_DATA = "data/tokenized"

PRESETS = {
    "base_smoke.yaml": {
        "run_name": "base-smoke",
        "seq_len": 128,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 128,
        "device_batch_size": 4,
        "total_batch_size_tokens": 8_192,
        "grad_accum_steps": 16,
        "max_steps": 500,
        "parameter_count": 4_604_544,
    },
    "tiny_20m_3090.yaml": {
        "run_name": "tiny-20m-3090",
        "seq_len": 512,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "device_batch_size": 4,
        "total_batch_size_tokens": 65_536,
        "grad_accum_steps": 32,
        "max_steps": 20_000,
        "parameter_count": 23_401_344,
    },
    "small_45m_3090.yaml": {
        "run_name": "small-45m-3090",
        "seq_len": 1_024,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "device_batch_size": 1,
        "total_batch_size_tokens": 65_536,
        "grad_accum_steps": 64,
        "max_steps": 50_000,
        "parameter_count": 42_476_032,
    },
}


@pytest.mark.parametrize(("filename", "expected"), PRESETS.items())
def test_base_preset_loads_with_production_artifacts_and_exact_token_budget(
    filename: str,
    expected: dict[str, int | str],
) -> None:
    config = load_config(CONFIG_DIR / filename)

    assert isinstance(config, ProjectConfig)
    assert config.run.name == expected["run_name"]
    assert config.run.device == "cuda"
    assert config.data.profile == "nanochat_climbmix"
    assert config.data.loader_strategy == "packed"
    assert config.data.tokenized_dir == TOKENIZED_DATA
    assert config.tokenizer.type == "regex_byte_bpe"
    assert config.tokenizer.vocab_size == 32_768
    assert config.tokenizer.artifact_dir == TOKENIZER_ARTIFACTS
    assert config.model.vocab_size == 32_768
    assert config.model.tie_weights is True
    assert config.model.seq_len == expected["seq_len"]
    assert config.model.n_layer == expected["n_layer"]
    assert config.model.n_head == expected["n_head"]
    assert config.model.n_embd == expected["n_embd"]
    assert config.train.device_batch_size == expected["device_batch_size"]
    assert config.train.total_batch_size_tokens == expected["total_batch_size_tokens"]
    assert config.train.grad_accum_steps == "auto"
    assert config.train.max_steps == expected["max_steps"]
    assert config.train.dtype == "float32"
    assert config.train.compile is False
    assert config.train.activation_checkpointing is False
    assert config.train.mfu_peak_flops_per_second == 35.58e12
    assert (
        config.train.mfu_peak_flops_basis
        == "NVIDIA GeForce RTX 3090 advertised FP32 peak (35.58 TFLOP/s)"
    )

    grad_accum_steps = derive_grad_accum_steps(
        device_batch_size=config.train.device_batch_size,
        seq_len=config.model.seq_len,
        total_batch_size_tokens=config.train.total_batch_size_tokens,
    )
    assert grad_accum_steps == expected["grad_accum_steps"]


@pytest.mark.parametrize(("filename", "expected"), PRESETS.items())
def test_base_preset_constructs_the_expected_tied_model_on_cpu(
    filename: str,
    expected: dict[str, int | str],
) -> None:
    config = load_config(CONFIG_DIR / filename)

    model = GPT(config.model)

    assert model.lm_head.weight is model.token_embedding.weight
    assert count_parameters(model) == expected["parameter_count"]
    assert count_parameters(model, trainable_only=True) == expected["parameter_count"]
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}


@pytest.mark.parametrize("filename", PRESETS)
def test_base_preset_pretrain_dry_run_requires_no_gpu_or_artifacts(
    filename: str,
    tmp_path: Path,
) -> None:
    run_name = f"dry-{Path(filename).stem.replace('_', '-')}"
    output_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.pretrain",
            "--config",
            str(CONFIG_DIR / filename),
            "--override",
            f"run.output_dir={output_dir}",
            "--override",
            f"run.name={run_name}",
            "--no-wandb",
            "--wandb-mode",
            "disabled",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    resolved_path = output_dir / run_name / "config.yaml"
    assert result.returncode == 0, result.stderr
    assert load_config(resolved_path).run.name == run_name
    assert not list((output_dir / run_name / "checkpoints").iterdir())


def test_first_sprint_smoke_remains_the_byte_tokenizer_fixture() -> None:
    config = load_config(CONFIG_DIR / "smoke.yaml")

    assert config.run.name == "smoke"
    assert config.run.device == "cpu"
    assert config.data.profile == "tiny_text"
    assert config.tokenizer.type == "byte"
    assert config.model.vocab_size == 265


def test_readme_documents_the_preset_boundary_and_3090_smoke_commands() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for filename in PRESETS:
        assert f"configs/{filename}" in readme
    assert "Hydra" in readme
    assert "three named presets" in readme
    assert "GradScaler" in readme
    assert (
        "python -m scripts.pretrain "
        "--config configs/tiny_20m_3090.yaml --dry-run" in readme
    )
    assert "--override run.name=tiny-20m-3090-smoke" in readme
    assert "--override train.max_steps=2" in readme
