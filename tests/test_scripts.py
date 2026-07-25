"""Subprocess coverage for the repository's command-module interfaces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scratch_llm.checkpoint import save_checkpoint
from scratch_llm.config import (
    GPTConfig,
    GenerationConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
    dump_config,
    load_config,
)
from scratch_llm.model import GPT
from scratch_llm.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tokenizer import VOCAB_SIZE, ByteTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
CONFIG_COMMANDS = (
    "scripts.train_tokenizer",
    "scripts.eval_tokenizer",
    "scripts.pretrain",
    "scripts.eval_base",
    "scripts.train_sft",
    "scripts.eval_chat",
)
UNIMPLEMENTED_CONFIG_COMMANDS = tuple(
    module for module in CONFIG_COMMANDS if module != "scripts.pretrain"
)
CHECKPOINT_COMMANDS = (
    "scripts.sample",
    "scripts.chat",
    "scripts.web_chat",
)
UNIMPLEMENTED_CHECKPOINT_COMMANDS = (
    "scripts.chat",
    "scripts.web_chat",
)
ROADMAP_COMMANDS = CONFIG_COMMANDS + CHECKPOINT_COMMANDS


def _run_module(module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("module", ROADMAP_COMMANDS)
def test_every_roadmap_command_has_dependency_light_help(module: str) -> None:
    result = _run_module(module, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("module", CONFIG_COMMANDS)
def test_config_commands_share_explicit_wandb_options(module: str) -> None:
    result = _run_module(module, "--help")

    assert result.returncode == 0, result.stderr
    assert "--wandb" in result.stdout
    assert "--no-wandb" in result.stdout
    assert "--wandb-mode {online,offline,disabled}" in result.stdout


def test_dry_run_applies_wandb_environment_then_cli_without_importing_wandb(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "runs"
    environment = os.environ.copy()
    environment.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_PROJECT": "environment-project",
            "WANDB_ENTITY": "environment-entity",
            "WANDB_RUN_GROUP": "environment-group",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.pretrain",
            "--config",
            str(SMOKE_CONFIG),
            "--override",
            f"run.output_dir={output_dir}",
            "--override",
            "run.name=tracking-dry-run",
            "--override",
            "tracking.wandb.project=cli-project",
            "--wandb",
            "--wandb-mode",
            "disabled",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    run_dir = output_dir / "tracking-dry-run"
    resolved = load_config(run_dir / "config.yaml")
    assert result.returncode == 0, result.stderr
    assert resolved.tracking.wandb.enabled is True
    assert resolved.tracking.wandb.mode == "disabled"
    assert resolved.tracking.wandb.project == "cli-project"
    assert resolved.tracking.wandb.entity == "environment-entity"
    assert resolved.tracking.wandb.group == "environment-group"
    assert (run_dir / "metrics" / "metrics.jsonl").is_file()
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("module", CONFIG_COMMANDS)
def test_config_command_dry_run_resolves_repeated_overrides_without_training(
    module: str,
    tmp_path: Path,
) -> None:
    command_name = module.removeprefix("scripts.")
    run_name = command_name.replace("_", "-")
    output_dir = tmp_path / "runs"

    result = _run_module(
        module,
        "--config",
        str(SMOKE_CONFIG),
        "--override",
        f"run.output_dir={output_dir}",
        "--override",
        "run.name=overridden-first",
        "--override",
        f"run.name={run_name}",
        "--dry-run",
    )

    run_dir = output_dir / run_name
    resolved_config = run_dir / "config.yaml"
    assert result.returncode == 0, result.stderr
    assert f"Run directory: {run_dir}" in result.stdout
    assert f"Resolved config: {resolved_config}" in result.stdout
    assert f"name: {run_name}" in result.stdout
    assert load_config(resolved_config).run.name == run_name
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "metrics").is_dir()
    assert not list((run_dir / "checkpoints").iterdir())
    assert sorted(path.name for path in (run_dir / "metrics").iterdir()) == [
        "metrics.jsonl",
        "summary.json",
    ]
    records = [
        json.loads(line)
        for line in (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records == [
        {
            "record_type": "config",
            "config": load_config(resolved_config).to_dict(),
        }
    ]
    summary = json.loads(
        (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "completed"
    assert summary["run"] == {
        "name": run_name,
        "output_dir": str(run_dir),
        "stage": command_name,
    }
    assert not list(run_dir.rglob("*.pt"))


def test_failed_pretrain_still_closes_valid_local_tracking_outputs(
    tmp_path: Path,
) -> None:
    config = load_config(SMOKE_CONFIG)
    config.run.name = "failing-pretrain"
    config.run.output_dir = str(tmp_path / "runs")
    config.data.profile = "not-yet-supported"
    config_path = dump_config(config, tmp_path / "failing.yaml")

    result = _run_module(
        "scripts.pretrain",
        "--config",
        str(config_path),
        "--no-wandb",
    )

    run_dir = Path(config.run.output_dir) / config.run.name
    metrics_path = run_dir / config.tracking.jsonl.path
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(
        (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert result.returncode != 0
    assert records == [
        {
            "record_type": "config",
            "config": config.to_dict(),
        }
    ]
    assert summary["status"] == "failed"
    assert summary["latest_step"] is None
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        *(
            (module, ("--config", str(SMOKE_CONFIG)))
            for module in UNIMPLEMENTED_CONFIG_COMMANDS
        ),
        *(
            (module, ("--checkpoint", "runs/missing/checkpoints/last.pt"))
            for module in UNIMPLEMENTED_CHECKPOINT_COMMANDS
        ),
    ],
)
def test_unimplemented_non_dry_run_commands_fail_explicitly(
    module: str,
    arguments: tuple[str, ...],
) -> None:
    result = _run_module(module, *arguments)

    assert result.returncode != 0
    assert "not implemented" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_sample_loads_a_tiny_checkpoint_and_prints_non_empty_text(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=4,
            grad_accum_steps=1,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
        generation=GenerationConfig(
            temperature=0.0,
            top_k=1,
            max_new_tokens=2,
            seed=31,
        ),
    )
    model = GPT(config.model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    checkpoint_path = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=ByteTokenizer(),
    )

    result = _run_module(
        "scripts.sample",
        "--checkpoint",
        str(checkpoint_path),
        "--prompt",
        "Hello",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "Hello\x00\x00\n"
    assert result.stdout.strip()
    assert "Traceback" not in result.stderr


def test_readme_documents_the_subprocess_tested_setup_and_smoke_commands() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "uv sync --extra dev" in readme
    assert "uv run --extra dev pytest" in readme
    assert (
        "uv run python -m scripts.pretrain "
        "--config configs/smoke.yaml --dry-run" in readme
    )
    assert "uv run python -m scripts.pretrain --config configs/smoke.yaml" in readme
    assert (
        "uv run python -m scripts.sample "
        "--checkpoint runs/smoke/checkpoints/last.pt" in readme
    )
    assert "--resume runs/smoke/checkpoints/step_000075.pt" in readme
    assert "metrics/metrics.jsonl" in readme
    assert "metrics/summary.json" in readme
    assert '"schema_version": 1' in readme
    assert '"latest_step": 200' in readme
    assert "`running`, `completed`, or `failed`" in readme
    assert "uv sync --extra dev --extra tracking" in readme
    assert "--override run.name=tracking-disabled-smoke" in readme
    assert "--override run.name=tracking-offline-smoke" in readme
    assert "--override run.name=tracking-online" in readme
    assert "WANDB_MODE=offline" in readme
    assert "WANDB_PROJECT=scratch-llm" in readme
    assert "WANDB_ENTITY=your-wandb-entity" in readme
    assert "WANDB_RUN_GROUP=3090-pretrain" in readme
    assert "wandb sync" in readme
    assert "pipeline-stage:pretrain" in readme
    assert "log_model_artifacts" in readme
    assert "log_prompts" in readme
    assert "log_responses" in readme
