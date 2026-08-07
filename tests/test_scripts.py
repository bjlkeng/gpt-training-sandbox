"""Subprocess coverage for the repository's command-module interfaces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.eval_base as eval_base_script
import scripts.sample as sample_script
from scratch_llm.training.checkpoint import save_checkpoint
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
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.tracking_state import TrackingState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
BASE_SMOKE_CONFIG = PROJECT_ROOT / "configs" / "base_smoke.yaml"
CONFIG_COMMANDS = (
    "scripts.benchmark_pretrain",
    "scripts.prepare_data",
    "scripts.train_tokenizer",
    "scripts.eval_tokenizer",
    "scripts.pretrain",
    "scripts.eval_base",
    "scripts.train_sft",
    "scripts.eval_chat",
)
UNIMPLEMENTED_CONFIG_COMMANDS = tuple(
    module
    for module in CONFIG_COMMANDS
    if module
    not in {
        "scripts.benchmark_pretrain",
        "scripts.eval_base",
        "scripts.eval_chat",
        "scripts.pretrain",
        "scripts.prepare_data",
        "scripts.eval_tokenizer",
        "scripts.train_tokenizer",
        "scripts.train_sft",
    }
)
CHECKPOINT_COMMANDS = (
    "scripts.sample",
    "scripts.chat",
    "scripts.web_chat",
)
CHAT_TRACKING_COMMANDS = ("scripts.chat", "scripts.web_chat")
ROADMAP_COMMANDS = (
    CONFIG_COMMANDS
    + CHECKPOINT_COMMANDS
    + (
        "scripts.compare_runs",
        "scripts.data_stats",
        "scripts.download_climbmix",
        "scripts.prepare_sft_data",
    )
)


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


def test_eval_chat_help_requires_explicit_unsafe_code_execution_opt_in() -> None:
    result = _run_module("scripts.eval_chat", "--help")

    assert result.returncode == 0, result.stderr
    assert "--allow-generated-code-execution" in result.stdout
    assert "not safe for malicious or adversarial code" in " ".join(
        result.stdout.split()
    )


@pytest.mark.parametrize("module", CONFIG_COMMANDS)
def test_config_commands_share_explicit_wandb_options(module: str) -> None:
    result = _run_module(module, "--help")

    assert result.returncode == 0, result.stderr
    assert "--wandb" in result.stdout
    assert "--no-wandb" in result.stdout
    assert "--wandb-mode {online,offline,disabled}" in result.stdout


@pytest.mark.parametrize("module", CHAT_TRACKING_COMMANDS)
def test_chat_commands_share_optional_tracking_config_options(module: str) -> None:
    result = _run_module(module, "--help")

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--override" in result.stdout
    assert "--wandb" in result.stdout
    assert "--no-wandb" in result.stdout
    assert "--wandb-mode {online,offline,disabled}" in result.stdout


def test_pretrain_help_and_validation_expose_legacy_resume_migration() -> None:
    help_result = _run_module("scripts.pretrain", "--help")
    invalid_result = _run_module(
        "scripts.pretrain",
        "--config",
        str(SMOKE_CONFIG),
        "--allow-non-exact-resume",
    )

    assert help_result.returncode == 0, help_result.stderr
    assert "--allow-non-exact-resume" in help_result.stdout
    assert invalid_result.returncode != 0
    assert "--allow-non-exact-resume requires --resume" in invalid_result.stderr
    assert "Traceback" not in invalid_result.stderr


def test_pretraining_benchmark_dry_run_is_gpu_and_artifact_free(
    tmp_path: Path,
) -> None:
    result = _run_module(
        "scripts.benchmark_pretrain",
        "--config",
        str(PROJECT_ROOT / "configs" / "base_smoke.yaml"),
        "--override",
        f"run.output_dir={tmp_path / 'runs'}",
        "--override",
        "run.name=benchmark-dry-run",
        "--warmup-steps",
        "1",
        "--timed-steps",
        "2",
        "--dry-run",
        "--no-wandb",
    )

    assert result.returncode == 0, result.stderr
    assert "Benchmark protocol: production_pretraining_optimizer_steps_v1" in (
        result.stdout
    )
    assert "Warmup steps: 1" in result.stdout
    assert "Timed steps: 2" in result.stdout
    assert not (
        tmp_path
        / "runs"
        / "benchmark-dry-run"
        / "metrics"
        / "throughput_benchmark.json"
    ).exists()


def test_data_preparation_dry_run_does_not_require_tokenizer_artifacts(
    tmp_path: Path,
) -> None:
    result = _run_module(
        "scripts.prepare_data",
        "--config",
        str(PROJECT_ROOT / "configs" / "base_smoke.yaml"),
        "--override",
        f"run.output_dir={tmp_path / 'runs'}",
        "--override",
        "run.name=data-prep-dry-run",
        "--batch-size",
        "64",
        "--dry-run",
        "--no-wandb",
    )

    assert result.returncode == 0, result.stderr
    assert "Tokenized output: data/tokenized" in result.stdout
    assert "Train shards: 16" in result.stdout
    assert "Encoding batch size: 64" in result.stdout
    assert not (tmp_path / "runs" / "data-prep-dry-run" / "artifacts").exists()


def test_eval_base_normalizes_modes_and_requires_core_bundle_only_for_core(
    tmp_path: Path,
) -> None:
    common = (
        "--config",
        str(SMOKE_CONFIG),
        "--override",
        f"run.output_dir={tmp_path / 'runs'}",
        "--override",
        "run.name=base-eval-modes",
    )

    dry_run = _run_module(
        "scripts.eval_base",
        *common,
        "--eval",
        "Sample,BPB,sample",
        "--dry-run",
    )
    unknown = _run_module(
        "scripts.eval_base",
        *common,
        "--eval",
        "bpb,unknown",
        "--dry-run",
    )
    missing_bundle = _run_module(
        "scripts.eval_base",
        *common,
        "--checkpoint",
        str(tmp_path / "missing.pt"),
        "--eval",
        "core",
    )
    unexpected_bundle = _run_module(
        "scripts.eval_base",
        *common,
        "--eval",
        "sample",
        "--core-bundle",
        str(tmp_path / "eval_bundle.zip"),
        "--dry-run",
    )
    core_dry_run = _run_module(
        "scripts.eval_base",
        *common,
        "--eval",
        "Core",
        "--core-bundle",
        str(tmp_path / "eval_bundle.zip"),
        "--max-per-task",
        "100",
        "--dry-run",
    )

    assert dry_run.returncode == 0, dry_run.stderr
    assert "Evaluation modes: sample,bpb" in dry_run.stdout
    assert unknown.returncode != 0
    assert "unknown evaluation mode 'unknown'" in unknown.stderr
    assert missing_bundle.returncode != 0
    assert "--core-bundle is required for --eval core" in missing_bundle.stderr
    assert unexpected_bundle.returncode != 0
    assert "--core-bundle requires --eval core" in unexpected_bundle.stderr
    assert core_dry_run.returncode == 0, core_dry_run.stderr
    assert "Evaluation modes: core" in core_dry_run.stdout
    assert f"CORE bundle: {tmp_path / 'eval_bundle.zip'}" in core_dry_run.stdout
    assert "Traceback" not in (
        unknown.stderr + missing_bundle.stderr + unexpected_bundle.stderr
    )


def test_eval_base_reuses_same_run_checkpoint_wandb_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TrackingState(backend="wandb", run_id="remote-123")
    source = ProjectConfig(
        run=RunConfig(name="tracked", output_dir=str(tmp_path / "runs"))
    )
    active = load_config(
        SMOKE_CONFIG,
        [
            "run.name=tracked",
            f"run.output_dir={tmp_path / 'runs'}",
            "tracking.wandb.enabled=true",
            "tracking.wandb.mode=offline",
        ],
    )
    monkeypatch.setattr(
        eval_base_script,
        "load_checkpoint_metadata",
        lambda path: SimpleNamespace(config=source, tracking=state),
    )

    selected = eval_base_script._resolve_wandb_evaluation_state(
        active,
        tmp_path / "last.pt",
    )

    assert selected == state
    forked = ProjectConfig(
        run=RunConfig(name="comparison-copy", output_dir=str(tmp_path / "runs")),
        tracking=active.tracking,
    )
    assert (
        eval_base_script._resolve_wandb_evaluation_state(
            forked,
            tmp_path / "last.pt",
        )
        is None
    )


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
    config_path = (
        BASE_SMOKE_CONFIG
        if module in {"scripts.benchmark_pretrain", "scripts.prepare_data"}
        else SMOKE_CONFIG
    )

    result = _run_module(
        module,
        "--config",
        str(config_path),
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


def test_sample_stops_on_generated_bos_without_rendering_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokenizer = ByteTokenizer()

    class _BosLogitsModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_seq_len = 8
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (
                    token_ids.shape[0],
                    token_ids.shape[1],
                    tokenizer.get_vocab_size(),
                ),
                -torch.inf,
                device=token_ids.device,
            )
            logits[:, -1, tokenizer.get_bos_token_id()] = self.anchor
            return logits

    checkpoint = SimpleNamespace(
        config=SimpleNamespace(
            generation=GenerationConfig(
                temperature=0,
                top_k=1,
                max_new_tokens=4,
                seed=31,
            )
        ),
        model=_BosLogitsModel(),
        tokenizer=tokenizer,
    )
    monkeypatch.setattr(
        sample_script,
        "load_model_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )

    exit_code = sample_script.main(
        [
            "--checkpoint",
            "unused.pt",
            "--prompt",
            "Hello",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "Hello\n"


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
    assert "--allow-non-exact-resume" in readme
    assert "checkpoint format version 5" in readme
    assert "Versions 3 and 4 remain exactly resumable" in readme
    assert "completed optimizer-step boundary" in readme
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
