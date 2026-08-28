"""Command-contract tests for dry-run, base initialization, and SFT resume."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.train_sft as train_sft
from scratch_llm.config import dump_config, load_config
from scratch_llm.model import GPT
from scratch_llm.training.checkpoint import load_model_checkpoint, save_checkpoint
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tokenization.tokenizer import ByteTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_SMOKE_CONFIG = PROJECT_ROOT / "configs" / "sft_smoke.yaml"


def test_dry_run_is_local_only_and_does_not_touch_checkpoint_data_or_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "scratch_llm.tracking.WandbTracker",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not initialize W&B"),
    )
    monkeypatch.setattr(
        "scratch_llm.training.sft.load_model_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not load a checkpoint"),
    )
    monkeypatch.setattr(
        "scratch_llm.training.sft.build_sft_conversation_sources",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not load SFT data"),
    )
    monkeypatch.setattr(
        "scratch_llm.training.sft.get_device",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not allocate a device"),
    )

    exit_code = train_sft.main(
        [
            "--config",
            str(SFT_SMOKE_CONFIG),
            "--override",
            f"run.output_dir={tmp_path / 'runs'}",
            "--override",
            "tracking.wandb.enabled=true",
            "--override",
            "tracking.wandb.mode=online",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    run_dir = tmp_path / "runs" / "sft-smoke"
    assert exit_code == 0
    assert "Resolved values:" in output
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics" / "metrics.jsonl").is_file()
    assert (run_dir / "metrics" / "summary.json").is_file()
    assert not list((run_dir / "checkpoints").glob("*.pt"))


def test_execution_requires_exactly_one_base_or_resume_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    common = [
        "--config",
        str(SFT_SMOKE_CONFIG),
        "--override",
        f"run.output_dir={tmp_path / 'runs'}",
    ]

    with pytest.raises(SystemExit, match="2"):
        train_sft.main(common)
    assert "requires a base checkpoint or --resume" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        train_sft.main(
            [
                *common,
                "--base-checkpoint",
                str(tmp_path / "base.pt"),
                "--resume",
                str(tmp_path / "sft.pt"),
            ]
        )
    assert "not allowed with argument" in capsys.readouterr().err


def test_resume_metadata_must_be_an_sft_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Metadata:
        training_stage = "pretrain"
        tracking = None
        config = None

    monkeypatch.setattr(
        train_sft, "load_checkpoint_metadata", lambda _path: _Metadata()
    )

    with pytest.raises(SystemExit, match="2"):
        train_sft.main(
            [
                "--config",
                str(SFT_SMOKE_CONFIG),
                "--override",
                f"run.output_dir={tmp_path / 'runs'}",
                "--resume",
                str(tmp_path / "pretrain.pt"),
            ]
        )

    assert "SFT resume requires an SFT checkpoint" in capsys.readouterr().err


def test_command_executes_a_bounded_local_base_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "scratch_llm.tracking.WandbTracker",
        lambda *_args, **_kwargs: pytest.fail("disabled SFT must not initialize W&B"),
    )
    config = load_config(SFT_SMOKE_CONFIG)
    config.run.name = "sft-cli"
    config.run.output_dir = str(tmp_path / "runs")
    config.model.n_layer = 1
    config.model.n_head = 1
    config.model.n_kv_head = 1
    config.model.n_embd = 8
    config.model.mlp_ratio = 2
    config.sft.max_steps = 1
    config.sft.eval_every = 1
    config.sft.eval_batches = 1
    config.sft.save_every = 1
    config.generation.max_new_tokens = 1
    base_path = tmp_path / "base.pt"
    config.sft.base_checkpoint = str(base_path)
    config.validate()
    tokenizer = ByteTokenizer()
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    save_checkpoint(
        base_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )
    config_path = dump_config(config, tmp_path / "sft.yaml")

    exit_code = train_sft.main(
        [
            "--config",
            str(config_path),
            "--base-checkpoint",
            str(base_path),
        ]
    )

    output = capsys.readouterr().out
    checkpoint_path = tmp_path / "runs" / "sft-cli" / "checkpoints" / "last.pt"
    assert exit_code == 0
    assert "Completed step 1" in output
    assert "Base checkpoint identity: sha256:" in output
    assert "Assistant validation BPB:" in output
    assert "Best checkpoint:" in output
    assert "Last checkpoint:" in output
    assert checkpoint_path.is_file()
    assert load_model_checkpoint(checkpoint_path).training_stage == "sft"
    assert (checkpoint_path.parent.parent / "metrics/sft_eval.json").is_file()
    assert (checkpoint_path.parent.parent / "metrics/sft_samples.md").is_file()
    assert "SFT evaluation:" in output
    assert "SFT samples:" in output
