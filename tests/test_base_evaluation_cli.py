"""Bounded subprocess integration for the standalone base evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scratch_llm.training.checkpoint import save_checkpoint
from scratch_llm.config import (
    DataConfig,
    GPTConfig,
    GenerationConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
    dump_config,
)
from scratch_llm.data.loaders import write_tokenized_parquet_shards
from scratch_llm.model import GPT
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.run import prepare_run
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer
from scratch_llm.tracking import build_tracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_eval_base_subprocess_runs_bpb_and_samples_then_rejects_conflicting_rerun(
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    pq.write_table(
        pa.table({"text": pa.array(["training text"], type=pa.string())}),
        parquet_dir / "shard_00000.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    pq.write_table(
        pa.table({"text": pa.array(["validation text"], type=pa.string())}),
        parquet_dir / "shard_06542.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    tokenizer = ByteTokenizer()
    tokenized_dir = tmp_path / "tokenized"
    write_tokenized_parquet_shards(
        parquet_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=1,
        batch_size=2,
    )
    config = ProjectConfig(
        run=RunConfig(
            name="bounded-base-eval",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        data=DataConfig(
            parquet_dir=str(parquet_dir),
            tokenized_dir=str(tokenized_dir),
            num_pretrain_train_shards=1,
        ),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=8,
            grad_accum_steps=1,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0.0,
            eval_tokens=8,
        ),
        generation=GenerationConfig(
            temperature=0.0,
            top_k=1,
            max_new_tokens=1,
            seed=17,
        ),
    )
    config_path = dump_config(config, tmp_path / "eval.yaml")
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    checkpoint_path = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )
    paths = prepare_run(config)
    with build_tracker(config, paths, stage="pretrain") as tracker:
        tracker.log({"train/loss": 2.0}, step=1)
    command = [
        sys.executable,
        "-m",
        "scripts.eval_base",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--eval",
        "bpb,sample",
        "--no-wandb",
    ]

    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    run_dir = Path(config.run.output_dir) / config.run.name
    report_path = run_dir / "metrics" / "base_eval.json"
    sample_path = run_dir / "metrics" / "base_samples.md"
    assert first.returncode == 0, first.stderr
    assert report_path.is_file()
    assert sample_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert (
        payload["requested_modes"]
        == payload["completed_modes"]
        == [
            "bpb",
            "sample",
        ]
    )
    assert payload["status"] == "completed"
    assert set(payload["results"]) == {
        "full_documents_v1",
        "nanochat_compat_v1",
    }
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text())
    assert summary["run"]["stage"] == "eval_base"
    assert summary["latest_step"] == 1
    assert summary["latest_metrics"]["train/loss"] == 2.0
    assert (
        summary["latest_metrics"]["eval/val_bpb"]
        == payload["results"]["nanochat_compat_v1"]["bpb"]
    )
    stable_report = report_path.read_bytes()

    rerun = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rerun.returncode != 0
    assert "different completed evaluation" in rerun.stderr
    assert "Traceback" not in rerun.stderr
    assert report_path.read_bytes() == stable_report
