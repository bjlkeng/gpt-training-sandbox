"""CPU command-level acceptance tests for chat evaluation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scripts._checkpoint_fixtures import create_tiny_sft_checkpoint
from scratch_llm.config import dump_config
from scratch_llm.data.hub import publish_local_parquet_cache
from scratch_llm.evaluation.chat.arc import get_arc_dataset_spec
from scratch_llm.evaluation.chat.categorical import (
    CHAT_CATEGORICAL_CONTEXT_POLICY_ID,
)
from scratch_llm.evaluation.chat.gsm8k import get_gsm8k_dataset_spec
from scratch_llm.evaluation.chat.humaneval import get_humaneval_dataset_spec
from scratch_llm.evaluation.chat.mmlu import get_mmlu_dataset_spec
from scratch_llm.training.checkpoint import load_checkpoint_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ALL_TASKS = (
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
)


def _publish_cache(
    cache_root: Path,
    source_dir: Path,
    name: str,
    spec,
    rows: list[dict[str, object]],
) -> None:
    path = source_dir / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    publish_local_parquet_cache(spec, cache_root, (path,))


def _chat_fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = create_tiny_sft_checkpoint(
        tmp_path / "sft.pt",
        seq_len=512,
        max_new_tokens=1,
    )
    source = load_checkpoint_metadata(checkpoint).config
    config = replace(
        source,
        run=replace(
            source.run,
            name="chat-eval-full",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        generation=replace(
            source.generation,
            max_new_tokens=1,
            temperature=0,
            top_k=1,
            top_p=None,
            seed=7,
        ),
    )
    config_path = dump_config(config, tmp_path / "config.yaml")
    cache_root = tmp_path / "cache"
    source_dir = tmp_path / "parquet"
    source_dir.mkdir()
    arc_row = {
        "answerKey": "A",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["alpha", "beta", "gamma", "delta"],
        },
        "question": "PRIVATE_ARC_QUESTION",
    }
    _publish_cache(
        cache_root,
        source_dir,
        "arc-easy",
        get_arc_dataset_spec("ARC-Easy"),
        [arc_row],
    )
    _publish_cache(
        cache_root,
        source_dir,
        "arc-challenge",
        get_arc_dataset_spec("ARC-Challenge"),
        [arc_row],
    )
    _publish_cache(
        cache_root,
        source_dir,
        "mmlu",
        get_mmlu_dataset_spec(),
        [
            {
                "answer": 0,
                "choices": ["a", "b", "c", "d"],
                "question": "PRIVATE_MMLU_QUESTION",
                "subject": "fixture_subject",
            },
            {
                "answer": 1,
                "choices": ["a", "b", "c", "d"],
                "question": "PRIVATE_OVERLENGTH_MMLU_" + "x" * 600,
                "subject": "fixture_subject",
            },
        ],
    )
    _publish_cache(
        cache_root,
        source_dir,
        "gsm8k",
        get_gsm8k_dataset_spec(),
        [
            {
                "answer": "PRIVATE_REASONING\n#### 2",
                "question": "PRIVATE_GSM8K_QUESTION",
            }
        ],
    )
    _publish_cache(
        cache_root,
        source_dir,
        "humaneval",
        get_humaneval_dataset_spec(),
        [
            {
                "canonical_solution": "\n    return 'PRIVATE_REFERENCE'\n",
                "entry_point": "fixture_function",
                "prompt": (
                    "import math\n\n"
                    "def fixture_function(value):\n"
                    '    """PRIVATE_HUMANEVAL_PROMPT"""\n'
                ),
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate(1) == 1  # PRIVATE_TEST\n"
                ),
            }
        ],
    )
    return checkpoint, config_path, cache_root, config.run.output_dir


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.eval_chat", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _tracking_records(output_dir: Path, run_name: str) -> list[dict[str, object]]:
    path = output_dir / run_name / "metrics" / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cpu_cli_full_filtered_bounded_failure_refusal_and_deterministic_rerun(
    tmp_path: Path,
) -> None:
    checkpoint, config, cache_root, raw_output_dir = _chat_fixtures(tmp_path)
    output_dir = Path(raw_output_dir)
    common = (
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--cache-root",
        str(cache_root),
        "--batch-size",
        "1",
        "--num-samples",
        "1",
        "--no-wandb",
    )

    refused = _run(*common)
    assert refused.returncode != 0
    assert "explicit generated-code execution opt-in" in refused.stderr
    assert "Traceback" not in refused.stderr
    assert not (output_dir / "chat-eval-full" / "metrics" / "chat_eval.json").exists()

    full = _run(*common, "--allow-generated-code-execution")
    assert full.returncode == 0, full.stderr
    report_path = output_dir / "chat-eval-full" / "metrics" / "chat_eval.json"
    original = report_path.read_bytes()
    payload = json.loads(original)
    assert payload["status"] == "completed"
    assert payload["scope"]["kind"] == "full"
    assert payload["scope"]["selected_tasks"] == [
        "ARC-Easy",
        "ARC-Challenge",
        "MMLU",
        "GSM8K",
        "HumanEval",
    ]
    assert payload["chatcore"]["chatcore_metric"] is not None
    assert payload["response_diagnostics"]["sample_count"] == 5
    mmlu_details = payload["tasks"][2]["details"]
    assert mmlu_details["counts"] == {
        "available": 2,
        "evaluated": 1,
        "excluded_overlength": 1,
        "passed": mmlu_details["counts"]["passed"],
        "selected": 2,
    }
    assert mmlu_details["prompt_preflight"]["policy_id"] == (
        CHAT_CATEGORICAL_CONTEXT_POLICY_ID
    )
    assert mmlu_details["prompt_preflight"]["model_max_seq_len"] == 512
    assert len(mmlu_details["prompt_preflight"]["excluded_examples"]) == 1
    assert "excluded_overlength=1" in full.stdout
    assert payload["tasks"][4]["details"]["scoring"]["outcome_counts"] == {
        "syntax_error": 1
    }
    serialized = original.decode("utf-8")
    for private in (
        "PRIVATE_ARC_QUESTION",
        "PRIVATE_MMLU_QUESTION",
        "PRIVATE_OVERLENGTH_MMLU_",
        "PRIVATE_GSM8K_QUESTION",
        "PRIVATE_HUMANEVAL_PROMPT",
        "PRIVATE_REFERENCE",
        "PRIVATE_TEST",
    ):
        assert private not in serialized
    records = _tracking_records(output_dir, "chat-eval-full")
    metric_records = [
        record for record in records if record["record_type"] == "metrics"
    ]
    artifact_records = [
        record for record in records if record["record_type"] == "artifact"
    ]
    expected_full_metrics = {
        "sft/chatcore_metric": payload["chatcore"]["chatcore_metric"],
        "sft/chatcore_cat": payload["chatcore"]["chatcore_cat"],
        **{
            f"sft/chatcore/{task['score']['task_name']}": task["score"][
                "centered_score"
            ]
            for task in payload["tasks"]
        },
    }
    assert [record["metrics"] for record in metric_records] == [expected_full_metrics]
    assert artifact_records[0]["path"] == "metrics/chat_eval.json"
    summary = json.loads(
        (output_dir / "chat-eval-full" / "metrics" / "summary.json").read_text()
    )
    assert summary["latest_metrics"] == expected_full_metrics
    tracking_text = json.dumps(records)
    for private in (
        "PRIVATE_ARC_QUESTION",
        "PRIVATE_MMLU_QUESTION",
        "PRIVATE_OVERLENGTH_MMLU_",
        "PRIVATE_GSM8K_QUESTION",
        "PRIVATE_HUMANEVAL_PROMPT",
        "PRIVATE_REFERENCE",
        "PRIVATE_TEST",
    ):
        assert private not in tracking_text

    repeated = _run(*common, "--allow-generated-code-execution")
    assert repeated.returncode == 0, repeated.stderr
    assert report_path.read_bytes() == original
    repeated_records = _tracking_records(output_dir, "chat-eval-full")
    assert sum(record["record_type"] == "metrics" for record in repeated_records) == 1
    assert sum(record["record_type"] == "artifact" for record in repeated_records) == 1

    filtered = _run(
        *common,
        "--override",
        "run.name=chat-eval-filtered",
        "--tasks",
        "ARC-Easy",
    )
    assert filtered.returncode == 0, filtered.stderr
    filtered_payload = json.loads(
        (output_dir / "chat-eval-filtered" / "metrics" / "chat_eval.json").read_text()
    )
    assert filtered_payload["scope"]["kind"] == "partial"
    assert filtered_payload["scope"]["bounded"] is False
    assert filtered_payload["chatcore"]["chatcore_metric"] is None
    filtered_metrics = next(
        record["metrics"]
        for record in _tracking_records(output_dir, "chat-eval-filtered")
        if record["record_type"] == "metrics"
    )
    assert set(filtered_metrics) == {"sft/chatcore/partial/ARC-Easy"}
    assert "sft/chatcore_metric" not in filtered_metrics
    assert "sft/chatcore_cat" not in filtered_metrics

    bounded = _run(
        *common,
        "--override",
        "run.name=chat-eval-bounded",
        "--max-problems",
        "1",
        "--allow-generated-code-execution",
    )
    assert bounded.returncode == 0, bounded.stderr
    bounded_payload = json.loads(
        (output_dir / "chat-eval-bounded" / "metrics" / "chat_eval.json").read_text()
    )
    assert bounded_payload["scope"]["kind"] == "bounded"
    assert (
        bounded_payload["scope"]["selected_tasks"] == payload["scope"]["selected_tasks"]
    )
    assert bounded_payload["chatcore"]["chatcore_metric"] is None
    assert bounded_payload["chatcore"]["chatcore_cat"] is None
    assert bounded_payload["tasks"][2]["details"]["counts"] == {
        "available": 2,
        "evaluated": 1,
        "excluded_overlength": 1,
        "passed": bounded_payload["tasks"][2]["details"]["counts"]["passed"],
        "selected": 2,
    }
    bounded_metrics = next(
        record["metrics"]
        for record in _tracking_records(output_dir, "chat-eval-bounded")
        if record["record_type"] == "metrics"
    )
    assert set(bounded_metrics) == {
        f"sft/chatcore/bounded/{task_name}" for task_name in _ALL_TASKS
    }
    assert "sft/chatcore_metric" not in bounded_metrics
    assert "sft/chatcore_cat" not in bounded_metrics

    empty_cache = tmp_path / "empty-cache"
    empty_cache.mkdir()
    failed = _run(
        "--config",
        str(config),
        "--override",
        "run.name=chat-eval-failed",
        "--checkpoint",
        str(checkpoint),
        "--cache-root",
        str(empty_cache),
        "--tasks",
        "MMLU",
        "--no-wandb",
    )
    assert failed.returncode != 0
    assert "Traceback" not in failed.stderr
    assert not (output_dir / "chat-eval-failed" / "metrics" / "chat_eval.json").exists()
    failed_records = _tracking_records(output_dir, "chat-eval-failed")
    assert not any(
        record["record_type"] in {"metrics", "artifact"} for record in failed_records
    )
    failed_summary = json.loads(
        (output_dir / "chat-eval-failed" / "metrics" / "summary.json").read_text()
    )
    assert failed_summary["status"] == "failed"
    assert failed_summary["latest_metrics"] == {}


def test_cli_dry_run_validates_scope_and_bounds_without_checkpoint_or_cache(
    tmp_path: Path,
) -> None:
    common = (
        "--config",
        str(PROJECT_ROOT / "configs" / "smoke.yaml"),
        "--override",
        f"run.output_dir={tmp_path / 'runs'}",
        "--override",
        "run.name=chat-eval-dry",
        "--dry-run",
        "--no-wandb",
    )

    valid = _run(*common, "--tasks", "MMLU,ARC-Easy")
    assert valid.returncode == 0, valid.stderr
    assert "Tasks: ARC-Easy,MMLU" in valid.stdout
    assert "Scope: partial" in valid.stdout
    assert not (
        tmp_path / "runs" / "chat-eval-dry" / "metrics" / "chat_eval.json"
    ).exists()

    duplicate = _run(*common, "--tasks", "MMLU,MMLU")
    unknown = _run(*common, "--tasks", "unknown")
    bad_batch = _run(*common, "--batch-size", "0")
    bad_limit = _run(*common, "--max-problems", "0")
    bad_samples = _run(*common, "--num-samples", "0")
    unsupported_top_p = _run(
        *common,
        "--override",
        "generation.top_p=0.9",
    )
    for result in (
        duplicate,
        unknown,
        bad_batch,
        bad_limit,
        bad_samples,
        unsupported_top_p,
    ):
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
