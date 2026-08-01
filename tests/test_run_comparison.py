"""Offline protocol-aware training-run comparison tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scratch_llm.config import ProjectConfig, RunConfig, TrainConfig, dump_config
from scratch_llm.comparison.pipeline import (
    RUN_COMPARISON_FORMAT,
    RUN_COMPARISON_FORMAT_VERSION,
    RunComparisonError,
    compare_training_runs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_run(
    root: Path,
    *,
    name: str,
    compatibility_bpb: float,
    full_document_bpb: float,
    manifest_identity: str = "manifest:shared",
    status: str = "completed",
    include_peak_memory: bool = True,
    max_steps: int = 2,
) -> Path:
    run_dir = root / name
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    config = ProjectConfig(
        run=RunConfig(name=name, output_dir=str(root)),
        train=TrainConfig(
            max_steps=max_steps,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
    )
    dump_config(config, run_dir / "config.yaml")
    training_records = [
        json.dumps(
            {"config": config.to_dict(), "record_type": "config"},
            sort_keys=True,
        )
    ]
    for step, loss in ((1, 2.0), (2, 1.5)):
        metrics: dict[str, object] = {
            "total_training_flops": float(step * 1000),
            "total_training_time": float(step * 2),
            "train/dt": 2.0,
            "train/loss": loss,
            "train/lrm": 1.0,
            "train/mfu": 0.25,
            "train/tok_per_sec": 100.0 + step,
        }
        if include_peak_memory:
            metrics["train/peak_memory_mib"] = 512.0 + step
        training_records.append(
            json.dumps(
                {"metrics": metrics, "record_type": "metrics", "step": step},
                sort_keys=True,
            )
        )
        training_records.append(
            json.dumps(
                {
                    "metrics": {
                        "min_val_bpb": compatibility_bpb + 0.1,
                        "min_val_bpb_full_documents": full_document_bpb + 0.1,
                        "val_bpb": compatibility_bpb + 0.2 / step,
                        "val_bpb_full_documents": full_document_bpb + 0.2 / step,
                    },
                    "record_type": "metrics",
                    "step": step,
                },
                sort_keys=True,
            )
        )
    (metrics_dir / "metrics.jsonl").write_text(
        "\n".join(training_records) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "summary.json").write_text(
        json.dumps(
            {
                "latest_metrics": {},
                "latest_step": 2,
                "run": {
                    "name": name,
                    "output_dir": str(run_dir),
                    "stage": "pretrain",
                },
                "schema_version": 1,
                "status": status,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_identity = f"checkpoint:{name}"
    tokenizer_identity = "tokenizer:shared"
    common = {
        "checkpoint_identity": checkpoint_identity,
        "counted_target_bytes": 10,
        "counted_target_tokens": 10,
        "processed_model_tokens": 16,
        "protocol_version": 1,
        "source_byte_retention": 1.0,
        "source_bytes": 10,
        "source_documents": 1,
        "source_token_retention": 1.0,
        "source_tokens": 10,
        "tokenizer_identity": tokenizer_identity,
        "unique_source_bytes": 10,
        "unique_source_tokens": 10,
        "validation_manifest_identity": manifest_identity,
    }
    (metrics_dir / "base_eval.json").write_text(
        json.dumps(
            {
                "bounded": False,
                "completed_modes": ["bpb"],
                "format": "scratch_llm_base_evaluation",
                "format_version": 2,
                "identities": {
                    "checkpoint": {"identity": checkpoint_identity, "step": 2},
                    "config": f"config:{name}",
                    "tokenizer": tokenizer_identity,
                    "validation_manifest": manifest_identity,
                },
                "max_per_task": None,
                "requested_modes": ["bpb"],
                "results": {
                    "full_documents_v1": {
                        **common,
                        "bpb": full_document_bpb,
                        "protocol_id": "full_documents_v1",
                        "reference_commit": None,
                        "reference_config": {"scope": "all-documents"},
                        "total_nats": full_document_bpb * math.log(2) * 10,
                    },
                    "nanochat_compat_v1": {
                        **common,
                        "bpb": compatibility_bpb,
                        "protocol_id": "nanochat_compat_v1",
                        "reference_commit": "pinned-commit",
                        "reference_config": {"eval_tokens": 16},
                        "total_nats": compatibility_bpb * math.log(2) * 10,
                    },
                },
                "run_kind": "full",
                "status": "completed",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_comparison_ranks_only_protocol_matched_results_and_keeps_full_separate(
    tmp_path: Path,
) -> None:
    first = _write_run(
        tmp_path / "runs",
        name="first",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    second = _write_run(
        tmp_path / "runs",
        name="second",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
        include_peak_memory=False,
    )
    output_dir = tmp_path / "comparison"

    artifacts = compare_training_runs((first, second), output_dir=output_dir)

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["format"] == RUN_COMPARISON_FORMAT
    assert payload["format_version"] == RUN_COMPARISON_FORMAT_VERSION
    assert [entry["run"] for entry in payload["rankings"]["nanochat_compat_v1"]] == [
        "second",
        "first",
    ]
    assert [entry["run"] for entry in payload["rankings"]["full_documents_v1"]] == [
        "first",
        "second",
    ]
    assert payload["runs"][1]["training"]["peak_memory_mib"] is None
    assert payload["runs"][0]["training"]["latest_compatibility_bpb"] == 1.6
    assert payload["runs"][0]["training"]["best_compatibility_bpb"] == 1.6
    assert payload["runs"][0]["training"]["latest_full_document_bpb"] == 1.1
    assert payload["runs"][0]["training"]["best_full_document_bpb"] == 1.1
    assert payload["aligned_steps"] == [1, 2]
    assert payload["step_series"]["2"]["first"]["train/loss"] == 1.5
    assert payload["step_series"]["2"]["second"]["train/loss"] == 1.5
    assert payload["step_series"]["2"]["first"]["val_bpb"] == 1.6
    assert artifacts.markdown_path.is_file()
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert markdown.index("Identity differences") < markdown.index("Numeric deltas")
    assert "Unavailable across all runs: `code_identity`." in markdown
    assert "Compatibility BPB" in markdown
    assert "Full-document BPB" in markdown
    assert "## Performance" in markdown
    assert "—" in markdown

    first_json = artifacts.json_path.read_bytes()
    first_markdown = artifacts.markdown_path.read_bytes()
    repeated = compare_training_runs((first, second), output_dir=output_dir)
    assert repeated.json_path.read_bytes() == first_json
    assert repeated.markdown_path.read_bytes() == first_markdown


def test_incomplete_or_identity_mismatched_runs_are_not_ranked(
    tmp_path: Path,
) -> None:
    first = _write_run(
        tmp_path / "runs",
        name="first",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    second = _write_run(
        tmp_path / "runs",
        name="second",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
        manifest_identity="manifest:different",
        status="failed",
    )

    artifacts = compare_training_runs(
        (second, first),
        output_dir=tmp_path / "comparison",
    )

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["rankings"] == {
        "full_documents_v1": [],
        "nanochat_compat_v1": [],
    }
    assert payload["runs"][0]["run"] == "first"
    assert payload["runs"][1]["run"] == "second"
    assert payload["runs"][1]["rankable"] is False
    assert "run status is failed" in payload["runs"][1]["ranking_blockers"]
    assert any(
        difference["field"] == "validation_manifest_identity"
        for difference in payload["identity_differences"]
    )


def test_completed_evaluation_does_not_hide_incomplete_training(
    tmp_path: Path,
) -> None:
    complete = _write_run(
        tmp_path / "runs",
        name="complete",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    interrupted_then_evaluated = _write_run(
        tmp_path / "runs",
        name="interrupted",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
        max_steps=3,
    )

    artifacts = compare_training_runs(
        (complete, interrupted_then_evaluated),
        output_dir=tmp_path / "comparison",
    )

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    interrupted = next(run for run in payload["runs"] if run["run"] == "interrupted")
    assert interrupted["status"] == "completed"
    assert interrupted["rankable"] is False
    assert "training stopped at step 2 of 3" in interrupted["ranking_blockers"]
    assert payload["rankings"] == {
        "full_documents_v1": [],
        "nanochat_compat_v1": [],
    }


def test_protocol_reference_mismatch_blocks_only_that_protocol_ranking(
    tmp_path: Path,
) -> None:
    first = _write_run(
        tmp_path / "runs",
        name="first",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    second = _write_run(
        tmp_path / "runs",
        name="second",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
    )
    report_path = second / "metrics" / "base_eval.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["results"]["nanochat_compat_v1"]["reference_config"] = {"eval_tokens": 32}
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = compare_training_runs(
        (first, second),
        output_dir=tmp_path / "comparison",
    )

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["rankings"]["nanochat_compat_v1"] == []
    assert [entry["run"] for entry in payload["rankings"]["full_documents_v1"]] == [
        "first",
        "second",
    ]


def test_truncated_metrics_fail_before_replacing_existing_reports(
    tmp_path: Path,
) -> None:
    first = _write_run(
        tmp_path / "runs",
        name="first",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    second = _write_run(
        tmp_path / "runs",
        name="second",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
    )
    (second / "metrics" / "metrics.jsonl").write_bytes(b'{"record_type":"metrics"')
    output_dir = tmp_path / "comparison"
    output_dir.mkdir()
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    json_path.write_text('{"stable": true}\n', encoding="utf-8")
    markdown_path.write_text("stable\n", encoding="utf-8")

    with pytest.raises(RunComparisonError, match="complete newline"):
        compare_training_runs((first, second), output_dir=output_dir)

    assert json_path.read_text(encoding="utf-8") == '{"stable": true}\n'
    assert markdown_path.read_text(encoding="utf-8") == "stable\n"


def test_jsonl_resolved_config_must_match_immutable_run_config(
    tmp_path: Path,
) -> None:
    first = _write_run(
        tmp_path / "runs",
        name="first",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    second = _write_run(
        tmp_path / "runs",
        name="second",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
    )
    metrics_path = second / "metrics" / "metrics.jsonl"
    records = metrics_path.read_text(encoding="utf-8").splitlines()
    config_record = json.loads(records[0])
    config_record["config"]["run"]["name"] = "not-second"
    records[0] = json.dumps(config_record, sort_keys=True)
    metrics_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(RunComparisonError, match="resolved configuration"):
        compare_training_runs(
            (first, second),
            output_dir=tmp_path / "comparison",
        )


def test_resumed_series_merges_same_step_records_and_advances_monotonically(
    tmp_path: Path,
) -> None:
    first = _write_run(
        tmp_path / "runs",
        name="first",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    second = _write_run(
        tmp_path / "runs",
        name="second",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
    )
    metrics_path = second / "metrics" / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "metrics": {
                        "total_training_flops": 3000.0,
                        "total_training_time": 6.0,
                        "train/loss": 1.25,
                        "train/mfu": 0.3,
                        "train/tok_per_sec": 104.0,
                    },
                    "record_type": "metrics",
                    "step": 3,
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.write(
            json.dumps(
                {
                    "metrics": {
                        "min_val_bpb": 1.2,
                        "val_bpb": 1.2,
                    },
                    "record_type": "metrics",
                    "step": 3,
                },
                sort_keys=True,
            )
            + "\n"
        )
    summary_path = second / "metrics" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["latest_step"] = 3
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifacts = compare_training_runs(
        (first, second),
        output_dir=tmp_path / "comparison",
    )

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    second_run = payload["runs"][1]
    assert payload["aligned_steps"] == [1, 2, 3]
    assert payload["step_series"]["3"]["second"]["train/loss"] == 1.25
    assert payload["step_series"]["3"]["second"]["val_bpb"] == 1.2
    assert second_run["training"]["latest_step"] == 3
    assert second_run["training"]["best_loss"] == 1.25
    assert second_run["training"]["best_compatibility_bpb"] == 1.2


def test_compare_runs_command_writes_reports_without_checkpoint_access(
    tmp_path: Path,
) -> None:
    first = _write_run(
        tmp_path / "runs",
        name="first",
        compatibility_bpb=1.5,
        full_document_bpb=1.0,
    )
    second = _write_run(
        tmp_path / "runs",
        name="second",
        compatibility_bpb=1.25,
        full_document_bpb=1.75,
    )
    output_dir = tmp_path / "comparison"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compare_runs",
            str(second),
            str(first),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"JSON report: {output_dir / 'comparison.json'}" in result.stdout
    assert f"Markdown report: {output_dir / 'comparison.md'}" in result.stdout
    assert "Traceback" not in result.stderr
