"""Tests for composing a validated CORE bundle with one model checkpoint."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch
from torch import nn

import pytest

import scratch_llm.core_evaluation_pipeline as pipeline
from scratch_llm.core_bundle import CoreBundleSpec, load_core_bundle
from scratch_llm.core_evaluation import CoreEvaluationError
from scratch_llm.core_evaluation_pipeline import evaluate_core_bundle
from scratch_llm.core_reporting import (
    render_core_comparison_markdown,
    write_core_comparison_markdown,
)
from scratch_llm.tokenizer import ByteTokenizer


class _OracleNextTokenModel(nn.Module):
    max_seq_len = 64

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*token_ids.shape, 265), device=token_ids.device)
        logits[:, :-1].scatter_(2, token_ids[:, 1:].unsqueeze(-1), 10.0)
        return logits


def _bundle(
    tmp_path: Path,
    *,
    num_fewshot: int = 0,
) -> tuple[Path, CoreBundleSpec]:
    path = tmp_path / "eval_bundle.zip"
    metadata = io.StringIO(newline="")
    writer = csv.writer(metadata, lineterminator="\n")
    writer.writerow(("Eval Task", "Random baseline"))
    writer.writerow(("language", 0))
    reference = io.StringIO(newline="")
    writer = csv.writer(reference, lineterminator="\n")
    writer.writerow(("Task", "Accuracy", "Centered"))
    writer.writerow(("language", 0.25, 0.25))
    writer.writerow(("CORE", "", 0.25))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "eval_bundle/core.yaml",
            """icl_tasks:
- label: language
  dataset_uri: fixtures/language.jsonl
  num_fewshot: [${num_fewshot}]
  icl_task_type: language_modeling
""".replace("${num_fewshot}", str(num_fewshot)),
        )
        archive.writestr("eval_bundle/eval_meta_data.csv", metadata.getvalue())
        archive.writestr(
            "eval_bundle/eval_data/fixtures/language.jsonl",
            "\n".join(
                json.dumps({"context": f"Question {index}", "continuation": "answer"})
                for index in range(2)
            )
            + "\n",
        )
        archive.writestr("eval_bundle/reference.csv", reference.getvalue())
    spec = CoreBundleSpec(
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        task_labels=("language",),
        reference_files={"reference": "eval_bundle/reference.csv"},
    )
    return path, spec


def test_evaluate_core_bundle_returns_complete_typed_bounded_result(
    tmp_path: Path,
) -> None:
    path, spec = _bundle(tmp_path)
    bundle = load_core_bundle(path, spec=spec)
    tokenizer = ByteTokenizer()

    result = evaluate_core_bundle(
        _OracleNextTokenModel(),
        tokenizer,
        bundle,
        checkpoint_identity="checkpoint",
        max_per_task=1,
        device="cpu",
        clock=iter((10.0, 12.0)).__next__,
    )

    assert result.run_kind == "bounded"
    assert result.max_per_task == 1
    assert result.core_metric == 1.0
    assert result.elapsed_seconds == 2.0
    assert result.tasks[0].correct_examples == 1
    assert result.tasks[0].evaluated_examples == 1
    assert result.tasks[0].available_examples == 2
    assert result.tasks[0].data_identity.startswith("sha256:")


def test_evaluate_core_bundle_preflights_bound_against_fewshot_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, spec = _bundle(tmp_path, num_fewshot=1)
    bundle = load_core_bundle(path, spec=spec)

    def unexpected_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("scope validation must precede task data loading")

    monkeypatch.setattr(pipeline, "load_core_task_examples", unexpected_load)

    with pytest.raises(CoreEvaluationError, match="at least 2"):
        evaluate_core_bundle(
            _OracleNextTokenModel(),
            ByteTokenizer(),
            bundle,
            checkpoint_identity="checkpoint",
            max_per_task=1,
            device="cpu",
        )


def test_core_comparison_markdown_labels_bounded_results_and_writes_atomically(
    tmp_path: Path,
) -> None:
    path, spec = _bundle(tmp_path)
    bundle = load_core_bundle(path, spec=spec)
    result = evaluate_core_bundle(
        _OracleNextTokenModel(),
        ByteTokenizer(),
        bundle,
        checkpoint_identity="checkpoint",
        max_per_task=1,
        device="cpu",
        clock=iter((0.0, 1.0)).__next__,
    )

    markdown = render_core_comparison_markdown(result)
    assert "Bounded estimate" in markdown
    assert "not comparable" in markdown
    assert "| language | 1 / 2 | 1.000000 | 0.00 | 1.000000 |" in markdown
    assert "| reference | 0.250000 | not comparable |" in markdown

    destination = write_core_comparison_markdown(result, tmp_path / "metrics")
    assert destination == tmp_path / "metrics" / "core_comparison.md"
    assert destination.read_text(encoding="utf-8") == markdown
