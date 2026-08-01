"""Tests for strict, read-only loading of pinned CORE bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from scratch_llm.evaluation.core.bundle import (
    CoreBundleError,
    CoreBundleSpec,
    load_core_bundle,
)
from scratch_llm.evaluation.core.examples import load_core_task_examples


TASKS = (
    ("choice", "multiple_choice", "choice.jsonl", 1, "\nAnswer: ", 25.0),
    ("schema", "schema", "schema.jsonl", 0, " ", 50.0),
    ("language", "language_modeling", "language.jsonl", 0, " ", 0.0),
)


def _write_bundle(
    path: Path,
    *,
    dataset_uri_override: str | None = None,
    malformed_choice: bool = False,
) -> CoreBundleSpec:
    config_lines = ["icl_tasks:"]
    for index, (label, task_type, member, fewshot, delimiter, _) in enumerate(TASKS):
        uri = dataset_uri_override if index == 0 and dataset_uri_override else member
        config_lines.extend(
            (
                f"- label: {label}",
                f"  dataset_uri: fixtures/{uri}",
                f"  num_fewshot: [{fewshot}]",
                f"  icl_task_type: {task_type}",
                f"  continuation_delimiter: {json.dumps(delimiter)}",
            )
        )
    metadata = io.StringIO(newline="")
    writer = csv.writer(metadata, lineterminator="\n")
    writer.writerow(("Eval Task", "Random baseline"))
    for label, _, _, _, _, baseline in TASKS:
        writer.writerow((label, baseline))
    writer.writerow(("unused_metadata_task", 50.0))
    reference = io.StringIO(newline="")
    writer = csv.writer(reference, lineterminator="\n")
    writer.writerow(("Task", "Accuracy", "Centered"))
    for label, _, _, _, _, _ in TASKS:
        writer.writerow((label, 0.5, 0.25))
    writer.writerow(("CORE", "", 0.25))
    choice = (
        {"query": "Question?", "choices": ["No", "Yes"], "gold": 1}
        if not malformed_choice
        else {"query": "Question?", "choices": ["Only"], "gold": 0}
    )
    members = {
        "eval_bundle/core.yaml": "\n".join(config_lines) + "\n",
        "eval_bundle/eval_meta_data.csv": metadata.getvalue(),
        "eval_bundle/eval_data/fixtures/choice.jsonl": json.dumps(choice) + "\n",
        "eval_bundle/eval_data/fixtures/schema.jsonl": json.dumps(
            {
                "context_options": ["Alice", "Bob"],
                "continuation": " won.",
                "gold": 0,
            }
        )
        + "\n",
        "eval_bundle/eval_data/fixtures/language.jsonl": json.dumps(
            {"context": "The answer is", "continuation": " yes"}
        )
        + "\n",
        "eval_bundle/reference.csv": reference.getvalue(),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, contents in members.items():
            archive.writestr(member, contents)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return CoreBundleSpec(
        archive_sha256=digest,
        task_labels=tuple(task[0] for task in TASKS),
        reference_files={"fixture-reference": "eval_bundle/reference.csv"},
    )


def test_load_core_bundle_validates_protocol_order_baselines_and_references(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval_bundle.zip"
    spec = _write_bundle(path)

    bundle = load_core_bundle(path, spec=spec)

    assert bundle.identity == f"sha256:{spec.archive_sha256}"
    assert tuple(task.label for task in bundle.tasks) == spec.task_labels
    assert [task.random_baseline_percent for task in bundle.tasks] == [25.0, 50.0, 0.0]
    assert bundle.tasks[0].continuation_delimiter == "\nAnswer: "
    assert bundle.config_identity.startswith("sha256:")
    assert bundle.metadata_identity.startswith("sha256:")
    assert bundle.reference_results[0].model_id == "fixture-reference"
    assert bundle.reference_results[0].core_metric == 0.25

    examples = load_core_task_examples(bundle, bundle.tasks[0])
    assert examples[0].choices == ("No", "Yes")
    assert examples[0].gold == 1
    assert examples.identity.startswith("sha256:")


def test_load_core_bundle_rejects_an_unpinned_archive(tmp_path: Path) -> None:
    path = tmp_path / "eval_bundle.zip"
    spec = _write_bundle(path)
    changed_spec = CoreBundleSpec(
        archive_sha256="0" * 64,
        task_labels=spec.task_labels,
        reference_files=spec.reference_files,
    )

    with pytest.raises(CoreBundleError, match="SHA-256"):
        load_core_bundle(path, spec=changed_spec)


def test_load_core_bundle_rejects_unsafe_dataset_paths(tmp_path: Path) -> None:
    path = tmp_path / "eval_bundle.zip"
    spec = _write_bundle(path, dataset_uri_override="../../outside.jsonl")

    with pytest.raises(CoreBundleError, match="safe relative path"):
        load_core_bundle(path, spec=spec)


def test_load_core_task_examples_validates_task_shapes(tmp_path: Path) -> None:
    path = tmp_path / "eval_bundle.zip"
    spec = _write_bundle(path, malformed_choice=True)
    bundle = load_core_bundle(path, spec=spec)

    with pytest.raises(CoreBundleError, match="at least two choices"):
        load_core_task_examples(bundle, bundle.tasks[0])
