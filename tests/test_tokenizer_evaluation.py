"""Tests for bounded, deterministic tokenizer evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scratch_llm.tokenization.bpe import RegexBPETokenizer, train_reference_bpe
from scratch_llm.config import GPTConfig, ProjectConfig, RunConfig, TokenizerConfig
from scratch_llm.tokenization.tokenizer import ByteTokenizer
import scratch_llm.evaluation.tokenizer as tokenizer_evaluation
from scratch_llm.evaluation.tokenizer import (
    collect_evaluation_corpora,
    evaluate_tokenizer,
    write_tokenizer_evaluation_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"


def _save_test_tokenizer(path: Path) -> RegexBPETokenizer:
    result = train_reference_bpe(
        ("abracadabra abracadabra", "bounded tokenizer fixture"),
        vocab_size=267,
    )
    tokenizer = RegexBPETokenizer(result)
    tokenizer.save(path)
    return tokenizer


def _benchmark_clock(
    category_count: int,
    *,
    encode_seconds: float = 2.0,
    decode_seconds: float = 4.0,
) -> Any:
    values: list[float] = []
    current = 0.0
    for _ in range(category_count):
        values.extend((current, current + encode_seconds))
        current += encode_seconds + 1.0
        values.extend((current, current + decode_seconds))
        current += decode_seconds + 1.0
    return iter(values).__next__


def test_evaluation_corpora_cover_fixed_and_bounded_climbmix_categories() -> None:
    corpora = collect_evaluation_corpora(
        PARQUET_FIXTURE_DIR,
        num_train_shards=1,
        max_documents=2,
        max_characters=40,
        document_char_cap=30,
        batch_size=1,
    )

    assert tuple(corpus.name for corpus in corpora) == (
        "news",
        "korean",
        "code",
        "math",
        "science",
        "climbmix-train",
        "climbmix-validation",
    )
    train, validation = corpora[-2:]
    assert train.documents == (
        "First synthetic training docum",
        "Unicode tr",
    )
    assert train.selected_shards == ("shard_00000.parquet",)
    assert validation.documents == (
        "Fixed validation document.",
        "",
    )
    assert validation.selected_shards == ("shard_06542.parquet",)
    assert train.source_dict() == {
        "character_count": 40,
        "data_dir": str(PARQUET_FIXTURE_DIR),
        "document_count": 2,
        "identifier": "climbmix-train",
        "kind": "climbmix",
        "limits": {
            "document_char_cap": 30,
            "max_characters": 40,
            "max_documents": 2,
        },
        "selected_shard_count": 1,
        "selected_shards": ["shard_00000.parquet"],
        "selection": {
            "num_train_shards": 1,
            "split": "train",
            "text_column": "text",
        },
        "utf8_bytes": 40,
    }


def test_evaluation_result_is_immutable_json_compatible_and_benchmarked() -> None:
    corpora = collect_evaluation_corpora(
        PARQUET_FIXTURE_DIR,
        num_train_shards=1,
        max_documents=2,
        max_characters=40,
        document_char_cap=30,
        batch_size=1,
    )

    result = evaluate_tokenizer(
        ByteTokenizer(),
        corpora,
        compare=False,
        benchmark_warmup_iterations=1,
        benchmark_iterations=2,
        clock=_benchmark_clock(len(corpora)),
    )
    payload = result.to_dict()

    assert payload["format"] == "scratch_llm_tokenizer_evaluation"
    assert payload["format_version"] == 1
    assert payload["tokenizer"] == {
        "identity": ByteTokenizer().get_identity(),
        "vocab_size": ByteTokenizer().get_vocab_size(),
    }
    assert [category["name"] for category in payload["categories"]] == [
        corpus.name for corpus in corpora
    ]
    assert all(category["round_trip"] for category in payload["categories"])
    assert all(
        category["bytes"] == category["tokens"]
        and category["bytes_per_token"] == 1.0
        and category["encode_tokens_per_second"] == category["tokens"]
        and category["decode_tokens_per_second"] == category["tokens"] / 2
        for category in payload["categories"]
    )
    assert payload["categories"][0]["comparisons"] == {
        "cl100k_base": {
            "detail": "comparison not requested",
            "relative_token_count_difference": None,
            "status": "skipped",
            "tokens": None,
            "vocab_size": None,
        },
        "gpt2": {
            "detail": "comparison not requested",
            "relative_token_count_difference": None,
            "status": "skipped",
            "tokens": None,
            "vocab_size": None,
        },
    }
    aggregate_tokens = payload["aggregate"]["tokens"]
    assert payload["aggregate"] == {
        "bytes": sum(len(corpus.text.encode("utf-8")) for corpus in corpora),
        "bytes_per_token": 1.0,
        "comparisons": {
            "cl100k_base": {
                "detail": "comparison not requested",
                "relative_token_count_difference": None,
                "status": "skipped",
                "tokens": None,
                "vocab_size": None,
            },
            "gpt2": {
                "detail": "comparison not requested",
                "relative_token_count_difference": None,
                "status": "skipped",
                "tokens": None,
                "vocab_size": None,
            },
        },
        "decode_tokens_per_second": aggregate_tokens / 14,
        "encode_tokens_per_second": aggregate_tokens / 7,
        "round_trip": True,
        "tokens": aggregate_tokens,
    }
    assert payload["benchmark"] == {
        "aggregate": {
            "decode": {
                "seconds": 28.0,
                "timed_token_count": aggregate_tokens * 2,
                "tokens_per_second": aggregate_tokens / 14,
            },
            "encode": {
                "seconds": 14.0,
                "timed_token_count": aggregate_tokens * 2,
                "tokens_per_second": aggregate_tokens / 7,
            },
        },
        "categories": [
            {
                "decode": {
                    "seconds": 4.0,
                    "timed_token_count": category["tokens"] * 2,
                    "tokens_per_second": category["tokens"] / 2,
                },
                "encode": {
                    "seconds": 2.0,
                    "timed_token_count": category["tokens"] * 2,
                    "tokens_per_second": category["tokens"],
                },
                "name": category["name"],
            }
            for category in payload["categories"]
        ],
        "protocol": {
            "clock": "monotonic",
            "denominator": "token IDs processed during timed calls",
            "timed_iterations": 2,
            "warmup_iterations": 1,
        },
    }
    with pytest.raises(FrozenInstanceError):
        result.vocab_size = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.categories[0].source.identifier = "changed"  # type: ignore[misc]


def test_comparisons_are_lazy_and_unavailable_dependency_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpora = collect_evaluation_corpora(
        PARQUET_FIXTURE_DIR,
        max_documents=1,
        max_characters=20,
        document_char_cap=20,
    )

    def missing_dependency() -> dict[str, Any]:
        raise ModuleNotFoundError("No module named 'tiktoken'", name="tiktoken")

    monkeypatch.setattr(
        tokenizer_evaluation,
        "_load_tiktoken_encodings",
        missing_dependency,
    )
    skipped = evaluate_tokenizer(
        ByteTokenizer(),
        corpora,
        compare=False,
        benchmark_warmup_iterations=0,
        benchmark_iterations=1,
        clock=_benchmark_clock(len(corpora)),
    )
    assert all(
        comparison["status"] == "skipped"
        for category in skipped.to_dict()["categories"]
        for comparison in category["comparisons"].values()
    )

    unavailable = evaluate_tokenizer(
        ByteTokenizer(),
        corpora,
        compare=True,
        benchmark_warmup_iterations=0,
        benchmark_iterations=1,
        clock=_benchmark_clock(len(corpora)),
    )
    assert all(
        comparison
        == {
            "detail": (
                "optional dependency 'tiktoken' is unavailable; install the "
                "tokenizer-comparison extra"
            ),
            "relative_token_count_difference": None,
            "status": "unavailable",
            "tokens": None,
            "vocab_size": None,
        }
        for category in unavailable.to_dict()["categories"]
        for comparison in category["comparisons"].values()
    )


def test_comparisons_record_baseline_counts_and_relative_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEncoding:
        def __init__(self, *, n_vocab: int, divisor: int) -> None:
            self.n_vocab = n_vocab
            self.divisor = divisor

        def encode(self, text: str) -> list[int]:
            return [0] * max(1, len(text.encode("utf-8")) // self.divisor)

    monkeypatch.setattr(
        tokenizer_evaluation,
        "_load_tiktoken_encodings",
        lambda: {
            "gpt2": FakeEncoding(n_vocab=50_257, divisor=2),
            "cl100k_base": FakeEncoding(n_vocab=100_277, divisor=4),
        },
    )
    corpus = tokenizer_evaluation.EvaluationCorpus(
        name="fixture",
        kind="builtin",
        identifier="test:fixture",
        documents=("abcdefgh",),
    )

    result = evaluate_tokenizer(
        ByteTokenizer(),
        (corpus,),
        compare=True,
        benchmark_warmup_iterations=0,
        benchmark_iterations=1,
        clock=_benchmark_clock(1),
    )
    comparisons = result.to_dict()["categories"][0]["comparisons"]

    assert comparisons == {
        "cl100k_base": {
            "detail": (
                "(baseline_tokens - project_tokens) / baseline_tokens; "
                "positive means the project tokenizer uses fewer tokens"
            ),
            "relative_token_count_difference": -3.0,
            "status": "measured",
            "tokens": 2,
            "vocab_size": 100_277,
        },
        "gpt2": {
            "detail": (
                "(baseline_tokens - project_tokens) / baseline_tokens; "
                "positive means the project tokenizer uses fewer tokens"
            ),
            "relative_token_count_difference": -1.0,
            "status": "measured",
            "tokens": 4,
            "vocab_size": 50_257,
        },
    }


def test_json_and_markdown_reports_are_atomic_deterministic_and_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpora = collect_evaluation_corpora(
        PARQUET_FIXTURE_DIR,
        max_documents=1,
        max_characters=20,
        document_char_cap=20,
    )
    result = evaluate_tokenizer(
        ByteTokenizer(),
        corpora,
        benchmark_warmup_iterations=0,
        benchmark_iterations=1,
        clock=_benchmark_clock(len(corpora)),
    )
    metrics_dir = tmp_path / "metrics"

    json_path, markdown_path = write_tokenizer_evaluation_reports(
        result,
        metrics_dir,
    )
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    payload = json.loads(first_json)
    markdown = first_markdown.decode("utf-8")

    assert json_path == metrics_dir / "tokenizer_eval.json"
    assert markdown_path == metrics_dir / "tokenizer_eval.md"
    assert payload == result.to_dict()
    assert first_json.endswith(b"\n")
    assert first_markdown.endswith(b"\n")
    assert "| news |" in markdown
    assert (
        f"| **Aggregate** | {payload['aggregate']['bytes']} | "
        f"{payload['aggregate']['tokens']} | 1.000 | pass |"
    ) in markdown
    assert (
        f"Encode: "
        f"{payload['benchmark']['aggregate']['encode']['tokens_per_second']:.3f} "
        "tokens/second"
    ) in markdown
    assert "Warmup iterations: 0" in markdown
    assert "Timed iterations: 1" in markdown

    write_tokenizer_evaluation_reports(result, metrics_dir)
    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown

    real_replace = os.replace
    replace_calls = 0

    def fail_second_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("interrupted report publication")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="interrupted report publication"):
        write_tokenizer_evaluation_reports(result, metrics_dir)

    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
    assert list(metrics_dir.glob(".*.tmp")) == []


def test_eval_tokenizer_command_is_bounded_and_writes_run_reports(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "tokenizer"
    tokenizer = _save_test_tokenizer(artifact_dir)
    config = ProjectConfig(
        run=RunConfig(
            name="tokenizer-eval",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tokenizer=TokenizerConfig(type="regex_byte_bpe", vocab_size=267),
        model=GPTConfig(vocab_size=267),
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.eval_tokenizer",
            "--config",
            str(config_path),
            "--tokenizer-artifacts",
            str(artifact_dir),
            "--data-dir",
            str(PARQUET_FIXTURE_DIR),
            "--num-train-shards",
            "1",
            "--max-documents",
            "2",
            "--max-characters",
            "40",
            "--document-char-cap",
            "30",
            "--batch-size",
            "1",
            "--benchmark-warmup",
            "0",
            "--benchmark-iterations",
            "1",
            "--no-wandb",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / "runs" / "tokenizer-eval"
    json_path = run_dir / "metrics" / "tokenizer_eval.json"
    markdown_path = run_dir / "metrics" / "tokenizer_eval.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "metrics" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    summary = json.loads(
        (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout
    assert f"Tokenizer: {artifact_dir}" in completed.stdout
    assert f"JSON report: {json_path}" in completed.stdout
    assert f"Markdown report: {markdown_path}" in completed.stdout
    assert payload["tokenizer"] == {
        "identity": tokenizer.get_identity(),
        "vocab_size": 267,
    }
    assert [category["name"] for category in payload["categories"]] == [
        "news",
        "korean",
        "code",
        "math",
        "science",
        "climbmix-train",
        "climbmix-validation",
    ]
    assert all(
        comparison["status"] == "skipped"
        for category in payload["categories"]
        for comparison in category["comparisons"].values()
    )
    assert payload["categories"][-2]["source"]["limits"] == {
        "document_char_cap": 30,
        "max_characters": 40,
        "max_documents": 2,
    }
    assert payload["benchmark"]["protocol"] == {
        "clock": "monotonic",
        "denominator": "token IDs processed during timed calls",
        "timed_iterations": 1,
        "warmup_iterations": 0,
    }
    aggregate = payload["aggregate"]
    assert [
        record["metrics"] for record in records if record["record_type"] == "metrics"
    ] == [
        {
            "tokenizer/vocab_size": 267,
            "tokenizer/bytes": aggregate["bytes"],
            "tokenizer/tokens": aggregate["tokens"],
            "tokenizer/bytes_per_token": aggregate["bytes_per_token"],
            "tokenizer/relative_diff_vs_gpt2": None,
            "tokenizer/relative_diff_vs_gpt4": None,
            "tokenizer/roundtrip_pass": True,
            "tokenizer/encode_tokens_per_sec": aggregate["encode_tokens_per_second"],
            "tokenizer/decode_tokens_per_sec": aggregate["decode_tokens_per_second"],
        }
    ]
    assert [
        (record["path"], record["name"], record["type"])
        for record in records
        if record["record_type"] == "artifact"
    ] == [("metrics/tokenizer_eval.json", "tokenizer_eval", "tokenizer")]
    assert markdown_path.is_file()
    assert summary["status"] == "completed"


def test_eval_tokenizer_command_fails_cleanly_before_writing_reports(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(
            name="missing-tokenizer",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tokenizer=TokenizerConfig(type="regex_byte_bpe", vocab_size=267),
        model=GPTConfig(vocab_size=267),
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.eval_tokenizer",
            "--config",
            str(config_path),
            "--tokenizer-artifacts",
            str(tmp_path / "missing"),
            "--data-dir",
            str(PARQUET_FIXTURE_DIR),
            "--no-wandb",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    metrics_dir = tmp_path / "runs" / "missing-tokenizer" / "metrics"
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    tracking_records = [
        json.loads(line)
        for line in (metrics_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert completed.returncode != 0
    assert "tokenizer artifact directory does not exist" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (metrics_dir / "tokenizer_eval.json").exists()
    assert not (metrics_dir / "tokenizer_eval.md").exists()
    assert [record["record_type"] for record in tracking_records] == ["config"]
    assert summary["status"] == "failed"


def test_readme_documents_tokenizer_evaluation_protocol_and_optional_extra() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "uv sync --extra tokenizer-comparison" in readme
    assert "python -m scripts.eval_tokenizer" in readme
    assert "--tokenizer-artifacts" in readme
    assert "--compare" in readme
    assert "tokenizer_eval.json" in readme
    assert "tokenizer_eval.md" in readme
    assert "monotonic clock" in readme
    assert "warmup" in readme
    assert "token IDs processed during timed calls" in readme
    assert "[project.optional-dependencies]" in pyproject
    assert 'tokenizer-comparison = ["tiktoken"]' in pyproject
