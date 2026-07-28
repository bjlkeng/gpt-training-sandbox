"""Differential tests for the incremental regex byte-BPE trainer."""

from __future__ import annotations

from collections.abc import Callable
import random
from pathlib import Path
import subprocess
import sys

import json
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from scratch_llm.bpe import (
    BPETrainingError,
    ReferenceBPETrainingResult,
    RegexBPETokenizer,
    apply_merge,
    count_pairs,
    select_best_pair,
    train_bpe,
    train_reference_bpe,
)
from scratch_llm.bpe_optimized import (
    BPETrainingBenchmark,
    IncrementalPairIndex,
    benchmark_bpe_trainers,
    train_optimized_bpe,
    write_bpe_training_benchmark,
)
from scratch_llm.config import (
    DataConfig,
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
)
from scratch_llm.tokenizer_artifacts import TOKENIZER_ARTIFACT_FILENAMES
from scratch_llm.tokenizer_training import TOKENIZER_TRAINING_REPORT_FILENAME


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"


def test_incremental_index_matches_full_recounts_after_every_merge() -> None:
    expected_chunks: tuple[tuple[int, ...], ...] = (
        (1, 1, 1, 2),
        (1, 2, 1, 2),
        (),
        (3, 1, 1, 3),
    )
    index = IncrementalPairIndex(expected_chunks)

    for new_token_id in range(10, 18):
        expected_counts = count_pairs(expected_chunks)
        assert index.pair_counts() == expected_counts
        if not expected_counts:
            break
        expected_pair = select_best_pair(expected_counts)
        assert index.best_pair() == expected_pair

        selected_count = index.merge_pair(expected_pair, new_token_id)
        expected_chunks = apply_merge(
            expected_chunks,
            expected_pair,
            new_token_id,
        )

        assert selected_count == expected_counts[expected_pair]
        assert index.chunks() == expected_chunks
        assert index.pair_counts() == count_pairs(expected_chunks)


def test_incremental_index_removes_overlap_and_stale_heap_entries() -> None:
    index = IncrementalPairIndex(((1, 1, 1),))

    assert index.pair_counts() == {(1, 1): 2}
    assert index.merge_pair((1, 1), 5) == 2
    assert index.chunks() == ((5, 1),)
    assert index.pair_counts() == {(5, 1): 1}
    assert index.best_pair() == (5, 1)
    assert index.merge_pair((5, 1), 6) == 1
    assert index.pair_counts() == {}
    assert index.chunks() == ((6,),)


def test_randomized_incremental_updates_match_reference_recounts() -> None:
    for seed in range(20):
        randomizer = random.Random(seed)
        expected_chunks = tuple(
            tuple(randomizer.randrange(6) for _ in range(randomizer.randrange(9)))
            for _ in range(randomizer.randrange(1, 8))
        )
        index = IncrementalPairIndex(expected_chunks)

        for new_token_id in range(20, 35):
            expected_counts = count_pairs(expected_chunks)
            assert index.pair_counts() == expected_counts
            if not expected_counts:
                break
            expected_pair = select_best_pair(expected_counts)
            assert index.best_pair() == expected_pair
            assert (
                index.merge_pair(expected_pair, new_token_id)
                == expected_counts[expected_pair]
            )
            expected_chunks = apply_merge(
                expected_chunks,
                expected_pair,
                new_token_id,
            )
            assert index.chunks() == expected_chunks
            assert index.pair_counts() == count_pairs(expected_chunks)


@pytest.mark.parametrize(
    "texts",
    [
        ("abab abab", "baba"),
        ("aaaaa", "aaa", "aa"),
        ("tie one", "tie two", "one two"),
        ("반복 반복 반복", "emoji 🚀🚀 and café"),
        ("same chunk",) * 5,
    ],
)
def test_optimized_training_is_exactly_equivalent_to_reference(
    texts: tuple[str, ...],
) -> None:
    reference = train_reference_bpe(texts, vocab_size=268)
    optimized = train_optimized_bpe(texts, vocab_size=268)

    assert optimized == reference
    assert optimized.merges == reference.merges
    assert dict(optimized.vocabulary) == dict(reference.vocabulary)
    assert dict(optimized.special_token_ids) == dict(reference.special_token_ids)

    reference_tokenizer = RegexBPETokenizer(reference)
    optimized_tokenizer = RegexBPETokenizer(optimized)
    assert optimized_tokenizer.get_identity() == reference_tokenizer.get_identity()
    for text in (*texts, "held-out Unicode: Καλημέρα 🌍"):
        assert optimized_tokenizer.encode(text) == reference_tokenizer.encode(text)
        assert optimized_tokenizer.decode(optimized_tokenizer.encode(text)) == text


def test_optimized_training_preserves_caps_one_pass_and_failure_semantics() -> None:
    consumed: list[str] = []

    def texts() -> object:
        for text in ("aaaa", "bbbb", "must not be consumed"):
            consumed.append(text)
            yield text

    optimized = train_optimized_bpe(
        texts(),  # type: ignore[arg-type]
        vocab_size=267,
        max_documents=2,
        max_characters=6,
    )
    reference = train_reference_bpe(
        ("aaaa", "bb"),
        vocab_size=267,
    )

    assert optimized == reference
    assert consumed == ["aaaa", "bbbb"]

    with pytest.raises(BPETrainingError, match="exhausted all adjacent pairs"):
        train_optimized_bpe(("a",), vocab_size=267)


def test_optimized_and_reference_artifacts_are_equivalent(tmp_path: Path) -> None:
    texts = ("artifact artifact artifact", "한국어와 emoji 🚀")
    reference_tokenizer = RegexBPETokenizer(train_reference_bpe(texts, vocab_size=271))
    optimized_tokenizer = RegexBPETokenizer(train_optimized_bpe(texts, vocab_size=271))
    reference_path = tmp_path / "reference"
    optimized_path = tmp_path / "optimized"

    reference_tokenizer.save(reference_path)
    optimized_tokenizer.save(optimized_path)

    for filename in TOKENIZER_ARTIFACT_FILENAMES[:-1]:
        assert (optimized_path / filename).read_bytes() == (
            reference_path / filename
        ).read_bytes()
    assert torch.equal(
        torch.load(
            optimized_path / "token_bytes.pt",
            map_location="cpu",
            weights_only=True,
        ),
        torch.load(
            reference_path / "token_bytes.pt",
            map_location="cpu",
            weights_only=True,
        ),
    )


def test_training_dispatcher_defaults_to_optimized_with_reference_fallback() -> None:
    texts = ("dispatcher dispatcher", "fallback")

    assert train_bpe(texts, vocab_size=269) == train_optimized_bpe(
        texts,
        vocab_size=269,
    )
    assert train_bpe(
        texts,
        vocab_size=269,
        algorithm="reference",
    ) == train_reference_bpe(texts, vocab_size=269)
    with pytest.raises(ValueError, match="algorithm"):
        train_bpe(texts, vocab_size=269, algorithm="native")


def test_bounded_benchmark_records_time_memory_and_exact_equivalence(
    tmp_path: Path,
) -> None:
    measurements = {
        "reference": (2.5, 12_000),
        "optimized": (1.25, 8_000),
    }

    def measure(
        algorithm: str,
        train: Callable[[], ReferenceBPETrainingResult],
    ) -> tuple[ReferenceBPETrainingResult, float, int]:
        result = train()
        seconds, peak_memory_bytes = measurements[algorithm]
        return result, seconds, peak_memory_bytes

    benchmark = benchmark_bpe_trainers(
        ("benchmark benchmark", "한국어 fixture", "not consumed"),
        vocab_size=269,
        max_documents=2,
        max_characters=30,
        measure=measure,
    )
    payload = benchmark.to_dict()

    assert isinstance(benchmark, BPETrainingBenchmark)
    assert payload == {
        "equivalent": True,
        "format": "scratch_llm_bpe_training_benchmark",
        "format_version": 1,
        "input": {
            "configured_max_characters": 30,
            "configured_max_documents": 2,
            "input_document_count": 3,
            "vocab_size": 269,
        },
        "measurements": [
            {
                "algorithm": "reference",
                "character_count": 30,
                "document_count": 2,
                "elapsed_seconds": 2.5,
                "merge_count": 4,
                "peak_memory_bytes": 12_000,
                "tokenizer_identity": payload["measurements"][0]["tokenizer_identity"],
            },
            {
                "algorithm": "optimized",
                "character_count": 30,
                "document_count": 2,
                "elapsed_seconds": 1.25,
                "merge_count": 4,
                "peak_memory_bytes": 8_000,
                "tokenizer_identity": payload["measurements"][0]["tokenizer_identity"],
            },
        ],
        "protocol": {
            "clock": "monotonic",
            "memory": "Python peak allocations measured independently with tracemalloc",
            "performance_assertion": "record values without a wall-clock ratio threshold",
        },
    }

    destination = tmp_path / "metrics" / "bpe_training_benchmark.json"
    assert write_bpe_training_benchmark(benchmark, destination) == destination
    first = destination.read_bytes()
    assert json.loads(first) == payload
    assert write_bpe_training_benchmark(benchmark, destination).read_bytes() == first


def test_real_bounded_benchmark_runs_without_a_performance_threshold() -> None:
    benchmark = benchmark_bpe_trainers(
        ("small benchmark corpus " * 4,),
        vocab_size=269,
    )

    assert benchmark.equivalent is True
    for measurement in benchmark.measurements:
        assert measurement.elapsed_seconds >= 0.0
        assert measurement.peak_memory_bytes > 0


def test_train_tokenizer_command_uses_optimized_path_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(
            name="optimized-tokenizer",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        data=DataConfig(
            profile="nanochat_climbmix",
            parquet_dir=str(PARQUET_FIXTURE_DIR),
            num_tokenizer_train_shards=1,
            doc_cap_chars=100,
        ),
        tokenizer=TokenizerConfig(
            type="regex_byte_bpe",
            vocab_size=270,
            max_chars=200,
            doc_cap=6,
        ),
        model=GPTConfig(vocab_size=270),
    )
    config_path = tmp_path / "train-tokenizer.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_tokenizer",
            "--config",
            str(config_path),
            "--algorithm",
            "optimized",
            "--benchmark-trainers",
            "--benchmark-vocab-size",
            "268",
            "--benchmark-max-documents",
            "3",
            "--benchmark-max-characters",
            "200",
            "--no-wandb",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / "runs" / "optimized-tokenizer"
    artifact_dir = run_dir / "artifacts" / "tokenizer"
    report_path = run_dir / "metrics" / TOKENIZER_TRAINING_REPORT_FILENAME
    benchmark_path = run_dir / "metrics" / "bpe_training_benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tokenizer = RegexBPETokenizer.load(artifact_dir)
    summary = json.loads(
        (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout
    assert f"Tokenizer artifacts: {artifact_dir}" in completed.stdout
    assert f"Training report: {report_path}" in completed.stdout
    assert f"Trainer benchmark: {benchmark_path}" in completed.stdout
    assert tokenizer.get_vocab_size() == 270
    assert report["algorithm"] == "optimized"
    assert report["vocab_size"] == 270
    assert report["merge_count"] == 5
    assert report["tokenizer_identity"] == tokenizer.get_identity()
    assert report["artifacts"] == {
        "directory": "artifacts/tokenizer",
        "files": list(TOKENIZER_ARTIFACT_FILENAMES),
    }
    assert report["corpus"]["selected_shards"] == ["shard_00000.parquet"]
    assert report["corpus"]["configured_max_characters"] == 200
    assert report["corpus"]["configured_max_documents"] == 6
    assert benchmark["equivalent"] is True
    assert [item["algorithm"] for item in benchmark["measurements"]] == [
        "reference",
        "optimized",
    ]
    assert summary["status"] == "completed"


def test_train_tokenizer_command_fails_without_partial_artifacts(
    tmp_path: Path,
) -> None:
    config = ProjectConfig(
        run=RunConfig(
            name="exhausted-tokenizer",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        data=DataConfig(
            profile="nanochat_climbmix",
            parquet_dir=str(PARQUET_FIXTURE_DIR),
            num_tokenizer_train_shards=1,
            doc_cap_chars=1,
        ),
        tokenizer=TokenizerConfig(
            type="regex_byte_bpe",
            vocab_size=300,
            max_chars=1,
            doc_cap=1,
        ),
        model=GPTConfig(vocab_size=300),
    )
    config_path = tmp_path / "train-tokenizer.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_tokenizer",
            "--config",
            str(config_path),
            "--algorithm",
            "optimized",
            "--no-wandb",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / "runs" / "exhausted-tokenizer"
    summary = json.loads(
        (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert completed.returncode != 0
    assert "exhausted all adjacent pairs" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (run_dir / "artifacts" / "tokenizer").exists()
    assert not (run_dir / "metrics" / TOKENIZER_TRAINING_REPORT_FILENAME).exists()
    assert summary["status"] == "failed"


def test_real_train_tokenizer_path_reaches_32768_on_bounded_parquet(
    tmp_path: Path,
) -> None:
    randomizer = random.Random(20_260_728)
    representative_text = "".join(
        randomizer.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(100_000)
    )
    data_dir = tmp_path / "parquet"
    data_dir.mkdir()
    pq.write_table(
        pa.table({"text": [representative_text]}),
        data_dir / "shard_00000.parquet",
    )
    config = ProjectConfig(
        run=RunConfig(
            name="tokenizer-32k",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        data=DataConfig(
            profile="nanochat_climbmix",
            parquet_dir=str(data_dir),
            num_tokenizer_train_shards=1,
            doc_cap_chars=100_000,
        ),
        tokenizer=TokenizerConfig(
            type="regex_byte_bpe",
            vocab_size=32_768,
            max_chars=100_000,
            doc_cap=1,
        ),
        model=GPTConfig(vocab_size=32_768),
    )
    config_path = tmp_path / "tokenizer-32k.yaml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_tokenizer",
            "--config",
            str(config_path),
            "--algorithm",
            "optimized",
            "--no-wandb",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    run_dir = tmp_path / "runs" / "tokenizer-32k"
    artifact_dir = run_dir / "artifacts" / "tokenizer"
    report = json.loads(
        (run_dir / "metrics" / TOKENIZER_TRAINING_REPORT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    tokenizer = RegexBPETokenizer.load(artifact_dir)
    assert completed.returncode == 0, completed.stderr
    assert tokenizer.get_vocab_size() == 32_768
    assert report["vocab_size"] == 32_768
    assert report["merge_count"] == 32_503
    assert report["corpus"]["document_count"] == 1
    assert report["corpus"]["character_count"] == 100_000


def test_readme_documents_optimized_training_and_32k_expectations() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m scripts.train_tokenizer" in readme
    assert "--algorithm optimized" in readme
    assert "--algorithm reference" in readme
    assert "--benchmark-trainers" in readme
    assert "bpe_training_benchmark.json" in readme
    assert "tokenizer_training.json" in readme
    assert "32,768" in readme
    assert "CPU" in readme
    assert "RAM" in readme
    assert "one process" in readme
    assert "incremental" in readme
