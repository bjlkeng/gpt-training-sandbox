"""Tests for restartable random-offset batches over tokenized memmaps."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch

from scratch_llm.data import loaders as data
from scratch_llm.data.loaders import (
    NextTokenDataset,
    RandomOffsetTokenLoader,
    RandomOffsetTokenLoaderStateError,
    TokenizedShardReader,
    TokenizedShardSource,
    TokenizedDataError,
    write_tokenized_shards,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer, VOCAB_SIZE
from scratch_llm.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(
    path: Path,
    *,
    train_texts: tuple[str, ...] = ("ABCDE", "uvwxyz"),
    val_texts: tuple[str, ...] = ("validation",),
) -> None:
    write_tokenized_shards(
        path,
        tokenizer=ByteTokenizer(),
        train_sources=[
            TokenizedShardSource(f"train-{index}", [text])
            for index, text in enumerate(train_texts)
        ],
        val_sources=[
            TokenizedShardSource(f"val-{index}", [text])
            for index, text in enumerate(val_texts)
        ],
    )


def _batch_windows(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> list[tuple[int, ...]]:
    return [
        tuple([*input_row.tolist(), target_row[-1].item()])
        for input_row, target_row in zip(inputs, targets, strict=True)
    ]


def _batches_equal(
    first: tuple[torch.Tensor, torch.Tensor],
    second: tuple[torch.Tensor, torch.Tensor],
) -> bool:
    return torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_random_loader_returns_valid_shifted_long_batches_without_crossing_shards(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    _write_dataset(dataset_dir)
    tokenizer = ByteTokenizer()

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = RandomOffsetTokenLoader(
            reader,
            split="train",
            batch_size=32,
            seq_len=2,
            seed=17,
        )
        inputs, targets = next(loader)

        assert inputs.shape == targets.shape == (32, 2)
        assert inputs.dtype == targets.dtype == torch.long
        assert torch.equal(targets[:, :-1], inputs[:, 1:])
        assert torch.all((0 <= inputs) & (inputs < VOCAB_SIZE))
        assert torch.all((0 <= targets) & (targets < VOCAB_SIZE))
        assert loader.valid_start_count == 7
        assert loader.position == 32

        legal_windows = {
            tuple(encoded[start : start + 3])
            for text in ("ABCDE", "uvwxyz")
            for encoded in [tokenizer.encode(text)]
            for start in range(len(encoded) - 2)
        }
        assert set(_batch_windows(inputs, targets)) <= legal_windows
        assert all(
            held is mapped
            for held, mapped in zip(
                loader._mapped_shards,  # noqa: SLF001 - verifies no corpus copy
                reader.shards("train"),
                strict=True,
            )
        )
        assert all(isinstance(shard, np.memmap) for shard in loader._mapped_shards)


def test_global_offset_mapping_covers_every_valid_start_and_shard_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    _write_dataset(dataset_dir)

    def enumerate_offsets(
        low: int,
        high: int,
        size: tuple[int, ...],
        *,
        generator: torch.Generator,
        dtype: torch.dtype,
        device: str,
    ) -> torch.Tensor:
        assert low == 0
        assert high == 7
        assert size == (7,)
        assert isinstance(generator, torch.Generator)
        assert dtype == torch.int64
        assert device == "cpu"
        return torch.arange(7, dtype=torch.int64)

    monkeypatch.setattr(data.torch, "randint", enumerate_offsets)

    with TokenizedShardReader(
        dataset_dir,
        tokenizer=ByteTokenizer(),
    ) as reader:
        loader = RandomOffsetTokenLoader(
            reader,
            split="train",
            batch_size=7,
            seq_len=2,
            seed=0,
        )
        inputs, targets = loader.next_batch()

    assert _batch_windows(inputs, targets) == [
        tuple(b"ABC"),
        tuple(b"BCD"),
        tuple(b"CDE"),
        tuple(b"uvw"),
        tuple(b"vwx"),
        tuple(b"wxy"),
        tuple(b"xyz"),
    ]


def test_seeded_batches_repeat_and_different_seeds_vary(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "tokenized"
    _write_dataset(dataset_dir)
    tokenizer = ByteTokenizer()

    with (
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as first_reader,
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as second_reader,
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as third_reader,
    ):
        first = RandomOffsetTokenLoader(
            first_reader,
            split="train",
            batch_size=8,
            seq_len=2,
            seed=123,
        )
        second = RandomOffsetTokenLoader(
            second_reader,
            split="train",
            batch_size=8,
            seq_len=2,
            seed=123,
        )
        third = RandomOffsetTokenLoader(
            third_reader,
            split="train",
            batch_size=8,
            seq_len=2,
            seed=124,
        )

        first_batches = [next(first) for _ in range(4)]
        second_batches = [next(second) for _ in range(4)]
        third_batches = [next(third) for _ in range(4)]

    assert all(
        _batches_equal(first_batch, second_batch)
        for first_batch, second_batch in zip(
            first_batches,
            second_batches,
            strict=True,
        )
    )
    assert any(
        not _batches_equal(first_batch, third_batch)
        for first_batch, third_batch in zip(
            first_batches,
            third_batches,
            strict=True,
        )
    )


def test_state_dict_resumes_the_exact_next_batches_in_a_fresh_reader(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    _write_dataset(dataset_dir)
    tokenizer = ByteTokenizer()

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as source_reader:
        source = RandomOffsetTokenLoader(
            source_reader,
            split="train",
            batch_size=5,
            seq_len=3,
            seed=89,
        )
        next(source)
        next(source)
        state = source.state_dict()
        expected = [next(source) for _ in range(3)]

    assert state.keys() == {
        "batch_size",
        "format",
        "format_version",
        "manifest_identity",
        "position",
        "rng_state",
        "seq_len",
        "split",
    }
    assert state["format"] == "scratch_llm_random_offset_loader_state"
    assert state["format_version"] == 1
    assert isinstance(state["manifest_identity"], str)
    assert str(state["manifest_identity"]).startswith("sha256:")
    assert state["split"] == "train"
    assert state["batch_size"] == 5
    assert state["seq_len"] == 3
    assert state["position"] == 10
    assert isinstance(state["rng_state"], list)
    assert state["rng_state"]

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as resumed_reader:
        resumed = RandomOffsetTokenLoader(
            resumed_reader,
            split="train",
            batch_size=5,
            seq_len=3,
            seed=999,
        )
        resumed.load_state_dict(state)
        actual = [next(resumed) for _ in range(3)]

    assert all(
        _batches_equal(expected_batch, actual_batch)
        for expected_batch, actual_batch in zip(expected, actual, strict=True)
    )
    assert resumed.position == 25


def test_json_state_resumes_the_exact_next_batch_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    state_path = tmp_path / "loader_state.json"
    _write_dataset(dataset_dir)

    with TokenizedShardReader(
        dataset_dir,
        tokenizer=ByteTokenizer(),
    ) as reader:
        loader = RandomOffsetTokenLoader(
            reader,
            split="train",
            batch_size=3,
            seq_len=2,
            seed=55,
        )
        next(loader)
        save_json(loader.state_dict(), state_path)
        expected_inputs, expected_targets = next(loader)
        expected_position = loader.position

    process_code = """
import json
from pathlib import Path
import sys
from scratch_llm.data.loaders import RandomOffsetTokenLoader, TokenizedShardReader
from scratch_llm.tokenization.tokenizer import ByteTokenizer

dataset_dir = Path(sys.argv[1])
state = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
with TokenizedShardReader(dataset_dir, tokenizer=ByteTokenizer()) as reader:
    loader = RandomOffsetTokenLoader(
        reader,
        split="train",
        batch_size=3,
        seq_len=2,
        seed=999,
    )
    loader.load_state_dict(state)
    inputs, targets = next(loader)
    print(json.dumps(
        {
            "inputs": inputs.tolist(),
            "position": loader.position,
            "targets": targets.tolist(),
        },
        sort_keys=True,
    ))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            process_code,
            str(dataset_dir),
            str(state_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "inputs": expected_inputs.tolist(),
        "position": expected_position,
        "targets": expected_targets.tolist(),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.__setitem__("format_version", 2), "format version"),
        (lambda state: state.__setitem__("split", "val"), "split"),
        (lambda state: state.__setitem__("split", 1), "split"),
        (lambda state: state.__setitem__("batch_size", 99), "batch_size"),
        (lambda state: state.__setitem__("batch_size", True), "batch_size"),
        (lambda state: state.__setitem__("seq_len", 99), "seq_len"),
        (lambda state: state.__setitem__("seq_len", True), "seq_len"),
        (lambda state: state.__setitem__("position", -1), "position"),
        (lambda state: state.__setitem__("position", 1), "position"),
        (lambda state: state.__setitem__("rng_state", []), "rng_state"),
        (lambda state: state.__setitem__("rng_state", [False]), "rng_state"),
        (lambda state: state.pop("position"), "fields"),
        (lambda state: state.__setitem__("unexpected", True), "fields"),
    ],
)
def test_malformed_or_incompatible_state_fails_without_mutating_the_loader(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    _write_dataset(dataset_dir)
    tokenizer = ByteTokenizer()

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = RandomOffsetTokenLoader(
            reader,
            split="train",
            batch_size=4,
            seq_len=2,
            seed=5,
        )
        invalid_state = deepcopy(loader.state_dict())
        mutation(invalid_state)
        original_state = loader.state_dict()

        with pytest.raises(RandomOffsetTokenLoaderStateError, match=message):
            loader.load_state_dict(invalid_state)

        assert loader.state_dict() == original_state


def test_state_from_a_changed_manifest_is_rejected_as_stale(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_dataset(first_dir, train_texts=("ABCDE",))
    _write_dataset(second_dir, train_texts=("ABCDF",))
    tokenizer = ByteTokenizer()

    with TokenizedShardReader(first_dir, tokenizer=tokenizer) as first_reader:
        first = RandomOffsetTokenLoader(
            first_reader,
            split="train",
            batch_size=2,
            seq_len=2,
            seed=1,
        )
        state = first.state_dict()

    with TokenizedShardReader(second_dir, tokenizer=tokenizer) as second_reader:
        second = RandomOffsetTokenLoader(
            second_reader,
            split="train",
            batch_size=2,
            seq_len=2,
            seed=1,
        )
        with pytest.raises(
            RandomOffsetTokenLoaderStateError,
            match="manifest identity",
        ):
            second.load_state_dict(state)


def test_short_shards_are_skipped_and_an_all_short_split_fails_actionably(
    tmp_path: Path,
) -> None:
    mixed_dir = tmp_path / "mixed"
    _write_dataset(mixed_dir, train_texts=("A", "ABCDE"))

    with TokenizedShardReader(
        mixed_dir,
        tokenizer=ByteTokenizer(),
    ) as mixed_reader:
        loader = RandomOffsetTokenLoader(
            mixed_reader,
            split="train",
            batch_size=16,
            seq_len=2,
            seed=7,
        )
        inputs, targets = next(loader)
        assert set(_batch_windows(inputs, targets)) <= {
            tuple(b"ABC"),
            tuple(b"BCD"),
            tuple(b"CDE"),
        }
        assert loader.valid_start_count == 3

    short_dir = tmp_path / "short"
    _write_dataset(short_dir, train_texts=("A", "BC"))
    with TokenizedShardReader(
        short_dir,
        tokenizer=ByteTokenizer(),
    ) as short_reader:
        with pytest.raises(
            ValueError,
            match=r"train.*no complete windows.*3 tokens.*lengths=\[1, 2\]",
        ):
            RandomOffsetTokenLoader(
                short_reader,
                split="train",
                batch_size=1,
                seq_len=2,
                seed=0,
            )


def test_empty_token_shards_fail_validation_before_loader_construction(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TokenizedDataError,
        match=r"train source 'empty'.*did not yield any tokens",
    ):
        write_tokenized_shards(
            tmp_path / "empty",
            tokenizer=ByteTokenizer(),
            train_sources=[TokenizedShardSource("empty", [""])],
            val_sources=[TokenizedShardSource("val", ["valid"])],
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("split", "validation"),
        ("batch_size", 0),
        ("batch_size", True),
        ("seq_len", -1),
        ("seq_len", 1.5),
        ("seed", -1),
        ("seed", "1"),
    ],
)
def test_loader_rejects_invalid_construction_settings(
    tmp_path: Path,
    argument: str,
    value: object,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    _write_dataset(dataset_dir)

    with TokenizedShardReader(
        dataset_dir,
        tokenizer=ByteTokenizer(),
    ) as reader:
        settings: dict[str, Any] = {
            "split": "train",
            "batch_size": 2,
            "seq_len": 2,
            "seed": 0,
        }
        settings[argument] = value
        with pytest.raises((TypeError, ValueError), match=argument):
            RandomOffsetTokenLoader(reader, **settings)


def test_closed_reader_fails_before_loader_construction(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "tokenized"
    _write_dataset(dataset_dir)
    reader = TokenizedShardReader(dataset_dir, tokenizer=ByteTokenizer())
    reader.close()

    with pytest.raises(RuntimeError, match="reader is closed"):
        RandomOffsetTokenLoader(
            reader,
            split="train",
            batch_size=2,
            seq_len=2,
            seed=0,
        )


def test_next_token_dataset_remains_available_for_the_tiny_text_path() -> None:
    dataset = NextTokenDataset([10, 11, 12, 13], seq_len=2)

    assert len(dataset) == 2
    inputs, targets = dataset[1]
    assert inputs.tolist() == [11, 12]
    assert targets.tolist() == [12, 13]


def test_readme_documents_random_loader_resume_and_memory_contract() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "RandomOffsetTokenLoader" in readme
    assert "`state_dict()`" in readme
    assert "`load_state_dict()`" in readme
    assert "never concatenates the token" in readme
    assert "only the sampled windows are copied" in readme
