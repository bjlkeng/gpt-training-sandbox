"""Tests for the tiny text fixture and next-token dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scratch_llm.data import NextTokenDataset
from scratch_llm.tokenizer import ByteTokenizer, VOCAB_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_TEXT_PATH = PROJECT_ROOT / "data" / "fixtures" / "tiny.txt"
FIXTURE_README_PATH = PROJECT_ROOT / "data" / "fixtures" / "README.md"


def test_tiny_text_fixture_is_nonempty_valid_utf8_with_provenance() -> None:
    fixture_bytes = TINY_TEXT_PATH.read_bytes()
    fixture_text = fixture_bytes.decode("utf-8", errors="strict")
    provenance = FIXTURE_README_PATH.read_text(encoding="utf-8")

    assert fixture_text.strip()
    assert any(ord(character) > 127 for character in fixture_text)
    assert "tiny.txt" in provenance
    assert "synthetic" in provenance.lower()
    assert "authored for this repository" in provenance.lower()


def test_next_token_sample_is_a_shifted_pair_of_long_tensors() -> None:
    token_ids = ByteTokenizer().encode("AéB!")
    dataset = NextTokenDataset(token_ids, seq_len=3)

    inputs, targets = dataset[0]

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert inputs.shape == targets.shape == (3,)
    assert inputs.tolist() == token_ids[:3]
    assert targets.tolist() == token_ids[1:4]
    assert torch.equal(targets[:-1], inputs[1:])
    assert torch.all((0 <= inputs) & (inputs < VOCAB_SIZE))
    assert torch.all((0 <= targets) & (targets < VOCAB_SIZE))


def test_dataset_length_and_boundary_windows_cover_each_valid_start() -> None:
    dataset = NextTokenDataset([10, 11, 12, 13, 14], seq_len=2)

    assert len(dataset) == 3
    assert tuple(tensor.tolist() for tensor in dataset[0]) == ([10, 11], [11, 12])
    assert tuple(tensor.tolist() for tensor in dataset[-1]) == (
        [12, 13],
        [13, 14],
    )

    with pytest.raises(IndexError, match="dataset index out of range"):
        dataset[len(dataset)]

    with pytest.raises(IndexError, match="dataset index out of range"):
        dataset[-len(dataset) - 1]


def test_token_stream_without_a_complete_shifted_window_is_empty() -> None:
    dataset = NextTokenDataset([10, 11, 12], seq_len=3)

    assert len(dataset) == 0
    with pytest.raises(IndexError, match="dataset index out of range"):
        dataset[0]


def test_dataloader_stacks_deterministic_shifted_batches() -> None:
    text = TINY_TEXT_PATH.read_text(encoding="utf-8")
    token_ids = ByteTokenizer().encode(text)
    seq_len = 8
    dataset = NextTokenDataset(token_ids, seq_len=seq_len)

    inputs, targets = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    assert inputs.shape == targets.shape == (4, seq_len)
    assert inputs.dtype == targets.dtype == torch.long
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert inputs[0].tolist() == token_ids[:seq_len]
    assert inputs[1].tolist() == token_ids[1 : seq_len + 1]
    assert torch.all((0 <= inputs) & (inputs < VOCAB_SIZE))
    assert torch.all((0 <= targets) & (targets < VOCAB_SIZE))


@pytest.mark.parametrize("token_ids", [[-1], [VOCAB_SIZE], [0, VOCAB_SIZE + 1]])
def test_dataset_rejects_out_of_vocabulary_ids(token_ids: list[int]) -> None:
    with pytest.raises(ValueError, match=r"token ID.*range \[0, 265\)"):
        NextTokenDataset(token_ids, seq_len=1)


@pytest.mark.parametrize("token_ids", [[True], [1.5], ["1"]])
def test_dataset_rejects_non_integer_ids(token_ids: list[object]) -> None:
    with pytest.raises(TypeError, match=r"token ID.*must be an integer"):
        NextTokenDataset(token_ids, seq_len=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("seq_len", [0, -1, True, 1.5])
def test_dataset_requires_a_positive_integer_sequence_length(
    seq_len: object,
) -> None:
    error_type = TypeError if isinstance(seq_len, (bool, float)) else ValueError

    with pytest.raises(error_type, match="seq_len"):
        NextTokenDataset([0, 1], seq_len=seq_len)  # type: ignore[arg-type]


def test_dataset_validates_ids_against_an_explicit_vocabulary_size() -> None:
    with pytest.raises(ValueError, match=r"token ID.*range \[0, 4\)"):
        NextTokenDataset([0, 1, 4], seq_len=1, vocab_size=4)


@pytest.mark.parametrize("vocab_size", [0, -1, True, 1.5])
def test_dataset_requires_a_positive_integer_vocabulary_size(
    vocab_size: object,
) -> None:
    error_type = TypeError if isinstance(vocab_size, (bool, float)) else ValueError

    with pytest.raises(error_type, match="vocab_size"):
        NextTokenDataset(
            [0, 1],
            seq_len=1,
            vocab_size=vocab_size,  # type: ignore[arg-type]
        )
