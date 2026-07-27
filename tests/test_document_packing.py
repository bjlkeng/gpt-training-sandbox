"""Tests for BOS-aware document-boundary packing."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from scratch_llm.data import (
    DocumentPackingTokenLoader,
    DocumentPackingTokenLoaderStateError,
    RandomOffsetTokenLoader,
    TokenizedDataError,
    TokenizedShardReader,
    TokenizedShardSource,
    create_token_loader,
    write_tokenized_shards,
)
from scratch_llm.tokenizer import ByteTokenizer
from scratch_llm.utils import save_json


def _write_dataset(
    path: Path,
    *,
    train_sources: tuple[tuple[str, ...], ...],
    val_sources: tuple[tuple[str, ...], ...] = (("validation",),),
) -> ByteTokenizer:
    tokenizer = ByteTokenizer()
    write_tokenized_shards(
        path,
        tokenizer=tokenizer,
        train_sources=[
            TokenizedShardSource(f"train-{index}", documents)
            for index, documents in enumerate(train_sources)
        ],
        val_sources=[
            TokenizedShardSource(f"val-{index}", documents)
            for index, documents in enumerate(val_sources)
        ],
    )
    return tokenizer


def _epoch_batches(
    loader: DocumentPackingTokenLoader,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch_count = math.ceil(loader.packed_example_count / loader.batch_size)
    return [next(loader) for _ in range(batch_count)]


def _batches_equal(
    first: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    second: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    return all(torch.equal(left, right) for left, right in zip(first, second))


def test_packing_marks_only_within_document_content_targets(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(
        dataset_dir,
        train_sources=(("AB", "C"),),
    )

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = DocumentPackingTokenLoader(
            reader,
            split="train",
            batch_size=1,
            seq_len=4,
            seed=0,
        )
        inputs, targets, loss_mask = next(loader)

    bos = tokenizer.get_bos_token_id()
    assert inputs.shape == targets.shape == loss_mask.shape == (1, 4)
    assert inputs.dtype == targets.dtype == torch.long
    assert loss_mask.dtype == torch.bool
    assert inputs.tolist() == [[bos, ord("A"), ord("B"), bos]]
    assert targets.tolist() == [[ord("A"), ord("B"), bos, ord("C")]]
    assert loss_mask.tolist() == [[True, True, False, True]]


def test_reader_exposes_validated_boundaries_without_changing_flat_tokens(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(
        dataset_dir,
        train_sources=(("A", "", "é"), ("BC",)),
    )

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        assert [
            (
                span.shard_index,
                span.document_index,
                span.start,
                span.stop,
                span.token_count,
            )
            for span in reader.document_spans("train")
        ] == [
            (0, 0, 0, 1, 1),
            (0, 1, 1, 1, 0),
            (0, 2, 1, 3, 2),
            (1, 0, 0, 2, 2),
        ]
        assert [shard.tolist() for shard in reader.shards("train")] == [
            tokenizer.encode("Aé"),
            tokenizer.encode("BC"),
        ]


def test_packing_covers_empty_unicode_exact_fit_and_residual_space(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    documents = ("AB", "", "é", "C")
    tokenizer = _write_dataset(dataset_dir, train_sources=(documents,))

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = DocumentPackingTokenLoader(
            reader,
            split="train",
            batch_size=2,
            seq_len=4,
            seed=11,
        )
        batches = _epoch_batches(loader)

    assert loader.packed_example_count == 2
    inputs = torch.cat([batch[0] for batch in batches])
    targets = torch.cat([batch[1] for batch in batches])
    loss_mask = torch.cat([batch[2] for batch in batches])
    expected_content = sorted(
        token_id for document in documents for token_id in tokenizer.encode(document)
    )
    bos = tokenizer.get_bos_token_id()

    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert sorted(targets[loss_mask].tolist()) == expected_content
    assert torch.all(targets[~loss_mask] == bos)
    assert (~loss_mask).any()
    assert "<|pad|>" not in tokenizer.get_special_tokens()


def test_oversized_documents_are_split_without_losing_or_duplicating_content(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    document = "abcdefghijk"
    tokenizer = _write_dataset(dataset_dir, train_sources=((document,),))

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = DocumentPackingTokenLoader(
            reader,
            split="train",
            batch_size=2,
            seq_len=4,
            seed=3,
        )
        batches = _epoch_batches(loader)

    assert loader.packed_example_count == 3
    targets = torch.cat([batch[1] for batch in batches])
    loss_mask = torch.cat([batch[2] for batch in batches])
    bos = tokenizer.get_bos_token_id()

    assert targets[loss_mask].tolist() == tokenizer.encode(document)
    assert torch.all(targets[~loss_mask] == bos)
    assert [int(mask.sum()) for mask in loss_mask] == [4, 4, 3, 0]


def test_seeded_document_order_repeats_and_different_seeds_vary(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(
        dataset_dir,
        train_sources=(("AA", "BB"), ("CC", "DD")),
    )

    with (
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as first_reader,
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as second_reader,
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as third_reader,
    ):
        first = DocumentPackingTokenLoader(
            first_reader,
            split="train",
            batch_size=1,
            seq_len=2,
            seed=19,
        )
        second = DocumentPackingTokenLoader(
            second_reader,
            split="train",
            batch_size=1,
            seq_len=2,
            seed=19,
        )
        third = DocumentPackingTokenLoader(
            third_reader,
            split="train",
            batch_size=1,
            seq_len=2,
            seed=20,
        )
        first_batches = _epoch_batches(first)
        second_batches = _epoch_batches(second)
        third_batches = _epoch_batches(third)

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


def test_state_resumes_exactly_across_shards_and_epoch_boundaries(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(
        dataset_dir,
        train_sources=(("AB", "CD"), ("EF", "GH")),
    )

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as source_reader:
        source = DocumentPackingTokenLoader(
            source_reader,
            split="train",
            batch_size=2,
            seq_len=2,
            seed=31,
        )
        next(source)
        state = json.loads(json.dumps(source.state_dict()))
        expected = [next(source) for _ in range(3)]

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as resumed_reader:
        resumed = DocumentPackingTokenLoader(
            resumed_reader,
            split="train",
            batch_size=2,
            seq_len=2,
            seed=999,
        )
        resumed.load_state_dict(state)
        actual = [next(resumed) for _ in range(3)]

    assert all(
        _batches_equal(expected_batch, actual_batch)
        for expected_batch, actual_batch in zip(expected, actual, strict=True)
    )
    assert resumed.position == source.position
    assert resumed.epoch == source.epoch
    assert resumed.row_position == source.row_position


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.__setitem__("format_version", 2), "format version"),
        (lambda state: state.__setitem__("batch_size", True), "batch_size"),
        (lambda state: state.__setitem__("row_position", 1), "row_position"),
        (lambda state: state.__setitem__("epoch_seed", -1), "epoch_seed"),
        (lambda state: state.__setitem__("rng_state", []), "rng_state"),
        (lambda state: state.pop("position"), "fields"),
    ],
)
def test_invalid_packing_state_fails_without_mutating_loader(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(dataset_dir, train_sources=(("AB", "CD"),))

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = DocumentPackingTokenLoader(
            reader,
            split="train",
            batch_size=2,
            seq_len=2,
            seed=1,
        )
        invalid_state = deepcopy(loader.state_dict())
        mutation(invalid_state)
        original_state = loader.state_dict()

        with pytest.raises(DocumentPackingTokenLoaderStateError, match=message):
            loader.load_state_dict(invalid_state)

        assert loader.state_dict() == original_state


def test_corrupt_document_boundaries_fail_before_packing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(dataset_dir, train_sources=(("AB", "CD"),))
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["shards"][0]["document_token_counts"] = [1, 1]
    save_json(manifest, manifest_path)

    with pytest.raises(TokenizedDataError, match="document token total"):
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer)


def test_loader_factory_selects_flat_or_packed_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(dataset_dir, train_sources=(("ABCDE",),))

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        flat = create_token_loader(
            reader,
            strategy="flat",
            split="train",
            batch_size=1,
            seq_len=2,
            seed=0,
        )
        packed = create_token_loader(
            reader,
            strategy="packed",
            split="train",
            batch_size=1,
            seq_len=2,
            seed=0,
        )
        with pytest.raises(ValueError, match="strategy"):
            create_token_loader(
                reader,
                strategy="unknown",  # type: ignore[arg-type]
                split="train",
                batch_size=1,
                seq_len=2,
                seed=0,
            )

    assert isinstance(flat, RandomOffsetTokenLoader)
    assert isinstance(packed, DocumentPackingTokenLoader)


def test_readme_documents_packing_policy_and_resume_contract() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "`DocumentPackingTokenLoader`" in readme
    assert '`strategy="packed"`' in readme
    assert "No pad token is introduced" in readme
    assert "boolean loss mask" in readme
    assert "exact next" in readme
