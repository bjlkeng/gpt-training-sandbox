"""Tests for BOS-aware document-boundary packing."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
from typing import Any

import pytest
import torch

from scratch_llm.data import (
    DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION,
    DocumentPackingTokenLoader,
    DocumentPackingTokenLoaderStateError,
    RandomOffsetTokenLoader,
    TokenizedDataError,
    TokenizedDocumentSpan,
    TokenizedShardReader,
    TokenizedShardSource,
    _plan_best_fit_document_rows,
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


def _reference_best_fit_rows(
    spans: tuple[TokenizedDocumentSpan, ...],
    *,
    order: list[int],
    seq_len: int,
) -> list[list[tuple[int, int, int, bool, bool]]]:
    """Small linear oracle preserving the original earliest-row tie behavior."""

    row_token_count = seq_len + 1
    rows: list[list[tuple[int, int, int, bool, bool]]] = []
    used_token_counts: list[int] = []

    def place(piece: tuple[int, int, int, bool, bool]) -> None:
        packed_token_count = piece[2] + 1
        best_index: int | None = None
        best_remaining: int | None = None
        for row_index, used in enumerate(used_token_counts):
            remaining = row_token_count - used - packed_token_count
            if remaining >= 0 and (
                best_remaining is None or remaining < best_remaining
            ):
                best_index = row_index
                best_remaining = remaining
        if best_index is None:
            rows.append([piece])
            used_token_counts.append(packed_token_count)
        else:
            rows[best_index].append(piece)
            used_token_counts[best_index] += packed_token_count

    for document_index in order:
        span = spans[document_index]
        remaining = span.token_count
        document_offset = 0
        while remaining > 0:
            piece_token_count = min(remaining, seq_len)
            is_continuation = document_offset > 0
            piece = (
                span.shard_index,
                span.start + document_offset,
                piece_token_count,
                is_continuation,
                piece_token_count == remaining,
            )
            if is_continuation:
                rows.append([piece])
                used_token_counts.append(piece_token_count + 1)
            else:
                place(piece)
            remaining -= piece_token_count
            document_offset += piece_token_count
        if span.token_count == 0:
            place((span.shard_index, span.start, 0, False, True))
    return rows


def _row_signature(
    rows: object,
) -> list[list[tuple[int, int, int, bool, bool]]]:
    return [
        [
            (
                piece.shard_index,
                piece.start,
                piece.token_count,
                piece.is_continuation,
                piece.is_document_end,
            )
            for piece in row.pieces
        ]
        for row in rows
    ]


@pytest.mark.parametrize("seed", range(12))
def test_capacity_indexed_planner_matches_linear_reference(seed: int) -> None:
    rng = random.Random(seed)
    seq_len = rng.randint(1, 12)
    token_counts = [
        0,
        seq_len,
        seq_len + 1,
        2 * seq_len,
        *(rng.randint(0, 4 * seq_len) for _ in range(60)),
    ]
    spans = tuple(
        TokenizedDocumentSpan(
            split="train",
            shard_index=index % 3,
            document_index=index,
            start=index * (4 * seq_len + 1),
            stop=index * (4 * seq_len + 1) + token_count,
        )
        for index, token_count in enumerate(token_counts)
    )
    order = list(range(len(spans)))
    rng.shuffle(order)

    actual, _ = _plan_best_fit_document_rows(
        spans,
        order=order,
        seq_len=seq_len,
    )
    expected = _reference_best_fit_rows(spans, order=order, seq_len=seq_len)

    assert _row_signature(actual) == expected


def test_capacity_index_uses_earliest_row_for_equal_best_fit() -> None:
    spans = tuple(
        TokenizedDocumentSpan(
            split="train",
            shard_index=0,
            document_index=index,
            start=index * 10,
            stop=index * 10 + token_count,
        )
        for index, token_count in enumerate((5, 5, 3))
    )

    rows, _ = _plan_best_fit_document_rows(
        spans,
        order=range(3),
        seq_len=9,
    )

    assert [[piece.start for piece in row.pieces] for row in rows] == [
        [0, 20],
        [10],
    ]


def test_planner_does_not_scan_existing_rows_for_each_placement() -> None:
    document_count = 20_000
    seq_len = 8
    spans = tuple(
        TokenizedDocumentSpan(
            split="train",
            shard_index=0,
            document_index=index,
            start=index * seq_len,
            stop=(index + 1) * seq_len,
        )
        for index in range(document_count)
    )

    rows, stats = _plan_best_fit_document_rows(
        spans,
        order=range(document_count),
        seq_len=seq_len,
    )

    assert len(rows) == document_count
    assert stats.capacity_searches == document_count
    assert stats.row_candidate_checks == 0


def test_loader_logs_planning_start_and_completion(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(dataset_dir, train_sources=(("AB", "CD"),))
    progress_messages: list[str] = []

    with (
        caplog.at_level("INFO", logger="scratch_llm.data"),
        TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader,
    ):
        DocumentPackingTokenLoader(
            reader,
            split="train",
            batch_size=1,
            seq_len=2,
            seed=0,
            planning_progress=progress_messages.append,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "packing planner started: documents=2" in message for message in messages
    )
    assert any(
        "packing planner completed: documents=2" in message for message in messages
    )
    assert progress_messages == [
        next(message for message in messages if "packing planner started" in message),
        next(message for message in messages if "packing planner completed" in message),
    ]


def test_packing_supervises_content_and_real_bos_boundaries(tmp_path: Path) -> None:
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
    assert loss_mask.tolist() == [[True, True, True, True]]


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
    documents = ("ABCDE", "", "é")
    tokenizer = _write_dataset(dataset_dir, train_sources=(documents,))

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = DocumentPackingTokenLoader(
            reader,
            split="train",
            batch_size=2,
            seq_len=5,
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
    supervised_content = targets[loss_mask & (targets != bos)]
    assert sorted(supervised_content.tolist()) == expected_content
    assert torch.any(loss_mask & (targets == bos))
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
    inputs = torch.cat([batch[0] for batch in batches])
    loss_mask = torch.cat([batch[2] for batch in batches])
    bos = tokenizer.get_bos_token_id()

    encoded = tokenizer.encode(document)
    assert inputs.tolist() == [
        [bos, *encoded[0:3]],
        encoded[3:7],
        encoded[7:11],
        [bos, bos, bos, bos],
    ]
    assert targets.tolist() == [
        encoded[0:4],
        encoded[4:8],
        [*encoded[8:11], bos],
        [bos, bos, bos, bos],
    ]
    assert targets[loss_mask & (targets != bos)].tolist() == encoded
    assert targets[2, -1].item() == bos
    assert loss_mask[2, -1].item()
    assert torch.all(targets[~loss_mask] == bos)
    assert [int(mask.sum()) for mask in loss_mask] == [4, 4, 4, 0]


def test_document_can_follow_a_continuation_with_a_supervised_bos_boundary(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "tokenized"
    tokenizer = _write_dataset(
        dataset_dir,
        train_sources=(("ABCDE", "FG"),),
    )

    with TokenizedShardReader(dataset_dir, tokenizer=tokenizer) as reader:
        loader = DocumentPackingTokenLoader(
            reader,
            split="train",
            batch_size=2,
            seq_len=4,
            seed=0,
        )
        inputs, targets, loss_mask = next(loader)

    bos = tokenizer.get_bos_token_id()
    assert loader.packed_example_count == 2
    assert inputs[1].tolist() == [ord("D"), ord("E"), bos, ord("F")]
    assert targets[1].tolist() == [ord("E"), bos, ord("F"), ord("G")]
    assert loss_mask[1].tolist() == [True, True, True, True]


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
        assert (
            state["format_version"] == DOCUMENT_PACKING_LOADER_STATE_FORMAT_VERSION == 2
        )
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
        (lambda state: state.__setitem__("format_version", 1), "format version"),
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
    normalized_readme = " ".join(readme.split())

    assert "`DocumentPackingTokenLoader`" in readme
    assert '`strategy="packed"`' in readme
    assert "No pad token is introduced" in readme
    assert "boolean loss mask" in readme
    assert "previous real token" in normalized_readme
    assert "real BOS boundary targets" in normalized_readme
    assert "exact next" in readme
