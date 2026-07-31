"""Frozen conformance tests for the pinned nanochat-compatible BPB protocol."""

from __future__ import annotations

import ast
from collections.abc import Iterator
import math
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from torch import nn

import scratch_llm.nanochat_bpb as nanochat_bpb
from scratch_llm.bpb import BPBAccumulation, BaseValidationResult
from scratch_llm.data import write_tokenized_parquet_shards
from scratch_llm.nanochat_bpb import (
    NANOCHAT_COMPAT_EVAL_METRIC,
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_TRAIN_METRIC,
    NANOCHAT_REFERENCE_COMMIT,
    NANOCHAT_REFERENCE_FILE_SHA256,
    NanochatCompatibilityConfig,
    NanochatCompatiblePacker,
    NanochatDocument,
    evaluate_nanochat_compatible_bpb,
    nanochat_compatible_metric_value,
)
from scratch_llm.tokenized_data import (
    TokenizedDataError,
    TokenizedShardReader,
    tokenized_manifest_identity,
)
from scratch_llm.tokenizer import ByteTokenizer
from scratch_llm.tokenizer_artifacts import build_token_byte_lengths
from tests.fixtures.bpb_conformance import BPB_CONFORMANCE_FIXTURE
from tests.fixtures.nanochat_compat_v1 import (
    BOS_TOKEN_ID,
    DOCUMENT_BATCHES,
    EXPECTED_INPUT_BATCH,
    EXPECTED_TARGET_BATCH,
    REFERENCE_COMMIT,
    REFERENCE_FILE_SHA256,
)


def _repeating_document_batches() -> Iterator[tuple[NanochatDocument, ...]]:
    while True:
        for batch in DOCUMENT_BATCHES:
            yield tuple(
                NanochatDocument(
                    source_document_index=document_index,
                    token_ids=token_ids,
                )
                for document_index, token_ids in batch
            )


def test_protocol_provenance_and_resolved_reference_config_are_frozen() -> None:
    config = NanochatCompatibilityConfig(
        device_batch_size=2,
        context_length=5,
        eval_tokens=23,
    )

    assert NANOCHAT_COMPAT_PROTOCOL_ID == "nanochat_compat_v1"
    assert NANOCHAT_REFERENCE_COMMIT == REFERENCE_COMMIT
    assert NANOCHAT_REFERENCE_FILE_SHA256 == REFERENCE_FILE_SHA256
    assert config.eval_steps == 2
    assert config.processed_eval_tokens == 20
    assert config.to_dict() == {
        "buffer_size": 1000,
        "device_batch_size": 2,
        "document_order": "final_validation_shard_row_group_document",
        "eval_steps": 2,
        "eval_tokens": 23,
        "max_seq_len": 5,
        "packing": {
            "crop": "first_shortest_prefix_discard_suffix",
            "largest_fit_tie": "first",
            "prepend_bos": True,
            "refill": "whole_tokenizer_batch_below_buffer_size",
            "selection": "first_largest_document_that_fits",
            "shortest_tie": "first",
        },
        "processed_eval_tokens": 20,
        "reference_files": REFERENCE_FILE_SHA256,
        "repeated_cycles": "restart_final_validation_shard_in_order",
        "row_capacity": 6,
        "split": "val",
        "tokenizer_batch_size": 128,
        "tokenizer_threads": 4,
        "world_size": 1,
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"device_batch_size": 0, "context_length": 5, "eval_tokens": 10}, "batch"),
        ({"device_batch_size": 1, "context_length": 0, "eval_tokens": 10}, "context"),
        ({"device_batch_size": 2, "context_length": 5, "eval_tokens": 9}, "step"),
    ],
)
def test_reference_config_rejects_non_evaluable_shapes(
    arguments: dict[str, int],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        NanochatCompatibilityConfig(**arguments)


def test_frozen_reference_matches_ties_refills_rows_and_repeated_cycles() -> None:
    token_bytes = torch.ones(100, dtype=torch.int32)
    token_bytes[BOS_TOKEN_ID] = 0
    packer = NanochatCompatiblePacker(
        _repeating_document_batches(),
        batch_size=2,
        context_length=5,
        bos_token_id=BOS_TOKEN_ID,
        token_bytes=token_bytes,
        buffer_size=4,
    )

    first = next(packer)
    second = next(packer)

    expected = (
        torch.tensor(EXPECTED_INPUT_BATCH),
        torch.tensor(EXPECTED_TARGET_BATCH),
    )
    for actual_batch in (first, second):
        torch.testing.assert_close(actual_batch[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual_batch[1], expected[1], rtol=0, atol=0)
    assert packer.counted_source_tokens == 16
    assert packer.counted_source_bytes == 16
    assert packer.unique_source_tokens == 8
    assert packer.unique_source_bytes == 8


def test_first_shortest_tie_crops_only_the_first_document_prefix() -> None:
    documents = (
        NanochatDocument(0, (10, 11, 12, 13, 14, 15)),
        NanochatDocument(1, (20, 21, 22, 23, 24, 25)),
    )

    def batches() -> Iterator[tuple[NanochatDocument, ...]]:
        while True:
            yield documents

    token_bytes = torch.ones(100, dtype=torch.int32)
    token_bytes[BOS_TOKEN_ID] = 0
    packer = NanochatCompatiblePacker(
        batches(),
        batch_size=1,
        context_length=4,
        bos_token_id=BOS_TOKEN_ID,
        token_bytes=token_bytes,
        buffer_size=2,
    )

    inputs, targets = next(packer)

    assert inputs.tolist() == [[BOS_TOKEN_ID, 10, 11, 12]]
    assert targets.tolist() == [[10, 11, 12, 13]]
    assert packer.counted_source_tokens == 4
    assert packer.unique_source_tokens == 4
    assert packer.unique_source_bytes == 4


class _RecordingUnitLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        assert loss_reduction == "none"
        self.batches.append((inputs.detach().cpu(), targets.detach().cpu()))
        return torch.ones_like(inputs, dtype=torch.float64) + self.anchor * 0


def _write_oversized_protocol_fixture(path: Path) -> tuple[list[str], Path]:
    fixture = BPB_CONFORMANCE_FIXTURE
    # 1,001 documents make the final 105-document tokenizer batch overshoot
    # the 1,000-document buffer. After the fitting document is popped, the
    # buffer remains exactly at the threshold and the oversized document is
    # cropped before a repeated cycle can refill it.
    documents = [fixture.documents[0], *([fixture.documents[1]] * 1000)]
    parquet_dir = path / "parquet"
    parquet_dir.mkdir()
    validation_path = parquet_dir / "shard_06542.parquet"
    pq.write_table(
        pa.table({"text": pa.array(documents, type=pa.string())}),
        validation_path,
        row_group_size=128,
        compression="NONE",
        use_dictionary=False,
    )
    pq.write_table(
        pa.table({"text": pa.array(["train"], type=pa.string())}),
        parquet_dir / "shard_00000.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    return documents, parquet_dir


def test_oversized_fixture_reports_exact_crop_coverage_and_bpb(
    tmp_path: Path,
) -> None:
    documents, parquet_dir = _write_oversized_protocol_fixture(tmp_path)
    tokenizer = ByteTokenizer()
    tokenized_dir = tmp_path / "tokenized"
    write_tokenized_parquet_shards(
        parquet_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=1,
        batch_size=128,
    )
    token_bytes = build_token_byte_lengths(tokenizer)
    model = _RecordingUnitLossModel()
    config = NanochatCompatibilityConfig(
        device_batch_size=1,
        context_length=32,
        eval_tokens=32,
    )

    with TokenizedShardReader(tokenized_dir, tokenizer=tokenizer) as reader:
        result = evaluate_nanochat_compatible_bpb(
            model,
            tokenizer,
            reader,
            token_bytes,
            parquet_dir=parquet_dir,
            checkpoint_identity="sha256:" + "1" * 64,
            config=config,
            device="cpu",
        )
        expected_manifest_identity = tokenized_manifest_identity(reader.manifest)

    first_tokens = tokenizer.encode(documents[0])
    oversized_tokens = tokenizer.encode(documents[1])
    crop_token_count = config.row_capacity - (1 + len(first_tokens)) - 1
    expected_row = [
        tokenizer.get_bos_token_id(),
        *first_tokens,
        tokenizer.get_bos_token_id(),
        *oversized_tokens[:crop_token_count],
    ]
    assert len(expected_row) == config.row_capacity
    recorded_inputs, recorded_targets = model.batches[0]
    assert recorded_inputs.tolist() == [expected_row[:-1]]
    assert recorded_targets.tolist() == [expected_row[1:]]

    source_tokens = sum(len(tokenizer.encode(document)) for document in documents)
    expected_retained = len(first_tokens) + crop_token_count
    assert result.protocol_id == NANOCHAT_COMPAT_PROTOCOL_ID
    assert result.protocol_version == 1
    assert result.reference_commit == REFERENCE_COMMIT
    assert result.reference_config == config.to_dict()
    assert result.validation_manifest_identity == expected_manifest_identity
    assert result.source_documents == 1001
    assert result.source_tokens == source_tokens
    assert result.source_bytes == sum(
        len(document.encode("utf-8")) for document in documents
    )
    assert result.processed_model_tokens == 32
    assert result.counted_target_tokens == expected_retained
    assert result.counted_target_bytes == expected_retained
    assert result.unique_source_tokens == expected_retained
    assert result.unique_source_bytes == expected_retained
    assert result.discarded_source_tokens == source_tokens - expected_retained
    assert result.discarded_source_bytes == result.source_bytes - expected_retained
    assert result.source_token_retention < 1
    assert result.source_byte_retention < 1
    assert result.total_nats == pytest.approx(float(expected_retained))
    assert result.bpb == pytest.approx(1 / math.log(2))


def test_raw_validation_order_must_match_the_tokenized_manifest(
    tmp_path: Path,
) -> None:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    validation_path = parquet_dir / "shard_06542.parquet"
    original_documents = ["first", "second"]
    pq.write_table(
        pa.table({"text": pa.array(original_documents, type=pa.string())}),
        validation_path,
        row_group_size=1,
        compression="NONE",
        use_dictionary=False,
    )
    pq.write_table(
        pa.table({"text": pa.array(["train"], type=pa.string())}),
        parquet_dir / "shard_00000.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    tokenizer = ByteTokenizer()
    tokenized_dir = tmp_path / "tokenized"
    write_tokenized_parquet_shards(
        parquet_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=1,
        batch_size=128,
    )
    pq.write_table(
        pa.table(
            {"text": pa.array(list(reversed(original_documents)), type=pa.string())}
        ),
        validation_path,
        row_group_size=1,
        compression="NONE",
        use_dictionary=False,
    )
    model = _RecordingUnitLossModel()

    with TokenizedShardReader(tokenized_dir, tokenizer=tokenizer) as reader:
        with pytest.raises(
            TokenizedDataError,
            match="tokenization does not match.*document 0",
        ):
            evaluate_nanochat_compatible_bpb(
                model,
                tokenizer,
                reader,
                build_token_byte_lengths(tokenizer),
                parquet_dir=parquet_dir,
                checkpoint_identity="checkpoint",
                config=NanochatCompatibilityConfig(1, 4, 4),
                device="cpu",
            )

    assert model.batches == []


def test_repeated_source_cycles_count_losses_without_inflating_unique_coverage() -> (
    None
):
    accumulation = BPBAccumulation(
        processed_model_tokens=16,
        counted_target_tokens=8,
        counted_target_bytes=8,
        total_nats=8.0,
    )

    result = BaseValidationResult.from_accumulation(
        accumulation,
        protocol_id=NANOCHAT_COMPAT_PROTOCOL_ID,
        protocol_version=1,
        reference_commit=REFERENCE_COMMIT,
        reference_config={"eval_steps": 2},
        checkpoint_identity="checkpoint",
        tokenizer_identity="tokenizer",
        validation_manifest_identity="manifest",
        source_documents=1,
        source_tokens=4,
        source_bytes=4,
        unique_source_tokens=4,
        unique_source_bytes=4,
    )

    assert result.counted_target_tokens == 8
    assert result.unique_source_tokens == 4
    assert result.source_token_retention == 1


@pytest.mark.parametrize(
    "invalid_key",
    [
        "val_bpb_full_documents",
        "eval/val_bpb_full_documents",
        "packed",
        "document_packing",
    ],
)
def test_metric_namespace_rejects_full_document_and_loader_aliases(
    invalid_key: str,
) -> None:
    result = BaseValidationResult.from_accumulation(
        BPBAccumulation(1, 1, 1, 1.0),
        protocol_id=NANOCHAT_COMPAT_PROTOCOL_ID,
        protocol_version=1,
        reference_commit=REFERENCE_COMMIT,
        reference_config={},
        checkpoint_identity="checkpoint",
        tokenizer_identity="tokenizer",
        validation_manifest_identity="manifest",
        source_documents=1,
        source_tokens=1,
        source_bytes=1,
        unique_source_tokens=1,
        unique_source_bytes=1,
    )

    assert NANOCHAT_COMPAT_TRAIN_METRIC == "val_bpb"
    assert NANOCHAT_COMPAT_EVAL_METRIC == "eval/val_bpb"
    assert (
        nanochat_compatible_metric_value(
            result,
            key=NANOCHAT_COMPAT_TRAIN_METRIC,
        )
        == result.bpb
    )
    assert (
        nanochat_compatible_metric_value(
            result,
            key=NANOCHAT_COMPAT_EVAL_METRIC,
        )
        == result.bpb
    )
    with pytest.raises(ValueError, match="reserved nanochat-compatible metric"):
        nanochat_compatible_metric_value(result, key=invalid_key)


def test_readme_documents_the_pinned_compatibility_domain() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert NANOCHAT_REFERENCE_COMMIT in readme
    assert "nanochat_compat_v1" in readme
    assert "1,000-document buffer" in readme
    assert "first largest document that fits" in readme
    assert "first shortest document" in readme
    assert "`val_bpb` and `eval/val_bpb`" in readme
    assert "cropped suffix is discarded" in readme


def test_production_protocol_imports_neither_nanochat_nor_training_packers() -> None:
    source_path = Path(nanochat_bpb.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
            imported_names.update(alias.name for alias in node.names)

    assert not any(
        module == "nanochat" or module.startswith("nanochat.")
        for module in imported_modules
    )
    assert {
        "DocumentPackingTokenLoader",
        "RandomOffsetTokenLoader",
        "create_token_loader",
    }.isdisjoint(imported_names)
