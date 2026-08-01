"""Conformance tests for finite continuation-aware full-document BPB."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch
from torch import nn

import scratch_llm.evaluation.full_document_bpb as full_document_bpb
from scratch_llm.data.loaders import (
    DocumentPackingTokenLoader,
    write_tokenized_parquet_shards,
)
from scratch_llm.evaluation.full_document_bpb import (
    FULL_DOCUMENT_EVAL_METRIC,
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_TRAIN_METRIC,
    FullDocumentProtocolConfig,
    FullDocumentValidationBatches,
    evaluate_full_document_bpb,
    full_document_metric_value,
)
from scratch_llm.evaluation.nanochat_bpb import (
    NanochatCompatibilityConfig,
    evaluate_nanochat_compatible_bpb,
)
from scratch_llm.data.tokenized import (
    TokenizedShardReader,
    TokenizedShardSource,
    tokenized_manifest_identity,
    write_tokenized_shards,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from scratch_llm.tokenization.artifacts import build_token_byte_lengths
from tests.fixtures.bpb_conformance import BPB_CONFORMANCE_FIXTURE


class _UnitLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        assert loss_reduction == "none"
        return torch.ones_like(inputs, dtype=torch.float64) + self.anchor * 0


def _write_multishard_validation_fixture(
    path: Path,
) -> tuple[ByteTokenizer, tuple[str, ...]]:
    tokenizer = ByteTokenizer()
    validation_documents = ("", "ABCD", "xy", "é", "ABCDEFGHI")
    write_tokenized_shards(
        path,
        tokenizer=tokenizer,
        train_sources=[TokenizedShardSource("train", ("train",))],
        val_sources=[
            TokenizedShardSource("val-0", validation_documents[:3]),
            TokenizedShardSource("val-1", validation_documents[3:]),
        ],
    )
    return tokenizer, validation_documents


def test_reference_config_is_frozen_and_has_no_upstream_commit() -> None:
    config = FullDocumentProtocolConfig(
        device_batch_size=4,
        context_length=4,
    )

    assert FULL_DOCUMENT_PROTOCOL_ID == "full_documents_v1"
    assert config.to_dict() == {
        "batch_padding": "bos_with_no_supervision",
        "device_batch_size": 4,
        "document_order": "manifest_shard_document",
        "max_seq_len": 4,
        "packing": {
            "continuation_context": "previous_ordinary_token",
            "first_window_context": "bos",
            "largest_fit_tie": "earliest_row",
            "ordinary_target_coverage": "exactly_once",
            "selection": "best_fit_complete_first_piece",
        },
        "row_capacity": 5,
        "seed": None,
        "termination": "one_validation_manifest_pass",
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"device_batch_size": 0, "context_length": 4}, "batch"),
        ({"device_batch_size": 1, "context_length": 0}, "context"),
    ],
)
def test_reference_config_rejects_invalid_shapes(
    arguments: dict[str, int],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FullDocumentProtocolConfig(**arguments)


def test_finite_batches_cover_empty_unicode_exact_residual_and_multishard_docs(
    tmp_path: Path,
) -> None:
    tokenizer, documents = _write_multishard_validation_fixture(tmp_path / "tokenized")
    token_bytes = build_token_byte_lengths(tokenizer)
    config = FullDocumentProtocolConfig(4, 4)

    with TokenizedShardReader(
        tmp_path / "tokenized",
        tokenizer=tokenizer,
    ) as reader:
        training_loader = DocumentPackingTokenLoader(
            reader,
            split="val",
            batch_size=2,
            seq_len=4,
            seed=73,
        )
        training_state = training_loader.state_dict()
        first = FullDocumentValidationBatches(
            reader,
            config=config,
            bos_token_id=tokenizer.get_bos_token_id(),
        )
        second = FullDocumentValidationBatches(
            reader,
            config=config,
            bos_token_id=tokenizer.get_bos_token_id(),
        )
        first_batches = list(first)
        second_batches = list(second)
        result = evaluate_full_document_bpb(
            _UnitLossModel(),
            tokenizer,
            reader,
            token_bytes,
            checkpoint_identity="checkpoint",
            config=config,
            device="cpu",
        )
        manifest_identity = tokenized_manifest_identity(reader.manifest)

        assert training_loader.state_dict() == training_state

    assert first.plan_signature == second.plan_signature
    assert first.plan_signature == (
        (
            (0, 0, 0, False, True),
            (0, 4, 2, False, True),
        ),
        ((0, 0, 4, False, True),),
        ((1, 0, 2, False, True),),
        ((1, 2, 4, False, False),),
        ((1, 6, 4, True, False),),
        ((1, 10, 1, True, True),),
    )
    assert len(first) == len(second) == 2
    assert len(first_batches) == len(second_batches) == 2
    for first_batch, second_batch in zip(
        first_batches,
        second_batches,
        strict=True,
    ):
        for first_tensor, second_tensor in zip(
            first_batch,
            second_batch,
            strict=True,
        ):
            torch.testing.assert_close(first_tensor, second_tensor, rtol=0, atol=0)
    with pytest.raises(StopIteration):
        next(first)

    bos = tokenizer.get_bos_token_id()
    inputs = torch.cat([batch[0] for batch in first_batches])
    targets = torch.cat([batch[1] for batch in first_batches])
    masks = torch.cat([batch[2] for batch in first_batches])
    assert inputs.shape == targets.shape == masks.shape == (8, 4)
    assert inputs[0].tolist() == [bos, bos, ord("x"), ord("y")]
    assert targets[0].tolist() == [bos, ord("x"), ord("y"), bos]
    assert masks[0].tolist() == [True, True, True, True]
    assert inputs[4].tolist() == [ord("D"), ord("E"), ord("F"), ord("G")]
    assert targets[4].tolist() == [ord("E"), ord("F"), ord("G"), ord("H")]
    assert masks[4].tolist() == [True, True, True, True]
    assert inputs[5].tolist() == [ord("H"), ord("I"), bos, bos]
    assert targets[5].tolist() == [ord("I"), bos, bos, bos]
    assert masks[5].tolist() == [True, True, False, False]
    assert torch.all(inputs[6:] == bos)
    assert torch.all(targets[6:] == bos)
    assert not bool(masks[6:].any().item())
    assert bool(((targets == bos) & masks).any().item())

    expected_tokens = [
        token_id for document in documents for token_id in tokenizer.encode(document)
    ]
    counted_targets = targets[masks]
    ordinary_targets = counted_targets[token_bytes[counted_targets] > 0].tolist()
    assert Counter(ordinary_targets) == Counter(expected_tokens)
    assert len(ordinary_targets) == len(expected_tokens) == 17

    assert result.protocol_id == FULL_DOCUMENT_PROTOCOL_ID
    assert result.protocol_version == 1
    assert result.reference_commit is None
    assert result.reference_config == config.to_dict()
    assert result.validation_manifest_identity == manifest_identity
    assert result.source_documents == 5
    assert result.source_tokens == 17
    assert result.source_bytes == 17
    assert result.processed_model_tokens == 32
    assert result.counted_target_tokens == 17
    assert result.counted_target_bytes == 17
    assert result.unique_source_tokens == 17
    assert result.unique_source_bytes == 17
    assert result.discarded_source_tokens == 0
    assert result.discarded_source_bytes == 0
    assert result.source_token_retention == 1
    assert result.source_byte_retention == 1
    assert result.total_nats == pytest.approx(17.0)


def _write_shared_oversized_parquet(path: Path) -> Path:
    parquet_dir = path / "parquet"
    parquet_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "text": pa.array(
                    [BPB_CONFORMANCE_FIXTURE.documents[1]],
                    type=pa.string(),
                )
            }
        ),
        parquet_dir / "shard_06542.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    pq.write_table(
        pa.table({"text": pa.array(["train"], type=pa.string())}),
        parquet_dir / "shard_00000.parquet",
        compression="NONE",
        use_dictionary=False,
    )
    return parquet_dir


def test_shared_oversized_fixture_contrasts_full_retention_with_compat_crop(
    tmp_path: Path,
) -> None:
    parquet_dir = _write_shared_oversized_parquet(tmp_path)
    tokenized_dir = tmp_path / "tokenized"
    tokenizer = ByteTokenizer()
    write_tokenized_parquet_shards(
        parquet_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=1,
        batch_size=128,
    )
    token_bytes = build_token_byte_lengths(tokenizer)
    model = _UnitLossModel()

    with TokenizedShardReader(tokenized_dir, tokenizer=tokenizer) as reader:
        compatibility = evaluate_nanochat_compatible_bpb(
            model,
            tokenizer,
            reader,
            token_bytes,
            parquet_dir=parquet_dir,
            checkpoint_identity="checkpoint",
            config=NanochatCompatibilityConfig(1, 8, 8),
            device="cpu",
        )
        complete = evaluate_full_document_bpb(
            model,
            tokenizer,
            reader,
            token_bytes,
            checkpoint_identity="checkpoint",
            config=FullDocumentProtocolConfig(1, 8),
            device="cpu",
        )

    assert compatibility.source_tokens == complete.source_tokens
    assert compatibility.source_bytes == complete.source_bytes
    assert compatibility.unique_source_tokens == 8
    assert compatibility.unique_source_bytes == 8
    assert compatibility.source_token_retention < 1
    assert compatibility.source_byte_retention < 1
    assert compatibility.discarded_source_tokens > 0
    assert compatibility.discarded_source_bytes > 0
    assert complete.unique_source_tokens == complete.source_tokens
    assert complete.unique_source_bytes == complete.source_bytes
    assert complete.source_token_retention == 1
    assert complete.source_byte_retention == 1
    assert complete.discarded_source_tokens == 0
    assert complete.discarded_source_bytes == 0


@pytest.mark.parametrize(
    "invalid_key",
    [
        "val_bpb",
        "eval/val_bpb",
        "packed",
        "nanochat_compat_v1",
    ],
)
def test_metric_namespace_rejects_unsuffixed_and_loader_aliases(
    invalid_key: str,
    tmp_path: Path,
) -> None:
    tokenizer, _ = _write_multishard_validation_fixture(tmp_path / "tokenized")
    with TokenizedShardReader(
        tmp_path / "tokenized",
        tokenizer=tokenizer,
    ) as reader:
        result = evaluate_full_document_bpb(
            _UnitLossModel(),
            tokenizer,
            reader,
            build_token_byte_lengths(tokenizer),
            checkpoint_identity="checkpoint",
            config=FullDocumentProtocolConfig(4, 4),
            device="cpu",
        )

    assert FULL_DOCUMENT_TRAIN_METRIC == "val_bpb_full_documents"
    assert FULL_DOCUMENT_EVAL_METRIC == "eval/val_bpb_full_documents"
    assert (
        full_document_metric_value(result, key=FULL_DOCUMENT_TRAIN_METRIC) == result.bpb
    )
    assert (
        full_document_metric_value(result, key=FULL_DOCUMENT_EVAL_METRIC) == result.bpb
    )
    with pytest.raises(ValueError, match="reserved full-document metric"):
        full_document_metric_value(result, key=invalid_key)


def test_readme_documents_distinct_complete_coverage_domain() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "full_documents_v1" in readme
    assert "`val_bpb_full_documents`" in readme
    assert "`eval/val_bpb_full_documents`" in readme
    assert "one finite manifest pass" in readme
    assert "previous ordinary token" in readme
    assert "exactly once" in readme


def test_production_protocol_imports_neither_compatibility_nor_training_loaders() -> (
    None
):
    source_path = Path(full_document_bpb.__file__)
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

    assert "scratch_llm.evaluation.nanochat_bpb" not in imported_modules
    assert {
        "DocumentPackingTokenLoader",
        "RandomOffsetTokenLoader",
        "create_token_loader",
    }.isdisjoint(imported_names)
