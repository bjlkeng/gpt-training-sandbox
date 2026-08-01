"""Tests for validated tokenized shard storage and memory-mapped reads."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scratch_llm.data import tokenized as tokenized_data
from scratch_llm.data.loaders import (
    TokenizedDataError,
    TokenizedShardReader,
    TokenizedShardSource,
    write_tokenized_parquet_shards,
    write_tokenized_shards,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer, Tokenizer
from scratch_llm.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"


class _FixedIdTokenizer(Tokenizer):
    def __init__(self, *, vocab_size: int, token_id: int) -> None:
        self._vocab_size = vocab_size
        self._token_id = token_id

    def encode(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]:
        if prepend is not None or append is not None:
            raise ValueError("test tokenizer does not expose special tokens")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [self._token_id] if text else []

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join("x" for _ in token_ids)

    def encode_special(self, token: str) -> int:
        raise ValueError(f"unsupported special token {token!r}")

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return b"x"

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_bos_token_id(self) -> int:
        return 0

    def get_special_tokens(self) -> set[str]:
        return set()

    def get_identity(self) -> str:
        return f"fixed:v1:{self._vocab_size}:{self._token_id}"


class _FailingByteTokenizer(ByteTokenizer):
    def encode(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]:
        if text == "fail":
            raise RuntimeError("intentional tokenizer failure")
        return super().encode(text, prepend=prepend, append=append)


class _DifferentIdentityByteTokenizer(ByteTokenizer):
    def get_identity(self) -> str:
        return "sha256:" + "0" * 64


class _DifferentSpecialsByteTokenizer(ByteTokenizer):
    def get_special_tokens(self) -> set[str]:
        tokens = super().get_special_tokens()
        tokens.remove("<|output_end|>")
        return tokens


def _write_byte_dataset(output_dir: Path) -> None:
    write_tokenized_shards(
        output_dir,
        tokenizer=ByteTokenizer(),
        train_sources=[TokenizedShardSource("train-source", ["train"])],
        val_sources=[TokenizedShardSource("val-source", ["validation"])],
    )


def _read_manifest(output_dir: Path) -> dict[str, Any]:
    value = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_parquet_documents_round_trip_through_manifest_and_memmaps(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    shutil.copytree(PARQUET_FIXTURE_DIR, raw_dir)
    output_dir = tmp_path / "tokenized"
    tokenizer = ByteTokenizer()

    manifest = write_tokenized_parquet_shards(
        raw_dir,
        output_dir,
        tokenizer=tokenizer,
        num_train_shards=2,
    )

    assert manifest.format_version == 1
    assert manifest.dtype == "uint16"
    assert manifest.vocab_size == tokenizer.get_vocab_size()
    assert manifest.tokenizer_identity == tokenizer.get_identity()
    assert [shard.filename for shard in manifest.splits["train"].shards] == [
        "train_000000.bin",
        "train_000001.bin",
    ]
    assert [shard.filename for shard in manifest.splits["val"].shards] == [
        "val_000000.bin"
    ]
    assert [shard.source_shards for shard in manifest.splits["train"].shards] == [
        ("shard_00000.parquet",),
        ("shard_00001.parquet",),
    ]
    assert manifest.splits["train"].shards[0].document_token_counts[-1] == 0

    shutil.rmtree(raw_dir)
    reader = TokenizedShardReader(output_dir, tokenizer=tokenizer)
    train_shards = reader.shards("train")
    val_shards = reader.shards("val")

    assert all(isinstance(shard, np.memmap) for shard in (*train_shards, *val_shards))
    assert all(shard.dtype == np.dtype("<u2") for shard in (*train_shards, *val_shards))
    expected_train = [
        tokenizer.encode(text)
        for text in [
            "First synthetic training document.",
            "Unicode train text: café ☕",
            "",
            "Second shard, first document.",
            "你好 from the tiny corpus.",
            "Last training document 🚀",
        ]
    ]
    expected_val = [
        tokenizer.encode(text)
        for text in [
            "Fixed validation document.",
            "",
            "Validation Unicode: Καλημέρα.",
        ]
    ]
    assert [shard.tolist() for shard in train_shards] == [
        expected_train[0] + expected_train[1] + expected_train[2],
        expected_train[3] + expected_train[4] + expected_train[5],
    ]
    assert val_shards[0].tolist() == sum(expected_val, [])
    assert reader.manifest == manifest


@pytest.mark.parametrize(
    ("vocab_size", "token_id", "expected_dtype", "expected_numpy_dtype"),
    [
        (65_535, 65_534, "uint16", np.dtype("<u2")),
        (65_536, 65_535, "uint32", np.dtype("<u4")),
    ],
)
def test_writer_selects_dtype_at_the_complete_vocabulary_boundary(
    tmp_path: Path,
    vocab_size: int,
    token_id: int,
    expected_dtype: str,
    expected_numpy_dtype: np.dtype[np.unsignedinteger],
) -> None:
    tokenizer = _FixedIdTokenizer(vocab_size=vocab_size, token_id=token_id)
    output_dir = tmp_path / f"tokens-{vocab_size}"

    manifest = write_tokenized_shards(
        output_dir,
        tokenizer=tokenizer,
        train_sources=[TokenizedShardSource("train-source", ["one", "two"])],
        val_sources=[TokenizedShardSource("val-source", ["validation"])],
    )
    reader = TokenizedShardReader(output_dir, tokenizer=tokenizer)

    assert manifest.dtype == expected_dtype
    assert reader.shards("train")[0].dtype == expected_numpy_dtype
    assert reader.shards("train")[0].tolist() == [token_id, token_id]


def test_source_order_and_output_bytes_are_reproducible(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    write_tokenized_shards(
        first_output,
        tokenizer=tokenizer,
        train_sources=[
            TokenizedShardSource("source-z", ["z"]),
            TokenizedShardSource("source-a", ["a"]),
        ],
        val_sources=[TokenizedShardSource("source-val", ["v"])],
    )
    write_tokenized_shards(
        second_output,
        tokenizer=tokenizer,
        train_sources=[
            TokenizedShardSource("source-a", ["a"]),
            TokenizedShardSource("source-z", ["z"]),
        ],
        val_sources=[TokenizedShardSource("source-val", ["v"])],
    )

    assert (first_output / "manifest.json").read_bytes() == (
        second_output / "manifest.json"
    ).read_bytes()
    for filename in ["train_000000.bin", "train_000001.bin", "val_000000.bin"]:
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()


def test_retry_and_overwrite_behavior_never_publishes_partial_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "tokenized"
    good_tokenizer = ByteTokenizer()

    with pytest.raises(RuntimeError, match="intentional tokenizer failure"):
        write_tokenized_shards(
            output_dir,
            tokenizer=_FailingByteTokenizer(),
            train_sources=[TokenizedShardSource("train", ["ready"])],
            val_sources=[TokenizedShardSource("val", ["fail"])],
        )

    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []

    original = write_tokenized_shards(
        output_dir,
        tokenizer=good_tokenizer,
        train_sources=[TokenizedShardSource("train", ["original"])],
        val_sources=[TokenizedShardSource("val", ["validation"])],
    )
    original_bytes = (output_dir / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match=r"overwrite=True"):
        write_tokenized_shards(
            output_dir,
            tokenizer=good_tokenizer,
            train_sources=[TokenizedShardSource("train", ["replacement"])],
            val_sources=[TokenizedShardSource("val", ["validation"])],
        )
    with pytest.raises(RuntimeError, match="intentional tokenizer failure"):
        write_tokenized_shards(
            output_dir,
            tokenizer=_FailingByteTokenizer(),
            train_sources=[TokenizedShardSource("train", ["replacement"])],
            val_sources=[TokenizedShardSource("val", ["fail"])],
            overwrite=True,
        )

    assert (output_dir / "manifest.json").read_bytes() == original_bytes
    assert (
        TokenizedShardReader(output_dir, tokenizer=good_tokenizer).manifest == original
    )

    replacement = write_tokenized_shards(
        output_dir,
        tokenizer=good_tokenizer,
        train_sources=[TokenizedShardSource("train", ["replacement"])],
        val_sources=[TokenizedShardSource("val", ["validation"])],
        overwrite=True,
    )

    assert replacement != original
    assert (
        TokenizedShardReader(output_dir, tokenizer=good_tokenizer).manifest
        == replacement
    )
    assert not any(
        path.name.endswith((".tmp", ".backup")) for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest.__setitem__("format_version", 2),
            r"unknown tokenized manifest version 2",
        ),
        (
            lambda manifest: manifest.__setitem__("dtype", "uint32"),
            r"dtype 'uint32'.*vocabulary size 265.*expected 'uint16'",
        ),
        (
            lambda manifest: manifest["splits"]["train"].__setitem__(  # type: ignore[index,union-attr]
                "token_count", 999
            ),
            r"train token_count is inconsistent",
        ),
        (
            lambda manifest: manifest.__setitem__(
                "token_count",
                manifest["token_count"] + 1,  # type: ignore[operator]
            ),
            r"manifest token_count is inconsistent",
        ),
        (
            lambda manifest: manifest["splits"]["train"]["shards"][0].__setitem__(  # type: ignore[index,union-attr]
                "filename", "../outside.bin"
            ),
            r"unsafe or noncanonical filename",
        ),
        (
            lambda manifest: manifest.__setitem__("unexpected", True),
            r"manifest fields.*unexpected=\['unexpected'\]",
        ),
        (
            lambda manifest: manifest["splits"]["val"]["shards"][0].__setitem__(  # type: ignore[index,union-attr]
                "source_shards", ["train-source"]
            ),
            r"source provenance reuses 'train-source'",
        ),
    ],
)
def test_reader_rejects_invalid_manifest_contracts(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    output_dir = tmp_path / "tokenized"
    _write_byte_dataset(output_dir)
    manifest = _read_manifest(output_dir)
    mutation(manifest)
    save_json(manifest, output_dir / "manifest.json")

    with pytest.raises(TokenizedDataError, match=message):
        TokenizedShardReader(output_dir, tokenizer=ByteTokenizer())


def test_reader_rejects_missing_truncated_and_checksum_corrupt_payloads(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing"
    _write_byte_dataset(missing_dir)
    (missing_dir / "train_000000.bin").unlink()
    with pytest.raises(TokenizedDataError, match=r"missing.*train_000000\.bin"):
        TokenizedShardReader(missing_dir, tokenizer=ByteTokenizer())

    truncated_dir = tmp_path / "truncated"
    _write_byte_dataset(truncated_dir)
    truncated_path = truncated_dir / "train_000000.bin"
    truncated_path.write_bytes(truncated_path.read_bytes()[:-1])
    with pytest.raises(TokenizedDataError, match=r"train_000000\.bin size mismatch"):
        TokenizedShardReader(truncated_dir, tokenizer=ByteTokenizer())

    corrupt_dir = tmp_path / "corrupt"
    _write_byte_dataset(corrupt_dir)
    corrupt_path = corrupt_dir / "train_000000.bin"
    payload = bytearray(corrupt_path.read_bytes())
    payload[0] ^= 1
    corrupt_path.write_bytes(payload)
    with pytest.raises(TokenizedDataError, match=r"checksum mismatch"):
        TokenizedShardReader(corrupt_dir, tokenizer=ByteTokenizer())


def test_reader_scans_for_out_of_range_ids_after_checksum_validation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "tokenized"
    _write_byte_dataset(output_dir)
    shard_path = output_dir / "train_000000.bin"
    payload = bytearray(shard_path.read_bytes())
    payload[:2] = np.asarray([265], dtype="<u2").tobytes()
    shard_path.write_bytes(payload)

    manifest = _read_manifest(output_dir)
    manifest["splits"]["train"]["shards"][0]["sha256"] = hashlib.sha256(  # type: ignore[index]
        payload
    ).hexdigest()
    save_json(manifest, output_dir / "manifest.json")

    with pytest.raises(
        TokenizedDataError,
        match=r"train_000000\.bin contains token IDs outside range \[0, 265\)",
    ):
        TokenizedShardReader(output_dir, tokenizer=ByteTokenizer())


def test_reader_rejects_runtime_tokenizer_mismatches(tmp_path: Path) -> None:
    output_dir = tmp_path / "tokenized"
    _write_byte_dataset(output_dir)

    with pytest.raises(TokenizedDataError, match=r"tokenizer identity mismatch"):
        TokenizedShardReader(
            output_dir,
            tokenizer=_DifferentIdentityByteTokenizer(),
        )
    with pytest.raises(TokenizedDataError, match=r"special-token IDs"):
        TokenizedShardReader(
            output_dir,
            tokenizer=_DifferentSpecialsByteTokenizer(),
        )


def test_reader_rejects_shard_symlinks_even_when_the_target_is_valid(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "tokenized"
    _write_byte_dataset(output_dir)
    shard_path = output_dir / "train_000000.bin"
    external_path = tmp_path / "external.bin"
    shard_path.replace(external_path)
    shard_path.symlink_to(external_path)

    with pytest.raises(TokenizedDataError, match=r"not a regular file"):
        TokenizedShardReader(output_dir, tokenizer=ByteTokenizer())


def test_reader_rejects_unreferenced_binary_payloads(tmp_path: Path) -> None:
    output_dir = tmp_path / "tokenized"
    _write_byte_dataset(output_dir)
    (output_dir / "stale.bin").write_bytes(b"stale")

    with pytest.raises(TokenizedDataError, match=r"unreferenced.*stale\.bin"):
        TokenizedShardReader(output_dir, tokenizer=ByteTokenizer())


def test_overwrite_publication_failure_restores_the_previous_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "tokenized"
    tokenizer = ByteTokenizer()
    _write_byte_dataset(output_dir)
    original_manifest = (output_dir / "manifest.json").read_bytes()
    real_replace = tokenized_data.os.replace

    def fail_replacement(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path == output_dir
            and source_path.is_dir()
            and source_path.name.endswith(".tmp")
        ):
            raise OSError("intentional publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(tokenized_data.os, "replace", fail_replacement)

    with pytest.raises(OSError, match="intentional publication failure"):
        write_tokenized_shards(
            output_dir,
            tokenizer=tokenizer,
            train_sources=[TokenizedShardSource("train-source", ["replacement"])],
            val_sources=[TokenizedShardSource("val-source", ["validation"])],
            overwrite=True,
        )

    assert (output_dir / "manifest.json").read_bytes() == original_manifest
    TokenizedShardReader(output_dir, tokenizer=tokenizer)
    assert not any(
        path.name.endswith((".tmp", ".backup")) for path in tmp_path.iterdir()
    )


def test_writer_rejects_split_leakage_and_out_of_range_ids_before_publication(
    tmp_path: Path,
) -> None:
    overlap_output = tmp_path / "overlap"
    with pytest.raises(TokenizedDataError, match=r"both train and val.*shared"):
        write_tokenized_shards(
            overlap_output,
            tokenizer=ByteTokenizer(),
            train_sources=[TokenizedShardSource("shared", ["train"])],
            val_sources=[TokenizedShardSource("shared", ["val"])],
        )
    assert not overlap_output.exists()

    invalid_id_output = tmp_path / "invalid-id"
    with pytest.raises(
        TokenizedDataError,
        match=r"must be in range \[0, 65535\).*65535",
    ):
        write_tokenized_shards(
            invalid_id_output,
            tokenizer=_FixedIdTokenizer(vocab_size=65_535, token_id=65_535),
            train_sources=[TokenizedShardSource("train", ["train"])],
            val_sources=[TokenizedShardSource("val", ["val"])],
        )
    assert not invalid_id_output.exists()
