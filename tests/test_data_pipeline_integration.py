"""Bounded offline regression coverage for the complete Phase 2 data path."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from scratch_llm.climbmix import (
    ClimbMixDownloadError,
    download_climbmix_target,
    download_climbmix_targets,
    plan_climbmix_downloads,
)
from scratch_llm.data import (
    DocumentPackingTokenLoader,
    RandomOffsetTokenLoader,
    TokenizedDataError,
    TokenizedShardReader,
    create_token_loader,
    list_parquet_files,
    select_parquet_files,
    write_tokenized_parquet_shards,
)
from scratch_llm.data_stats import compute_raw_data_statistics
from scratch_llm.tokenizer import ByteTokenizer
from scratch_llm.utils import save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        expected_size: int | None = None,
    ) -> None:
        self.status = 200
        self.headers: Mapping[str, str] = {
            "Content-Length": str(len(body) if expected_size is None else expected_size)
        }
        self._body = BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


class _InterruptingResponse(_FakeResponse):
    def __init__(self, body: bytes) -> None:
        super().__init__(body, expected_size=len(body) + 1)
        self._read_once = False

    def read(self, size: int = -1) -> bytes:
        if self._read_once:
            raise OSError("offline transport interrupted")
        self._read_once = True
        return super().read(size)


class _ScriptedOpener:
    def __init__(self, *responses: _FakeResponse) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, int | None, float]] = []

    @contextmanager
    def __call__(
        self,
        url: str,
        *,
        offset: int | None,
        timeout: float,
    ) -> Iterator[_FakeResponse]:
        self.calls.append((url, offset, timeout))
        try:
            response = next(self._responses)
        except StopIteration as error:
            raise AssertionError("unexpected network request") from error
        yield response


class _DifferentIdentityTokenizer(ByteTokenizer):
    def get_identity(self) -> str:
        return "sha256:" + "0" * 64


def _copy_raw_fixture(destination: Path) -> Path:
    shutil.copytree(PARQUET_FIXTURE_DIR, destination)
    return destination


def _batches_equal(
    first: tuple[torch.Tensor, ...],
    second: tuple[torch.Tensor, ...],
) -> bool:
    return len(first) == len(second) and all(
        torch.equal(left, right) for left, right in zip(first, second, strict=True)
    )


def test_local_parquet_to_restartable_flat_and_packed_batches(
    tmp_path: Path,
) -> None:
    raw_dir = _copy_raw_fixture(tmp_path / "raw")
    tokenized_dir = tmp_path / "tokenized"
    tokenizer = ByteTokenizer()

    discovered = list_parquet_files(raw_dir)
    assert [path.name for path in discovered] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_06542.parquet",
    ]
    assert [path.name for path in select_parquet_files(discovered, "train")] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
    ]
    assert [path.name for path in select_parquet_files(discovered, "val")] == [
        "shard_06542.parquet"
    ]

    statistics = compute_raw_data_statistics(
        raw_dir,
        num_train_shards=2,
        include_validation=True,
        batch_size=2,
    )
    assert statistics.train.documents == 6
    assert statistics.train.utf8_bytes == 147
    assert statistics.validation.documents == 3
    assert statistics.validation.utf8_bytes == 63

    manifest = write_tokenized_parquet_shards(
        raw_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=2,
        batch_size=2,
    )
    assert manifest.token_count == 210
    assert manifest.document_count == 9
    assert manifest.tokenizer_identity == tokenizer.get_identity()
    assert [shard.source_shards for shard in manifest.splits["train"].shards] == [
        ("shard_00000.parquet",),
        ("shard_00001.parquet",),
    ]
    assert [shard.source_shards for shard in manifest.splits["val"].shards] == [
        ("shard_06542.parquet",)
    ]

    shutil.rmtree(raw_dir)
    with pytest.raises(FileNotFoundError, match="parquet data directory"):
        compute_raw_data_statistics(raw_dir)

    with TokenizedShardReader(tokenized_dir, tokenizer=tokenizer) as reader:
        assert reader.manifest == manifest
        assert all(
            isinstance(shard, np.memmap)
            for split in ("train", "val")
            for shard in reader.shards(split)
        )
        assert sum(len(shard) for shard in reader.shards("train")) == 147
        assert sum(len(shard) for shard in reader.shards("val")) == 63

        flat = create_token_loader(
            reader,
            strategy="flat",
            split="train",
            batch_size=3,
            seq_len=8,
            seed=17,
        )
        packed = create_token_loader(
            reader,
            strategy="packed",
            split="train",
            batch_size=2,
            seq_len=8,
            seed=23,
        )
        assert isinstance(flat, RandomOffsetTokenLoader)
        assert isinstance(packed, DocumentPackingTokenLoader)

        flat_batch = next(flat)
        flat_state = json.loads(json.dumps(flat.state_dict()))
        expected_flat = next(flat)
        packed_batch = next(packed)
        packed_state = json.loads(json.dumps(packed.state_dict()))
        expected_packed = next(packed)

    flat_inputs, flat_targets = flat_batch
    assert flat_inputs.shape == flat_targets.shape == (3, 8)
    assert torch.equal(flat_targets[:, :-1], flat_inputs[:, 1:])
    packed_inputs, packed_targets, loss_mask = packed_batch
    assert packed_inputs.shape == packed_targets.shape == loss_mask.shape == (2, 8)
    assert torch.equal(packed_targets[:, :-1], packed_inputs[:, 1:])
    assert loss_mask.dtype == torch.bool
    for values in (flat_inputs, flat_targets, packed_inputs, packed_targets):
        assert torch.all((0 <= values) & (values < tokenizer.get_vocab_size()))

    with TokenizedShardReader(tokenized_dir, tokenizer=tokenizer) as reader:
        resumed_flat = create_token_loader(
            reader,
            strategy="flat",
            split="train",
            batch_size=3,
            seq_len=8,
            seed=999,
        )
        resumed_packed = create_token_loader(
            reader,
            strategy="packed",
            split="train",
            batch_size=2,
            seq_len=8,
            seed=999,
        )
        assert isinstance(resumed_flat, RandomOffsetTokenLoader)
        assert isinstance(resumed_packed, DocumentPackingTokenLoader)
        resumed_flat.load_state_dict(flat_state)
        resumed_packed.load_state_dict(packed_state)

        assert _batches_equal(expected_flat, next(resumed_flat))
        assert _batches_equal(expected_packed, next(resumed_packed))


def test_fake_downloads_feed_discovery_while_interrupted_parts_do_not(
    tmp_path: Path,
) -> None:
    downloaded_dir = tmp_path / "downloads"
    train_payload = (PARQUET_FIXTURE_DIR / "shard_00000.parquet").read_bytes()
    interrupted_payload = (PARQUET_FIXTURE_DIR / "shard_00001.parquet").read_bytes()
    validation_payload = (PARQUET_FIXTURE_DIR / "shard_06542.parquet").read_bytes()
    targets = plan_climbmix_downloads(
        num_train_shards=1,
        include_val=True,
        data_dir=downloaded_dir,
        base_url="https://offline.invalid/climbmix",
    )
    opener = _ScriptedOpener(
        _FakeResponse(train_payload),
        _FakeResponse(validation_payload),
    )

    summary = download_climbmix_targets(
        targets,
        opener=opener,
        backoff_base=0,
        sleep=lambda _: None,
    )

    assert summary.downloaded_shards == summary.ready_shards == 2
    assert [path.name for path in list_parquet_files(downloaded_dir)] == [
        "shard_00000.parquet",
        "shard_06542.parquet",
    ]
    interrupted_target = plan_climbmix_downloads(
        num_train_shards=2,
        include_val=False,
        data_dir=downloaded_dir,
        base_url="https://offline.invalid/climbmix",
    )[1]

    with pytest.raises(
        ClimbMixDownloadError,
        match=r"failed after 1 attempt.*restartable partial retained",
    ):
        download_climbmix_target(
            interrupted_target,
            opener=_ScriptedOpener(_InterruptingResponse(interrupted_payload)),
            max_attempts=1,
            backoff_base=0,
            sleep=lambda _: None,
        )

    assert interrupted_target.temporary_path.read_bytes() == interrupted_payload
    assert not interrupted_target.destination.exists()
    assert [path.name for path in list_parquet_files(downloaded_dir)] == [
        "shard_00000.parquet",
        "shard_06542.parquet",
    ]

    manifest = write_tokenized_parquet_shards(
        downloaded_dir,
        tmp_path / "tokenized",
        tokenizer=ByteTokenizer(),
        num_train_shards=2,
    )
    assert manifest.splits["train"].shards[0].source_shards == ("shard_00000.parquet",)
    assert manifest.splits["val"].shards[0].source_shards == ("shard_06542.parquet",)


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("manifest", r"unknown tokenized manifest version 999"),
        ("payload", r"train_000000\.bin checksum mismatch"),
        ("tokenizer", r"tokenizer identity mismatch"),
        ("split-leakage", r"source provenance reuses"),
    ),
)
def test_corruption_and_mismatches_fail_at_manifest_reader_boundary(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    raw_dir = _copy_raw_fixture(tmp_path / "raw")
    tokenized_dir = tmp_path / "tokenized"
    tokenizer = ByteTokenizer()
    write_tokenized_parquet_shards(
        raw_dir,
        tokenized_dir,
        tokenizer=tokenizer,
        num_train_shards=2,
    )
    runtime_tokenizer = tokenizer

    if failure == "manifest":
        manifest = json.loads(
            (tokenized_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["format_version"] = 999
        save_json(manifest, tokenized_dir / "manifest.json")
    elif failure == "payload":
        payload_path = tokenized_dir / "train_000000.bin"
        payload = bytearray(payload_path.read_bytes())
        payload[0] ^= 1
        payload_path.write_bytes(payload)
    elif failure == "tokenizer":
        runtime_tokenizer = _DifferentIdentityTokenizer()
    elif failure == "split-leakage":
        manifest = json.loads(
            (tokenized_dir / "manifest.json").read_text(encoding="utf-8")
        )
        train_source = manifest["splits"]["train"]["shards"][0]["source_shards"][0]
        manifest["splits"]["val"]["shards"][0]["source_shards"] = [train_source]
        save_json(manifest, tokenized_dir / "manifest.json")
    else:
        raise AssertionError(f"unhandled failure fixture {failure!r}")

    with pytest.raises(TokenizedDataError, match=message):
        TokenizedShardReader(tokenized_dir, tokenizer=runtime_tokenizer)


def test_data_commands_smoke_in_subprocess_without_network_or_gpu(
    tmp_path: Path,
) -> None:
    raw_dir = _copy_raw_fixture(tmp_path / "raw")
    report_path = tmp_path / "metrics" / "data_stats.json"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""

    download = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.download_climbmix",
            "--num-train-shards",
            "2",
            "--include-val",
            "--data-dir",
            str(raw_dir),
            "--base-url",
            "http://127.0.0.1:1",
            "--max-attempts",
            "1",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert download.returncode == 0
    assert json.loads(download.stdout) == {
        "data_dir": str(raw_dir),
        "downloaded_shards": 0,
        "planned_shards": 3,
        "ready_shards": 3,
        "skipped_shards": 3,
        "total_bytes": sum(path.stat().st_size for path in list_parquet_files(raw_dir)),
    }

    statistics = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.data_stats",
            "--data-dir",
            str(raw_dir),
            "--num-train-shards",
            "2",
            "--include-val",
            "--batch-size",
            "2",
            "--output",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert statistics.returncode == 0
    assert statistics.stderr == ""
    assert "total: 3 shards, 9 documents, 192 characters, 210 UTF-8 bytes" in (
        statistics.stdout
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["total"] == {
        "characters": 192,
        "documents": 9,
        "selected_shard_count": 3,
        "utf8_bytes": 210,
    }
