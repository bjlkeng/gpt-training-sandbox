"""Verified, atomic Hugging Face parquet cache tests for SFT sources."""

from __future__ import annotations

from contextlib import AbstractContextManager
import io
import json
from pathlib import Path
import shutil
from urllib.request import Request

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from scratch_llm.chat.datasets import get_sft_dataset_spec
from scratch_llm.chat.hub import (
    HUB_PARQUET_CACHE_FORMAT,
    HubParquetCacheError,
    HubParquetDiscoveryError,
    RemoteParquetShard,
    discover_hub_parquet,
    load_hub_parquet_cache,
    prepare_hub_parquet_cache,
    publish_local_parquet_cache,
)


class _JSONResponse(io.BytesIO):
    def __enter__(self) -> _JSONResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _json_opener(
    payload: object,
    *,
    observed: list[str] | None = None,
):
    encoded = json.dumps(payload).encode("utf-8")

    def open_response(
        request: Request,
        *,
        timeout: float,
    ) -> AbstractContextManager[_JSONResponse]:
        assert timeout > 0
        if observed is not None:
            observed.append(request.full_url)
        return _JSONResponse(encoded)

    return open_response


def _smoltalk_rows(prefix: str, count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"{prefix} user {index}"},
                {"role": "assistant", "content": f"{prefix} assistant {index}"},
            ]
        }
        for index in range(count)
    ]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _local_discovery(paths: tuple[Path, ...]):
    shards = tuple(
        RemoteParquetShard(
            url=path.resolve().as_uri(),
            remote_filename=path.name,
            expected_size=path.stat().st_size,
        )
        for path in paths
    )

    def discover(_spec):
        return shards

    return discover


def _copy_download(shard: RemoteParquetShard, destination: Path) -> None:
    source = Path(shard.url.removeprefix("file://"))
    shutil.copyfile(source, destination)


def test_official_discovery_filters_and_validates_exact_source_identity() -> None:
    spec = get_sft_dataset_spec("mmlu", "test")
    observed: list[str] = []
    payload = {
        "parquet_files": [
            {
                "dataset": "cais/mmlu",
                "config": "all",
                "split": "auxiliary_train",
                "url": "https://huggingface.co/ignored.parquet",
                "filename": "0000.parquet",
                "size": 5,
            },
            {
                "dataset": "cais/mmlu",
                "config": "all",
                "split": "test",
                "url": "https://huggingface.co/test-1.parquet",
                "filename": "0001.parquet",
                "size": 7,
            },
            {
                "dataset": "cais/mmlu",
                "config": "all",
                "split": "test",
                "url": "https://huggingface.co/test-0.parquet",
                "filename": "0000.parquet",
                "size": 6,
            },
        ],
        "pending": [],
        "failed": [],
        "partial": False,
    }

    shards = discover_hub_parquet(
        spec,
        opener=_json_opener(payload, observed=observed),
    )

    assert [shard.remote_filename for shard in shards] == [
        "0000.parquet",
        "0001.parquet",
    ]
    assert [shard.expected_size for shard in shards] == [6, 7]
    assert observed == [
        "https://datasets-server.huggingface.co/parquet?dataset=cais%2Fmmlu"
    ]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {"parquet_files": [], "pending": [], "failed": [], "partial": False},
            "no parquet",
        ),
        (
            {"parquet_files": [], "pending": ["job"], "failed": [], "partial": True},
            "incomplete",
        ),
        (
            {
                "parquet_files": [
                    {
                        "dataset": "wrong/repo",
                        "config": "default",
                        "split": "train",
                        "url": "https://huggingface.co/a.parquet",
                        "filename": "a.parquet",
                        "size": 1,
                    }
                ],
                "pending": [],
                "failed": [],
                "partial": False,
            },
            "dataset identity",
        ),
    ],
)
def test_discovery_rejects_empty_incomplete_or_conflicting_responses(
    payload: object,
    match: str,
) -> None:
    spec = get_sft_dataset_spec("smoltalk", "train")
    with pytest.raises(HubParquetDiscoveryError, match=match):
        discover_hub_parquet(spec, opener=_json_opener(payload))


def test_cache_downloads_to_staging_and_publishes_verified_manifest(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_rows(first, _smoltalk_rows("first"))
    _write_rows(second, _smoltalk_rows("second", count=3))
    spec = get_sft_dataset_spec("smoltalk", "train")
    calls: list[str] = []

    def downloader(shard: RemoteParquetShard, destination: Path) -> None:
        assert destination.parent.name.endswith(".tmp")
        calls.append(shard.remote_filename)
        _copy_download(shard, destination)

    cached = prepare_hub_parquet_cache(
        spec,
        tmp_path / "cache",
        discovery=_local_discovery((first, second)),
        downloader=downloader,
    )

    manifest_path = cached.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == ["first.parquet", "second.parquet"]
    assert cached.row_count == 5
    assert cached.source_identity.startswith("sha256:")
    assert manifest["format"] == HUB_PARQUET_CACHE_FORMAT
    assert manifest["source"]["repository"] == "HuggingFaceTB/smol-smoltalk"
    assert manifest["row_count"] == 5
    assert [path.name for path in cached.shard_paths] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
    ]
    assert not list((tmp_path / "cache").glob(".*.tmp"))

    def unexpected_discovery(_spec):
        raise AssertionError("a verified cache must be reusable offline")

    reused = prepare_hub_parquet_cache(
        spec,
        tmp_path / "cache",
        discovery=unexpected_discovery,
        downloader=downloader,
    )
    assert reused == cached


def test_failed_or_invalid_download_never_publishes_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_rows(source, _smoltalk_rows("source"))
    spec = get_sft_dataset_spec("smoltalk", "train")

    def corrupt_download(_shard: RemoteParquetShard, destination: Path) -> None:
        destination.write_bytes(b"not parquet")

    with pytest.raises(HubParquetCacheError, match="parquet"):
        prepare_hub_parquet_cache(
            spec,
            tmp_path / "cache",
            discovery=_local_discovery((source,)),
            downloader=corrupt_download,
        )

    assert not (tmp_path / "cache" / spec.cache_key).exists()
    assert not list((tmp_path / "cache").glob(".*.tmp"))


def test_cache_rejects_partial_corrupt_conflicting_empty_and_mixed_schema_data(
    tmp_path: Path,
) -> None:
    spec = get_sft_dataset_spec("smoltalk", "train")
    cache_root = tmp_path / "partial-cache"
    partial = cache_root / spec.cache_key
    partial.mkdir(parents=True)
    (partial / "shard_00000.parquet").write_bytes(b"partial")
    with pytest.raises(HubParquetCacheError, match="partial.*manifest"):
        prepare_hub_parquet_cache(
            spec,
            cache_root,
            discovery=lambda _spec: (),
            downloader=_copy_download,
        )

    good = tmp_path / "good.parquet"
    _write_rows(good, _smoltalk_rows("good"))
    cached = publish_local_parquet_cache(spec, tmp_path / "corrupt-cache", (good,))
    cached.shard_paths[0].write_bytes(b"corrupt")
    with pytest.raises(HubParquetCacheError, match="size|checksum|parquet"):
        load_hub_parquet_cache(spec, tmp_path / "corrupt-cache")

    conflict = publish_local_parquet_cache(
        spec,
        tmp_path / "conflict-cache",
        (good,),
    )
    manifest_path = conflict.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_identity"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HubParquetCacheError, match="conflicting source identity"):
        load_hub_parquet_cache(spec, tmp_path / "conflict-cache")

    empty = tmp_path / "empty.parquet"
    empty_schema = pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                        ]
                    )
                ),
            )
        ]
    )
    pq.write_table(pa.Table.from_pylist([], schema=empty_schema), empty)
    with pytest.raises(HubParquetCacheError, match="zero rows|empty"):
        publish_local_parquet_cache(spec, tmp_path / "empty-cache", (empty,))

    mismatched = tmp_path / "mismatched.parquet"
    pq.write_table(pa.table({"wrong": ["column"]}), mismatched)
    with pytest.raises(HubParquetCacheError, match="required columns"):
        publish_local_parquet_cache(
            spec,
            tmp_path / "schema-cache",
            (mismatched,),
        )

    other_schema = tmp_path / "other-schema.parquet"
    rows = _smoltalk_rows("other")
    for row in rows:
        row["extra"] = 1
    _write_rows(other_schema, rows)
    with pytest.raises(HubParquetCacheError, match="schemas do not match"):
        publish_local_parquet_cache(
            spec,
            tmp_path / "mixed-cache",
            (good, other_schema),
        )


def test_local_publication_rejects_different_data_for_an_existing_contract(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_rows(first, _smoltalk_rows("first"))
    _write_rows(second, _smoltalk_rows("different"))
    spec = get_sft_dataset_spec("smoltalk", "train")

    publish_local_parquet_cache(spec, tmp_path / "cache", (first,))

    with pytest.raises(HubParquetCacheError, match="conflicts.*local parquet"):
        publish_local_parquet_cache(spec, tmp_path / "cache", (second,))
