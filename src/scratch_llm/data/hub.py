"""Verified, atomic parquet acquisition for public SFT Hub datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import BinaryIO, Final, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scratch_llm._validation import require_positive_integer, require_positive_real
from scratch_llm.identity import canonical_json_identity, file_identity
from scratch_llm.utils import save_json


HUB_PARQUET_CACHE_FORMAT: Final = "scratch_llm_sft_hub_parquet"
HUB_PARQUET_CACHE_VERSION: Final = 1
HUGGING_FACE_PARQUET_ENDPOINT: Final = "https://datasets-server.huggingface.co/parquet"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DATASET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256_IDENTITY = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_FILENAME = "manifest.json"
_MANIFEST_FIELDS: Final = frozenset(
    {
        "dataset_identity",
        "format",
        "format_version",
        "row_count",
        "schema_fingerprint",
        "shards",
        "source",
        "source_identity",
    }
)
_SHARD_FIELDS: Final = frozenset(
    {
        "expected_size",
        "local_filename",
        "remote_filename",
        "row_count",
        "schema_fingerprint",
        "sha256",
        "size_bytes",
        "url",
    }
)


class HubParquetError(RuntimeError):
    """Base error for SFT Hub parquet discovery and cache validation."""


class HubParquetDiscoveryError(HubParquetError):
    """The Hub did not return a complete, matching parquet listing."""


class HubParquetCacheError(HubParquetError):
    """A parquet cache could not be safely published or reused."""


class DiscoveryResponseOpener(Protocol):
    """Injectable HTTP boundary for the Hugging Face listing request."""

    def __call__(
        self,
        request: Request,
        *,
        timeout: float,
    ) -> AbstractContextManager[BinaryIO]:
        """Open one binary response."""


@dataclass(frozen=True, slots=True)
class HubDatasetSpec:
    """Pinned repository/subset/split contract for one normalized SFT source."""

    dataset: str
    repository: str
    subset: str
    split: str
    adapter_version: str
    reference_commit: str
    required_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not _DATASET_NAME.fullmatch(
            self.dataset
        ):
            raise ValueError(
                "dataset must use lowercase letters, digits, underscores, or hyphens"
            )
        if not isinstance(self.repository, str):
            raise TypeError("repository must be a string")
        repository_parts = self.repository.split("/")
        if len(repository_parts) != 2 or not all(
            _IDENTIFIER.fullmatch(part) for part in repository_parts
        ):
            raise ValueError("repository must be a validated owner/name identity")
        for label, value in (
            ("subset", self.subset),
            ("split", self.split),
            ("adapter_version", self.adapter_version),
        ):
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{label} must be a validated identifier")
        if (
            not isinstance(self.reference_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.reference_commit) is None
        ):
            raise ValueError("reference_commit must be a full lowercase Git SHA")
        if (
            not isinstance(self.required_columns, tuple)
            or not self.required_columns
            or not all(
                isinstance(column, str) and _IDENTIFIER.fullmatch(column)
                for column in self.required_columns
            )
            or len(set(self.required_columns)) != len(self.required_columns)
        ):
            raise ValueError(
                "required_columns must be a non-empty tuple of unique identifiers"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible source contract."""

        return {
            "adapter_version": self.adapter_version,
            "dataset": self.dataset,
            "repository": self.repository,
            "reference_commit": self.reference_commit,
            "required_columns": list(self.required_columns),
            "split": self.split,
            "subset": self.subset,
        }

    @property
    def source_identity(self) -> str:
        """Return a stable identity for the adapter and Hub coordinates."""

        return canonical_json_identity(self.to_dict())

    @property
    def cache_key(self) -> str:
        """Return a filesystem-safe cache directory name."""

        digest = self.source_identity.removeprefix("sha256:")[:12]
        return f"{self.dataset}--{self.subset}--{self.split}--{digest}"


@dataclass(frozen=True, slots=True)
class RemoteParquetShard:
    """One discovered remote parquet object before local publication."""

    url: str
    remote_filename: str
    expected_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or urlparse(self.url).scheme not in {
            "https",
            "file",
        }:
            raise HubParquetDiscoveryError(
                "parquet URL must use the https or explicit local file scheme"
            )
        if (
            not isinstance(self.remote_filename, str)
            or Path(self.remote_filename).name != self.remote_filename
            or not self.remote_filename.endswith(".parquet")
        ):
            raise HubParquetDiscoveryError(
                "remote parquet filename must be a safe .parquet basename"
            )
        try:
            require_positive_integer(self.expected_size, name="expected_size")
        except (TypeError, ValueError) as error:
            raise HubParquetDiscoveryError(
                "remote parquet expected_size must be a positive integer"
            ) from error


@dataclass(frozen=True, slots=True)
class CachedHubParquetDataset:
    """A completely verified local parquet source and immutable manifest facts."""

    spec: HubDatasetSpec
    directory: Path
    shard_paths: tuple[Path, ...]
    row_count: int
    schema_fingerprint: str
    source_identity: str

    @property
    def manifest_path(self) -> Path:
        """Return the completion manifest written last during staging."""

        return self.directory / _MANIFEST_FILENAME


ParquetDiscovery = Callable[[HubDatasetSpec], Sequence[RemoteParquetShard]]
ParquetDownloader = Callable[[RemoteParquetShard, Path], None]


def discover_hub_parquet(
    spec: HubDatasetSpec,
    *,
    opener: DiscoveryResponseOpener | None = None,
    timeout: float = 30.0,
) -> tuple[RemoteParquetShard, ...]:
    """List complete auto-converted parquet shards through the official API."""

    if not isinstance(spec, HubDatasetSpec):
        raise TypeError("spec must be a HubDatasetSpec")
    try:
        timeout = require_positive_real(timeout, name="timeout")
    except (TypeError, ValueError) as error:
        raise HubParquetDiscoveryError("timeout must be positive") from error
    if opener is None:
        opener = _open_url

    request_url = (
        f"{HUGGING_FACE_PARQUET_ENDPOINT}?{urlencode({'dataset': spec.repository})}"
    )
    request = Request(
        request_url,
        headers={"User-Agent": "scratch-llm/0.1 parquet-discovery"},
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (
        HTTPError,
        URLError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise HubParquetDiscoveryError(
            f"could not discover parquet files for {spec.repository}: {error}"
        ) from error

    if not isinstance(payload, Mapping):
        raise HubParquetDiscoveryError("Hub parquet response must be a JSON object")
    parquet_files = payload.get("parquet_files")
    pending = payload.get("pending")
    failed = payload.get("failed")
    partial = payload.get("partial")
    if not isinstance(parquet_files, list):
        raise HubParquetDiscoveryError(
            "Hub parquet response must contain a parquet_files list"
        )
    if (
        not isinstance(pending, list)
        or not isinstance(failed, list)
        or not isinstance(partial, bool)
    ):
        raise HubParquetDiscoveryError(
            "Hub parquet response has invalid pending/failed/partial status"
        )
    if partial or pending or failed:
        raise HubParquetDiscoveryError(
            "Hub parquet conversion is incomplete; retry after pending/failed jobs clear"
        )

    selected: list[RemoteParquetShard] = []
    for index, raw_file in enumerate(parquet_files):
        if not isinstance(raw_file, Mapping):
            raise HubParquetDiscoveryError(f"parquet_files[{index}] must be an object")
        repository = raw_file.get("dataset")
        if repository != spec.repository:
            raise HubParquetDiscoveryError(
                f"parquet_files[{index}] has conflicting dataset identity "
                f"{repository!r}; expected {spec.repository!r}"
            )
        subset = raw_file.get("config")
        split = raw_file.get("split")
        if not isinstance(subset, str) or not isinstance(split, str):
            raise HubParquetDiscoveryError(
                f"parquet_files[{index}] config and split must be strings"
            )
        if subset != spec.subset or split != spec.split:
            continue
        try:
            selected.append(
                RemoteParquetShard(
                    url=raw_file.get("url"),  # type: ignore[arg-type]
                    remote_filename=raw_file.get("filename"),  # type: ignore[arg-type]
                    expected_size=raw_file.get("size"),  # type: ignore[arg-type]
                )
            )
        except HubParquetDiscoveryError as error:
            raise HubParquetDiscoveryError(
                f"parquet_files[{index}] is invalid: {error}"
            ) from error

    selected.sort(key=lambda shard: (shard.remote_filename, shard.url))
    filenames = [shard.remote_filename for shard in selected]
    if len(filenames) != len(set(filenames)):
        raise HubParquetDiscoveryError(
            "matching Hub parquet files contain duplicate filenames"
        )
    if not selected:
        raise HubParquetDiscoveryError(
            f"no parquet files found for {spec.repository}/{spec.subset}/{spec.split}"
        )
    return tuple(selected)


def download_hub_parquet_shard(
    shard: RemoteParquetShard,
    destination: Path,
    *,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Stream one HTTPS shard into its private staging destination."""

    if urlparse(shard.url).scheme != "https":
        raise HubParquetCacheError("default downloader accepts only HTTPS URLs")
    timeout = require_positive_real(timeout, name="timeout")
    chunk_size = require_positive_integer(chunk_size, name="chunk_size")
    request = Request(
        shard.url,
        headers={"User-Agent": "scratch-llm/0.1 parquet-download"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            with destination.open("xb") as output:
                while chunk := response.read(chunk_size):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except (HTTPError, URLError, OSError) as error:
        raise HubParquetCacheError(
            f"failed to download {shard.remote_filename}: {error}"
        ) from error
    actual_size = destination.stat().st_size
    if actual_size != shard.expected_size:
        raise HubParquetCacheError(
            f"downloaded size for {shard.remote_filename} is {actual_size}, "
            f"expected {shard.expected_size}"
        )


def prepare_hub_parquet_cache(
    spec: HubDatasetSpec,
    cache_root: str | Path,
    *,
    discovery: ParquetDiscovery | None = None,
    downloader: ParquetDownloader | None = None,
) -> CachedHubParquetDataset:
    """Reuse a verified cache or atomically publish a newly downloaded one."""

    if not isinstance(spec, HubDatasetSpec):
        raise TypeError("spec must be a HubDatasetSpec")
    root = Path(cache_root)
    destination = root / spec.cache_key
    if destination.exists():
        return _load_cache_directory(spec, destination)

    if discovery is None:
        discovery = discover_hub_parquet
    if downloader is None:
        downloader = download_hub_parquet_shard
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=root,
            prefix=f".{spec.cache_key}.",
            suffix=".tmp",
        )
    )
    try:
        remote_shards = tuple(discovery(spec))
        if not remote_shards:
            raise HubParquetCacheError("parquet discovery returned no shards")
        if not all(isinstance(shard, RemoteParquetShard) for shard in remote_shards):
            raise HubParquetCacheError(
                "parquet discovery returned an unsupported shard descriptor"
            )
        remote_names = [shard.remote_filename for shard in remote_shards]
        if len(remote_names) != len(set(remote_names)):
            raise HubParquetCacheError(
                "parquet discovery returned duplicate remote filenames"
            )

        local_paths: list[Path] = []
        for index, shard in enumerate(remote_shards):
            local_path = staging / f"shard_{index:05d}.parquet"
            downloader(shard, local_path)
            if not local_path.is_file():
                raise HubParquetCacheError(
                    f"downloader did not produce a regular file for "
                    f"{shard.remote_filename}"
                )
            local_paths.append(local_path)

        manifest = _build_manifest(spec, remote_shards, tuple(local_paths))
        save_json(manifest, staging / _MANIFEST_FILENAME)
        os.replace(staging, destination)
    except HubParquetError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except (OSError, ValueError, pa.ArrowException) as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise HubParquetCacheError(
            f"could not publish parquet cache for {spec.dataset}: {error}"
        ) from error
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _load_cache_directory(spec, destination)


def publish_local_parquet_cache(
    spec: HubDatasetSpec,
    cache_root: str | Path,
    parquet_paths: Sequence[str | Path],
) -> CachedHubParquetDataset:
    """Inject local parquet files through the same staging and manifest checks."""

    sources = tuple(Path(path).resolve() for path in parquet_paths)
    if not sources:
        raise HubParquetCacheError("at least one local parquet file is required")
    for source in sources:
        if not source.is_file():
            raise HubParquetCacheError(
                f"local parquet source is not a regular file: {source}"
            )
        if source.suffix != ".parquet":
            raise HubParquetCacheError(
                f"local parquet source must end in .parquet: {source}"
            )
        if source.stat().st_size <= 0:
            raise HubParquetCacheError(f"local parquet source is empty: {source}")

    shards = tuple(
        RemoteParquetShard(
            url=source.as_uri(),
            remote_filename=source.name,
            expected_size=source.stat().st_size,
        )
        for source in sources
    )
    if len({shard.remote_filename for shard in shards}) != len(shards):
        raise HubParquetCacheError("local parquet basenames must be unique")

    def local_discovery(_spec: HubDatasetSpec) -> tuple[RemoteParquetShard, ...]:
        return shards

    def local_downloader(shard: RemoteParquetShard, destination: Path) -> None:
        source_path = Path(urlparse(shard.url).path)
        try:
            shutil.copyfile(source_path, destination)
        except OSError as error:
            raise HubParquetCacheError(
                f"could not stage local parquet {source_path}: {error}"
            ) from error

    cached = prepare_hub_parquet_cache(
        spec,
        cache_root,
        discovery=local_discovery,
        downloader=local_downloader,
    )
    local_identities = tuple(file_identity(source) for source in sources)
    cached_identities = tuple(file_identity(path) for path in cached.shard_paths)
    if local_identities != cached_identities:
        raise HubParquetCacheError(
            "existing parquet cache conflicts with the requested local parquet data"
        )
    return cached


def load_hub_parquet_cache(
    spec: HubDatasetSpec,
    cache_root: str | Path,
) -> CachedHubParquetDataset:
    """Load and fully revalidate one expected cached source."""

    destination = Path(cache_root) / spec.cache_key
    if not destination.exists():
        raise HubParquetCacheError(
            f"parquet cache does not exist for {spec.dataset}: {destination}"
        )
    return _load_cache_directory(spec, destination)


def _build_manifest(
    spec: HubDatasetSpec,
    remote_shards: tuple[RemoteParquetShard, ...],
    local_paths: tuple[Path, ...],
) -> dict[str, object]:
    if len(remote_shards) != len(local_paths):
        raise AssertionError("remote and local shard counts must agree")
    shard_records: list[dict[str, object]] = []
    reference_schema: pa.Schema | None = None
    schema_fingerprint: str | None = None
    total_rows = 0
    for remote, path in zip(remote_shards, local_paths, strict=True):
        actual_size = path.stat().st_size
        if actual_size != remote.expected_size:
            raise HubParquetCacheError(
                f"staged size for {remote.remote_filename} is {actual_size}, "
                f"expected {remote.expected_size}"
            )
        schema, row_count = _inspect_parquet(path, spec)
        if reference_schema is None:
            reference_schema = schema
            schema_fingerprint = _schema_identity(schema)
        elif not schema.equals(reference_schema, check_metadata=False):
            raise HubParquetCacheError(
                "parquet shard schemas do not match within one source"
            )
        if row_count <= 0:
            raise HubParquetCacheError(f"parquet shard {path.name} has zero rows")
        total_rows += row_count
        shard_records.append(
            {
                "expected_size": remote.expected_size,
                "local_filename": path.name,
                "remote_filename": remote.remote_filename,
                "row_count": row_count,
                "schema_fingerprint": _schema_identity(schema),
                "sha256": file_identity(path),
                "size_bytes": actual_size,
                "url": remote.url,
            }
        )
    if total_rows <= 0 or schema_fingerprint is None:
        raise HubParquetCacheError("parquet source is empty")

    identity_payload = {
        "shards": [
            {
                key: shard[key]
                for key in (
                    "local_filename",
                    "row_count",
                    "schema_fingerprint",
                    "sha256",
                    "size_bytes",
                )
            }
            for shard in shard_records
        ],
        "source_identity": spec.source_identity,
    }
    return {
        "dataset_identity": canonical_json_identity(identity_payload),
        "format": HUB_PARQUET_CACHE_FORMAT,
        "format_version": HUB_PARQUET_CACHE_VERSION,
        "row_count": total_rows,
        "schema_fingerprint": schema_fingerprint,
        "shards": shard_records,
        "source": spec.to_dict(),
        "source_identity": spec.source_identity,
    }


def _load_cache_directory(
    spec: HubDatasetSpec,
    destination: Path,
) -> CachedHubParquetDataset:
    if not destination.is_dir():
        raise HubParquetCacheError(
            f"parquet cache path is not a directory: {destination}"
        )
    manifest_path = destination / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise HubParquetCacheError(
            f"partial parquet cache has no completion manifest: {destination}"
        )
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HubParquetCacheError(
            f"parquet cache manifest is invalid: {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS:
        raise HubParquetCacheError("parquet cache manifest fields are invalid")
    if (
        manifest["format"] != HUB_PARQUET_CACHE_FORMAT
        or manifest["format_version"] != HUB_PARQUET_CACHE_VERSION
    ):
        raise HubParquetCacheError("parquet cache manifest format is unsupported")
    if manifest["source_identity"] != spec.source_identity:
        raise HubParquetCacheError("parquet cache has a conflicting source identity")
    if manifest["source"] != spec.to_dict():
        raise HubParquetCacheError("parquet cache source contract is conflicting")
    dataset_identity = manifest["dataset_identity"]
    schema_identity = manifest["schema_fingerprint"]
    if not isinstance(dataset_identity, str) or not _SHA256_IDENTITY.fullmatch(
        dataset_identity
    ):
        raise HubParquetCacheError("parquet cache dataset identity is invalid")
    if not isinstance(schema_identity, str) or not _SHA256_IDENTITY.fullmatch(
        schema_identity
    ):
        raise HubParquetCacheError("parquet cache schema fingerprint is invalid")
    manifest_row_count = _manifest_positive_int(
        manifest["row_count"],
        label="row_count",
    )

    raw_shards = manifest["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise HubParquetCacheError("parquet cache manifest has no shards")
    expected_files = {_MANIFEST_FILENAME}
    shard_paths: list[Path] = []
    identity_shards: list[dict[str, object]] = []
    local_filenames: set[str] = set()
    remote_filenames: set[str] = set()
    reference_schema: pa.Schema | None = None
    total_rows = 0
    for index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping) or set(raw_shard) != _SHARD_FIELDS:
            raise HubParquetCacheError(f"parquet cache shard record {index} is invalid")
        local_filename = raw_shard["local_filename"]
        if (
            not isinstance(local_filename, str)
            or Path(local_filename).name != local_filename
            or not local_filename.endswith(".parquet")
        ):
            raise HubParquetCacheError(
                f"parquet cache shard record {index} has an unsafe filename"
            )
        path = destination / local_filename
        if local_filename in local_filenames:
            raise HubParquetCacheError(
                "parquet cache manifest contains duplicate local filenames"
            )
        local_filenames.add(local_filename)
        expected_files.add(local_filename)
        if not path.is_file():
            raise HubParquetCacheError(f"cached parquet shard is missing: {path}")
        size_bytes = _manifest_positive_int(
            raw_shard["size_bytes"],
            label=f"shards[{index}].size_bytes",
        )
        expected_size = _manifest_positive_int(
            raw_shard["expected_size"],
            label=f"shards[{index}].expected_size",
        )
        remote_filename = raw_shard["remote_filename"]
        url = raw_shard["url"]
        try:
            remote = RemoteParquetShard(
                url=url,  # type: ignore[arg-type]
                remote_filename=remote_filename,  # type: ignore[arg-type]
                expected_size=expected_size,
            )
        except HubParquetDiscoveryError as error:
            raise HubParquetCacheError(
                f"parquet cache shard record {index} has invalid remote metadata: "
                f"{error}"
            ) from error
        if remote.remote_filename in remote_filenames:
            raise HubParquetCacheError(
                "parquet cache manifest contains duplicate remote filenames"
            )
        remote_filenames.add(remote.remote_filename)
        if expected_size != size_bytes:
            raise HubParquetCacheError(
                f"shards[{index}] expected and actual sizes are inconsistent"
            )
        if path.stat().st_size != size_bytes:
            raise HubParquetCacheError(
                f"cached parquet shard size does not match manifest: {path}"
            )
        sha256 = raw_shard["sha256"]
        if not isinstance(sha256, str) or not _SHA256_IDENTITY.fullmatch(sha256):
            raise HubParquetCacheError(
                f"shards[{index}].sha256 is not a SHA-256 identity"
            )
        if file_identity(path) != sha256:
            raise HubParquetCacheError(
                f"cached parquet shard checksum does not match manifest: {path}"
            )
        schema, row_count = _inspect_parquet(path, spec)
        if reference_schema is None:
            reference_schema = schema
        elif not schema.equals(reference_schema, check_metadata=False):
            raise HubParquetCacheError("cached parquet shard schemas do not match")
        recorded_rows = _manifest_positive_int(
            raw_shard["row_count"],
            label=f"shards[{index}].row_count",
        )
        if row_count != recorded_rows:
            raise HubParquetCacheError(
                f"cached parquet row count does not match manifest: {path}"
            )
        shard_schema_identity = _schema_identity(schema)
        if raw_shard["schema_fingerprint"] != shard_schema_identity:
            raise HubParquetCacheError(
                f"cached parquet schema does not match manifest: {path}"
            )
        total_rows += row_count
        shard_paths.append(path)
        identity_shards.append(
            {
                "local_filename": local_filename,
                "row_count": row_count,
                "schema_fingerprint": shard_schema_identity,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )

    actual_files = {path.name for path in destination.iterdir()}
    if actual_files != expected_files:
        raise HubParquetCacheError(
            "parquet cache contains missing or unexpected files; "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    if manifest_row_count != total_rows:
        raise HubParquetCacheError("parquet cache total row count is inconsistent")
    if reference_schema is None:
        raise HubParquetCacheError("parquet cache is empty")
    schema_fingerprint = _schema_identity(reference_schema)
    if schema_identity != schema_fingerprint:
        raise HubParquetCacheError("parquet cache schema fingerprint is inconsistent")
    expected_identity = canonical_json_identity(
        {
            "shards": identity_shards,
            "source_identity": spec.source_identity,
        }
    )
    if dataset_identity != expected_identity:
        raise HubParquetCacheError("parquet cache dataset identity is inconsistent")
    return CachedHubParquetDataset(
        spec=spec,
        directory=destination,
        shard_paths=tuple(shard_paths),
        row_count=total_rows,
        schema_fingerprint=schema_fingerprint,
        source_identity=expected_identity,
    )


def _inspect_parquet(path: Path, spec: HubDatasetSpec) -> tuple[pa.Schema, int]:
    try:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        row_count = parquet.metadata.num_rows
    except (OSError, pa.ArrowException) as error:
        raise HubParquetCacheError(
            f"parquet shard is corrupt or unreadable: {path}: {error}"
        ) from error
    missing = sorted(set(spec.required_columns) - set(schema.names))
    if missing:
        raise HubParquetCacheError(
            f"parquet shard is missing required columns {missing}: {path}"
        )
    return schema, row_count


def _schema_identity(schema: pa.Schema) -> str:
    return "sha256:" + hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _manifest_positive_int(value: object, *, label: str) -> int:
    try:
        return require_positive_integer(value, name=label)
    except (TypeError, ValueError) as error:
        raise HubParquetCacheError(f"{label} must be a positive integer") from error


def _open_url(
    request: Request,
    *,
    timeout: float,
) -> AbstractContextManager[BinaryIO]:
    return urlopen(request, timeout=timeout)  # type: ignore[return-value] # noqa: S310


__all__ = [
    "HUB_PARQUET_CACHE_FORMAT",
    "HUB_PARQUET_CACHE_VERSION",
    "HUGGING_FACE_PARQUET_ENDPOINT",
    "CachedHubParquetDataset",
    "HubDatasetSpec",
    "HubParquetCacheError",
    "HubParquetDiscoveryError",
    "HubParquetError",
    "RemoteParquetShard",
    "discover_hub_parquet",
    "download_hub_parquet_shard",
    "load_hub_parquet_cache",
    "prepare_hub_parquet_cache",
    "publish_local_parquet_cache",
]
