"""Validated, atomic storage for tokenized document shards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Final, Literal

import numpy as np

from scratch_llm._validation import (
    JsonValueValidator,
    require_integer,
    require_positive_integer,
)
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.utils import save_json


TOKENIZED_SHARD_FORMAT: Final = "scratch_llm_tokenized_shards"
TOKENIZED_SHARD_FORMAT_VERSION: Final = 1
TOKENIZED_MANIFEST_NAME: Final = "manifest.json"
_MANIFEST_SCHEMA_LABEL: Final = f"version {TOKENIZED_SHARD_FORMAT_VERSION}"
_SPLITS: Final = ("train", "val")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_UINT32_VOCAB_SIZE = 2**32
_MANIFEST_KEYS = frozenset(
    {
        "byte_count",
        "byte_order",
        "document_count",
        "dtype",
        "format",
        "format_version",
        "special_token_ids",
        "splits",
        "token_count",
        "tokenizer_identity",
        "vocab_size",
    }
)
_SPLIT_KEYS = frozenset({"byte_count", "document_count", "shards", "token_count"})
_SHARD_KEYS = frozenset(
    {
        "byte_count",
        "document_count",
        "document_token_counts",
        "filename",
        "index",
        "sha256",
        "source_shards",
        "token_count",
    }
)

TokenDType = Literal["uint16", "uint32"]
TokenizedSplit = Literal["train", "val"]


class TokenizedDataError(ValueError):
    """A tokenized dataset violates its committed storage contract."""


_JSON_VALUES = JsonValueValidator(TokenizedDataError)


@dataclass(frozen=True)
class TokenizedShardSource:
    """One provenance identity and its single-pass stream of documents."""

    identity: str
    documents: Iterable[str]


@dataclass(frozen=True)
class TokenizedShardManifest:
    """Validated metadata for one binary token payload."""

    index: int
    filename: str
    token_count: int
    document_count: int
    document_token_counts: tuple[int, ...]
    byte_count: int
    sha256: str
    source_shards: tuple[str, ...]


@dataclass(frozen=True)
class TokenizedDocumentSpan:
    """One validated document boundary within a mapped token shard."""

    split: TokenizedSplit
    shard_index: int
    document_index: int
    start: int
    stop: int

    @property
    def token_count(self) -> int:
        """Return the number of ordinary tokens in this document."""

        return self.stop - self.start


@dataclass(frozen=True)
class TokenizedSplitManifest:
    """Validated totals and ordered shards for one dataset split."""

    token_count: int
    document_count: int
    byte_count: int
    shards: tuple[TokenizedShardManifest, ...]


@dataclass(frozen=True)
class TokenizedDatasetManifest:
    """Versioned contract for a complete tokenized dataset."""

    format_version: int
    dtype: TokenDType
    vocab_size: int
    tokenizer_identity: str
    special_token_ids: Mapping[str, int]
    token_count: int
    document_count: int
    byte_count: int
    splits: Mapping[str, TokenizedSplitManifest]

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible manifest representation."""

        return {
            "byte_count": self.byte_count,
            "byte_order": "little",
            "document_count": self.document_count,
            "dtype": self.dtype,
            "format": TOKENIZED_SHARD_FORMAT,
            "format_version": self.format_version,
            "special_token_ids": dict(self.special_token_ids),
            "splits": {
                split: {
                    "byte_count": split_manifest.byte_count,
                    "document_count": split_manifest.document_count,
                    "shards": [
                        {
                            "byte_count": shard.byte_count,
                            "document_count": shard.document_count,
                            "document_token_counts": list(shard.document_token_counts),
                            "filename": shard.filename,
                            "index": shard.index,
                            "sha256": shard.sha256,
                            "source_shards": list(shard.source_shards),
                            "token_count": shard.token_count,
                        }
                        for shard in split_manifest.shards
                    ],
                    "token_count": split_manifest.token_count,
                }
                for split, split_manifest in self.splits.items()
            },
            "token_count": self.token_count,
            "tokenizer_identity": self.tokenizer_identity,
            "vocab_size": self.vocab_size,
        }


def tokenized_manifest_identity(manifest: TokenizedDatasetManifest) -> str:
    """Return the canonical SHA-256 identity of a validated manifest."""

    if not isinstance(manifest, TokenizedDatasetManifest):
        raise TypeError(
            "manifest must be a TokenizedDatasetManifest, got "
            f"{type(manifest).__name__}"
        )
    payload = json.dumps(
        manifest.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def write_tokenized_shards(
    output_dir: str | os.PathLike[str],
    *,
    tokenizer: Tokenizer,
    train_sources: Iterable[TokenizedShardSource],
    val_sources: Iterable[TokenizedShardSource],
    overwrite: bool = False,
) -> TokenizedDatasetManifest:
    """Stream documents into a staged dataset and atomically publish it.

    Each source becomes one deterministically numbered payload. Existing output
    is rejected by default. With ``overwrite=True``, a complete existing
    directory remains available until a fully validated replacement is ready.
    """

    if not isinstance(overwrite, bool):
        raise TypeError(f"overwrite must be a boolean, got {type(overwrite).__name__}")
    destination = _validate_output_directory(output_dir)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise NotADirectoryError(
                f"tokenized output path is not a regular directory: {destination}"
            )
        if not overwrite:
            raise FileExistsError(
                f"tokenized output already exists; pass overwrite=True to replace it: "
                f"{destination}"
            )

    vocab_size, tokenizer_identity, special_token_ids = _tokenizer_metadata(tokenizer)
    dtype = _dtype_for_vocab_size(vocab_size)
    normalized_sources = {
        "train": _normalize_sources(train_sources, split="train"),
        "val": _normalize_sources(val_sources, split="val"),
    }
    train_identities = {source.identity for source in normalized_sources["train"]}
    val_identities = {source.identity for source in normalized_sources["val"]}
    overlapping_identities = sorted(train_identities & val_identities)
    if overlapping_identities:
        raise TokenizedDataError(
            "source identities must not appear in both train and val splits: "
            f"{overlapping_identities}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
    )
    try:
        split_manifests = {
            split: _write_split(
                staging_dir,
                split=split,
                sources=normalized_sources[split],
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                dtype=dtype,
            )
            for split in _SPLITS
        }
        manifest = TokenizedDatasetManifest(
            format_version=TOKENIZED_SHARD_FORMAT_VERSION,
            dtype=dtype,
            vocab_size=vocab_size,
            tokenizer_identity=tokenizer_identity,
            special_token_ids=MappingProxyType(dict(special_token_ids)),
            token_count=sum(
                split_manifest.token_count
                for split_manifest in split_manifests.values()
            ),
            document_count=sum(
                split_manifest.document_count
                for split_manifest in split_manifests.values()
            ),
            byte_count=sum(
                split_manifest.byte_count for split_manifest in split_manifests.values()
            ),
            splits=MappingProxyType(split_manifests),
        )
        save_json(manifest.to_dict(), staging_dir / TOKENIZED_MANIFEST_NAME)

        with TokenizedShardReader(staging_dir, tokenizer=tokenizer):
            pass
        _publish_staged_directory(
            staging_dir,
            destination,
            overwrite=overwrite,
        )
        return manifest
    finally:
        if staging_dir.exists():
            _remove_private_path(staging_dir)


class TokenizedShardReader:
    """Validate a complete tokenized dataset before exposing read-only memmaps."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        tokenizer: Tokenizer,
    ) -> None:
        manifest_path, dataset_dir = _resolve_manifest_path(path)
        manifest = _load_manifest(manifest_path)
        _validate_tokenizer_match(manifest, tokenizer)
        _reject_unreferenced_payloads(dataset_dir, manifest)

        opened: dict[str, list[np.memmap]] = {split: [] for split in _SPLITS}
        try:
            for split in _SPLITS:
                for shard in manifest.splits[split].shards:
                    opened[split].append(
                        _validate_and_map_shard(
                            dataset_dir,
                            shard,
                            dtype=manifest.dtype,
                            vocab_size=manifest.vocab_size,
                        )
                    )
        except BaseException:
            _close_memmaps(opened)
            raise

        self.manifest_path = manifest_path
        self.dataset_dir = dataset_dir
        self.manifest = manifest
        self._shards = {split: tuple(arrays) for split, arrays in opened.items()}
        self._document_spans = {
            split: _document_spans(manifest.splits[split], split=split)
            for split in _SPLITS
        }
        self._closed = False

    def shards(self, split: TokenizedSplit) -> tuple[np.memmap, ...]:
        """Return the validated, ordered memmaps for ``split``."""

        if self._closed:
            raise RuntimeError("tokenized shard reader is closed")
        if split not in _SPLITS:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        return self._shards[split]

    def document_spans(
        self,
        split: TokenizedSplit,
    ) -> tuple[TokenizedDocumentSpan, ...]:
        """Return ordered, validated document offsets for ``split``."""

        if self._closed:
            raise RuntimeError("tokenized shard reader is closed")
        if split not in _SPLITS:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        return self._document_spans[split]

    def close(self) -> None:
        """Close every mapped shard. Repeated calls are harmless."""

        if not self._closed:
            _close_memmaps(self._shards)
            self._closed = True

    def __enter__(self) -> TokenizedShardReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self.close()


def _write_split(
    directory: Path,
    *,
    split: str,
    sources: Sequence[TokenizedShardSource],
    tokenizer: Tokenizer,
    vocab_size: int,
    dtype: TokenDType,
) -> TokenizedSplitManifest:
    shards = tuple(
        _write_source(
            directory,
            split=split,
            index=index,
            source=source,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            dtype=dtype,
        )
        for index, source in enumerate(sources)
    )
    return TokenizedSplitManifest(
        token_count=sum(shard.token_count for shard in shards),
        document_count=sum(shard.document_count for shard in shards),
        byte_count=sum(shard.byte_count for shard in shards),
        shards=shards,
    )


def _write_source(
    directory: Path,
    *,
    split: str,
    index: int,
    source: TokenizedShardSource,
    tokenizer: Tokenizer,
    vocab_size: int,
    dtype: TokenDType,
) -> TokenizedShardManifest:
    filename = f"{split}_{index:06d}.bin"
    path = directory / filename
    numpy_dtype = _numpy_dtype(dtype)
    digest = hashlib.sha256()
    document_token_counts: list[int] = []
    token_count = 0

    try:
        documents = iter(source.documents)
    except TypeError as error:
        raise TypeError(
            f"{split} source {source.identity!r} documents must be iterable"
        ) from error

    with path.open("xb") as output_file:
        for document_index, document in enumerate(documents):
            token_ids = tokenizer.encode(document)
            _validate_encoded_ids(
                token_ids,
                source=source.identity,
                document_index=document_index,
                vocab_size=vocab_size,
            )
            payload = np.asarray(token_ids, dtype=numpy_dtype).tobytes(order="C")
            output_file.write(payload)
            digest.update(payload)
            document_token_counts.append(len(token_ids))
            token_count += len(token_ids)
        output_file.flush()
        os.fsync(output_file.fileno())

    if not document_token_counts:
        raise TokenizedDataError(
            f"{split} source {source.identity!r} did not yield any documents"
        )
    if token_count == 0:
        raise TokenizedDataError(
            f"{split} source {source.identity!r} did not yield any tokens"
        )

    byte_count = token_count * numpy_dtype.itemsize
    if path.stat().st_size != byte_count:
        raise TokenizedDataError(
            f"{filename} write size does not match its encoded token count"
        )
    return TokenizedShardManifest(
        index=index,
        filename=filename,
        token_count=token_count,
        document_count=len(document_token_counts),
        document_token_counts=tuple(document_token_counts),
        byte_count=byte_count,
        sha256=digest.hexdigest(),
        source_shards=(source.identity,),
    )


def _validate_encoded_ids(
    token_ids: object,
    *,
    source: str,
    document_index: int,
    vocab_size: int,
) -> None:
    if not isinstance(token_ids, list):
        raise TypeError(
            f"tokenizer.encode must return a list, got {type(token_ids).__name__} "
            f"for {source!r} document {document_index}"
        )
    for position, token_id in enumerate(token_ids):
        label = f"token ID at {source!r} document {document_index}, position {position}"
        normalized = require_integer(token_id, name=label)
        if not 0 <= normalized < vocab_size:
            raise TokenizedDataError(
                f"{label} must be in range [0, {vocab_size}); got {normalized}"
            )


def _normalize_sources(
    sources: Iterable[TokenizedShardSource],
    *,
    split: str,
) -> tuple[TokenizedShardSource, ...]:
    if isinstance(sources, (str, bytes)):
        raise TypeError(f"{split}_sources must contain TokenizedShardSource values")
    try:
        normalized = tuple(sources)
    except TypeError as error:
        raise TypeError(f"{split}_sources must be iterable") from error
    if not normalized:
        raise TokenizedDataError(f"{split}_sources must contain at least one shard")

    identities: set[str] = set()
    for position, source in enumerate(normalized):
        if not isinstance(source, TokenizedShardSource):
            raise TypeError(
                f"{split}_sources item {position} must be a TokenizedShardSource, "
                f"got {type(source).__name__}"
            )
        identity = source.identity
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(
                f"{split}_sources item {position} identity must be a non-empty string"
            )
        if identity in identities:
            raise TokenizedDataError(
                f"{split}_sources contains duplicate identity {identity!r}"
            )
        identities.add(identity)
    return tuple(sorted(normalized, key=lambda source: source.identity))


def _tokenizer_metadata(
    tokenizer: Tokenizer,
) -> tuple[int, str, Mapping[str, int]]:
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    vocab_size = require_positive_integer(
        tokenizer.get_vocab_size(),
        name="tokenizer vocabulary size",
    )
    if vocab_size > _MAX_UINT32_VOCAB_SIZE:
        raise ValueError(
            "tokenizer vocabulary size must be in range "
            f"[1, {_MAX_UINT32_VOCAB_SIZE}], got {vocab_size}"
        )

    identity = tokenizer.get_identity()
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("tokenizer identity must be a non-empty string")

    special_tokens = tokenizer.get_special_tokens()
    if not isinstance(special_tokens, set):
        raise TypeError(
            "tokenizer special tokens must be returned as a set, "
            f"got {type(special_tokens).__name__}"
        )
    special_token_ids: dict[str, int] = {}
    used_ids: set[int] = set()
    for token in sorted(special_tokens):
        if not isinstance(token, str) or not token:
            raise ValueError("tokenizer special tokens must be non-empty strings")
        label = f"special token ID for {token!r}"
        token_id = require_integer(
            tokenizer.encode_special(token),
            name=label,
        )
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"{label} must be in range [0, {vocab_size}); got {token_id}"
            )
        if token_id in used_ids:
            raise ValueError(f"special token ID {token_id} is assigned more than once")
        special_token_ids[token] = token_id
        used_ids.add(token_id)
    return vocab_size, identity, MappingProxyType(special_token_ids)


def _dtype_for_vocab_size(vocab_size: int) -> TokenDType:
    return "uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32"


def _numpy_dtype(dtype: TokenDType) -> np.dtype[np.unsignedinteger]:
    return np.dtype("<u2" if dtype == "uint16" else "<u4")


def _validate_output_directory(path: str | os.PathLike[str]) -> Path:
    try:
        destination = Path(path)
    except TypeError as error:
        raise TypeError(
            f"output_dir must be path-like, got {type(path).__name__}"
        ) from error
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError(f"output_dir must name a dataset directory, got {path!r}")
    return destination


def _publish_staged_directory(
    staging_dir: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if not destination.exists():
        os.replace(staging_dir, destination)
        return
    if not overwrite:
        raise FileExistsError(
            f"tokenized output appeared while writing and was not replaced: "
            f"{destination}"
        )
    if destination.is_symlink() or not destination.is_dir():
        raise NotADirectoryError(
            f"tokenized output path is not a regular directory: {destination}"
        )

    backup = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".backup",
        )
    )
    backup.rmdir()
    os.replace(destination, backup)
    try:
        os.replace(staging_dir, destination)
    except BaseException:
        if not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise
    else:
        _remove_private_path(backup)


def _remove_private_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _resolve_manifest_path(
    path: str | os.PathLike[str],
) -> tuple[Path, Path]:
    try:
        candidate = Path(path)
    except TypeError as error:
        raise TypeError(f"path must be path-like, got {type(path).__name__}") from error
    if candidate.is_symlink():
        raise TokenizedDataError(
            f"tokenized dataset path must not be a symlink: {path}"
        )
    if candidate.is_dir() or candidate.name != TOKENIZED_MANIFEST_NAME:
        dataset_dir = candidate
        manifest_path = dataset_dir / TOKENIZED_MANIFEST_NAME
    else:
        manifest_path = candidate
        dataset_dir = candidate.parent

    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TokenizedDataError(
            f"tokenized manifest is missing or not a regular file: {manifest_path}"
        )
    return manifest_path, dataset_dir


def _load_manifest(path: Path) -> TokenizedDatasetManifest:
    try:
        with path.open(encoding="utf-8") as manifest_file:
            value = json.load(
                manifest_file,
                object_pairs_hook=_JSON_VALUES.duplicate_object_hook(label="manifest"),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TokenizedDataError(
            f"could not read tokenized manifest {path}: {error}"
        ) from error
    return _parse_manifest(value)


def _parse_manifest(value: object) -> TokenizedDatasetManifest:
    root = _JSON_VALUES.require_object(
        value,
        label="manifest",
        expected_keys=_MANIFEST_KEYS,
        schema_label=_MANIFEST_SCHEMA_LABEL,
    )
    format_name = _JSON_VALUES.require_string(
        root["format"],
        label="manifest format",
    )
    if format_name != TOKENIZED_SHARD_FORMAT:
        raise TokenizedDataError(f"unknown tokenized manifest format {format_name!r}")
    format_version = _JSON_VALUES.require_integer(
        root["format_version"],
        label="manifest format_version",
        minimum=1,
    )
    if format_version != TOKENIZED_SHARD_FORMAT_VERSION:
        raise TokenizedDataError(
            f"unknown tokenized manifest version {format_version}; "
            f"expected {TOKENIZED_SHARD_FORMAT_VERSION}"
        )
    byte_order = _JSON_VALUES.require_string(
        root["byte_order"],
        label="manifest byte_order",
    )
    if byte_order != "little":
        raise TokenizedDataError(
            f"manifest byte_order must be 'little', got {byte_order!r}"
        )

    vocab_size = _JSON_VALUES.require_integer(
        root["vocab_size"],
        label="manifest vocab_size",
        minimum=1,
        maximum=_MAX_UINT32_VOCAB_SIZE,
    )
    raw_dtype = _JSON_VALUES.require_string(
        root["dtype"],
        label="manifest dtype",
    )
    if raw_dtype == "uint16":
        dtype: TokenDType = "uint16"
    elif raw_dtype == "uint32":
        dtype = "uint32"
    else:
        raise TokenizedDataError(f"unsupported manifest dtype {raw_dtype!r}")
    expected_dtype = _dtype_for_vocab_size(vocab_size)
    if dtype != expected_dtype:
        raise TokenizedDataError(
            f"manifest dtype {dtype!r} is inconsistent with vocabulary size "
            f"{vocab_size}; expected {expected_dtype!r}"
        )

    tokenizer_identity = _JSON_VALUES.require_string(
        root["tokenizer_identity"],
        label="manifest tokenizer_identity",
    )
    special_token_ids = _parse_special_token_ids(
        root["special_token_ids"],
        vocab_size=vocab_size,
    )

    raw_splits = _JSON_VALUES.require_object(
        root["splits"],
        label="manifest splits",
        expected_keys=frozenset(_SPLITS),
        schema_label=_MANIFEST_SCHEMA_LABEL,
    )
    splits = {
        split: _parse_split(
            raw_splits[split],
            split=split,
            dtype=dtype,
        )
        for split in _SPLITS
    }
    _validate_source_provenance(splits)

    token_count = _JSON_VALUES.require_integer(
        root["token_count"],
        label="manifest token_count",
        minimum=1,
    )
    document_count = _JSON_VALUES.require_integer(
        root["document_count"],
        label="manifest document_count",
        minimum=1,
    )
    byte_count = _JSON_VALUES.require_integer(
        root["byte_count"],
        label="manifest byte_count",
        minimum=1,
    )
    _require_total(
        token_count,
        sum(split.token_count for split in splits.values()),
        label="manifest token_count",
    )
    _require_total(
        document_count,
        sum(split.document_count for split in splits.values()),
        label="manifest document_count",
    )
    _require_total(
        byte_count,
        sum(split.byte_count for split in splits.values()),
        label="manifest byte_count",
    )

    return TokenizedDatasetManifest(
        format_version=format_version,
        dtype=dtype,
        vocab_size=vocab_size,
        tokenizer_identity=tokenizer_identity,
        special_token_ids=MappingProxyType(special_token_ids),
        token_count=token_count,
        document_count=document_count,
        byte_count=byte_count,
        splits=MappingProxyType(splits),
    )


def _parse_special_token_ids(
    value: object,
    *,
    vocab_size: int,
) -> dict[str, int]:
    values = _JSON_VALUES.require_object(
        value,
        label="manifest special_token_ids",
    )
    parsed: dict[str, int] = {}
    used_ids: set[int] = set()
    for token, raw_token_id in values.items():
        if not token:
            raise TokenizedDataError("manifest special-token names must not be empty")
        token_id = _JSON_VALUES.require_integer(
            raw_token_id,
            label=f"manifest special_token_ids[{token!r}]",
            minimum=0,
            maximum=vocab_size - 1,
        )
        if token_id in used_ids:
            raise TokenizedDataError(
                f"manifest special token ID {token_id} is assigned more than once"
            )
        parsed[token] = token_id
        used_ids.add(token_id)
    return parsed


def _validate_source_provenance(
    splits: Mapping[str, TokenizedSplitManifest],
) -> None:
    locations: dict[str, str] = {}
    for split in _SPLITS:
        for shard in splits[split].shards:
            location = f"{split}/{shard.filename}"
            for source in shard.source_shards:
                previous = locations.get(source)
                if previous is not None:
                    raise TokenizedDataError(
                        f"manifest source provenance reuses {source!r} in "
                        f"{previous} and {location}"
                    )
                locations[source] = location


def _parse_split(
    value: object,
    *,
    split: str,
    dtype: TokenDType,
) -> TokenizedSplitManifest:
    parsed = _JSON_VALUES.require_object(
        value,
        label=f"manifest {split} split",
        expected_keys=_SPLIT_KEYS,
        schema_label=_MANIFEST_SCHEMA_LABEL,
    )
    raw_shards = _JSON_VALUES.require_list(
        parsed["shards"],
        label=f"manifest {split} shards",
        non_empty=True,
    )
    shards = tuple(
        _parse_shard(
            raw_shard,
            split=split,
            expected_index=index,
            dtype=dtype,
        )
        for index, raw_shard in enumerate(raw_shards)
    )
    token_count = _JSON_VALUES.require_integer(
        parsed["token_count"],
        label=f"manifest {split} token_count",
        minimum=1,
    )
    document_count = _JSON_VALUES.require_integer(
        parsed["document_count"],
        label=f"manifest {split} document_count",
        minimum=1,
    )
    byte_count = _JSON_VALUES.require_integer(
        parsed["byte_count"],
        label=f"manifest {split} byte_count",
        minimum=1,
    )
    _require_total(
        token_count,
        sum(shard.token_count for shard in shards),
        label=f"manifest {split} token_count",
    )
    _require_total(
        document_count,
        sum(shard.document_count for shard in shards),
        label=f"manifest {split} document_count",
    )
    _require_total(
        byte_count,
        sum(shard.byte_count for shard in shards),
        label=f"manifest {split} byte_count",
    )
    return TokenizedSplitManifest(
        token_count=token_count,
        document_count=document_count,
        byte_count=byte_count,
        shards=shards,
    )


def _parse_shard(
    value: object,
    *,
    split: str,
    expected_index: int,
    dtype: TokenDType,
) -> TokenizedShardManifest:
    label = f"manifest {split} shard {expected_index}"
    parsed = _JSON_VALUES.require_object(
        value,
        label=label,
        expected_keys=_SHARD_KEYS,
        schema_label=_MANIFEST_SCHEMA_LABEL,
    )
    index = _JSON_VALUES.require_integer(
        parsed["index"],
        label=f"{label} index",
        minimum=0,
    )
    if index != expected_index:
        raise TokenizedDataError(f"{label} index must be {expected_index}, got {index}")
    filename = _JSON_VALUES.require_string(
        parsed["filename"],
        label=f"{label} filename",
    )
    expected_filename = f"{split}_{expected_index:06d}.bin"
    if filename != expected_filename or Path(filename).name != filename:
        raise TokenizedDataError(
            f"{label} contains unsafe or noncanonical filename {filename!r}; "
            f"expected {expected_filename!r}"
        )

    token_count = _JSON_VALUES.require_integer(
        parsed["token_count"],
        label=f"{label} token_count",
        minimum=1,
    )
    document_count = _JSON_VALUES.require_integer(
        parsed["document_count"],
        label=f"{label} document_count",
        minimum=1,
    )
    byte_count = _JSON_VALUES.require_integer(
        parsed["byte_count"],
        label=f"{label} byte_count",
        minimum=1,
    )
    expected_byte_count = token_count * _numpy_dtype(dtype).itemsize
    _require_total(
        byte_count,
        expected_byte_count,
        label=f"{label} byte_count",
    )

    raw_document_counts = _JSON_VALUES.require_list(
        parsed["document_token_counts"],
        label=f"{label} document_token_counts",
    )
    document_token_counts = tuple(
        _JSON_VALUES.require_integer(
            count,
            label=f"{label} document_token_counts[{index}]",
            minimum=0,
        )
        for index, count in enumerate(raw_document_counts)
    )
    _require_total(
        len(document_token_counts),
        document_count,
        label=f"{label} document_count",
    )
    _require_total(
        sum(document_token_counts),
        token_count,
        label=f"{label} document token total",
    )

    sha256 = _JSON_VALUES.require_string(
        parsed["sha256"],
        label=f"{label} sha256",
    )
    if _SHA256.fullmatch(sha256) is None:
        raise TokenizedDataError(
            f"{label} sha256 must contain 64 lowercase hexadecimal characters"
        )
    raw_sources = _JSON_VALUES.require_list(
        parsed["source_shards"],
        label=f"{label} source_shards",
        non_empty=True,
    )
    source_shards = tuple(
        _JSON_VALUES.require_string(
            source,
            label=f"{label} source_shards[{index}]",
        )
        for index, source in enumerate(raw_sources)
    )
    if len(set(source_shards)) != len(source_shards):
        raise TokenizedDataError(f"{label} source_shards contains duplicates")

    return TokenizedShardManifest(
        index=index,
        filename=filename,
        token_count=token_count,
        document_count=document_count,
        document_token_counts=document_token_counts,
        byte_count=byte_count,
        sha256=sha256,
        source_shards=source_shards,
    )


def _require_total(actual: int, expected: int, *, label: str) -> None:
    if actual != expected:
        raise TokenizedDataError(
            f"{label} is inconsistent: declared {actual}, computed {expected}"
        )


def _validate_tokenizer_match(
    manifest: TokenizedDatasetManifest,
    tokenizer: Tokenizer,
) -> None:
    vocab_size, identity, special_token_ids = _tokenizer_metadata(tokenizer)
    if identity != manifest.tokenizer_identity:
        raise TokenizedDataError(
            "tokenizer identity mismatch: manifest contains "
            f"{manifest.tokenizer_identity!r}, runtime provides {identity!r}"
        )
    if vocab_size != manifest.vocab_size:
        raise TokenizedDataError(
            f"tokenizer vocabulary mismatch: manifest contains "
            f"{manifest.vocab_size}, runtime provides {vocab_size}"
        )
    if dict(special_token_ids) != dict(manifest.special_token_ids):
        raise TokenizedDataError(
            "tokenizer special-token IDs do not match the manifest"
        )


def _reject_unreferenced_payloads(
    directory: Path,
    manifest: TokenizedDatasetManifest,
) -> None:
    referenced = {
        shard.filename for split in _SPLITS for shard in manifest.splits[split].shards
    }
    present = {path.name for path in directory.iterdir() if path.suffix == ".bin"}
    unexpected = sorted(present - referenced)
    if unexpected:
        raise TokenizedDataError(
            f"tokenized dataset contains unreferenced payloads: {unexpected}"
        )


def _validate_and_map_shard(
    directory: Path,
    shard: TokenizedShardManifest,
    *,
    dtype: TokenDType,
    vocab_size: int,
) -> np.memmap:
    path = directory / shard.filename
    if path.is_symlink() or not path.is_file():
        raise TokenizedDataError(
            f"tokenized shard is missing or not a regular file: {path}"
        )
    actual_size = path.stat().st_size
    if actual_size != shard.byte_count:
        raise TokenizedDataError(
            f"{shard.filename} size mismatch: manifest declares "
            f"{shard.byte_count} bytes, found {actual_size}"
        )
    actual_digest = _hash_file(path)
    if actual_digest != shard.sha256:
        raise TokenizedDataError(
            f"{shard.filename} checksum mismatch: manifest contains "
            f"{shard.sha256}, computed {actual_digest}"
        )

    array = np.memmap(
        path,
        dtype=_numpy_dtype(dtype),
        mode="r",
        shape=(shard.token_count,),
    )
    if int(array.max()) >= vocab_size:
        _close_memmap(array)
        raise TokenizedDataError(
            f"{shard.filename} contains token IDs outside range [0, {vocab_size})"
        )
    return array


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _close_memmap(array: np.memmap) -> None:
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def _close_memmaps(shards: Mapping[str, Sequence[np.memmap]]) -> None:
    for arrays in shards.values():
        for array in arrays:
            _close_memmap(array)


def _document_spans(
    manifest: TokenizedSplitManifest,
    *,
    split: TokenizedSplit,
) -> tuple[TokenizedDocumentSpan, ...]:
    spans: list[TokenizedDocumentSpan] = []
    for shard in manifest.shards:
        start = 0
        for document_index, token_count in enumerate(shard.document_token_counts):
            stop = start + token_count
            spans.append(
                TokenizedDocumentSpan(
                    split=split,
                    shard_index=shard.index,
                    document_index=document_index,
                    start=start,
                    stop=stop,
                )
            )
            start = stop
    return tuple(spans)


__all__ = [
    "TOKENIZED_MANIFEST_NAME",
    "TOKENIZED_SHARD_FORMAT",
    "TOKENIZED_SHARD_FORMAT_VERSION",
    "TokenizedDataError",
    "TokenizedDatasetManifest",
    "TokenizedDocumentSpan",
    "TokenizedShardManifest",
    "TokenizedShardReader",
    "TokenizedShardSource",
    "TokenizedSplitManifest",
    "tokenized_manifest_identity",
    "write_tokenized_shards",
]
