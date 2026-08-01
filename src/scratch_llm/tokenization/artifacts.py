"""Versioned, deterministic artifacts for the regex byte-BPE tokenizer."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import os
from os import PathLike
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import torch

from scratch_llm._validation import JsonValueValidator, require_non_negative_integer
from scratch_llm.tokenization.tokenizer import (
    BYTE_VOCAB_SIZE,
    NANOCHAT_SPECIAL_TOKENS,
    Tokenizer,
)
from scratch_llm.utils import atomic_write, save_json

if TYPE_CHECKING:
    from scratch_llm.tokenization.bpe import ReferenceBPETrainingResult


TOKENIZER_ARTIFACT_FORMAT: Final = "scratch_llm_regex_byte_bpe"
TOKENIZER_ARTIFACT_VERSION: Final = 1
TOKENIZER_ARTIFACT_FILENAMES: Final = (
    "tokenizer.json",
    "merges.json",
    "vocab.json",
    "special_tokens.json",
    "token_bytes.pt",
)
TOKEN_BYTE_LENGTHS_DTYPE: Final = torch.int32

_JSON_FILENAMES: Final = TOKENIZER_ARTIFACT_FILENAMES[:-1]
_TOKEN_BYTES_FILENAME: Final = TOKENIZER_ARTIFACT_FILENAMES[-1]
_ARTIFACT_SCHEMA_LABEL: Final = f"artifact version {TOKENIZER_ARTIFACT_VERSION}"


class TokenizerArtifactError(ValueError):
    """A tokenizer artifact set is missing, unsafe, or internally inconsistent."""


_JSON_VALUES = JsonValueValidator(TokenizerArtifactError)


def regex_bpe_identity(result: ReferenceBPETrainingResult) -> str:
    """Return the stable identity of a complete regex byte-BPE token mapping."""

    encoded = json.dumps(
        _identity_payload(result),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_token_byte_lengths(tokenizer: Tokenizer) -> torch.Tensor:
    """Build CPU ``int32`` raw-byte lengths, with every special token zeroed."""

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    vocab_size = require_non_negative_integer(
        tokenizer.get_vocab_size(),
        name="tokenizer vocabulary size",
    )

    special_ids = {
        tokenizer.encode_special(token) for token in tokenizer.get_special_tokens()
    }
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or not 0 <= token_id < vocab_size
        for token_id in special_ids
    ):
        raise ValueError("tokenizer returned an invalid special-token ID")

    lengths: list[int] = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            lengths.append(0)
            continue
        raw_token_bytes = tokenizer.decode_single_token_bytes(token_id)
        if not isinstance(raw_token_bytes, bytes):
            raise TypeError(
                f"decode_single_token_bytes must return bytes for token ID {token_id}"
            )
        byte_length = len(raw_token_bytes)
        if byte_length > torch.iinfo(TOKEN_BYTE_LENGTHS_DTYPE).max:
            raise ValueError(
                f"raw byte length for token ID {token_id} does not fit "
                f"{TOKEN_BYTE_LENGTHS_DTYPE}"
            )
        lengths.append(byte_length)
    return torch.tensor(lengths, dtype=TOKEN_BYTE_LENGTHS_DTYPE, device="cpu")


def save_token_byte_lengths(
    tokenizer: Tokenizer,
    path: str | PathLike[str],
) -> Path:
    """Atomically write a CPU-loadable ``token_bytes.pt`` tensor."""

    buffer = io.BytesIO()
    torch.save(build_token_byte_lengths(tokenizer), buffer)
    return atomic_write(path, buffer.getvalue())


def save_regex_bpe_artifacts(
    tokenizer: Tokenizer,
    result: ReferenceBPETrainingResult,
    path: str | PathLike[str],
) -> Path:
    """Stage and atomically publish one complete tokenizer artifact directory."""

    destination = _artifact_directory(path, must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(
                f"tokenizer artifact path is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise FileExistsError(
                "tokenizer artifact directory already exists and is not empty: "
                f"{destination}"
            )

    identity = regex_bpe_identity(result)
    if (
        tokenizer.get_identity() != identity
        or tokenizer.get_vocab_size() != result.vocab_size
    ):
        raise TokenizerArtifactError(
            "tokenizer identity or vocabulary does not match the training result"
        )
    token_byte_lengths = build_token_byte_lengths(tokenizer)
    if not torch.equal(token_byte_lengths, _expected_token_byte_lengths(result)):
        raise TokenizerArtifactError(
            "tokenizer raw bytes or special tokens do not match the training result"
        )
    try:
        merges = _merge_records(result)
        vocabulary = _vocabulary_records(result)
        special_tokens = _special_token_records(result)
    except (KeyError, TypeError, ValueError) as error:
        raise TokenizerArtifactError(
            f"training result cannot be serialized as a contiguous vocabulary: {error}"
        ) from error
    common = {
        "format": TOKENIZER_ARTIFACT_FORMAT,
        "format_version": TOKENIZER_ARTIFACT_VERSION,
        "tokenizer_identity": identity,
    }
    documents = {
        "tokenizer.json": {
            **common,
            "artifact_type": "tokenizer",
            "mergeable_vocab_size": result.mergeable_vocab_size,
            "merges": merges,
            "special_tokens": special_tokens,
            "training": {
                "character_count": result.character_count,
                "chunk_count": result.chunk_count,
                "document_count": result.document_count,
            },
            "vocab_size": result.vocab_size,
            "vocabulary": vocabulary,
        },
        "merges.json": {
            **common,
            "artifact_type": "merges",
            "merges": merges,
        },
        "vocab.json": {
            **common,
            "artifact_type": "vocabulary",
            "vocab_size": result.vocab_size,
            "vocabulary": vocabulary,
        },
        "special_tokens.json": {
            **common,
            "artifact_type": "special_tokens",
            "special_tokens": special_tokens,
        },
    }
    _parse_tokenizer_document(documents["tokenizer.json"])

    staging_path = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
    )
    try:
        for filename in _JSON_FILENAMES:
            save_json(documents[filename], staging_path / filename)
        save_token_byte_lengths(tokenizer, staging_path / _TOKEN_BYTES_FILENAME)
        _fsync_directory(staging_path)
        os.replace(staging_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise
    return destination


def load_regex_bpe_training_result(
    path: str | PathLike[str],
) -> ReferenceBPETrainingResult:
    """Load the authoritative artifact and reconstruct an immutable result."""

    directory = _artifact_directory(path, must_exist=True)
    _require_complete_file_set(directory)
    tokenizer_document = _load_json_document(directory / "tokenizer.json")
    result = _parse_tokenizer_document(tokenizer_document)
    _validate_redundant_json_documents(
        directory,
        tokenizer_document=tokenizer_document,
    )
    _validate_token_byte_lengths(directory / _TOKEN_BYTES_FILENAME, result)
    return result


def _parse_tokenizer_document(
    document: dict[str, object],
) -> ReferenceBPETrainingResult:
    from scratch_llm.tokenization.bpe import BPEMerge, ReferenceBPETrainingResult

    _validate_common_document(document, artifact_type="tokenizer")
    _JSON_VALUES.require_object(
        document,
        label="tokenizer.json",
        expected_keys=frozenset(
            {
                "artifact_type",
                "format",
                "format_version",
                "mergeable_vocab_size",
                "merges",
                "special_tokens",
                "tokenizer_identity",
                "training",
                "vocab_size",
                "vocabulary",
            }
        ),
        schema_label=_ARTIFACT_SCHEMA_LABEL,
    )

    vocab_size = _JSON_VALUES.require_integer(
        document.get("vocab_size"),
        label="vocab_size",
        minimum=0,
    )
    mergeable_vocab_size = _JSON_VALUES.require_integer(
        document.get("mergeable_vocab_size"),
        label="mergeable_vocab_size",
        minimum=BYTE_VOCAB_SIZE,
    )
    raw_merges = _JSON_VALUES.require_list(
        document.get("merges"),
        label="merges",
    )
    expected_mergeable_vocab_size = BYTE_VOCAB_SIZE + len(raw_merges)
    if mergeable_vocab_size != expected_mergeable_vocab_size:
        raise TokenizerArtifactError(
            "mergeable_vocab_size must equal 256 plus the merge count; "
            f"expected {expected_mergeable_vocab_size}, got {mergeable_vocab_size}"
        )
    expected_vocab_size = mergeable_vocab_size + len(NANOCHAT_SPECIAL_TOKENS)
    if vocab_size != expected_vocab_size:
        raise TokenizerArtifactError(
            "vocab_size must equal mergeable_vocab_size plus the canonical "
            f"special-token count; expected {expected_vocab_size}, got {vocab_size}"
        )

    merges: list[BPEMerge] = []
    for rank, raw_merge in enumerate(raw_merges):
        merge_document = _JSON_VALUES.require_object(
            raw_merge,
            label=f"merge at rank {rank}",
            expected_keys=frozenset(
                {"count", "left_id", "rank", "right_id", "token_id"}
            ),
            schema_label=_ARTIFACT_SCHEMA_LABEL,
        )
        stored_rank = _JSON_VALUES.require_integer(
            merge_document.get("rank"),
            label="rank",
            minimum=0,
        )
        if stored_rank != rank:
            raise TokenizerArtifactError(
                "merge ranks must be contiguous from zero; "
                f"expected rank {rank}, got {stored_rank}"
            )
        token_id = _JSON_VALUES.require_integer(
            merge_document.get("token_id"),
            label="token_id",
            minimum=0,
        )
        expected_token_id = BYTE_VOCAB_SIZE + rank
        if token_id != expected_token_id:
            raise TokenizerArtifactError(
                "merge token IDs must be contiguous from 256 in rank order; "
                f"expected {expected_token_id}, got {token_id}"
            )
        left_id = _JSON_VALUES.require_integer(
            merge_document.get("left_id"),
            label="left_id",
            minimum=0,
        )
        right_id = _JSON_VALUES.require_integer(
            merge_document.get("right_id"),
            label="right_id",
            minimum=0,
        )
        if left_id >= token_id or right_id >= token_id:
            raise TokenizerArtifactError(
                "merge inputs must refer to earlier token IDs; "
                f"rank {rank} has pair ({left_id}, {right_id}) for token {token_id}"
            )
        merges.append(
            BPEMerge(
                pair=(left_id, right_id),
                token_id=token_id,
                count=_JSON_VALUES.require_integer(
                    merge_document.get("count"),
                    label="count",
                    minimum=1,
                ),
            )
        )

    raw_vocabulary = _JSON_VALUES.require_list(
        document.get("vocabulary"),
        label="vocabulary",
    )
    if len(raw_vocabulary) != vocab_size:
        raise TokenizerArtifactError(
            "vocabulary IDs must be contiguous across vocab_size; "
            f"expected {vocab_size} entries, got {len(raw_vocabulary)}"
        )
    vocabulary: dict[int, bytes] = {}
    for position, raw_token in enumerate(raw_vocabulary):
        token_document = _JSON_VALUES.require_object(
            raw_token,
            label=f"vocabulary entry at position {position}",
            expected_keys=frozenset({"bytes_hex", "id"}),
            schema_label=_ARTIFACT_SCHEMA_LABEL,
        )
        token_id = _JSON_VALUES.require_integer(
            token_document.get("id"),
            label="id",
            minimum=0,
        )
        if token_id != position:
            raise TokenizerArtifactError(
                "vocabulary IDs must be contiguous from zero; "
                f"expected {position}, got {token_id}"
            )
        raw_hex = token_document.get("bytes_hex")
        if not isinstance(raw_hex, str):
            raise TokenizerArtifactError(
                f"vocabulary bytes_hex for token ID {token_id} must be a string"
            )
        try:
            token_bytes = bytes.fromhex(raw_hex)
        except ValueError as error:
            raise TokenizerArtifactError(
                f"vocabulary bytes_hex for token ID {token_id} is invalid"
            ) from error
        if token_bytes.hex() != raw_hex:
            raise TokenizerArtifactError(
                f"vocabulary bytes_hex for token ID {token_id} is not canonical"
            )
        vocabulary[token_id] = token_bytes

    for token_id in range(BYTE_VOCAB_SIZE):
        if vocabulary[token_id] != bytes([token_id]):
            raise TokenizerArtifactError(
                f"raw bytes for base byte token {token_id} do not match its ID"
            )
    for merge in merges:
        expected_bytes = vocabulary[merge.left_id] + vocabulary[merge.right_id]
        if vocabulary[merge.token_id] != expected_bytes:
            raise TokenizerArtifactError(
                "raw bytes for merge token "
                f"{merge.token_id} do not match its ranked input pair"
            )

    raw_special_tokens = _JSON_VALUES.require_list(
        document.get("special_tokens"),
        label="special_tokens",
    )
    expected_special_tokens = [
        {"id": mergeable_vocab_size + offset, "token": token}
        for offset, token in enumerate(NANOCHAT_SPECIAL_TOKENS)
    ]
    if raw_special_tokens != expected_special_tokens:
        raise TokenizerArtifactError(
            "special tokens must exactly match the ordered nanochat vocabulary "
            "at contiguous final IDs"
        )
    special_token_ids = {
        token: mergeable_vocab_size + offset
        for offset, token in enumerate(NANOCHAT_SPECIAL_TOKENS)
    }
    for token, token_id in special_token_ids.items():
        if vocabulary[token_id] != token.encode("utf-8"):
            raise TokenizerArtifactError(
                f"raw bytes for special token {token!r} do not match its UTF-8 bytes"
            )

    training = _JSON_VALUES.require_object(
        document.get("training"),
        label="training metadata",
        expected_keys=frozenset({"character_count", "chunk_count", "document_count"}),
        schema_label=_ARTIFACT_SCHEMA_LABEL,
    )
    result = ReferenceBPETrainingResult(
        vocab_size=vocab_size,
        mergeable_vocab_size=mergeable_vocab_size,
        merges=tuple(merges),
        vocabulary=MappingProxyType(vocabulary),
        special_token_ids=MappingProxyType(special_token_ids),
        document_count=_JSON_VALUES.require_integer(
            training.get("document_count"),
            label="document_count",
            minimum=0,
        ),
        character_count=_JSON_VALUES.require_integer(
            training.get("character_count"),
            label="character_count",
            minimum=0,
        ),
        chunk_count=_JSON_VALUES.require_integer(
            training.get("chunk_count"),
            label="chunk_count",
            minimum=0,
        ),
    )
    expected_identity = document["tokenizer_identity"]
    if expected_identity != regex_bpe_identity(result):
        raise TokenizerArtifactError(
            "tokenizer identity does not match the canonical token mapping"
        )
    return result


def _validate_redundant_json_documents(
    directory: Path,
    *,
    tokenizer_document: Mapping[str, object],
) -> None:
    specifications = (
        (
            "merges.json",
            "merges",
            {
                "artifact_type",
                "format",
                "format_version",
                "merges",
                "tokenizer_identity",
            },
        ),
        (
            "vocab.json",
            "vocabulary",
            {
                "artifact_type",
                "format",
                "format_version",
                "tokenizer_identity",
                "vocab_size",
                "vocabulary",
            },
        ),
        (
            "special_tokens.json",
            "special_tokens",
            {
                "artifact_type",
                "format",
                "format_version",
                "special_tokens",
                "tokenizer_identity",
            },
        ),
    )
    for filename, field, expected_keys in specifications:
        document = _load_json_document(directory / filename)
        artifact_type = {
            "merges.json": "merges",
            "vocab.json": "vocabulary",
            "special_tokens.json": "special_tokens",
        }[filename]
        _validate_common_document(document, artifact_type=artifact_type)
        _JSON_VALUES.require_object(
            document,
            label=filename,
            expected_keys=frozenset(expected_keys),
            schema_label=_ARTIFACT_SCHEMA_LABEL,
        )
        if document["tokenizer_identity"] != tokenizer_document["tokenizer_identity"]:
            raise TokenizerArtifactError(
                f"{filename} is inconsistent with tokenizer.json identity"
            )
        if document.get(field) != tokenizer_document[field]:
            raise TokenizerArtifactError(
                f"{filename} is inconsistent with tokenizer.json field {field}"
            )
        if (
            filename == "vocab.json"
            and document.get("vocab_size") != tokenizer_document["vocab_size"]
        ):
            raise TokenizerArtifactError(
                "vocab.json is inconsistent with tokenizer.json vocab_size"
            )


def _validate_token_byte_lengths(
    path: Path,
    result: ReferenceBPETrainingResult,
) -> None:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise TokenizerArtifactError(
            f"token_bytes.pt could not be safely loaded: {error}"
        ) from error
    if type(value) is not torch.Tensor:
        raise TokenizerArtifactError("token_bytes.pt must contain exactly one tensor")
    if value.device.type != "cpu":
        raise TokenizerArtifactError("token_bytes.pt must load onto the CPU")
    if value.layout != torch.strided:
        raise TokenizerArtifactError("token_bytes.pt must contain a dense tensor")
    if value.dtype != TOKEN_BYTE_LENGTHS_DTYPE:
        raise TokenizerArtifactError(
            f"token_bytes.pt must use dtype {TOKEN_BYTE_LENGTHS_DTYPE}"
        )
    if tuple(value.shape) != (result.vocab_size,):
        raise TokenizerArtifactError(
            "token_bytes.pt must have shape "
            f"({result.vocab_size},), got {tuple(value.shape)}"
        )
    expected = _expected_token_byte_lengths(result)
    if not torch.equal(value, expected):
        raise TokenizerArtifactError(
            "token_bytes.pt values are inconsistent with stored raw token bytes "
            "or special-token zeroing"
        )


def _expected_token_byte_lengths(
    result: ReferenceBPETrainingResult,
) -> torch.Tensor:
    special_ids = set(result.special_token_ids.values())
    return torch.tensor(
        [
            0 if token_id in special_ids else len(result.vocabulary[token_id])
            for token_id in range(result.vocab_size)
        ],
        dtype=TOKEN_BYTE_LENGTHS_DTYPE,
        device="cpu",
    )


def _identity_payload(result: ReferenceBPETrainingResult) -> dict[str, object]:
    return {
        "format": TOKENIZER_ARTIFACT_FORMAT,
        "format_version": TOKENIZER_ARTIFACT_VERSION,
        "merges": [
            {
                "left_id": merge.left_id,
                "rank": rank,
                "right_id": merge.right_id,
                "token_id": merge.token_id,
            }
            for rank, merge in enumerate(result.merges)
        ],
        "special_tokens": _special_token_records(result),
        "vocabulary": _vocabulary_records(result),
    }


def _merge_records(
    result: ReferenceBPETrainingResult,
) -> list[dict[str, int]]:
    return [
        {
            "count": merge.count,
            "left_id": merge.left_id,
            "rank": rank,
            "right_id": merge.right_id,
            "token_id": merge.token_id,
        }
        for rank, merge in enumerate(result.merges)
    ]


def _vocabulary_records(
    result: ReferenceBPETrainingResult,
) -> list[dict[str, int | str]]:
    return [
        {"bytes_hex": result.vocabulary[token_id].hex(), "id": token_id}
        for token_id in range(result.vocab_size)
    ]


def _special_token_records(
    result: ReferenceBPETrainingResult,
) -> list[dict[str, int | str]]:
    return [
        {"id": result.special_token_ids[token], "token": token}
        for token in result.special_token_ids
    ]


def _artifact_directory(
    path: str | PathLike[str],
    *,
    must_exist: bool,
) -> Path:
    try:
        directory = Path(path)
    except TypeError as error:
        raise TypeError(
            "tokenizer artifact path must be a string or path-like value"
        ) from error
    if ".." in directory.parts:
        raise TokenizerArtifactError(
            "tokenizer artifact path must not contain a '..' traversal component"
        )
    if directory.name in {"", ".", ".."}:
        raise TokenizerArtifactError(
            "tokenizer artifact path must name a dedicated directory"
        )
    if directory.is_symlink():
        raise TokenizerArtifactError(
            f"tokenizer artifact directory must not be a symlink: {directory}"
        )
    if must_exist:
        if not directory.exists():
            raise FileNotFoundError(
                f"tokenizer artifact directory does not exist: {directory}"
            )
        if not directory.is_dir():
            raise TokenizerArtifactError(
                f"tokenizer artifact path is not a directory: {directory}"
            )
    return directory


def _require_complete_file_set(directory: Path) -> None:
    entries = {entry.name: entry for entry in directory.iterdir()}
    expected = set(TOKENIZER_ARTIFACT_FILENAMES)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unknown = sorted(set(entries) - expected)
        raise TokenizerArtifactError(
            "tokenizer artifact directory has an incomplete or unknown file set; "
            f"missing={missing}, unknown={unknown}"
        )
    for name, entry in entries.items():
        if entry.is_symlink() or not entry.is_file():
            raise TokenizerArtifactError(
                f"tokenizer artifact must be a regular file, not a symlink: {name}"
            )


def _load_json_document(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as json_file:
            value = json.load(
                json_file,
                object_pairs_hook=_JSON_VALUES.duplicate_object_hook(
                    label=f"tokenizer artifact {path.name}"
                ),
            )
    except TokenizerArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TokenizerArtifactError(
            f"could not load tokenizer artifact {path.name}: {error}"
        ) from error
    return _JSON_VALUES.require_object(
        value,
        label=f"tokenizer artifact {path.name}",
    )


def _validate_common_document(
    document: Mapping[str, object],
    *,
    artifact_type: str,
) -> None:
    if document.get("format") != TOKENIZER_ARTIFACT_FORMAT:
        raise TokenizerArtifactError(
            f"unknown tokenizer artifact format {document.get('format')!r}"
        )
    version = document.get("format_version")
    if version != TOKENIZER_ARTIFACT_VERSION or isinstance(version, bool):
        raise TokenizerArtifactError(
            f"unknown tokenizer artifact format version {version!r}"
        )
    if document.get("artifact_type") != artifact_type:
        raise TokenizerArtifactError(
            f"expected {artifact_type!r} tokenizer artifact, "
            f"got {document.get('artifact_type')!r}"
        )
    identity = document.get("tokenizer_identity")
    if (
        not isinstance(identity, str)
        or not identity.startswith("sha256:")
        or len(identity) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in identity[7:])
    ):
        raise TokenizerArtifactError(
            "tokenizer_identity must be a sha256-prefixed digest"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    file_descriptor = os.open(path, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


__all__ = [
    "TOKENIZER_ARTIFACT_FILENAMES",
    "TOKENIZER_ARTIFACT_FORMAT",
    "TOKENIZER_ARTIFACT_VERSION",
    "TOKEN_BYTE_LENGTHS_DTYPE",
    "TokenizerArtifactError",
    "build_token_byte_lengths",
    "load_regex_bpe_training_result",
    "regex_bpe_identity",
    "save_regex_bpe_artifacts",
    "save_token_byte_lengths",
]
