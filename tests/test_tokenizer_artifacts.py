"""Tests for persisted regex byte-BPE artifacts and BPB byte lengths."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch

import scratch_llm.tokenizer_artifacts as tokenizer_artifacts
from scratch_llm.bpe import RegexBPETokenizer, train_reference_bpe
from scratch_llm.tokenizer import BYTE_VOCAB_SIZE, NANOCHAT_SPECIAL_TOKENS
from scratch_llm.tokenizer_artifacts import (
    TOKENIZER_ARTIFACT_FILENAMES,
    TOKENIZER_ARTIFACT_FORMAT,
    TOKENIZER_ARTIFACT_VERSION,
    TOKEN_BYTE_LENGTHS_DTYPE,
    TokenizerArtifactError,
    build_token_byte_lengths,
    save_regex_bpe_artifacts,
)


def _trained_tokenizer() -> RegexBPETokenizer:
    result = train_reference_bpe(
        ("aa aa 🚀", "aa"),
        vocab_size=BYTE_VOCAB_SIZE + 3 + len(NANOCHAT_SPECIAL_TOKENS),
    )
    return RegexBPETokenizer(result)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_save_load_writes_deterministic_complete_equivalent_artifacts(
    tmp_path: Path,
) -> None:
    tokenizer = _trained_tokenizer()
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"

    tokenizer.save(first_path)
    tokenizer.save(second_path)

    assert {entry.name for entry in first_path.iterdir()} == set(
        TOKENIZER_ARTIFACT_FILENAMES
    )
    for filename in TOKENIZER_ARTIFACT_FILENAMES:
        assert not filename.startswith(".")
    for filename in (
        "tokenizer.json",
        "merges.json",
        "vocab.json",
        "special_tokens.json",
    ):
        assert (first_path / filename).read_bytes() == (
            second_path / filename
        ).read_bytes()

    tokenizer_document = json.loads(
        (first_path / "tokenizer.json").read_text(encoding="utf-8")
    )
    assert tokenizer_document["format"] == TOKENIZER_ARTIFACT_FORMAT
    assert tokenizer_document["format_version"] == TOKENIZER_ARTIFACT_VERSION
    assert tokenizer_document["tokenizer_identity"] == tokenizer.get_identity()

    loaded = RegexBPETokenizer.load(first_path)
    fixtures = (
        "aa aa 🚀",
        "안녕하세요",
        "<|bos|> remains ordinary text",
        "",
    )
    assert loaded.get_vocab_size() == tokenizer.get_vocab_size()
    assert loaded.get_bos_token_id() == tokenizer.get_bos_token_id()
    assert loaded.get_special_tokens() == tokenizer.get_special_tokens()
    assert loaded.get_identity() == tokenizer.get_identity()
    for token_id in range(tokenizer.get_vocab_size()):
        assert loaded.decode_single_token_bytes(
            token_id
        ) == tokenizer.decode_single_token_bytes(token_id)
    for text in fixtures:
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == text


def test_token_byte_lengths_use_raw_bytes_and_zero_special_targets(
    tmp_path: Path,
) -> None:
    tokenizer = _trained_tokenizer()
    artifact_path = tmp_path / "artifacts"

    expected = build_token_byte_lengths(tokenizer)
    tokenizer.save(artifact_path)
    stored = torch.load(
        artifact_path / "token_bytes.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert type(stored) is torch.Tensor
    assert stored.device.type == "cpu"
    assert stored.dtype == TOKEN_BYTE_LENGTHS_DTYPE == torch.int32
    assert stored.shape == (tokenizer.get_vocab_size(),)
    assert torch.equal(stored, expected)

    special_ids = {tokenizer.encode_special(token) for token in NANOCHAT_SPECIAL_TOKENS}
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in special_ids:
            assert stored[token_id].item() == 0
        else:
            assert stored[token_id].item() == len(
                tokenizer.decode_single_token_bytes(token_id)
            )

    merge_id = BYTE_VOCAB_SIZE
    bos_id = tokenizer.get_bos_token_id()
    targets = torch.tensor([ord("a"), merge_id, bos_id, -100])
    losses_nats = torch.tensor(
        [math.log(2), 4 * math.log(2), 100 * math.log(2), 1000 * math.log(2)]
    )
    target_bytes = torch.zeros_like(targets)
    unmasked = targets >= 0
    target_bytes[unmasked] = stored[targets[unmasked]].to(targets.dtype)
    counted = target_bytes > 0
    bpb = losses_nats[counted].sum() / math.log(2) / target_bytes[counted].sum()

    assert tokenizer.decode_single_token_bytes(merge_id) == b"aa"
    assert target_bytes.tolist() == [1, 2, 0, 0]
    assert bpb.item() == pytest.approx(5 / 3)


def test_token_byte_lengths_preserve_multibyte_tokens_without_utf8_decoding() -> None:
    tokenizer = RegexBPETokenizer(
        train_reference_bpe(
            ("🚀",),
            vocab_size=BYTE_VOCAB_SIZE + 3 + len(NANOCHAT_SPECIAL_TOKENS),
        )
    )

    lengths = build_token_byte_lengths(tokenizer)

    assert tokenizer.decode_single_token_bytes(BYTE_VOCAB_SIZE) == b"\x9a\x80"
    assert tokenizer.decode_single_token_bytes(BYTE_VOCAB_SIZE + 1) == b"\x9f\x9a\x80"
    assert tokenizer.decode_single_token_bytes(BYTE_VOCAB_SIZE + 2) == "🚀".encode()
    assert lengths[BYTE_VOCAB_SIZE : BYTE_VOCAB_SIZE + 3].tolist() == [2, 3, 4]


def test_loaded_identity_is_stable_in_an_isolated_process(tmp_path: Path) -> None:
    tokenizer = _trained_tokenizer()
    artifact_path = tmp_path / "artifacts"
    tokenizer.save(artifact_path)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                "from scratch_llm.bpe import RegexBPETokenizer; "
                "print(RegexBPETokenizer.load(sys.argv[1]).get_identity())"
            ),
            str(artifact_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == tokenizer.get_identity()


def test_identity_depends_on_token_mapping_not_training_counters() -> None:
    small = RegexBPETokenizer(
        train_reference_bpe(
            ("aa",),
            vocab_size=BYTE_VOCAB_SIZE + 1 + len(NANOCHAT_SPECIAL_TOKENS),
        )
    )
    repeated = RegexBPETokenizer(
        train_reference_bpe(
            ("aa aa",),
            vocab_size=BYTE_VOCAB_SIZE + 1 + len(NANOCHAT_SPECIAL_TOKENS),
        )
    )

    assert small.get_identity() == repeated.get_identity()


def test_load_rejects_unknown_versions_and_fields(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts"
    _trained_tokenizer().save(artifact_path)
    tokenizer_document = _load_json(artifact_path / "tokenizer.json")
    tokenizer_document["format_version"] = 999
    _write_json(artifact_path / "tokenizer.json", tokenizer_document)

    with pytest.raises(TokenizerArtifactError, match="unknown.*version"):
        RegexBPETokenizer.load(artifact_path)

    _trained_tokenizer().save(tmp_path / "second")
    second_document = _load_json(tmp_path / "second" / "tokenizer.json")
    second_document["unexpected"] = True
    _write_json(tmp_path / "second" / "tokenizer.json", second_document)

    with pytest.raises(TokenizerArtifactError, match="unknown fields"):
        RegexBPETokenizer.load(tmp_path / "second")


def test_load_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts"
    _trained_tokenizer().save(artifact_path)
    tokenizer_path = artifact_path / "tokenizer.json"
    text = tokenizer_path.read_text(encoding="utf-8")
    tokenizer_path.write_text(
        text.replace(
            '  "artifact_type": "tokenizer",',
            '  "artifact_type": "tokenizer",\n  "artifact_type": "tokenizer",',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TokenizerArtifactError, match="duplicate key"):
        RegexBPETokenizer.load(artifact_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("rank", "merge ranks must be contiguous"),
        ("token_id", "merge token IDs must be contiguous"),
        ("dependency", "merge inputs must refer to earlier"),
        ("raw_bytes", "raw bytes.*merge"),
        ("vocab_id", "vocabulary IDs must be contiguous"),
    ),
)
def test_load_rejects_invalid_merge_and_vocabulary_invariants(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    artifact_path = tmp_path / mutation
    _trained_tokenizer().save(artifact_path)
    tokenizer_document = _load_json(artifact_path / "tokenizer.json")
    merges = tokenizer_document["merges"]
    vocabulary = tokenizer_document["vocabulary"]
    assert isinstance(merges, list)
    assert isinstance(vocabulary, list)
    assert len(merges) >= 1
    assert isinstance(merges[0], dict)
    assert isinstance(vocabulary[BYTE_VOCAB_SIZE], dict)

    if mutation == "rank":
        merges[0]["rank"] = 1
    elif mutation == "token_id":
        merges[0]["token_id"] = BYTE_VOCAB_SIZE + 1
    elif mutation == "dependency":
        merges[0]["left_id"] = BYTE_VOCAB_SIZE
    elif mutation == "raw_bytes":
        vocabulary[BYTE_VOCAB_SIZE]["bytes_hex"] = b"wrong".hex()
    elif mutation == "vocab_id":
        vocabulary[BYTE_VOCAB_SIZE]["id"] = BYTE_VOCAB_SIZE + 1
    else:
        raise AssertionError(f"unhandled mutation {mutation}")
    _write_json(artifact_path / "tokenizer.json", tokenizer_document)

    with pytest.raises(TokenizerArtifactError, match=message):
        RegexBPETokenizer.load(artifact_path)


@pytest.mark.parametrize("mutation", ("reordered", "unknown", "wrong_id"))
def test_load_rejects_noncanonical_special_tokens(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact_path = tmp_path / mutation
    _trained_tokenizer().save(artifact_path)
    tokenizer_document = _load_json(artifact_path / "tokenizer.json")
    special_tokens = tokenizer_document["special_tokens"]
    assert isinstance(special_tokens, list)
    assert isinstance(special_tokens[0], dict)

    if mutation == "reordered":
        special_tokens[0], special_tokens[1] = special_tokens[1], special_tokens[0]
    elif mutation == "unknown":
        special_tokens[0]["token"] = "<|unknown|>"
    elif mutation == "wrong_id":
        special_tokens[0]["id"] = 0
    else:
        raise AssertionError(f"unhandled mutation {mutation}")
    _write_json(artifact_path / "tokenizer.json", tokenizer_document)

    with pytest.raises(
        TokenizerArtifactError,
        match="special tokens must exactly match the ordered nanochat vocabulary",
    ):
        RegexBPETokenizer.load(artifact_path)


@pytest.mark.parametrize(
    ("filename", "field"),
    (
        ("merges.json", "merges"),
        ("vocab.json", "vocabulary"),
        ("special_tokens.json", "special_tokens"),
    ),
)
def test_load_rejects_cross_file_inconsistencies(
    tmp_path: Path,
    filename: str,
    field: str,
) -> None:
    artifact_path = tmp_path / filename.removesuffix(".json")
    _trained_tokenizer().save(artifact_path)
    document = _load_json(artifact_path / filename)
    values = document[field]
    assert isinstance(values, list)
    values.pop()
    _write_json(artifact_path / filename, document)

    with pytest.raises(TokenizerArtifactError, match=f"{filename}.*inconsistent"):
        RegexBPETokenizer.load(artifact_path)


@pytest.mark.parametrize("corruption", ("dtype", "shape", "special", "ordinary"))
def test_load_rejects_invalid_token_byte_lengths(
    tmp_path: Path,
    corruption: str,
) -> None:
    artifact_path = tmp_path / corruption
    tokenizer = _trained_tokenizer()
    tokenizer.save(artifact_path)
    tensor_path = artifact_path / "token_bytes.pt"
    lengths = torch.load(tensor_path, map_location="cpu", weights_only=True)

    if corruption == "dtype":
        lengths = lengths.to(torch.int64)
    elif corruption == "shape":
        lengths = lengths[:-1]
    elif corruption == "special":
        lengths[tokenizer.get_bos_token_id()] = 1
    elif corruption == "ordinary":
        lengths[BYTE_VOCAB_SIZE] += 1
    else:
        raise AssertionError(f"unhandled corruption {corruption}")
    torch.save(lengths, tensor_path)

    with pytest.raises(TokenizerArtifactError, match="token_bytes.pt"):
        RegexBPETokenizer.load(artifact_path)


def test_load_rejects_incomplete_unknown_and_unsafe_paths(tmp_path: Path) -> None:
    tokenizer = _trained_tokenizer()

    missing_path = tmp_path / "missing"
    tokenizer.save(missing_path)
    (missing_path / "merges.json").unlink()
    with pytest.raises(TokenizerArtifactError, match="missing=.*merges.json"):
        RegexBPETokenizer.load(missing_path)

    unknown_path = tmp_path / "unknown"
    tokenizer.save(unknown_path)
    (unknown_path / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TokenizerArtifactError, match="unknown=.*extra.json"):
        RegexBPETokenizer.load(unknown_path)

    symlink_path = tmp_path / "symlink-target"
    tokenizer.save(symlink_path)
    symlink = tmp_path / "artifacts-link"
    symlink.symlink_to(symlink_path, target_is_directory=True)
    with pytest.raises(TokenizerArtifactError, match="must not be a symlink"):
        RegexBPETokenizer.load(symlink)

    traversed_path = symlink_path / ".." / symlink_path.name
    with pytest.raises(TokenizerArtifactError, match="must not contain.*\\.\\."):
        RegexBPETokenizer.load(traversed_path)


def test_load_uses_safe_tensor_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "artifacts"
    _trained_tokenizer().save(artifact_path)
    real_load = torch.load
    calls: list[dict[str, object]] = []

    def recording_load(
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object:
        calls.append({"map_location": map_location, "weights_only": weights_only})
        return real_load(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )

    monkeypatch.setattr(torch, "load", recording_load)

    RegexBPETokenizer.load(artifact_path)

    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_save_rejects_a_tokenizer_and_training_result_mismatch(
    tmp_path: Path,
) -> None:
    tokenizer = _trained_tokenizer()
    different_result = train_reference_bpe(
        ("bb bb 🚀", "bb"),
        vocab_size=BYTE_VOCAB_SIZE + 3 + len(NANOCHAT_SPECIAL_TOKENS),
    )
    destination = tmp_path / "artifacts"

    with pytest.raises(TokenizerArtifactError, match="tokenizer.*training result"):
        save_regex_bpe_artifacts(tokenizer, different_result, destination)

    assert not destination.exists()


def test_save_is_no_clobber_and_accepts_an_existing_empty_directory(
    tmp_path: Path,
) -> None:
    tokenizer = _trained_tokenizer()
    empty_destination = tmp_path / "empty"
    empty_destination.mkdir()
    tokenizer.save(empty_destination)
    assert {entry.name for entry in empty_destination.iterdir()} == set(
        TOKENIZER_ARTIFACT_FILENAMES
    )

    occupied_destination = tmp_path / "occupied"
    occupied_destination.mkdir()
    sentinel = occupied_destination / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        tokenizer.save(occupied_destination)

    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert {entry.name for entry in occupied_destination.iterdir()} == {"sentinel.txt"}


def test_failed_save_never_publishes_a_partial_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifacts"

    def fail_token_bytes(*args: object, **kwargs: object) -> Path:
        raise OSError("simulated token tensor write failure")

    monkeypatch.setattr(
        tokenizer_artifacts,
        "save_token_byte_lengths",
        fail_token_bytes,
    )

    with pytest.raises(OSError, match="simulated"):
        _trained_tokenizer().save(destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".artifacts.*.tmp")) == []


def test_save_and_load_reject_symlinked_artifact_files(tmp_path: Path) -> None:
    tokenizer = _trained_tokenizer()
    real_path = tmp_path / "real"
    link_path = tmp_path / "linked"
    tokenizer.save(real_path)
    link_path.symlink_to(real_path, target_is_directory=True)

    with pytest.raises(TokenizerArtifactError, match="must not be a symlink"):
        tokenizer.save(link_path)

    artifact_path = tmp_path / "artifacts"
    tokenizer.save(artifact_path)
    merges_path = artifact_path / "merges.json"
    outside_path = tmp_path / "outside.json"
    outside_path.write_bytes(merges_path.read_bytes())
    merges_path.unlink()
    merges_path.symlink_to(outside_path)

    with pytest.raises(TokenizerArtifactError, match="regular file.*symlink"):
        RegexBPETokenizer.load(artifact_path)


def test_token_byte_builder_never_decodes_special_token_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _trained_tokenizer()
    special_ids = {tokenizer.encode_special(token) for token in NANOCHAT_SPECIAL_TOKENS}
    decoded_ids: list[int] = []
    real_decode = tokenizer.decode_single_token_bytes

    def recording_decode(token_id: int) -> bytes:
        if token_id in special_ids:
            raise AssertionError("special token bytes must not be decoded")
        decoded_ids.append(token_id)
        return real_decode(token_id)

    monkeypatch.setattr(tokenizer, "decode_single_token_bytes", recording_decode)

    lengths = build_token_byte_lengths(tokenizer)

    assert all(lengths[token_id].item() == 0 for token_id in special_ids)
    assert set(decoded_ids) == set(range(tokenizer.get_vocab_size())) - special_ids


def test_readme_documents_artifact_atomicity_identity_and_dtype() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "## Regex byte-BPE artifacts" in readme
    assert "one atomic rename" in readme
    assert "stable\n`sha256:` tokenizer identity" in readme
    assert "`torch.int32` dtype" in readme
    assert "every special-token entry is exactly zero" in readme
