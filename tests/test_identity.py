"""Shared content-identity tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scratch_llm.config import ProjectConfig, RunConfig
from scratch_llm.identity import file_identity, project_config_identity


def test_file_identity_streams_the_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    content = b"checkpoint\x00content\n"
    path.write_bytes(content)

    assert file_identity(path) == "sha256:" + hashlib.sha256(content).hexdigest()

    with pytest.raises(FileNotFoundError, match="regular file"):
        file_identity(tmp_path / "missing.bin")


def test_project_config_identity_uses_canonical_resolved_config() -> None:
    config = ProjectConfig(run=RunConfig(name="identity-fixture"))
    encoded = json.dumps(
        config.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert project_config_identity(config) == (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    assert project_config_identity(config) != project_config_identity(
        ProjectConfig(run=RunConfig(name="different"))
    )
