"""Shared deterministic identities for files and resolved project configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scratch_llm.config import ProjectConfig


def file_identity(path: str | Path) -> str:
    """Return a streaming SHA-256 identity for one regular file."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"identity source is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def project_config_identity(config: ProjectConfig) -> str:
    """Return the canonical identity of one fully resolved project config."""

    if not isinstance(config, ProjectConfig):
        raise TypeError("config must be a ProjectConfig")
    encoded = json.dumps(
        config.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["file_identity", "project_config_identity"]
