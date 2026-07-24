"""Smoke tests for the installable project layout and dependency contract."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import scratch_llm
import scripts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CORE_DEPENDENCIES = {
    "numpy",
    "omegaconf",
    "pandas",
    "pyarrow",
    "regex",
    "torch",
    "tqdm",
}
EXPECTED_OPTIONAL_DEPENDENCIES = {
    "demo": {"gradio"},
    "dev": {"matplotlib", "mypy", "pytest", "ruff"},
    "tracking": {"wandb"},
    "web": {"fastapi", "pydantic", "uvicorn", "websockets"},
}
FORBIDDEN_CORE_DEPENDENCIES = {
    "accelerate",
    "datasets",
    "deepspeed",
    "lightning",
    "tokenizers",
    "transformers",
    "trl",
}
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
EXTRA_MARKER = re.compile(r"extra\s*==\s*['\"]([^'\"]+)['\"]")


def _normalized_requirement_name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    assert match is not None, f"Unable to parse requirement: {requirement}"
    return match.group(1).lower().replace("_", "-")


def test_packages_import_from_the_planned_layout() -> None:
    assert (
        Path(scratch_llm.__file__).resolve().parent
        == PROJECT_ROOT / "src" / "scratch_llm"
    )
    assert Path(scripts.__file__).resolve().parent == PROJECT_ROOT / "scripts"


def test_editable_install_exposes_packages_outside_the_repository(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import scratch_llm, scripts"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_distribution_metadata_separates_core_and_optional_dependencies() -> None:
    metadata = distribution("scratch-llm").metadata
    requirements = metadata.get_all("Requires-Dist") or []
    extras = set(metadata.get_all("Provides-Extra") or [])

    core_dependencies = {
        _normalized_requirement_name(requirement)
        for requirement in requirements
        if EXTRA_MARKER.search(requirement) is None
    }
    optional_dependencies: dict[str, set[str]] = {extra: set() for extra in extras}
    for requirement in requirements:
        marker = EXTRA_MARKER.search(requirement)
        if marker is not None:
            optional_dependencies[marker.group(1)].add(
                _normalized_requirement_name(requirement)
            )

    assert metadata["Requires-Python"] == ">=3.10"
    assert core_dependencies == EXPECTED_CORE_DEPENDENCIES
    assert core_dependencies.isdisjoint(FORBIDDEN_CORE_DEPENDENCIES)
    assert optional_dependencies == EXPECTED_OPTIONAL_DEPENDENCIES

    ruff_requirements = [
        requirement
        for requirement in requirements
        if _normalized_requirement_name(requirement) == "ruff"
    ]
    assert len(ruff_requirements) == 1
    assert ruff_requirements[0].startswith("ruff==0.15.22;")
