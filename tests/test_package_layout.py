"""Smoke tests for the installable project layout and dependency contract."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib.util import find_spec
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
    "tokenizer-comparison": {"tiktoken"},
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
EXPECTED_EVALUATION_MODULES = {
    "base",
    "base_pipeline",
    "base_tracking",
    "bpb",
    "full_document_bpb",
    "nanochat_bpb",
    "sampling",
    "tokenizer",
    "tokenizer_tracking",
}
EXPECTED_CORE_EVALUATION_MODULES = {
    "bundle",
    "examples",
    "pipeline",
    "prompting",
    "reporting",
    "results",
    "scoring",
    "tracking",
}
OBSOLETE_TOP_LEVEL_EVALUATION_MODULES = {
    "base_evaluation",
    "base_evaluation_pipeline",
    "base_evaluation_tracking",
    "base_sampling",
    "bpb",
    "core_bundle",
    "core_evaluation",
    "core_evaluation_pipeline",
    "core_examples",
    "core_prompting",
    "core_reporting",
    "core_scoring",
    "full_document_bpb",
    "nanochat_bpb",
    "tokenizer_evaluation",
    "tokenizer_tracking",
}
EXPECTED_DOMAIN_MODULES = {
    "comparison": {"loading", "model", "pipeline", "reporting"},
    "data": {"climbmix", "loaders", "preparation", "statistics", "tokenized"},
    "diagnostics": {
        "accelerator_memory",
        "oom",
        "resource_estimation",
        "throughput",
        "throughput_runtime",
    },
    "tokenization": {
        "artifacts",
        "bpe",
        "optimized_bpe",
        "regex_chunking",
        "tokenizer",
        "training",
    },
    "training": {
        "best_checkpoint",
        "checkpoint",
        "loop",
        "optim",
        "pretraining",
        "rng_state",
        "telemetry",
    },
}
OBSOLETE_TOP_LEVEL_DOMAIN_MODULES = {
    "_run_comparison_loading",
    "_run_comparison_model",
    "_run_comparison_reporting",
    "accelerator_memory",
    "best_checkpoint",
    "bpe",
    "bpe_optimized",
    "checkpoint",
    "climbmix",
    "data_preparation",
    "data_stats",
    "oom_diagnostics",
    "optim",
    "pretraining",
    "regex_chunking",
    "resource_estimation",
    "rng_state",
    "run_comparison",
    "throughput_benchmark",
    "throughput_benchmark_runtime",
    "tokenized_data",
    "tokenizer",
    "tokenizer_artifacts",
    "tokenizer_training",
    "training_telemetry",
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


def test_evaluation_modules_are_grouped_by_scope() -> None:
    evaluation_directory = PROJECT_ROOT / "src" / "scratch_llm" / "evaluation"
    core_directory = evaluation_directory / "core"

    for module_name in EXPECTED_EVALUATION_MODULES:
        spec = find_spec(f"scratch_llm.evaluation.{module_name}")
        assert spec is not None
        assert Path(spec.origin or "").resolve().parent == evaluation_directory
    for module_name in EXPECTED_CORE_EVALUATION_MODULES:
        spec = find_spec(f"scratch_llm.evaluation.core.{module_name}")
        assert spec is not None
        assert Path(spec.origin or "").resolve().parent == core_directory

    for module_name in OBSOLETE_TOP_LEVEL_EVALUATION_MODULES:
        assert find_spec(f"scratch_llm.{module_name}") is None


def test_domain_modules_are_grouped_by_responsibility() -> None:
    source_directory = PROJECT_ROOT / "src" / "scratch_llm"

    for package_name, module_names in EXPECTED_DOMAIN_MODULES.items():
        package_directory = source_directory / package_name
        package_spec = find_spec(f"scratch_llm.{package_name}")
        assert package_spec is not None
        assert package_spec.submodule_search_locations is not None
        assert Path(package_spec.origin or "").resolve().parent == package_directory
        for module_name in module_names:
            spec = find_spec(f"scratch_llm.{package_name}.{module_name}")
            assert spec is not None
            assert Path(spec.origin or "").resolve().parent == package_directory

    for module_name in OBSOLETE_TOP_LEVEL_DOMAIN_MODULES:
        assert find_spec(f"scratch_llm.{module_name}") is None


def test_editable_install_exposes_packages_outside_the_repository(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import scratch_llm, scripts; "
                "import scratch_llm.comparison.pipeline; "
                "import scratch_llm.data.loaders; "
                "import scratch_llm.diagnostics.throughput; "
                "import scratch_llm.evaluation.core.pipeline; "
                "import scratch_llm.tokenization.tokenizer; "
                "import scratch_llm.training.pretraining"
            ),
        ],
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
