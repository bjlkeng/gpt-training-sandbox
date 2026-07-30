"""Config-only model, token-budget, and training-memory estimates."""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import scratch_llm.resource_estimation as resource_estimation
from scratch_llm.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
)
from scratch_llm.config import GPTConfig, load_config
from scratch_llm.model import GPT
from scratch_llm.resource_estimation import (
    RESOURCE_ESTIMATE_FORMAT,
    RESOURCE_ESTIMATE_FORMAT_VERSION,
    compare_memory_estimate,
    estimate_gpt_model_size,
    estimate_token_budget,
    estimate_training_resources,
    render_training_resource_estimate,
    summarize_module_parameters,
)
from scratch_llm.utils import count_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
PRESET_COUNTS = {
    "base_smoke.yaml": (4_604_544, True),
    "tiny_20m_3090.yaml": (23_401_344, True),
    "small_45m_3090.yaml": (42_476_032, False),
}
MIB = 1024**2


@pytest.mark.parametrize(
    ("filename", "expected"),
    PRESET_COUNTS.items(),
)
def test_preset_model_estimates_match_constructed_tied_models(
    filename: str,
    expected: tuple[int, bool],
) -> None:
    config = load_config(CONFIG_DIR / filename)

    estimate = estimate_gpt_model_size(config.model)
    model = GPT(config.model)
    actual = summarize_module_parameters(model)

    assert estimate.unique_parameters == expected[0]
    assert estimate.trainable_parameters == expected[0]
    assert estimate.non_trainable_parameters == 0
    assert actual.unique_parameters == expected[0]
    assert actual.trainable_parameters == expected[0]
    assert actual.non_trainable_parameters == 0
    assert count_parameters(model) == estimate.unique_parameters
    assert estimate.tie_weights is True
    assert estimate.output_head_parameters == 0
    assert estimate.embedding_dominated is expected[1]
    assert estimate.embedding_fraction == pytest.approx(
        estimate.embedding_parameters / estimate.unique_parameters
    )
    assert sum(estimate.component_parameters.values()) == expected[0]


@pytest.mark.parametrize("bias", [False, True])
def test_tied_untied_and_trainability_counts_are_deduplicated(
    bias: bool,
) -> None:
    values = {
        "vocab_size": 17,
        "seq_len": 4,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "mlp_ratio": 3,
        "bias": bias,
    }
    tied_config = GPTConfig(**values, tie_weights=True)
    untied_config = GPTConfig(**values, tie_weights=False)
    tied = GPT(tied_config)
    untied = GPT(untied_config)

    tied_estimate = estimate_gpt_model_size(tied_config)
    untied_estimate = estimate_gpt_model_size(untied_config)

    assert summarize_module_parameters(tied).unique_parameters == (
        tied_estimate.unique_parameters
    )
    assert summarize_module_parameters(untied).unique_parameters == (
        untied_estimate.unique_parameters
    )
    assert untied_estimate.unique_parameters - tied_estimate.unique_parameters == (
        tied_config.vocab_size * tied_config.n_embd
    )
    assert untied_estimate.output_head_parameters == (
        tied_config.vocab_size * tied_config.n_embd
    )

    untied.position_embedding.weight.requires_grad_(False)
    summary = summarize_module_parameters(untied)
    assert summary.unique_parameters == untied_estimate.unique_parameters
    assert summary.non_trainable_parameters == (
        untied_config.seq_len * untied_config.n_embd
    )
    assert (
        summary.trainable_parameters + summary.non_trainable_parameters
        == summary.unique_parameters
    )


def test_token_budget_uses_exact_training_accumulation_and_marks_targets_unknown() -> (
    None
):
    automatic = estimate_token_budget(
        device_batch_size=4,
        seq_len=512,
        total_batch_size_tokens=65_536,
        grad_accum_steps="auto",
    )
    explicit = estimate_token_budget(
        device_batch_size=4,
        seq_len=512,
        total_batch_size_tokens=65_536,
        grad_accum_steps=32,
    )

    assert automatic == explicit
    assert automatic.processed_model_tokens_per_microbatch == 2_048
    assert automatic.grad_accum_steps == 32
    assert automatic.processed_model_tokens_per_optimizer_step == 65_536
    assert automatic.configured_total_batch_size_tokens == 65_536
    assert automatic.maximum_supervised_targets_per_microbatch == 2_048
    assert automatic.maximum_supervised_targets_per_optimizer_step == 65_536
    assert automatic.supervised_target_tokens_per_microbatch is None
    assert automatic.supervised_target_tokens_per_optimizer_step is None
    assert automatic.supervised_targets_are_data_and_mask_dependent is True

    with pytest.raises(ValueError, match="must be divisible"):
        estimate_token_budget(
            device_batch_size=3,
            seq_len=5,
            total_batch_size_tokens=32,
            grad_accum_steps="auto",
        )
    with pytest.raises(ValueError, match="contradicts"):
        estimate_token_budget(
            device_batch_size=4,
            seq_len=512,
            total_batch_size_tokens=65_536,
            grad_accum_steps=31,
        )


def test_memory_components_and_headroom_follow_documented_arithmetic() -> None:
    config = load_config(CONFIG_DIR / "base_smoke.yaml")
    result = estimate_training_resources(config)
    memory = result.memory
    model = result.model
    batch = config.train.device_batch_size
    sequence = config.model.seq_len
    channels = config.model.n_embd
    layers = config.model.n_layer
    heads = config.model.n_head
    ratio = config.model.mlp_ratio
    vocab = config.model.vocab_size
    dtype_bytes = 4

    expected_activations = (
        layers * batch * sequence * channels * (8 + 2 * ratio) * dtype_bytes
        + layers * 2 * batch * heads * sequence**2 * dtype_bytes
        + layers * sequence**2
        + batch * sequence * channels * dtype_bytes
    )
    expected_logits = 2 * batch * sequence * vocab * dtype_bytes + batch * sequence * (
        8 + 4
    )

    assert memory.parameter_bytes == model.unique_parameters * dtype_bytes
    assert memory.gradient_bytes == model.trainable_parameters * dtype_bytes
    assert memory.optimizer_state_bytes == model.trainable_parameters * 8
    assert memory.activation_bytes == expected_activations
    assert memory.logits_loss_workspace_bytes == expected_logits
    expected_subtotal = (
        memory.parameter_bytes
        + memory.gradient_bytes
        + memory.optimizer_state_bytes
        + expected_activations
        + expected_logits
    )
    assert memory.subtotal_bytes == expected_subtotal
    assert memory.allocator_headroom_bytes == max(
        math.ceil(expected_subtotal * 0.20),
        512 * MIB,
    )
    assert memory.total_bytes == (
        memory.subtotal_bytes + memory.allocator_headroom_bytes
    )

    payload = memory.to_dict()
    assert set(payload["components"]) == {
        "activations",
        "allocator_headroom",
        "gradients",
        "logits_loss_workspace",
        "optimizer_states",
        "parameters",
    }
    for component in payload["components"].values():
        assert component["bytes"] >= 0
        assert component["mib"] == component["bytes"] / MIB
    assert payload["classification"] == "conservative_estimate_not_observed"
    assert payload["assumptions"]["automatic_mixed_precision"] is False
    assert payload["assumptions"]["activation_checkpointing"] is False
    assert payload["assumptions"]["optimizer"] == "AdamW"
    assert payload["assumptions"]["optimizer_state_dtype"] == "float32"
    assert payload["assumptions"]["attention"] == (
        "manual_materialized_scores_and_probabilities"
    )


def test_dtype_batch_and_sequence_changes_affect_only_documented_terms() -> None:
    float32_config = load_config(CONFIG_DIR / "tiny_20m_3090.yaml")
    bfloat16_config = copy.deepcopy(float32_config)
    bfloat16_config.train.dtype = "bfloat16"
    float32 = estimate_training_resources(float32_config)
    bfloat16 = estimate_training_resources(bfloat16_config)

    assert bfloat16.memory.parameter_bytes * 2 == float32.memory.parameter_bytes
    assert bfloat16.memory.gradient_bytes * 2 == float32.memory.gradient_bytes
    assert bfloat16.memory.optimizer_state_bytes == float32.memory.optimizer_state_bytes
    assert bfloat16.memory.activation_bytes < float32.memory.activation_bytes
    assert (
        bfloat16.memory.logits_loss_workspace_bytes
        < float32.memory.logits_loss_workspace_bytes
    )
    assert bfloat16.memory.to_dict()["assumptions"]["parameter_dtype"] == "bfloat16"

    smaller_batch_config = copy.deepcopy(float32_config)
    smaller_batch_config.train.device_batch_size = 2
    smaller_batch = estimate_training_resources(smaller_batch_config)
    assert smaller_batch.memory.parameter_bytes == float32.memory.parameter_bytes
    assert smaller_batch.memory.activation_bytes < float32.memory.activation_bytes
    assert (
        smaller_batch.memory.logits_loss_workspace_bytes
        < float32.memory.logits_loss_workspace_bytes
    )
    assert (
        smaller_batch.tokens.processed_model_tokens_per_microbatch
        == float32.tokens.processed_model_tokens_per_microbatch // 2
    )

    shorter_sequence_config = load_config(CONFIG_DIR / "small_45m_3090.yaml")
    long_sequence = estimate_training_resources(shorter_sequence_config)
    shorter_sequence_config.model.seq_len = 512
    short_sequence = estimate_training_resources(shorter_sequence_config)
    assert short_sequence.model.position_embedding_parameters * 2 == (
        long_sequence.model.position_embedding_parameters
    )
    assert (
        short_sequence.memory.activation_bytes < long_sequence.memory.activation_bytes
    )
    assert (
        short_sequence.memory.logits_loss_workspace_bytes
        < long_sequence.memory.logits_loss_workspace_bytes
    )


def test_unsupported_checkpointing_and_signed_64_bit_overflow_fail_explicitly() -> None:
    checkpointed = load_config(CONFIG_DIR / "base_smoke.yaml")
    checkpointed.train.activation_checkpointing = True
    with pytest.raises(ValueError, match="activation checkpointing"):
        estimate_training_resources(checkpointed)

    oversized = load_config(CONFIG_DIR / "base_smoke.yaml")
    oversized.model.vocab_size = 2**62
    oversized.tokenizer.vocab_size = 2**62
    with pytest.raises(OverflowError, match="signed 64-bit"):
        estimate_training_resources(oversized)


def test_resource_json_and_human_summary_are_stable_finite_and_unambiguous() -> None:
    config = load_config(CONFIG_DIR / "tiny_20m_3090.yaml")
    first = estimate_training_resources(config)
    second = estimate_training_resources(config)

    assert first == second
    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    assert payload == first.to_dict()
    assert payload["format"] == RESOURCE_ESTIMATE_FORMAT
    assert payload["format_version"] == RESOURCE_ESTIMATE_FORMAT_VERSION
    _assert_finite_json(payload)

    rendered = render_training_resource_estimate(first)
    assert "Conservative training resource estimate" in rendered
    assert "not observed usage" in rendered
    assert "bytes" in rendered
    assert "MiB" in rendered
    assert "processed model tokens" in rendered
    assert "supervised target count: data/mask dependent" in rendered
    assert "automatic mixed precision: disabled" in rendered
    assert "activation checkpointing: disabled" in rendered


def test_memory_comparison_keeps_estimated_and_observed_names_distinct() -> None:
    result = estimate_training_resources(load_config(CONFIG_DIR / "base_smoke.yaml"))
    snapshot = AcceleratorMemorySnapshot(
        device=torch.device("cuda:0"),
        available=True,
        allocated_bytes=100,
        reserved_bytes=120,
        peak_allocated_bytes=200,
        peak_reserved_bytes=240,
        capacity_bytes=24 * 1024**3,
    )

    comparison = compare_memory_estimate(result.memory, snapshot)

    assert comparison.estimated_total_bytes == result.memory.total_bytes
    assert comparison.observed_peak_allocated_bytes == 200
    assert comparison.observed_peak_reserved_bytes == 240
    assert comparison.observed_peak_allocated_mib == 200 / MIB
    assert comparison.observed_peak_reserved_mib == 240 / MIB
    assert comparison.estimate_minus_observed_peak_allocated_bytes == (
        result.memory.total_bytes - 200
    )
    assert comparison.estimate_minus_observed_peak_reserved_bytes == (
        result.memory.total_bytes - 240
    )
    assert comparison.to_dict()["comparison"] == "estimate_vs_observed_snapshot"

    unavailable = compare_memory_estimate(
        result.memory,
        collect_accelerator_memory("cpu"),
    )
    assert unavailable.observed_available is False
    assert unavailable.observed_peak_allocated_bytes is None
    assert unavailable.estimate_minus_observed_peak_allocated_bytes is None
    assert "unavailable" in unavailable.to_dict()["observed"]["reason"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_comparison_reports_estimate_and_actual_without_ratio_contract() -> None:
    result = estimate_training_resources(load_config(CONFIG_DIR / "base_smoke.yaml"))

    comparison = compare_memory_estimate(
        result.memory,
        collect_accelerator_memory("cuda"),
    )

    assert comparison.observed_available is True
    assert comparison.observed_peak_allocated_bytes is not None
    assert comparison.observed_peak_reserved_bytes is not None
    assert comparison.estimated_total_bytes == result.memory.total_bytes


def test_pretrain_dry_run_prints_stable_machine_and_human_resource_summaries(
    tmp_path: Path,
) -> None:
    config_path = CONFIG_DIR / "base_smoke.yaml"
    output_dir = tmp_path / "runs"
    command = [
        sys.executable,
        "-m",
        "scripts.pretrain",
        "--config",
        str(config_path),
        "--override",
        f"run.output_dir={output_dir}",
        "--override",
        "run.name=resource-dry-run",
        "--no-wandb",
        "--wandb-mode",
        "disabled",
        "--dry-run",
    ]

    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == second.returncode == 0
    first_payload = _resource_payload(first.stdout)
    second_payload = _resource_payload(second.stdout)
    assert first_payload == second_payload
    assert (
        first_payload == estimate_training_resources(load_config(config_path)).to_dict()
    )
    assert "Conservative training resource estimate" in first.stdout
    assert "not observed usage" in first.stdout
    assert "Memory components:" in first.stdout
    assert not list((output_dir / "resource-dry-run" / "checkpoints").iterdir())


def test_estimator_has_no_gpt_construction_dependency_and_readme_documents_scope() -> (
    None
):
    source_path = Path(resource_estimation.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "scratch_llm.model" not in imported_modules
    assert "GPT" not in imported_names
    assert "conservative planning estimate" in readme
    assert "not observed CUDA usage" in readme
    assert "20% allocator/headroom" in readme
    assert "512 MiB minimum" in readme
    assert "data- and mask-dependent" in readme


def _resource_payload(stdout: str) -> dict[str, object]:
    prefix = "Resource estimate JSON: "
    matches = [
        line.removeprefix(prefix)
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(matches) == 1
    payload = json.loads(matches[0])
    assert isinstance(payload, dict)
    return payload


def _assert_finite_json(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_json(item)
