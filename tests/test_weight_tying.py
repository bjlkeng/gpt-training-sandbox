"""Untied token-embedding construction, training, and evidence contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    TokenizerConfig,
    TrainConfig,
)
from scratch_llm.diagnostics.resource_estimation import (
    estimate_gpt_model_size,
    estimate_training_resources,
    summarize_module_parameters,
)
from scratch_llm.model import GPT
from scratch_llm.training.checkpoint import (
    CheckpointError,
    load_model_checkpoint,
    save_checkpoint,
)
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.tokenization.tokenizer import VOCAB_SIZE, ByteTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model_config(**overrides: object) -> GPTConfig:
    values: dict[str, object] = {
        "vocab_size": 24,
        "seq_len": 6,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 8,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "bias": False,
    }
    values.update(overrides)
    return GPTConfig(**values)  # type: ignore[arg-type]


def _checkpoint_components(
    *, tie_weights: bool
) -> tuple[
    ProjectConfig,
    ByteTokenizer,
    GPT,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    config = ProjectConfig(
        run=RunConfig(device="cpu"),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=_model_config(
            vocab_size=VOCAB_SIZE,
            seq_len=4,
            tie_weights=tie_weights,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=4,
            grad_accum_steps=1,
            max_steps=1,
            warmup_steps=0,
            warmdown_ratio=0.0,
        ),
    )
    model = GPT(config.model)
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    return config, ByteTokenizer(), model, optimizer, scheduler


def test_tied_and_untied_construction_have_explicit_storage_and_initialization() -> (
    None
):
    torch.manual_seed(601)
    tied = GPT(_model_config(tie_weights=True))
    torch.manual_seed(601)
    untied = GPT(_model_config(tie_weights=False))

    assert tied.token_embedding.weight is tied.lm_head.weight
    assert tied.token_embedding.weight.data_ptr() == tied.lm_head.weight.data_ptr()
    assert untied.token_embedding.weight is not untied.lm_head.weight
    assert untied.token_embedding.weight.data_ptr() != untied.lm_head.weight.data_ptr()
    torch.testing.assert_close(
        tied.token_embedding.weight,
        untied.token_embedding.weight,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        untied.token_embedding.weight,
        untied.lm_head.weight,
    )
    linear_bound = 1 / math.sqrt(untied.config.n_embd)
    assert untied.lm_head.weight.abs().max().item() <= linear_bound


def test_default_tied_mode_preserves_explicit_tied_logits() -> None:
    torch.manual_seed(607)
    implicit = GPT(_model_config()).eval()
    torch.manual_seed(607)
    explicit = GPT(_model_config(tie_weights=True)).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])

    with torch.inference_mode():
        torch.testing.assert_close(implicit(tokens), explicit(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("tie_weights", [True, False])
def test_forward_loss_gradients_and_optimizer_cover_unique_parameters_once(
    tie_weights: bool,
) -> None:
    torch.manual_seed(613)
    config = _model_config(tie_weights=tie_weights)
    model = GPT(config)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])

    logits = model(tokens)
    loss = model(tokens, targets)
    loss.backward()
    optimizer = build_optimizer(model, TrainConfig())
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]

    assert logits.shape == (2, 6, config.vocab_size)
    assert loss.ndim == 0
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    assert len({id(parameter) for parameter in optimizer_parameters}) == len(
        optimizer_parameters
    )
    assert {id(parameter) for parameter in optimizer_parameters} == {
        id(parameter) for parameter in model.parameters()
    }
    if tie_weights:
        assert model.token_embedding.weight.grad is model.lm_head.weight.grad
    else:
        assert model.token_embedding.weight.grad is not model.lm_head.weight.grad


def test_untied_parameter_and_training_memory_deltas_are_exact() -> None:
    tied_model = _model_config(
        vocab_size=VOCAB_SIZE,
        seq_len=4,
        tie_weights=True,
    )
    untied_model = _model_config(
        vocab_size=VOCAB_SIZE,
        seq_len=4,
        tie_weights=False,
    )
    tied_size = estimate_gpt_model_size(tied_model)
    untied_size = estimate_gpt_model_size(untied_model)
    parameter_delta = tied_model.vocab_size * tied_model.n_embd

    assert (
        untied_size.unique_parameters - tied_size.unique_parameters == parameter_delta
    )
    assert untied_size.output_head_parameters == parameter_delta
    assert summarize_module_parameters(GPT(tied_model)).unique_parameters == (
        tied_size.unique_parameters
    )
    assert summarize_module_parameters(GPT(untied_model)).unique_parameters == (
        untied_size.unique_parameters
    )

    def project(model: GPTConfig) -> ProjectConfig:
        return ProjectConfig(
            tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
            model=model,
            train=TrainConfig(
                device_batch_size=1,
                total_batch_size_tokens=model.seq_len,
                grad_accum_steps=1,
            ),
        )

    tied_memory = estimate_training_resources(project(tied_model)).memory
    untied_memory = estimate_training_resources(project(untied_model)).memory
    assert untied_memory.parameter_bytes - tied_memory.parameter_bytes == (
        parameter_delta * tied_memory.bytes_per_dtype_element
    )
    assert untied_memory.gradient_bytes - tied_memory.gradient_bytes == (
        parameter_delta * tied_memory.bytes_per_dtype_element
    )
    assert untied_memory.optimizer_state_bytes - tied_memory.optimizer_state_bytes == (
        parameter_delta * 2 * 4
    )


@pytest.mark.parametrize("tie_weights", [True, False])
def test_full_checkpoint_round_trip_preserves_weight_topology_and_logits(
    tmp_path: Path,
    tie_weights: bool,
) -> None:
    torch.manual_seed(617)
    config, tokenizer, model, optimizer, scheduler = _checkpoint_components(
        tie_weights=tie_weights
    )
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(tokens)
    checkpoint_path = save_checkpoint(
        tmp_path / f"{tie_weights}.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )

    loaded = load_model_checkpoint(checkpoint_path)

    assert loaded.config.model.tie_weights is tie_weights
    assert (
        loaded.model.token_embedding.weight is loaded.model.lm_head.weight
    ) is tie_weights
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(tokens), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("source_tied", "configured_tied", "actual"),
    [(True, False, "shared"), (False, True, "independent")],
)
def test_checkpoint_rejects_both_weight_sharing_config_mismatches(
    tmp_path: Path,
    source_tied: bool,
    configured_tied: bool,
    actual: str,
) -> None:
    config, tokenizer, model, optimizer, scheduler = _checkpoint_components(
        tie_weights=source_tied
    )
    checkpoint_path = save_checkpoint(
        tmp_path / "source.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["config"]["model"]["tie_weights"] = configured_tied
    mismatched = tmp_path / "mismatched.pt"
    torch.save(payload, mismatched)

    configured = str(configured_tied).lower()
    with pytest.raises(
        CheckpointError,
        match=rf"tie_weights={configured}.*found {actual}",
    ):
        load_model_checkpoint(mismatched)


@pytest.mark.parametrize("tie_weights", [True, False])
def test_tied_and_untied_models_tiny_overfit(tie_weights: bool) -> None:
    torch.manual_seed(619)
    model = GPT(
        _model_config(
            tie_weights=tie_weights,
            vocab_size=16,
            n_embd=16,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])

    for _ in range(60):
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.inference_mode():
        assert model(inputs, targets).item() < 0.1


def test_weight_tying_documentation_and_bounded_report_are_reproducible() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    directory = PROJECT_ROOT / "comparisons" / "gpt-training-sandbox-as7-5-untied"
    report = (directory / "README.md").read_text(encoding="utf-8")
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    offline = json.loads(
        (directory / "offline-run-comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )

    assert "### Tied and untied token embeddings" in readme
    assert "32K" in readme
    assert "experimental" in report
    assert payload["controls"]["changed_config_fields"] == [
        "model.tie_weights",
        "run.name",
    ]
    assert payload["deltas"]["unique_parameters"] == (
        payload["controls"]["vocab_size"] * payload["controls"]["embedding_width"]
    )
    assert payload["runs"]["tied"]["tie_weights"] is True
    assert payload["runs"]["untied"]["tie_weights"] is False
    assert len(offline["runs"]) == 2
    assert offline["rankings"] == {
        "full_documents_v1": [],
        "nanochat_compat_v1": [],
    }
