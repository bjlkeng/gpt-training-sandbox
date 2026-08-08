"""Block-boundary activation checkpointing math and mode coverage."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import scratch_llm.model as model_module
from scratch_llm.config import GPTConfig
from scratch_llm.model import GPT
from scratch_llm.training.activation_checkpointing import (
    configure_activation_checkpointing,
)
from scratch_llm.training.loop import run_training_steps


def _config(*, dropout: float = 0.0) -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        seq_len=6,
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_ratio=2,
        dropout=dropout,
    )


def _forward_backward(
    model: GPT,
    tokens: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    embedding_outputs: list[torch.Tensor] = []

    def retain_embedding_grad(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        embedding_outputs.append(output)

    handle = model.token_embedding.register_forward_hook(retain_embedding_grad)
    try:
        torch.manual_seed(seed)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.reshape(-1),
        )
        loss.backward()
    finally:
        handle.remove()
    embedding_grad = embedding_outputs[0].grad
    assert embedding_grad is not None
    parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return logits.detach(), embedding_grad.detach().clone(), parameter_grads


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_checkpointed_blocks_match_forward_input_and_parameter_gradients(
    dropout: float,
) -> None:
    torch.manual_seed(107)
    ordinary = GPT(_config(dropout=dropout))
    checkpointed = GPT(_config(dropout=dropout))
    checkpointed.load_state_dict(ordinary.state_dict(), strict=True)
    configure_activation_checkpointing(checkpointed, enabled=True)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])

    ordinary_result = _forward_backward(ordinary, tokens, seed=109)
    checkpointed_result = _forward_backward(checkpointed, tokens, seed=109)

    for actual, expected in zip(
        checkpointed_result[:2],
        ordinary_result[:2],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert set(checkpointed_result[2]) == set(ordinary_result[2])
    for name, expected in ordinary_result[2].items():
        torch.testing.assert_close(
            checkpointed_result[2][name],
            expected,
            rtol=2e-5,
            atol=1e-6,
        )


def test_disabled_mode_preserves_the_exact_model_path_and_state_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = GPT(_config())
    keys = set(model.state_dict())
    called = False

    def unexpected_checkpoint(*_args: object, **_kwargs: object) -> torch.Tensor:
        nonlocal called
        called = True
        raise AssertionError("disabled path invoked checkpoint")

    monkeypatch.setattr(model_module, "checkpoint", unexpected_checkpoint)
    selection = configure_activation_checkpointing(model, enabled=False)

    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 3, 32)
    assert called is False
    assert set(model.state_dict()) == keys
    assert selection.to_dict() == {
        "block_boundary": True,
        "effective": False,
        "requested": False,
        "use_reentrant": False,
    }


@pytest.mark.parametrize("mode", ["eval", "no_grad", "inference"])
def test_non_training_modes_never_invoke_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    model = GPT(_config())
    configure_activation_checkpointing(model, enabled=True)
    monkeypatch.setattr(
        model_module,
        "checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-training mode invoked checkpoint")
        ),
    )
    tokens = torch.tensor([[1, 2, 3]])

    if mode == "eval":
        model.eval()
        output = model(tokens)
    elif mode == "no_grad":
        model.train()
        with torch.no_grad():
            output = model(tokens)
    else:
        model.train()
        with torch.inference_mode():
            output = model(tokens)

    assert output.shape == (1, 3, 32)


class _FailOnRecomputation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("fixture recomputation failure")
        return torch.sin(x) * self.scale


def test_recomputation_failure_is_actionable_and_leaves_step_unchanged() -> None:
    model = GPT(_config())
    model.blocks = nn.ModuleList([_FailOnRecomputation()])
    configure_activation_checkpointing(model, enabled=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    initial_epoch = scheduler.last_epoch
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with pytest.raises(
        RuntimeError,
        match="activation checkpoint block 0 failed.*fixture recomputation failure",
    ):
        run_training_steps(
            model,
            [(tokens, tokens)],
            optimizer,
            scheduler,
            max_steps=1,
            grad_accum_steps=1,
            grad_clip=1.0,
            device="cpu",
        )

    assert scheduler.last_epoch == initial_epoch
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
