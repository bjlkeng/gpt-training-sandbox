"""Canonical-state and failure-policy coverage for optional torch.compile."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

from scratch_llm.config import TrainConfig
from scratch_llm.model import GPT
from scratch_llm.training.compilation import (
    COMPILE_CONSTRUCTION_FAILED,
    COMPILE_EXECUTION_FAILED,
    CompileRuntimeError,
    build_compile_runtime,
    warmup_compiled_training,
)
from scratch_llm.training.loop import run_training_steps
from scratch_llm.training.precision import build_precision_policy


class _Proxy(nn.Module):
    def __init__(self, model: nn.Module, *, fail: bool = False) -> None:
        super().__init__()
        object.__setattr__(self, "model", model)
        self.fail = fail

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("fixture compiled execution failed")
        model = self.model
        assert isinstance(model, nn.Module)
        return model(*args, **kwargs)  # type: ignore[no-any-return]


class _FailBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, loss: torch.Tensor) -> torch.Tensor:
        del ctx
        return loss

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        raise RuntimeError("fixture compiled backward failed")


class _BackwardFailProxy(_Proxy):
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        loss = super().forward(*args, **kwargs)
        return _FailBackward.apply(loss)


def _model() -> GPT:
    from scratch_llm.config import GPTConfig

    return GPT(
        GPTConfig(
            vocab_size=32,
            seq_len=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
        )
    )


def _clock(*values: float):
    iterator: Iterator[float] = iter(values)
    return lambda: next(iterator)


def test_disabled_compile_is_an_exact_execution_noop() -> None:
    model = _model()
    called = False

    def compiler(*_args: object, **_kwargs: object) -> nn.Module:
        nonlocal called
        called = True
        return _Proxy(model)

    runtime = build_compile_runtime(
        model,
        TrainConfig(compile=False),
        compiler=compiler,
    )

    assert runtime.canonical_model is model
    assert runtime.execution_model is model
    assert runtime.selection.to_dict() == {
        "backend": "inductor",
        "compile_duration_seconds": 0.0,
        "dynamic": False,
        "effective": False,
        "fallback_reason": None,
        "fullgraph": False,
        "mode": "default",
        "observed_recompilations": 0,
        "requested": False,
    }
    assert called is False


def test_compiled_execution_keeps_canonical_state_and_optimizer_ownership() -> None:
    model = _model()
    canonical_keys = set(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())
    calls: list[dict[str, object]] = []

    def compiler(module: nn.Module, **kwargs: object) -> nn.Module:
        assert module is model
        calls.append(kwargs)
        return _Proxy(module)

    runtime = build_compile_runtime(
        model,
        TrainConfig(
            compile=True,
            compile_backend="eager",
            compile_mode="reduce-overhead",
            compile_fullgraph=True,
            compile_dynamic=True,
        ),
        compiler=compiler,
        clock=_clock(1.0, 1.25, 2.0, 2.75),
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    loss = runtime.execution_model(tokens, tokens)
    loss.backward()
    optimizer.step()

    assert calls == [
        {
            "backend": "eager",
            "dynamic": True,
            "fullgraph": True,
            "mode": "reduce-overhead",
        }
    ]
    assert set(runtime.canonical_model.state_dict()) == canonical_keys
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == {id(parameter) for parameter in model.parameters()}
    assert runtime.selection.effective is True
    assert runtime.selection.compile_duration_seconds == pytest.approx(1.0)


@pytest.mark.parametrize("policy", ["eager", "error"])
def test_compiler_construction_failure_obeys_policy(policy: str) -> None:
    model = _model()

    def fail(*_args: object, **_kwargs: object) -> nn.Module:
        raise RuntimeError("compiler missing")

    config = TrainConfig(
        compile=True,
        compile_fallback_policy=policy,  # type: ignore[arg-type]
    )
    if policy == "error":
        with pytest.raises(CompileRuntimeError, match=COMPILE_CONSTRUCTION_FAILED):
            build_compile_runtime(model, config, compiler=fail)
        return

    runtime = build_compile_runtime(model, config, compiler=fail)
    assert runtime.execution_model is model
    assert runtime.selection.effective is False
    assert runtime.selection.fallback_reason == COMPILE_CONSTRUCTION_FAILED


@pytest.mark.parametrize("policy", ["eager", "error"])
def test_lazy_compiled_execution_failure_obeys_policy(policy: str) -> None:
    model = _model()
    runtime = build_compile_runtime(
        model,
        TrainConfig(
            compile=True,
            compile_fallback_policy=policy,  # type: ignore[arg-type]
        ),
        compiler=lambda module, **_kwargs: _Proxy(module, fail=True),
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    if policy == "error":
        with pytest.raises(CompileRuntimeError, match=COMPILE_EXECUTION_FAILED):
            runtime.execution_model(tokens)
        return

    output = runtime.execution_model(tokens)
    assert output.shape == (1, 4, 32)
    assert runtime.selection.effective is False
    assert runtime.selection.fallback_reason == COMPILE_EXECUTION_FAILED


def test_recompilation_counter_is_reported_separately_from_initial_graph() -> None:
    model = _model()
    graphs = 10
    runtime = build_compile_runtime(
        model,
        TrainConfig(compile=True),
        compiler=lambda module, **_kwargs: _Proxy(module),
        graph_counter=lambda: graphs,
    )
    tokens = torch.tensor([[1, 2, 3, 4]])

    runtime.execution_model(tokens)
    graphs = 13

    assert runtime.selection.observed_recompilations == 2


def test_available_device_torch_compile_smoke_runs_two_optimizer_steps() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model().to(device)
    canonical_keys = set(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = build_compile_runtime(
        model,
        TrainConfig(
            compile=True,
            compile_backend="eager",
            compile_fallback_policy="error",
        ),
    )
    tokens = torch.tensor([[1, 2, 3, 4]], device=device)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = runtime.execution_model(tokens, tokens)
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    assert runtime.selection.effective is True
    assert set(model.state_dict()) == canonical_keys


def test_strict_lazy_failure_leaves_optimizer_scheduler_and_parameters_unchanged() -> (
    None
):
    model = _model()
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    initial_scheduler_epoch = scheduler.last_epoch
    runtime = build_compile_runtime(
        model,
        TrainConfig(compile=True, compile_fallback_policy="error"),
        compiler=lambda module, **_kwargs: _Proxy(module, fail=True),
    )
    tokens = torch.tensor([[1, 2, 3, 4]])

    with pytest.raises(CompileRuntimeError, match=COMPILE_EXECUTION_FAILED):
        run_training_steps(
            runtime.execution_model,
            [(tokens, tokens)],
            optimizer,
            scheduler,
            max_steps=1,
            grad_accum_steps=1,
            grad_clip=1.0,
            device="cpu",
        )

    assert scheduler.last_epoch == initial_scheduler_epoch
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["eager", "error"])
def test_full_forward_backward_warmup_handles_lazy_backward_failure(
    policy: str,
) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = build_compile_runtime(
        model,
        TrainConfig(
            compile=True,
            compile_fallback_policy=policy,  # type: ignore[arg-type]
        ),
        compiler=lambda module, **_kwargs: _BackwardFailProxy(module),
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    precision = build_precision_policy(dtype="float32", device="cpu")

    if policy == "error":
        with pytest.raises(CompileRuntimeError, match=COMPILE_EXECUTION_FAILED):
            warmup_compiled_training(
                runtime,
                optimizer,
                inputs=tokens,
                targets=tokens,
                precision=precision,
                device="cpu",
            )
    else:
        warmup_compiled_training(
            runtime,
            optimizer,
            inputs=tokens,
            targets=tokens,
            precision=precision,
            device="cpu",
        )
        assert runtime.selection.effective is False
        assert runtime.selection.fallback_reason == COMPILE_EXECUTION_FAILED
    assert all(parameter.grad is None for parameter in model.parameters())
