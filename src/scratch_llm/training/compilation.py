"""Execution-only torch.compile adapter that preserves canonical model state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import torch
from torch import nn
from torch.optim import Optimizer

from scratch_llm.config import SFTConfig, TrainConfig
from scratch_llm.training.precision import PrecisionPolicy
from scratch_llm.training.rng_state import preserve_global_rng_state


COMPILE_CONSTRUCTION_FAILED = "compile_construction_failed"
COMPILE_EXECUTION_FAILED = "compile_execution_failed"


class CompileRuntimeError(RuntimeError):
    """A strict compile request could not execute safely."""


class ModelCompiler(Protocol):
    """The supported subset of ``torch.compile`` used by the adapter."""

    def __call__(
        self,
        model: nn.Module,
        *,
        backend: str,
        mode: str,
        fullgraph: bool,
        dynamic: bool,
    ) -> nn.Module: ...


@dataclass
class CompileSelection:
    """Mutable observations with a stable JSON representation."""

    requested: bool
    effective: bool
    backend: str
    mode: str
    fullgraph: bool
    dynamic: bool
    compile_duration_seconds: float = 0.0
    fallback_reason: str | None = None
    _initial_graph_count: int | None = field(default=None, repr=False)
    _graph_counter: Callable[[], int | None] | None = field(default=None, repr=False)

    @property
    def observed_recompilations(self) -> int | None:
        if not self.requested:
            return 0
        if self._initial_graph_count is None or self._graph_counter is None:
            return None
        current = self._graph_counter()
        if current is None:
            return None
        return max(0, current - self._initial_graph_count - 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "compile_duration_seconds": self.compile_duration_seconds,
            "dynamic": self.dynamic,
            "effective": self.effective,
            "fallback_reason": self.fallback_reason,
            "fullgraph": self.fullgraph,
            "mode": self.mode,
            "observed_recompilations": self.observed_recompilations,
            "requested": self.requested,
        }


@dataclass(frozen=True)
class CompileRuntime:
    """Canonical artifact owner, execution module, and observations."""

    canonical_model: nn.Module
    execution_model: nn.Module
    selection: CompileSelection
    fallback_policy: str


class _CompiledExecution(nn.Module):
    """Catch lazy compiler failures without registering artifact-owned modules."""

    _canonical_model: nn.Module
    _compiled_model: nn.Module

    def __init__(
        self,
        canonical_model: nn.Module,
        compiled_model: nn.Module,
        selection: CompileSelection,
        *,
        fallback_policy: str,
        clock: Callable[[], float],
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_canonical_model", canonical_model)
        object.__setattr__(self, "_compiled_model", compiled_model)
        self.selection = selection
        self.fallback_policy = fallback_policy
        self.clock = clock
        self._observed_execution = False
        config = getattr(canonical_model, "config", None)
        if config is not None:
            self.config = config
        max_seq_len = getattr(canonical_model, "max_seq_len", None)
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self.selection.effective:
            return self._canonical_model(*args, **kwargs)
        started = self.clock() if not self._observed_execution else None
        try:
            result = self._compiled_model(*args, **kwargs)
        except Exception as error:
            if started is not None:
                self.selection.compile_duration_seconds += self.clock() - started
            if self.fallback_policy == "error":
                raise CompileRuntimeError(
                    f"{COMPILE_EXECUTION_FAILED}: {type(error).__name__}: {error}"
                ) from error
            self.selection.effective = False
            self.selection.fallback_reason = COMPILE_EXECUTION_FAILED
            return self._canonical_model(*args, **kwargs)
        if started is not None:
            self.selection.compile_duration_seconds += self.clock() - started
            self._observed_execution = True
        return result

    def create_kv_cache(self, *, batch_size: int, capacity: int) -> Any:
        """Delegate external inference-cache ownership to the canonical model."""

        factory = getattr(self._canonical_model, "create_kv_cache", None)
        if not callable(factory):
            raise TypeError("canonical model does not expose create_kv_cache")
        return factory(batch_size=batch_size, capacity=capacity)

    @property
    def canonical_model(self) -> nn.Module:
        """Expose the artifact owner for read-only identity and byte accounting."""

        return self._canonical_model

    def train(self, mode: bool = True) -> _CompiledExecution:
        super().train(mode)
        self._canonical_model.train(mode)
        if self._compiled_model is not self._canonical_model:
            self._compiled_model.train(mode)
        return self


def build_compile_runtime(
    model: nn.Module,
    config: TrainConfig | SFTConfig,
    *,
    compiler: ModelCompiler | None = None,
    clock: Callable[[], float] = perf_counter,
    graph_counter: Callable[[], int | None] | None = None,
) -> CompileRuntime:
    """Build an execution path while retaining ``model`` as artifact owner."""

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(config, (TrainConfig, SFTConfig)):
        raise TypeError("config must be a TrainConfig or SFTConfig")
    config.validate()
    if not callable(clock):
        raise TypeError("clock must be callable")
    selection = CompileSelection(
        requested=config.compile,
        effective=False,
        backend=config.compile_backend,
        mode=config.compile_mode,
        fullgraph=config.compile_fullgraph,
        dynamic=config.compile_dynamic,
    )
    if not config.compile:
        return CompileRuntime(model, model, selection, config.compile_fallback_policy)

    active_compiler = torch.compile if compiler is None else compiler
    if not callable(active_compiler):
        raise TypeError("compiler must be callable")
    active_graph_counter = (
        _dynamo_graph_count if graph_counter is None else graph_counter
    )
    selection._graph_counter = active_graph_counter
    selection._initial_graph_count = active_graph_counter()
    started = clock()
    try:
        compiled = active_compiler(
            model,
            backend=config.compile_backend,
            mode=config.compile_mode,
            fullgraph=config.compile_fullgraph,
            dynamic=config.compile_dynamic,
        )
        if not isinstance(compiled, nn.Module):
            raise TypeError("compiler did not return an nn.Module")
    except Exception as error:
        selection.compile_duration_seconds = clock() - started
        if config.compile_fallback_policy == "error":
            raise CompileRuntimeError(
                f"{COMPILE_CONSTRUCTION_FAILED}: {type(error).__name__}: {error}"
            ) from error
        selection.fallback_reason = COMPILE_CONSTRUCTION_FAILED
        return CompileRuntime(model, model, selection, config.compile_fallback_policy)
    selection.compile_duration_seconds = clock() - started
    selection.effective = True
    execution = _CompiledExecution(
        model,
        compiled,
        selection,
        fallback_policy=config.compile_fallback_policy,
        clock=clock,
    )
    return CompileRuntime(model, execution, selection, config.compile_fallback_policy)


def warmup_compiled_training(
    runtime: CompileRuntime,
    optimizer: Optimizer,
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    precision: PrecisionPolicy,
    device: str | torch.device,
    clock: Callable[[], float] = perf_counter,
) -> None:
    """Compile forward/backward on a deterministic dummy batch before step zero."""

    if not isinstance(runtime, CompileRuntime):
        raise TypeError("runtime must be a CompileRuntime")
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be an Optimizer")
    if not isinstance(precision, PrecisionPolicy):
        raise TypeError("precision must be a PrecisionPolicy")
    if not runtime.selection.requested or not runtime.selection.effective:
        return
    construction_duration = runtime.selection.compile_duration_seconds
    runtime.execution_model.train()
    started = clock()
    try:
        with preserve_global_rng_state(device), precision.autocast():
            loss = runtime.execution_model(inputs, targets)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("compiled training warmup must return a scalar loss")
            precision.backward(loss)
    except Exception as error:
        optimizer.zero_grad(set_to_none=True)
        runtime.selection.compile_duration_seconds = construction_duration + (
            clock() - started
        )
        if runtime.fallback_policy == "error":
            if isinstance(error, CompileRuntimeError):
                raise
            raise CompileRuntimeError(
                f"{COMPILE_EXECUTION_FAILED}: {type(error).__name__}: {error}"
            ) from error
        runtime.selection.effective = False
        runtime.selection.fallback_reason = COMPILE_EXECUTION_FAILED
        _warmup_eager_fallback(
            runtime,
            optimizer,
            inputs=inputs,
            targets=targets,
            precision=precision,
            device=device,
        )
        return
    finally:
        optimizer.zero_grad(set_to_none=True)
    runtime.selection.compile_duration_seconds = construction_duration + (
        clock() - started
    )


def _warmup_eager_fallback(
    runtime: CompileRuntime,
    optimizer: Optimizer,
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    precision: PrecisionPolicy,
    device: str | torch.device,
) -> None:
    try:
        with preserve_global_rng_state(device), precision.autocast():
            loss = runtime.canonical_model(inputs, targets)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("eager training warmup must return a scalar loss")
            precision.backward(loss)
    finally:
        optimizer.zero_grad(set_to_none=True)


def format_compile_selection(selection: CompileSelection) -> str:
    """Render one stable progress line for requested and observed state."""

    reason = selection.fallback_reason or "none"
    recompilations = selection.observed_recompilations
    return (
        f"torch.compile: requested={selection.requested} "
        f"effective={selection.effective} backend={selection.backend} "
        f"mode={selection.mode} fullgraph={selection.fullgraph} "
        f"dynamic={selection.dynamic} "
        f"compile_duration_seconds={selection.compile_duration_seconds:.6f} "
        f"observed_recompilations={recompilations} fallback_reason={reason}"
    )


def _dynamo_graph_count() -> int | None:
    try:
        dynamo = getattr(torch, "_dynamo")
        counters = dynamo.utils.counters
        return int(counters["stats"]["unique_graphs"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
