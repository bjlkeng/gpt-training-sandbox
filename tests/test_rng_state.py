"""Deterministic CPU/fake coverage for exact global RNG state."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch

from scratch_llm.training.rng_state import (
    RNGStateError,
    TrainingRNGState,
    capture_training_rng_state,
    restore_training_rng_state,
)
from scratch_llm.training.checkpoint import ExactTrainingState
from scratch_llm.training import pretraining


class FakeCudaRNGBackend:
    """Narrow deterministic stand-in that never touches a real CUDA runtime."""

    def __init__(self, *, device_count: int = 2) -> None:
        self.count = device_count
        generator = torch.Generator(device="cpu").manual_seed(10)
        self.states = [generator.get_state().clone()]
        generator.manual_seed(11)
        self.states.append(generator.get_state().clone())
        self.calls: list[str] = []

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return True

    def device_count(self) -> int:
        self.calls.append("device_count")
        return self.count

    def get_rng_state_all(self) -> list[torch.Tensor]:
        self.calls.append("get_rng_state_all")
        return [state.clone() for state in self.states[: self.count]]

    def set_rng_state_all(self, states: list[torch.Tensor]) -> None:
        self.calls.append("set_rng_state_all")
        self.states = [state.clone() for state in states]


def _global_cpu_state() -> tuple[object, tuple[Any, ...], torch.Tensor]:
    return (
        random.getstate(),
        np.random.get_state(legacy=True),
        torch.get_rng_state().clone(),
    )


def _assert_global_cpu_state_equal(
    actual: tuple[object, tuple[Any, ...], torch.Tensor],
    expected: tuple[object, tuple[Any, ...], torch.Tensor],
) -> None:
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_cpu_rng_capture_and_restore_replays_every_global_stream() -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    state = capture_training_rng_state("cpu")
    expected = (
        [random.random() for _ in range(4)],
        np.random.random(4),
        torch.rand(4),
    )

    for _ in range(8):
        random.random()
        np.random.random()
        torch.rand(())
    restore_training_rng_state(state, device="cpu")
    actual = (
        [random.random() for _ in range(4)],
        np.random.random(4),
        torch.rand(4),
    )

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_cpu_capture_and_restore_never_queries_cuda() -> None:
    backend = FakeCudaRNGBackend()

    state = capture_training_rng_state("cpu", cuda_backend=backend)
    restore_training_rng_state(state, device="cpu", cuda_backend=backend)

    assert state.backend == "cpu"
    assert state.cuda_states == ()
    assert backend.calls == []


def test_cuda_device_count_mismatch_fails_before_mutating_any_stream() -> None:
    capture_backend = FakeCudaRNGBackend(device_count=2)
    state = capture_training_rng_state("cuda:0", cuda_backend=capture_backend)
    restore_backend = FakeCudaRNGBackend(device_count=1)
    before = _global_cpu_state()

    with pytest.raises(RNGStateError, match=r"saved 2 CUDA.*runtime exposes 1"):
        restore_training_rng_state(
            state,
            device="cuda:0",
            cuda_backend=restore_backend,
        )

    _assert_global_cpu_state_equal(_global_cpu_state(), before)
    assert "set_rng_state_all" not in restore_backend.calls


def test_malformed_serialized_state_fails_without_global_rng_mutation() -> None:
    state = capture_training_rng_state("cpu")
    malformed = deepcopy(state.to_dict())
    malformed["torch_cpu_state"] = []
    before = _global_cpu_state()

    with pytest.raises(RNGStateError, match="torch_cpu_state"):
        TrainingRNGState.from_dict(malformed)

    _assert_global_cpu_state_equal(_global_cpu_state(), before)


def test_loader_and_rng_restore_roll_back_as_one_transaction() -> None:
    class PartiallyMutatingLoader:
        def __init__(self) -> None:
            self.state = {"format": "fake_loader", "position": 0}

        def state_dict(self) -> dict[str, object]:
            return dict(self.state)

        def load_state_dict(self, state: dict[str, object]) -> None:
            self.state = dict(state)
            if state["position"] == 99:
                raise ValueError("injected state failure")

    loader = PartiallyMutatingLoader()
    continuation = ExactTrainingState(
        loader_format="fake_loader",
        loader_state={"format": "fake_loader", "position": 99},
        rng_state=capture_training_rng_state("cpu"),
        tracker_step=0,
        total_training_time_seconds=0.0,
        total_training_flops=0.0,
    )
    before_rng = _global_cpu_state()

    with pytest.raises(
        pretraining.PretrainingError,
        match="could not restore exact training continuation",
    ):
        pretraining._restore_exact_continuation(
            loader,
            continuation,
            device=torch.device("cpu"),
        )

    assert loader.state == {"format": "fake_loader", "position": 0}
    _assert_global_cpu_state_equal(_global_cpu_state(), before_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_restores_each_available_device_stream() -> None:
    state = capture_training_rng_state("cuda")
    expected = [
        torch.rand(4, device=f"cuda:{index}") for index in range(len(state.cuda_states))
    ]

    for index in range(len(state.cuda_states)):
        torch.rand(8, device=f"cuda:{index}")
    restore_training_rng_state(state, device="cuda")
    actual = [
        torch.rand(4, device=f"cuda:{index}") for index in range(len(state.cuda_states))
    ]

    for actual_values, expected_values in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
