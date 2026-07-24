"""Tests for deterministic common runtime utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scratch_llm.utils import (
    Timer,
    atomic_write,
    autodetect_device_type,
    count_parameters,
    format_bytes,
    format_num,
    get_device,
    load_json,
    save_json,
    set_seed,
    timer,
)


def test_set_seed_repeats_python_numpy_and_torch_random_streams() -> None:
    set_seed(1337)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    set_seed(1337)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_set_seed_reaches_all_cuda_rngs_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_seeds: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)

    set_seed(17)

    assert cuda_seeds
    assert set(cuda_seeds) == {17}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_set_seed_repeats_each_available_cuda_random_stream() -> None:
    set_seed(29)
    first = [
        torch.rand(4, device=f"cuda:{index}").cpu()
        for index in range(torch.cuda.device_count())
    ]

    set_seed(29)
    second = [
        torch.rand(4, device=f"cuda:{index}").cpu()
        for index in range(torch.cuda.device_count())
    ]

    assert all(torch.equal(left, right) for left, right in zip(first, second))


@pytest.mark.parametrize("seed", [True, -1, 2**32])
def test_set_seed_rejects_values_outside_the_shared_rng_range(seed: object) -> None:
    error_type = TypeError if isinstance(seed, bool) else ValueError

    with pytest.raises(error_type, match="seed"):
        set_seed(seed)  # type: ignore[arg-type]


def test_device_autodetection_falls_back_to_cpu_without_an_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert autodetect_device_type() == "cpu"
    assert get_device() == torch.device("cpu")
    assert get_device("auto") == torch.device("cpu")


def test_device_autodetection_prefers_cuda_then_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert autodetect_device_type() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert autodetect_device_type() == "mps"


@pytest.mark.parametrize("device_request", ["cuda", "cuda:0"])
def test_get_device_rejects_an_unavailable_explicit_cuda_request(
    monkeypatch: pytest.MonkeyPatch, device_request: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        get_device(device_request)


def test_get_device_validates_an_explicit_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert get_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match=r"CUDA device index 1.*1 device"):
        get_device("cuda:1")


def test_count_parameters_deduplicates_tied_weights_and_filters_trainable() -> None:
    model = nn.Module()
    model.embedding = nn.Embedding(5, 3)
    model.output = nn.Linear(3, 5, bias=False)
    model.output.weight = model.embedding.weight
    model.frozen = nn.Parameter(torch.ones(2), requires_grad=False)

    assert count_parameters(model) == 17
    assert count_parameters(model, trainable_only=True) == 15
    assert count_parameters(nn.Module()) == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1.00K"),
        (1_250, "1.25K"),
        (1_000_000, "1.00M"),
        (-2_500_000_000, "-2.50B"),
        (1_000_000_000_000, "1.00T"),
    ],
)
def test_format_num_uses_stable_decimal_boundaries(value: int, expected: str) -> None:
    assert format_num(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.00 KiB"),
        (1536, "1.50 KiB"),
        (1024**2, "1.00 MiB"),
        (1024**3, "1.00 GiB"),
    ],
)
def test_format_bytes_uses_stable_binary_boundaries(value: int, expected: str) -> None:
    assert format_bytes(value) == expected


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
def test_format_num_rejects_non_finite_or_boolean_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="value"):
        format_num(value)  # type: ignore[arg-type]


def test_format_bytes_rejects_a_negative_byte_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_bytes(-1)


def test_json_helpers_round_trip_canonical_utf8_text(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "metrics.json"
    payload = {"z": 3, "a": {"unicode": "café", "enabled": True}}

    result = save_json(payload, destination)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "enabled": true,\n    "unicode": "café"\n  },\n  "z": 3\n}\n'
    )
    assert load_json(destination) == payload


def test_atomic_write_replaces_existing_text_without_leaving_a_temp_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "state.txt"
    destination.write_text("old", encoding="utf-8")

    result = atomic_write(destination, "new\n")

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_preserves_destination_and_cleans_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "state.txt"
    destination.write_text("stable", encoding="utf-8")

    def fail_replace(source: object, target: object) -> None:
        raise OSError(f"cannot replace {source} with {target}")

    monkeypatch.setattr("scratch_llm.utils.os.replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        atomic_write(destination, "partial")

    assert destination.read_text(encoding="utf-8") == "stable"
    assert list(tmp_path.iterdir()) == [destination]


def test_save_json_serialization_failure_does_not_touch_existing_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "metrics.json"
    destination.write_text("stable", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        save_json({"loss": float("nan")}, destination)

    assert destination.read_text(encoding="utf-8") == "stable"
    assert list(tmp_path.iterdir()) == [destination]


def test_timer_uses_the_monotonic_clock_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([10.0, 10.25, 11.0])
    monkeypatch.setattr("scratch_llm.utils.time.monotonic", lambda: next(ticks))
    measurement = Timer()

    assert measurement.elapsed == 0.0
    measurement.start()
    assert measurement.running is True
    assert measurement.elapsed == pytest.approx(0.25)
    assert measurement.stop() == pytest.approx(1.0)
    assert measurement.running is False
    assert measurement.elapsed == pytest.approx(1.0)


def test_timer_context_accumulates_intervals_from_an_injected_clock() -> None:
    ticks = iter([2.0, 2.5, 4.0, 5.25])

    def clock() -> float:
        return next(ticks)

    measurement = Timer(clock=clock)

    with measurement:
        assert measurement.running is True
    measurement.start()
    assert measurement.stop() == pytest.approx(1.75)


def test_timer_context_helper_stops_after_an_exception() -> None:
    ticks = iter([7.0, 7.125])

    with pytest.raises(RuntimeError, match="boom"):
        with timer(clock=lambda: next(ticks)) as measurement:
            raise RuntimeError("boom")

    assert measurement.running is False
    assert measurement.elapsed == pytest.approx(0.125)


def test_timer_rejects_invalid_state_transitions() -> None:
    measurement = Timer(clock=lambda: 1.0)

    with pytest.raises(RuntimeError, match="not running"):
        measurement.stop()

    measurement.start()
    with pytest.raises(RuntimeError, match="already running"):
        measurement.start()
