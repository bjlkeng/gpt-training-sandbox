"""Tests for side-effect-free accelerator memory statistics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from scratch_llm.diagnostics.accelerator_memory import (
    AcceleratorMemoryError,
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
    reset_accelerator_memory_peak,
)


class FakeCudaMemoryBackend:
    """Deterministic stand-in for the narrow CUDA memory API boundary."""

    def __init__(
        self,
        *,
        available: bool = True,
        device_count: int = 2,
        current_device: int = 0,
    ) -> None:
        self.available = available
        self.count = device_count
        self.current = current_device
        self.calls: list[tuple[str, object | None]] = []
        self.values: dict[int, dict[str, int | None]] = {
            0: {
                "allocated": 2 * 1024**2,
                "reserved": 3 * 1024**2,
                "peak_allocated": 4 * 1024**2,
                "peak_reserved": 5 * 1024**2,
                "capacity": 24 * 1024**3,
            },
            1: {
                "allocated": 7 * 1024**2,
                "reserved": 8 * 1024**2,
                "peak_allocated": 9 * 1024**2,
                "peak_reserved": 10 * 1024**2,
                "capacity": 12 * 1024**3,
            },
        }

    def is_available(self) -> bool:
        self.calls.append(("is_available", None))
        return self.available

    def device_count(self) -> int:
        self.calls.append(("device_count", None))
        return self.count

    def current_device(self) -> int:
        self.calls.append(("current_device", None))
        return self.current

    def memory_allocated(self, device: torch.device) -> int:
        self.calls.append(("memory_allocated", device))
        return self._value(device, "allocated")

    def memory_reserved(self, device: torch.device) -> int:
        self.calls.append(("memory_reserved", device))
        return self._value(device, "reserved")

    def max_memory_allocated(self, device: torch.device) -> int:
        self.calls.append(("max_memory_allocated", device))
        return self._value(device, "peak_allocated")

    def max_memory_reserved(self, device: torch.device) -> int:
        self.calls.append(("max_memory_reserved", device))
        return self._value(device, "peak_reserved")

    def device_capacity(self, device: torch.device) -> int | None:
        self.calls.append(("device_capacity", device))
        index = self._index(device)
        value = self.values[index]["capacity"]
        assert value is None or isinstance(value, int)
        return value

    def reset_peak_memory_stats(self, device: torch.device) -> None:
        self.calls.append(("reset_peak_memory_stats", device))

    def _value(self, device: torch.device, name: str) -> int:
        value = self.values[self._index(device)][name]
        return value  # type: ignore[return-value]

    @staticmethod
    def _index(device: torch.device) -> int:
        assert device.index is not None
        return device.index


def test_cuda_snapshot_reports_one_resolved_device_and_derived_mib() -> None:
    backend = FakeCudaMemoryBackend()

    snapshot = collect_accelerator_memory("cuda:1", backend=backend)

    assert snapshot == AcceleratorMemorySnapshot(
        device=torch.device("cuda:1"),
        available=True,
        allocated_bytes=7 * 1024**2,
        reserved_bytes=8 * 1024**2,
        peak_allocated_bytes=9 * 1024**2,
        peak_reserved_bytes=10 * 1024**2,
        capacity_bytes=12 * 1024**3,
    )
    assert snapshot.allocated_mib == 7.0
    assert snapshot.reserved_mib == 8.0
    assert snapshot.peak_allocated_mib == 9.0
    assert snapshot.peak_reserved_mib == 10.0
    assert snapshot.capacity_mib == 12 * 1024
    assert ("current_device", None) not in backend.calls
    assert not any(name == "synchronize" for name, _ in backend.calls)
    assert not any(name == "reset_peak_memory_stats" for name, _ in backend.calls)

    with pytest.raises(FrozenInstanceError):
        snapshot.allocated_bytes = 0  # type: ignore[misc]


def test_bare_cuda_resolves_the_current_device_and_capacity_may_be_unknown() -> None:
    backend = FakeCudaMemoryBackend(current_device=1)
    backend.values[1]["capacity"] = None

    snapshot = collect_accelerator_memory(torch.device("cuda"), backend=backend)

    assert snapshot.device == torch.device("cuda:1")
    assert snapshot.capacity_bytes is None
    assert snapshot.capacity_mib is None
    assert backend.calls[:3] == [
        ("is_available", None),
        ("device_count", None),
        ("current_device", None),
    ]


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu:0"])
def test_non_cuda_devices_are_unavailable_without_touching_the_backend(
    device: str,
) -> None:
    backend = FakeCudaMemoryBackend()

    snapshot = collect_accelerator_memory(device, backend=backend)

    assert snapshot.device == torch.device(device)
    assert snapshot.available is False
    assert snapshot.unavailable_reason == (
        f"memory statistics are unavailable for device type {snapshot.device.type!r}"
    )
    assert snapshot.allocated_bytes is None
    assert snapshot.reserved_bytes is None
    assert snapshot.peak_allocated_bytes is None
    assert snapshot.peak_reserved_bytes is None
    assert snapshot.capacity_bytes is None
    assert snapshot.allocated_mib is None
    assert backend.calls == []
    assert reset_accelerator_memory_peak(device, backend=backend) is False
    assert backend.calls == []


def test_cuda_unavailable_does_not_query_or_reset_cuda_state() -> None:
    backend = FakeCudaMemoryBackend(available=False)

    snapshot = collect_accelerator_memory("cuda:1", backend=backend)

    assert snapshot.available is False
    assert snapshot.unavailable_reason == "CUDA is not available"
    assert backend.calls == [("is_available", None)]

    backend.calls.clear()
    assert reset_accelerator_memory_peak("cuda:1", backend=backend) is False
    assert backend.calls == [("is_available", None)]


def test_cuda_device_index_is_validated_before_reading_memory() -> None:
    backend = FakeCudaMemoryBackend(device_count=2)

    with pytest.raises(ValueError, match=r"CUDA device index 2.*2 devices"):
        collect_accelerator_memory("cuda:2", backend=backend)

    assert backend.calls == [
        ("is_available", None),
        ("device_count", None),
    ]


def test_reset_is_explicit_and_targets_only_the_resolved_device() -> None:
    backend = FakeCudaMemoryBackend(current_device=1)

    assert reset_accelerator_memory_peak("cuda", backend=backend) is True

    assert backend.calls == [
        ("is_available", None),
        ("device_count", None),
        ("current_device", None),
        ("reset_peak_memory_stats", torch.device("cuda:1")),
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("allocated", True, "allocated bytes must be an integer"),
        ("reserved", -1, "reserved bytes must be non-negative"),
        ("peak_allocated", 1.5, "peak allocated bytes must be an integer"),
        ("capacity", 0, "capacity bytes must be positive"),
    ],
)
def test_malformed_backend_values_fail_as_one_domain_error(
    field: str,
    value: Any,
    message: str,
) -> None:
    backend = FakeCudaMemoryBackend()
    backend.values[0][field] = value

    with pytest.raises(AcceleratorMemoryError, match=message):
        collect_accelerator_memory("cuda:0", backend=backend)


def test_inconsistent_memory_values_are_rejected() -> None:
    backend = FakeCudaMemoryBackend()
    backend.values[0]["peak_allocated"] = 1

    with pytest.raises(
        AcceleratorMemoryError,
        match="peak allocated bytes.*current allocated bytes",
    ):
        collect_accelerator_memory("cuda:0", backend=backend)


def test_backend_errors_include_the_device_and_operation() -> None:
    backend = FakeCudaMemoryBackend()

    def fail_reserved(device: torch.device) -> int:
        raise RuntimeError(f"driver failed for {device}")

    backend.memory_reserved = fail_reserved  # type: ignore[method-assign]

    with pytest.raises(
        AcceleratorMemoryError,
        match=r"collect accelerator memory for cuda:0.*driver failed",
    ):
        collect_accelerator_memory("cuda:0", backend=backend)


def test_snapshot_rejects_mixed_available_and_unavailable_states() -> None:
    with pytest.raises(ValueError, match="unavailable_reason"):
        AcceleratorMemorySnapshot(
            device=torch.device("cuda:0"),
            available=True,
            unavailable_reason="driver missing",
            allocated_bytes=1,
            reserved_bytes=1,
            peak_allocated_bytes=1,
            peak_reserved_bytes=1,
            capacity_bytes=2,
        )

    with pytest.raises(ValueError, match="must be absent"):
        AcceleratorMemorySnapshot(
            device=torch.device("cpu"),
            available=False,
            unavailable_reason="not CUDA",
            allocated_bytes=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_real_cuda_backend_satisfies_the_public_contract() -> None:
    device = torch.device("cuda", torch.cuda.current_device())

    snapshot = collect_accelerator_memory(device)

    assert snapshot.available is True
    assert snapshot.device == device
    assert snapshot.allocated_bytes is not None
    assert snapshot.reserved_bytes is not None
    assert snapshot.peak_allocated_bytes is not None
    assert snapshot.peak_reserved_bytes is not None
    assert snapshot.capacity_bytes is not None
