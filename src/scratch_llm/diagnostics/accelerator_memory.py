"""Side-effect-free accelerator memory statistics.

Collection and peak reset are deliberately separate operations.  The public
helpers do not synchronize devices, estimate future allocations, or emit
telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from scratch_llm._validation import (
    require_integer,
    require_non_negative_integer,
    require_positive_integer,
)


_BYTES_PER_MIB = 1024**2


class AcceleratorMemoryError(RuntimeError):
    """Raised when an accelerator backend returns an invalid result or fails."""


class CudaMemoryBackend(Protocol):
    """Narrow CUDA API used by memory collection and peak reset."""

    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def current_device(self) -> int: ...

    def memory_allocated(self, device: torch.device) -> int: ...

    def memory_reserved(self, device: torch.device) -> int: ...

    def max_memory_allocated(self, device: torch.device) -> int: ...

    def max_memory_reserved(self, device: torch.device) -> int: ...

    def device_capacity(self, device: torch.device) -> int | None: ...

    def reset_peak_memory_stats(self, device: torch.device) -> None: ...


class _TorchCudaMemoryBackend:
    """Adapter around the subset of ``torch.cuda`` used by this module."""

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def memory_allocated(self, device: torch.device) -> int:
        return torch.cuda.memory_allocated(device)

    def memory_reserved(self, device: torch.device) -> int:
        return torch.cuda.memory_reserved(device)

    def max_memory_allocated(self, device: torch.device) -> int:
        return torch.cuda.max_memory_allocated(device)

    def max_memory_reserved(self, device: torch.device) -> int:
        return torch.cuda.max_memory_reserved(device)

    def device_capacity(self, device: torch.device) -> int:
        return torch.cuda.get_device_properties(device).total_memory

    def reset_peak_memory_stats(self, device: torch.device) -> None:
        torch.cuda.reset_peak_memory_stats(device)


@dataclass(frozen=True)
class AcceleratorMemorySnapshot:
    """Immutable memory counters for one resolved device.

    Byte fields are authoritative.  MiB properties are derived so serialized
    or tracked consumers cannot observe disagreeing unit conversions.
    """

    device: torch.device
    available: bool
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    capacity_bytes: int | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, torch.device):
            raise TypeError(
                f"device must be a torch.device, got {type(self.device).__name__}"
            )
        if not isinstance(self.available, bool):
            raise TypeError(
                f"available must be a boolean, got {type(self.available).__name__}"
            )

        fields = {
            "allocated bytes": self.allocated_bytes,
            "reserved bytes": self.reserved_bytes,
            "peak allocated bytes": self.peak_allocated_bytes,
            "peak reserved bytes": self.peak_reserved_bytes,
            "capacity bytes": self.capacity_bytes,
        }
        if not self.available:
            if not isinstance(self.unavailable_reason, str) or not (
                self.unavailable_reason.strip()
            ):
                raise ValueError(
                    "unavailable_reason must explain an unavailable snapshot"
                )
            present = [name for name, value in fields.items() if value is not None]
            if present:
                raise ValueError(
                    "memory counters must be absent for an unavailable snapshot; "
                    f"present={present}"
                )
            return

        if self.unavailable_reason is not None:
            raise ValueError(
                "unavailable_reason must be absent for an available snapshot"
            )
        for name in (
            "allocated bytes",
            "reserved bytes",
            "peak allocated bytes",
            "peak reserved bytes",
        ):
            _require_byte_count(fields[name], label=name)
        if self.capacity_bytes is not None:
            _require_byte_count(
                self.capacity_bytes,
                label="capacity bytes",
                positive=True,
            )
        _validate_counter_relationships(
            allocated=self.allocated_bytes,
            reserved=self.reserved_bytes,
            peak_allocated=self.peak_allocated_bytes,
            peak_reserved=self.peak_reserved_bytes,
            capacity=self.capacity_bytes,
        )

    @property
    def allocated_mib(self) -> float | None:
        """Current allocated memory in mebibytes."""

        return _to_mib(self.allocated_bytes)

    @property
    def reserved_mib(self) -> float | None:
        """Current reserved memory in mebibytes."""

        return _to_mib(self.reserved_bytes)

    @property
    def peak_allocated_mib(self) -> float | None:
        """Peak allocated memory in mebibytes."""

        return _to_mib(self.peak_allocated_bytes)

    @property
    def peak_reserved_mib(self) -> float | None:
        """Peak reserved memory in mebibytes."""

        return _to_mib(self.peak_reserved_bytes)

    @property
    def capacity_mib(self) -> float | None:
        """Device capacity in mebibytes when exposed by the backend."""

        return _to_mib(self.capacity_bytes)


def collect_accelerator_memory(
    device: str | torch.device,
    *,
    backend: CudaMemoryBackend | None = None,
) -> AcceleratorMemorySnapshot:
    """Collect current and peak memory counters without mutating backend state.

    CPU, MPS, and unsupported accelerator types return an explicit unavailable
    snapshot without consulting CUDA.  An unavailable CUDA runtime is handled
    the same way after the single availability check.
    """

    requested = _coerce_device(device)
    if requested.type != "cuda":
        return _unavailable(
            requested,
            f"memory statistics are unavailable for device type {requested.type!r}",
        )

    cuda = _TorchCudaMemoryBackend() if backend is None else backend
    if not _cuda_is_available(cuda, requested):
        return _unavailable(requested, "CUDA is not available")

    resolved = _resolve_cuda_device(requested, cuda)
    try:
        allocated = _backend_byte_count(
            cuda.memory_allocated(resolved),
            label="allocated bytes",
        )
        reserved = _backend_byte_count(
            cuda.memory_reserved(resolved),
            label="reserved bytes",
        )
        peak_allocated = _backend_byte_count(
            cuda.max_memory_allocated(resolved),
            label="peak allocated bytes",
        )
        peak_reserved = _backend_byte_count(
            cuda.max_memory_reserved(resolved),
            label="peak reserved bytes",
        )
        capacity_value = cuda.device_capacity(resolved)
        capacity = (
            None
            if capacity_value is None
            else _backend_byte_count(
                capacity_value,
                label="capacity bytes",
                positive=True,
            )
        )
        _validate_counter_relationships(
            allocated=allocated,
            reserved=reserved,
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
            capacity=capacity,
            error_type=AcceleratorMemoryError,
        )
    except AcceleratorMemoryError:
        raise
    except Exception as error:
        raise AcceleratorMemoryError(
            f"failed to collect accelerator memory for {resolved}: {error}"
        ) from error

    return AcceleratorMemorySnapshot(
        device=resolved,
        available=True,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        capacity_bytes=capacity,
    )


def reset_accelerator_memory_peak(
    device: str | torch.device,
    *,
    backend: CudaMemoryBackend | None = None,
) -> bool:
    """Reset peak counters for one CUDA device and report whether it occurred.

    ``False`` means the requested device type or CUDA runtime does not expose
    resettable counters.  No CUDA API is touched for non-CUDA devices.
    """

    requested = _coerce_device(device)
    if requested.type != "cuda":
        return False

    cuda = _TorchCudaMemoryBackend() if backend is None else backend
    if not _cuda_is_available(cuda, requested):
        return False

    resolved = _resolve_cuda_device(requested, cuda)
    try:
        cuda.reset_peak_memory_stats(resolved)
    except Exception as error:
        raise AcceleratorMemoryError(
            f"failed to reset peak accelerator memory for {resolved}: {error}"
        ) from error
    return True


def _coerce_device(device: str | torch.device) -> torch.device:
    if not isinstance(device, (str, torch.device)):
        raise TypeError(
            f"device must be a string or torch.device, got {type(device).__name__}"
        )
    try:
        return torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device request {device!r}: {error}") from error


def _cuda_is_available(
    backend: CudaMemoryBackend,
    device: torch.device,
) -> bool:
    try:
        available = backend.is_available()
    except Exception as error:
        raise AcceleratorMemoryError(
            f"failed to check CUDA availability for {device}: {error}"
        ) from error
    if not isinstance(available, bool):
        raise AcceleratorMemoryError(
            f"CUDA availability must be a boolean, got {type(available).__name__}"
        )
    return available


def _resolve_cuda_device(
    requested: torch.device,
    backend: CudaMemoryBackend,
) -> torch.device:
    try:
        count = backend.device_count()
        try:
            count = require_positive_integer(count, name="CUDA device count")
        except (TypeError, ValueError) as error:
            raise AcceleratorMemoryError(
                f"CUDA device count must be a positive integer, got {count!r}"
            ) from error
        index = requested.index
        if index is None:
            index = backend.current_device()
            try:
                index = require_integer(index, name="current CUDA device index")
            except TypeError as error:
                raise AcceleratorMemoryError(
                    "current CUDA device index must be an integer, "
                    f"got {type(index).__name__}"
                ) from error
    except AcceleratorMemoryError:
        raise
    except Exception as error:
        raise AcceleratorMemoryError(
            f"failed to resolve CUDA device {requested}: {error}"
        ) from error

    if index < 0 or index >= count:
        noun = "device" if count == 1 else "devices"
        raise ValueError(
            f"CUDA device index {index} is unavailable; found {count} {noun}"
        )
    return torch.device("cuda", index)


def _backend_byte_count(
    value: object,
    *,
    label: str,
    positive: bool = False,
) -> int:
    try:
        return _require_byte_count(value, label=label, positive=positive)
    except (TypeError, ValueError) as error:
        raise AcceleratorMemoryError(str(error)) from error


def _require_byte_count(
    value: object,
    *,
    label: str,
    positive: bool = False,
) -> int:
    if positive:
        return require_positive_integer(value, name=label)
    return require_non_negative_integer(value, name=label)


def _validate_counter_relationships(
    *,
    allocated: int | None,
    reserved: int | None,
    peak_allocated: int | None,
    peak_reserved: int | None,
    capacity: int | None,
    error_type: type[Exception] = ValueError,
) -> None:
    assert allocated is not None
    assert reserved is not None
    assert peak_allocated is not None
    assert peak_reserved is not None
    if reserved < allocated:
        raise error_type(
            "reserved bytes must be greater than or equal to allocated bytes"
        )
    if peak_allocated < allocated:
        raise error_type(
            "peak allocated bytes must be greater than or equal to "
            "current allocated bytes"
        )
    if peak_reserved < reserved:
        raise error_type(
            "peak reserved bytes must be greater than or equal to "
            "current reserved bytes"
        )
    if peak_reserved < peak_allocated:
        raise error_type(
            "peak reserved bytes must be greater than or equal to peak allocated bytes"
        )
    if capacity is not None and peak_reserved > capacity:
        raise error_type(
            "capacity bytes must be greater than or equal to peak reserved bytes"
        )


def _unavailable(
    device: torch.device,
    reason: str,
) -> AcceleratorMemorySnapshot:
    return AcceleratorMemorySnapshot(
        device=device,
        available=False,
        unavailable_reason=reason,
    )


def _to_mib(value: int | None) -> float | None:
    return None if value is None else value / _BYTES_PER_MIB


__all__ = [
    "AcceleratorMemoryError",
    "AcceleratorMemorySnapshot",
    "CudaMemoryBackend",
    "collect_accelerator_memory",
    "reset_accelerator_memory_peak",
]
