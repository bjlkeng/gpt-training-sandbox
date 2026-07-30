"""Validated, transactional global RNG capture and restoration."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, cast

import numpy as np
import torch


class RNGStateError(RuntimeError):
    """A saved RNG state is malformed or incompatible with this runtime."""


class CudaRNGBackend(Protocol):
    """Narrow CUDA RNG API used for deterministic fake coverage."""

    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def get_rng_state_all(self) -> list[torch.Tensor]: ...

    def set_rng_state_all(self, states: list[torch.Tensor]) -> None: ...


class _TorchCudaRNGBackend:
    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def get_rng_state_all(self) -> list[torch.Tensor]:
        return torch.cuda.get_rng_state_all()

    def set_rng_state_all(self, states: list[torch.Tensor]) -> None:
        torch.cuda.set_rng_state_all(states)


_SERIALIZED_KEYS = frozenset(
    {
        "backend",
        "cuda_states",
        "numpy_algorithm",
        "numpy_cached_gaussian",
        "numpy_has_gauss",
        "numpy_keys",
        "numpy_position",
        "python_gauss",
        "python_internal",
        "python_version",
        "torch_cpu_state",
    }
)


@dataclass(frozen=True)
class TrainingRNGState:
    """Immutable JSON-compatible state for every configured stochastic stream."""

    backend: Literal["cpu", "cuda"]
    python_version: int
    python_internal: tuple[int, ...]
    python_gauss: float | None
    numpy_algorithm: str
    numpy_keys: tuple[int, ...]
    numpy_position: int
    numpy_has_gauss: int
    numpy_cached_gaussian: float
    torch_cpu_state: tuple[int, ...]
    cuda_states: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.backend not in ("cpu", "cuda"):
            raise RNGStateError(f"unsupported RNG backend {self.backend!r}")
        if self.backend == "cpu" and self.cuda_states:
            raise RNGStateError("CPU RNG state must not contain CUDA streams")
        if self.backend == "cuda" and not self.cuda_states:
            raise RNGStateError(
                "CUDA RNG state must contain at least one device stream"
            )
        _validate_python_state(self)
        _validate_numpy_state(self)
        _validate_torch_state(self.torch_cpu_state, label="torch_cpu_state")
        for index, state in enumerate(self.cuda_states):
            _validate_byte_values(state, label=f"cuda_states[{index}]")

    def to_dict(self) -> dict[str, object]:
        """Return a deep JSON-compatible representation."""

        return {
            "backend": self.backend,
            "cuda_states": [list(state) for state in self.cuda_states],
            "numpy_algorithm": self.numpy_algorithm,
            "numpy_cached_gaussian": self.numpy_cached_gaussian,
            "numpy_has_gauss": self.numpy_has_gauss,
            "numpy_keys": list(self.numpy_keys),
            "numpy_position": self.numpy_position,
            "python_gauss": self.python_gauss,
            "python_internal": list(self.python_internal),
            "python_version": self.python_version,
            "torch_cpu_state": list(self.torch_cpu_state),
        }

    @classmethod
    def from_dict(cls, value: object) -> TrainingRNGState:
        """Validate and reconstruct serialized state without changing globals."""

        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise RNGStateError("RNG state must be an object with string keys")
        if set(value) != _SERIALIZED_KEYS:
            missing = sorted(_SERIALIZED_KEYS - set(value))
            unexpected = sorted(set(value) - _SERIALIZED_KEYS)
            raise RNGStateError(
                "RNG state fields do not match format version 1; "
                f"missing={missing}, unexpected={unexpected}"
            )
        backend = value["backend"]
        if backend not in ("cpu", "cuda"):
            raise RNGStateError("RNG state backend must be 'cpu' or 'cuda'")
        python_gauss_value = value["python_gauss"]
        if python_gauss_value is not None and not _is_real(python_gauss_value):
            raise RNGStateError("RNG state python_gauss must be a number or null")
        numpy_algorithm = value["numpy_algorithm"]
        if not isinstance(numpy_algorithm, str) or not numpy_algorithm:
            raise RNGStateError("RNG state numpy_algorithm must be a string")
        numpy_cached = value["numpy_cached_gaussian"]
        if not _is_real(numpy_cached):
            raise RNGStateError(
                "RNG state numpy_cached_gaussian must be a finite number"
            )
        return cls(
            backend=backend,
            python_version=_require_integer(
                value["python_version"],
                label="python_version",
                minimum=0,
            ),
            python_internal=_integer_tuple(
                value["python_internal"],
                label="python_internal",
                minimum=0,
            ),
            python_gauss=(
                None if python_gauss_value is None else float(python_gauss_value)
            ),
            numpy_algorithm=numpy_algorithm,
            numpy_keys=_integer_tuple(
                value["numpy_keys"],
                label="numpy_keys",
                minimum=0,
                maximum=2**32 - 1,
            ),
            numpy_position=_require_integer(
                value["numpy_position"],
                label="numpy_position",
                minimum=0,
            ),
            numpy_has_gauss=_require_integer(
                value["numpy_has_gauss"],
                label="numpy_has_gauss",
                minimum=0,
                maximum=1,
            ),
            numpy_cached_gaussian=float(numpy_cached),
            torch_cpu_state=_byte_tuple(
                value["torch_cpu_state"],
                label="torch_cpu_state",
            ),
            cuda_states=_nested_byte_tuple(
                value["cuda_states"],
                label="cuda_states",
            ),
        )


def capture_training_rng_state(
    device: str | torch.device,
    *,
    cuda_backend: CudaRNGBackend | None = None,
) -> TrainingRNGState:
    """Capture configured streams without touching CUDA on a CPU path."""

    requested = _coerce_device(device)
    python_state = random.getstate()
    numpy_state = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    torch_cpu_state = _tensor_bytes(torch.get_rng_state(), label="torch CPU RNG")
    cuda_states: tuple[tuple[int, ...], ...] = ()
    backend_name: Literal["cpu", "cuda"] = "cpu"
    if requested.type == "cuda":
        cuda = _TorchCudaRNGBackend() if cuda_backend is None else cuda_backend
        count = _validate_cuda_runtime(requested, cuda)
        try:
            raw_cuda_states = cuda.get_rng_state_all()
        except Exception as error:
            raise RNGStateError(
                f"could not capture CUDA RNG states: {error}"
            ) from error
        if not isinstance(raw_cuda_states, list) or len(raw_cuda_states) != count:
            raise RNGStateError(
                f"CUDA runtime reported {count} devices but returned "
                f"{len(raw_cuda_states) if isinstance(raw_cuda_states, list) else 'invalid'} "
                "RNG states"
            )
        cuda_states = tuple(
            _tensor_bytes(state, label=f"CUDA RNG state {index}")
            for index, state in enumerate(raw_cuda_states)
        )
        backend_name = "cuda"
    elif requested.type != "cpu":
        raise RNGStateError(
            f"exact RNG continuation does not support device type {requested.type!r}"
        )

    python_internal = python_state[1]
    return TrainingRNGState(
        backend=backend_name,
        python_version=int(python_state[0]),
        python_internal=tuple(int(value) for value in python_internal),
        python_gauss=(None if python_state[2] is None else float(python_state[2])),
        numpy_algorithm=str(numpy_state[0]),
        numpy_keys=tuple(int(value) for value in numpy_state[1].tolist()),
        numpy_position=int(numpy_state[2]),
        numpy_has_gauss=int(numpy_state[3]),
        numpy_cached_gaussian=float(numpy_state[4]),
        torch_cpu_state=torch_cpu_state,
        cuda_states=cuda_states,
    )


def restore_training_rng_state(
    state: TrainingRNGState,
    *,
    device: str | torch.device,
    cuda_backend: CudaRNGBackend | None = None,
) -> None:
    """Transactionally install saved streams for a compatible runtime."""

    if not isinstance(state, TrainingRNGState):
        raise TypeError(f"state must be a TrainingRNGState, got {type(state).__name__}")
    requested = _coerce_device(device)
    if requested.type != state.backend:
        raise RNGStateError(
            f"saved RNG backend {state.backend!r} is incompatible with "
            f"requested device type {requested.type!r}"
        )

    cuda: CudaRNGBackend | None = None
    prior_cuda: list[torch.Tensor] | None = None
    candidate_cuda: list[torch.Tensor] | None = None
    if state.backend == "cuda":
        cuda = _TorchCudaRNGBackend() if cuda_backend is None else cuda_backend
        count = _validate_cuda_runtime(requested, cuda)
        if count != len(state.cuda_states):
            raise RNGStateError(
                f"saved {len(state.cuda_states)} CUDA RNG streams but runtime "
                f"exposes {count} devices"
            )
        candidate_cuda = [_bytes_tensor(values) for values in state.cuda_states]
        try:
            prior_cuda = cuda.get_rng_state_all()
        except Exception as error:
            raise RNGStateError(
                f"could not capture CUDA RNG rollback state: {error}"
            ) from error
        if not isinstance(prior_cuda, list) or len(prior_cuda) != count:
            raise RNGStateError("CUDA rollback state count is inconsistent")

    prior_python = random.getstate()
    prior_numpy = cast(tuple[Any, ...], np.random.get_state(legacy=True))
    prior_torch = torch.get_rng_state().clone()
    try:
        _apply_cpu_state(state)
        if cuda is not None and candidate_cuda is not None:
            cuda.set_rng_state_all(candidate_cuda)
    except Exception as error:
        random.setstate(prior_python)
        np.random.set_state(prior_numpy)
        torch.set_rng_state(prior_torch)
        if cuda is not None and prior_cuda is not None:
            try:
                cuda.set_rng_state_all(prior_cuda)
            except Exception:
                pass
        if isinstance(error, RNGStateError):
            raise
        raise RNGStateError(f"could not restore RNG state: {error}") from error


@contextmanager
def preserve_global_rng_state(
    device: str | torch.device | None = None,
) -> Iterator[None]:
    """Preserve caller streams around model-only or inspection work."""

    preserved_device: str | torch.device
    if device is None:
        preserved_device = "cuda" if torch.cuda.is_initialized() else "cpu"
    else:
        preserved_device = device
    state = capture_training_rng_state(preserved_device)
    try:
        yield
    finally:
        restore_training_rng_state(state, device=preserved_device)


def _apply_cpu_state(state: TrainingRNGState) -> None:
    random.setstate(
        (
            state.python_version,
            tuple(state.python_internal),
            state.python_gauss,
        )
    )
    np.random.set_state(
        (
            state.numpy_algorithm,
            np.asarray(state.numpy_keys, dtype=np.uint32),
            state.numpy_position,
            state.numpy_has_gauss,
            state.numpy_cached_gaussian,
        )
    )
    torch.set_rng_state(_bytes_tensor(state.torch_cpu_state))


def _validate_python_state(state: TrainingRNGState) -> None:
    _validate_integer_values(state.python_internal, label="python_internal")
    if state.python_gauss is not None and not math.isfinite(state.python_gauss):
        raise RNGStateError("python_gauss must be finite")
    probe = random.Random()
    try:
        probe.setstate(
            (
                state.python_version,
                tuple(state.python_internal),
                state.python_gauss,
            )
        )
    except Exception as error:
        raise RNGStateError(f"invalid Python RNG state: {error}") from error


def _validate_numpy_state(state: TrainingRNGState) -> None:
    if not state.numpy_algorithm:
        raise RNGStateError("numpy_algorithm must be non-empty")
    if state.numpy_has_gauss not in (0, 1):
        raise RNGStateError("numpy_has_gauss must be 0 or 1")
    if not math.isfinite(state.numpy_cached_gaussian):
        raise RNGStateError("numpy_cached_gaussian must be finite")
    if any(not 0 <= value <= 2**32 - 1 for value in state.numpy_keys):
        raise RNGStateError("numpy_keys values must fit uint32")
    probe = np.random.RandomState()
    try:
        probe.set_state(
            (
                state.numpy_algorithm,
                np.asarray(state.numpy_keys, dtype=np.uint32),
                state.numpy_position,
                state.numpy_has_gauss,
                state.numpy_cached_gaussian,
            )
        )
    except Exception as error:
        raise RNGStateError(f"invalid NumPy RNG state: {error}") from error


def _validate_torch_state(values: tuple[int, ...], *, label: str) -> None:
    _validate_byte_values(values, label=label)
    probe = torch.Generator(device="cpu")
    try:
        probe.set_state(_bytes_tensor(values))
    except Exception as error:
        raise RNGStateError(f"invalid {label}: {error}") from error


def _validate_byte_values(values: tuple[int, ...], *, label: str) -> None:
    if not values:
        raise RNGStateError(f"{label} must be a non-empty byte list")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255
        for value in values
    ):
        raise RNGStateError(f"{label} must contain only integer bytes")


def _validate_integer_values(values: tuple[int, ...], *, label: str) -> None:
    if not values or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise RNGStateError(f"{label} must contain integer values")


def _validate_cuda_runtime(
    requested: torch.device,
    backend: CudaRNGBackend,
) -> int:
    try:
        available = backend.is_available()
        count = backend.device_count() if available is True else 0
    except Exception as error:
        raise RNGStateError(f"could not inspect CUDA RNG runtime: {error}") from error
    if available is not True:
        raise RNGStateError("CUDA is unavailable for exact RNG continuation")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RNGStateError(f"CUDA device count must be positive, got {count!r}")
    if requested.index is not None and not 0 <= requested.index < count:
        raise RNGStateError(
            f"requested CUDA device index {requested.index} is unavailable; "
            f"runtime exposes {count}"
        )
    return count


def _coerce_device(device: str | torch.device) -> torch.device:
    try:
        return torch.device(device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise RNGStateError(f"invalid RNG device {device!r}: {error}") from error


def _tensor_bytes(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise RNGStateError(f"{label} must be a Tensor")
    if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
        raise RNGStateError(f"{label} must be a one-dimensional CPU uint8 Tensor")
    return tuple(int(item) for item in value.tolist())


def _bytes_tensor(values: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.uint8, device="cpu")


def _integer_tuple(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RNGStateError(f"RNG state {label} must be a non-empty list")
    return tuple(
        _require_integer(
            item,
            label=f"{label}[{index}]",
            minimum=minimum,
            maximum=maximum,
        )
        for index, item in enumerate(value)
    )


def _byte_tuple(value: object, *, label: str) -> tuple[int, ...]:
    return _integer_tuple(value, label=label, minimum=0, maximum=255)


def _nested_byte_tuple(value: object, *, label: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise RNGStateError(f"RNG state {label} must be a list")
    return tuple(
        _byte_tuple(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )


def _require_integer(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RNGStateError(f"RNG state {label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise RNGStateError(f"RNG state {label} is outside its valid range")
    return value


def _is_real(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = [
    "CudaRNGBackend",
    "RNGStateError",
    "TrainingRNGState",
    "capture_training_rng_state",
    "preserve_global_rng_state",
    "restore_training_rng_state",
]
