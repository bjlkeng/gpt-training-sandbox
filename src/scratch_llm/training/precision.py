"""One explicit autocast and loss-scaling policy for shared training code."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from scratch_llm.config import TrainDType


PrecisionDeviceType = Literal["cpu", "cuda", "mps"]
_SUPPORTED_DTYPES = frozenset({"float32", "float16", "bfloat16"})
_CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_KEYS = frozenset(
    {
        "device_type",
        "dtype",
        "format_version",
        "scaler_enabled",
        "scaler_state",
    }
)


class PrecisionError(RuntimeError):
    """A requested runtime precision policy is unsupported or inconsistent."""


class _GradScaler(Protocol):
    def is_enabled(self) -> bool: ...

    def scale(self, loss: Tensor) -> Tensor: ...

    def unscale_(self, optimizer: Optimizer) -> None: ...

    def step(self, optimizer: Optimizer) -> Any: ...

    def update(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class PrecisionCheckpointState:
    """Versioned exact-resume state for autocast and dynamic loss scaling."""

    dtype: TrainDType
    device_type: PrecisionDeviceType
    scaler_enabled: bool
    scaler_state: dict[str, Any] | None

    def __post_init__(self) -> None:
        _validate_descriptor(
            dtype=self.dtype,
            device_type=self.device_type,
            scaler_enabled=self.scaler_enabled,
        )
        if self.scaler_enabled:
            if not isinstance(self.scaler_state, dict):
                raise ValueError(
                    "enabled precision scaler requires a complete state dictionary"
                )
            if not all(isinstance(key, str) for key in self.scaler_state):
                raise ValueError("precision scaler-state keys must be strings")
            object.__setattr__(self, "scaler_state", deepcopy(self.scaler_state))
        elif self.scaler_state is not None:
            raise ValueError("disabled precision scaler must have null state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "dtype": self.dtype,
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "scaler_enabled": self.scaler_enabled,
            "scaler_state": deepcopy(self.scaler_state),
        }

    @classmethod
    def from_dict(cls, value: object) -> PrecisionCheckpointState:
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise PrecisionError("checkpoint precision state must be an object")
        if set(value) != _CHECKPOINT_KEYS:
            missing = sorted(_CHECKPOINT_KEYS - set(value))
            unexpected = sorted(set(value) - _CHECKPOINT_KEYS)
            raise PrecisionError(
                "checkpoint precision fields do not match format version 1; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if value["format_version"] != _CHECKPOINT_FORMAT_VERSION:
            raise PrecisionError(
                "unsupported precision checkpoint format version "
                f"{value['format_version']!r}; expected {_CHECKPOINT_FORMAT_VERSION}"
            )
        dtype = value["dtype"]
        device_type = value["device_type"]
        scaler_enabled = value["scaler_enabled"]
        if dtype not in _SUPPORTED_DTYPES:
            raise PrecisionError(f"invalid checkpoint precision dtype {dtype!r}")
        if device_type not in {"cpu", "cuda", "mps"}:
            raise PrecisionError(
                f"invalid checkpoint precision device type {device_type!r}"
            )
        if not isinstance(scaler_enabled, bool):
            raise PrecisionError("checkpoint scaler_enabled must be a boolean")
        scaler_state = value["scaler_state"]
        if scaler_state is not None and not isinstance(scaler_state, dict):
            raise PrecisionError(
                "checkpoint precision scaler_state must be an object or null"
            )
        try:
            return cls(
                dtype=cast(TrainDType, dtype),
                device_type=cast(PrecisionDeviceType, device_type),
                scaler_enabled=scaler_enabled,
                scaler_state=cast(dict[str, Any] | None, scaler_state),
            )
        except (TypeError, ValueError) as error:
            raise PrecisionError(
                f"invalid checkpoint precision state: {error}"
            ) from error


def _validate_descriptor(
    *,
    dtype: str,
    device_type: str,
    scaler_enabled: bool,
) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_DTYPES))
        raise ValueError(f"dtype must be one of {supported}, got {dtype!r}")
    if device_type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"precision device type must be cpu, cuda, or mps, got {device_type!r}"
        )
    expected_scaler = dtype == "float16"
    if scaler_enabled is not expected_scaler:
        raise ValueError(
            f"{dtype} on {device_type} requires scaler_enabled={expected_scaler}"
        )
    if dtype == "float16" and device_type != "cuda":
        raise ValueError("float16 training with GradScaler requires CUDA")
    if dtype == "bfloat16" and device_type not in {"cpu", "cuda"}:
        raise ValueError("bfloat16 autocast requires CPU or CUDA")


class PrecisionPolicy:
    """Runtime owner for autocast, backward scaling, and optimizer stepping."""

    def __init__(
        self,
        *,
        dtype: TrainDType,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        scaler: _GradScaler | None,
        autocast_factory: Callable[..., AbstractContextManager[Any]] = torch.autocast,
    ) -> None:
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        if not callable(autocast_factory):
            raise TypeError("autocast_factory must be callable")
        scaler_enabled = scaler is not None and scaler.is_enabled()
        _validate_descriptor(
            dtype=dtype,
            device_type=device.type,
            scaler_enabled=scaler_enabled,
        )
        expected_autocast = dtype != "float32"
        if (autocast_dtype is not None) is not expected_autocast:
            raise ValueError(
                f"{dtype} requires autocast_dtype to be "
                f"{'set' if expected_autocast else 'null'}"
            )
        self._dtype = dtype
        self._device = device
        self._autocast_dtype = autocast_dtype
        self._scaler = scaler
        self._autocast_factory = autocast_factory

    @property
    def dtype(self) -> TrainDType:
        return self._dtype

    @property
    def device_type(self) -> PrecisionDeviceType:
        return cast(PrecisionDeviceType, self._device.type)

    @property
    def autocast_enabled(self) -> bool:
        return self._autocast_dtype is not None

    @property
    def scaler_enabled(self) -> bool:
        return self._scaler is not None and self._scaler.is_enabled()

    def autocast(self) -> AbstractContextManager[Any]:
        if self._autocast_dtype is None:
            return nullcontext()
        return self._autocast_factory(
            device_type=self._device.type,
            dtype=self._autocast_dtype,
            enabled=True,
        )

    def backward(self, loss: Tensor) -> None:
        if not isinstance(loss, Tensor):
            raise TypeError("loss must be a Tensor")
        if self._scaler is None:
            loss.backward()
        else:
            self._scaler.scale(loss).backward()

    def unscale_(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def step_and_update(self, optimizer: Optimizer) -> bool:
        """Step once and report whether GradScaler invoked the optimizer."""

        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if self._scaler is None:
            optimizer.step()
            return True

        applied = False

        def mark_applied(
            _optimizer: Optimizer,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            nonlocal applied
            applied = True

        hook = optimizer.register_step_post_hook(mark_applied)
        try:
            self._scaler.step(optimizer)
        finally:
            hook.remove()
        self._scaler.update()
        return applied

    def checkpoint_state(self) -> PrecisionCheckpointState:
        scaler_state = None
        if self._scaler is not None:
            scaler_state = deepcopy(self._scaler.state_dict())
        return PrecisionCheckpointState(
            dtype=self._dtype,
            device_type=self.device_type,
            scaler_enabled=self.scaler_enabled,
            scaler_state=scaler_state,
        )

    def validate_checkpoint_state(self, state: PrecisionCheckpointState) -> None:
        if not isinstance(state, PrecisionCheckpointState):
            raise TypeError("state must be a PrecisionCheckpointState")
        expected = (self._dtype, self.device_type, self.scaler_enabled)
        actual = (state.dtype, state.device_type, state.scaler_enabled)
        if actual != expected:
            raise PrecisionError(
                "checkpoint precision policy is incompatible with the requested "
                f"runtime: checkpoint={actual}, requested={expected}"
            )

    def load_checkpoint_state(self, state: PrecisionCheckpointState) -> None:
        self.validate_checkpoint_state(state)
        if self._scaler is not None:
            if state.scaler_state is None:  # pragma: no cover - state validates.
                raise PrecisionError("enabled checkpoint scaler lost its state")
            self._scaler.load_state_dict(deepcopy(state.scaler_state))


def build_precision_policy(
    *,
    dtype: TrainDType,
    device: str | torch.device,
) -> PrecisionPolicy:
    """Validate and construct the supported single-device precision policy."""

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise PrecisionError(f"invalid precision device {device!r}: {error}") from error
    if dtype == "float32":
        return PrecisionPolicy(
            dtype=dtype,
            device=resolved_device,
            autocast_dtype=None,
            scaler=None,
        )
    if dtype == "float16":
        if resolved_device.type != "cuda":
            raise PrecisionError("float16 training with GradScaler requires CUDA")
        if not torch.amp.autocast_mode.is_autocast_available("cuda"):
            raise PrecisionError("float16 CUDA autocast is unavailable in this runtime")
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        return PrecisionPolicy(
            dtype=dtype,
            device=resolved_device,
            autocast_dtype=torch.float16,
            scaler=cast(_GradScaler, scaler),
        )
    if dtype == "bfloat16":
        if resolved_device.type not in {"cpu", "cuda"}:
            raise PrecisionError("bfloat16 autocast requires CPU or CUDA")
        if not torch.amp.autocast_mode.is_autocast_available(resolved_device.type):
            raise PrecisionError(
                f"bfloat16 {resolved_device.type} autocast is unavailable"
            )
        if resolved_device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise PrecisionError("bfloat16 is unsupported by the requested CUDA device")
        return PrecisionPolicy(
            dtype=dtype,
            device=resolved_device,
            autocast_dtype=torch.bfloat16,
            scaler=None,
        )
    raise PrecisionError(f"unsupported training dtype {dtype!r}")


def legacy_float32_precision_state(
    device: str | torch.device,
) -> PrecisionCheckpointState:
    """Return the only precision policy supported by pre-AMP checkpoints."""

    resolved = torch.device(device)
    return PrecisionCheckpointState(
        dtype="float32",
        device_type=cast(PrecisionDeviceType, resolved.type),
        scaler_enabled=False,
        scaler_state=None,
    )


__all__ = [
    "PrecisionCheckpointState",
    "PrecisionError",
    "PrecisionPolicy",
    "build_precision_policy",
    "legacy_float32_precision_state",
]
