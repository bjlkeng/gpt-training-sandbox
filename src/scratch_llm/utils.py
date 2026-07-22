"""Deterministic runtime utilities shared by training and model code."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch
from torch import nn


_MAX_SHARED_SEED = 2**32 - 1
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps"})
_NUMBER_SUFFIXES = ("", "K", "M", "B", "T", "Q")
_BYTE_SUFFIXES = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, Torch, and every available CUDA generator."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    if not 0 <= seed <= _MAX_SHARED_SEED:
        raise ValueError(
            f"seed must be in range [0, {_MAX_SHARED_SEED}], got {seed}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and bool(backend.is_available())


def autodetect_device_type() -> str:
    """Return the best available local Torch device type."""

    if torch.cuda.is_available():
        return "cuda"
    if _mps_is_available():
        return "mps"
    return "cpu"


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an automatic or explicit device request and validate availability."""

    if device is None or device == "auto":
        return torch.device(autodetect_device_type())
    if not isinstance(device, (str, torch.device)):
        raise TypeError(
            "device must be a string, torch.device, or None, "
            f"got {type(device).__name__}"
        )

    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device request {device!r}: {error}") from error

    if resolved.type not in _SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_DEVICE_TYPES))
        raise ValueError(
            f"unsupported device type {resolved.type!r}; expected one of: {supported}"
        )

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {str(resolved)!r} was requested but CUDA is not available"
            )
        if resolved.index is not None:
            device_count = torch.cuda.device_count()
            if resolved.index >= device_count:
                noun = "device" if device_count == 1 else "devices"
                raise RuntimeError(
                    f"CUDA device index {resolved.index} is unavailable; "
                    f"found {device_count} {noun}"
                )
    elif resolved.type == "mps" and not _mps_is_available():
        raise RuntimeError(
            f"MPS device {str(resolved)!r} was requested but MPS is not available"
        )

    return resolved


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count unique parameter objects, optionally only those requiring gradients."""

    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
    if not isinstance(trainable_only, bool):
        raise TypeError(
            "trainable_only must be a boolean, "
            f"got {type(trainable_only).__name__}"
        )

    seen: set[int] = set()
    total = 0
    for _, parameter in module.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if not trainable_only or parameter.requires_grad:
            total += parameter.numel()
    return total


def format_num(value: int | float) -> str:
    """Format a finite number with decimal ML-oriented magnitude suffixes."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"value must be a number, got {type(value).__name__}")
    try:
        scaled = float(value)
    except OverflowError as error:
        raise ValueError(f"value must be finite, got {value}") from error
    if not math.isfinite(scaled):
        raise ValueError(f"value must be finite, got {value}")

    suffix_index = 0
    while (
        suffix_index < len(_NUMBER_SUFFIXES) - 1
        and float(f"{abs(scaled):.2f}") >= 1000
    ):
        scaled /= 1000
        suffix_index += 1

    suffix = _NUMBER_SUFFIXES[suffix_index]
    if not suffix:
        if isinstance(value, int):
            return str(value)
        return f"{scaled:.2f}".rstrip("0").rstrip(".")
    return f"{scaled:.2f}{suffix}"


def format_bytes(num_bytes: int) -> str:
    """Format a non-negative byte count with IEC binary units."""

    if not isinstance(num_bytes, int) or isinstance(num_bytes, bool):
        raise TypeError(
            f"num_bytes must be an integer, got {type(num_bytes).__name__}"
        )
    if num_bytes < 0:
        raise ValueError(f"num_bytes must be non-negative, got {num_bytes}")

    scaled = float(num_bytes)
    suffix_index = 0
    while scaled >= 1024 and suffix_index < len(_BYTE_SUFFIXES) - 1:
        scaled /= 1024
        suffix_index += 1

    suffix = _BYTE_SUFFIXES[suffix_index]
    if suffix_index == 0:
        return f"{num_bytes} {suffix}"
    return f"{scaled:.2f} {suffix}"


def atomic_write(
    path: str | os.PathLike[str],
    data: str | bytes,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace ``path`` with text or bytes written beside it."""

    if not isinstance(data, (str, bytes)):
        raise TypeError(f"data must be text or bytes, got {type(data).__name__}")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        if isinstance(data, str):
            text_file = os.fdopen(
                file_descriptor,
                mode="w",
                encoding=encoding,
                newline="",
            )
            file_descriptor = -1
            with text_file:
                text_file.write(data)
                text_file.flush()
                os.fsync(text_file.fileno())
        else:
            binary_file = os.fdopen(file_descriptor, mode="wb")
            file_descriptor = -1
            with binary_file:
                binary_file.write(data)
                binary_file.flush()
                os.fsync(binary_file.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if file_descriptor >= 0:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return destination


def save_json(value: Any, path: str | os.PathLike[str]) -> Path:
    """Serialize canonical, human-readable JSON and replace ``path`` atomically."""

    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not valid deterministic JSON: {error}") from error
    return atomic_write(path, f"{serialized}\n")


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load UTF-8 JSON from ``path``."""

    with Path(path).open(encoding="utf-8") as json_file:
        return json.load(json_file)


class Timer:
    """Accumulate elapsed intervals measured by a monotonic clock."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        if clock is not None and not callable(clock):
            raise TypeError(f"clock must be callable, got {type(clock).__name__}")
        self._clock = time.monotonic if clock is None else clock
        self._elapsed = 0.0
        self._started_at: float | None = None

    @property
    def running(self) -> bool:
        """Return whether an interval is currently being measured."""

        return self._started_at is not None

    @property
    def elapsed(self) -> float:
        """Return accumulated seconds, including the current interval."""

        if self._started_at is None:
            return self._elapsed
        return self._elapsed + (self._clock() - self._started_at)

    def start(self) -> Timer:
        """Start a new interval and return this timer."""

        if self.running:
            raise RuntimeError("timer is already running")
        self._started_at = self._clock()
        return self

    def stop(self) -> float:
        """Stop the current interval and return total elapsed seconds."""

        if self._started_at is None:
            raise RuntimeError("timer is not running")
        self._elapsed += self._clock() - self._started_at
        self._started_at = None
        return self._elapsed

    def reset(self) -> None:
        """Clear elapsed time, restarting the current interval when running."""

        self._elapsed = 0.0
        if self.running:
            self._started_at = self._clock()

    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


@contextmanager
def timer(*, clock: Callable[[], float] | None = None) -> Iterator[Timer]:
    """Yield a running :class:`Timer` and stop it on context exit."""

    measurement = Timer(clock=clock)
    with measurement:
        yield measurement


__all__ = [
    "Timer",
    "atomic_write",
    "autodetect_device_type",
    "count_parameters",
    "format_bytes",
    "format_num",
    "get_device",
    "load_json",
    "save_json",
    "set_seed",
    "timer",
]
