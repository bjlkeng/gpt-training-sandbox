"""Shared validation helpers for low-level runtime APIs."""

from __future__ import annotations


def require_positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer or raise an actionable input error."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_positive_real(value: object, *, name: str) -> float:
    """Return a positive real number or raise an actionable input error."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return numeric


__all__ = ["require_positive_integer", "require_positive_real"]
