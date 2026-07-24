"""Shared scalar validators and structured configuration errors."""

from __future__ import annotations

from typing import NoReturn


class ConfigValidationError(ValueError):
    """A configuration error tied to a dotted field path."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _fail(path: str, message: str) -> NoReturn:
    raise ConfigValidationError(path, message)


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


def _require_non_empty(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")


def _require_int(value: object, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "must be an integer")


def _require_positive_int(value: object, path: str) -> None:
    try:
        require_positive_integer(value, name=path)
    except (TypeError, ValueError):
        _fail(path, "must be a positive integer")


def _require_non_negative_int(value: object, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(path, "must be a non-negative integer")


def _require_real(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, "must be a number")
    return float(value)


def _require_positive_real(value: object, path: str) -> None:
    try:
        require_positive_real(value, name=path)
    except TypeError:
        _fail(path, "must be a number")
    except ValueError:
        _fail(path, "must be greater than zero")


def _require_non_negative_real(value: object, path: str) -> None:
    if _require_real(value, path) < 0:
        _fail(path, "must be non-negative")


def _require_unit_interval(
    value: object, path: str, *, include_zero: bool = True
) -> None:
    numeric = _require_real(value, path)
    lower_bound_satisfied = numeric >= 0 if include_zero else numeric > 0
    if not lower_bound_satisfied or numeric > 1:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        _fail(path, f"must be in {interval}")


def _require_half_open_unit_interval(value: object, path: str) -> None:
    numeric = _require_real(value, path)
    if not 0 <= numeric < 1:
        _fail(path, "must be in [0, 1)")


def _require_choice(value: object, path: str, choices: frozenset[str]) -> None:
    if value not in choices:
        options = ", ".join(sorted(choices))
        _fail(path, f"must be one of: {options}")


__all__ = [
    "ConfigValidationError",
    "require_positive_integer",
    "require_positive_real",
]
