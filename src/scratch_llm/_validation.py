"""Shared scalar and structured-value validation primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import AbstractSet, NoReturn


class ConfigValidationError(ValueError):
    """A configuration error tied to a dotted field path."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


class JsonValueValidator:
    """Validate JSON-shaped values while preserving a domain error type."""

    def __init__(self, error_factory: Callable[[str], Exception]) -> None:
        if not callable(error_factory):
            raise TypeError("error_factory must be callable")
        self._error_factory = error_factory

    def require_object(
        self,
        value: object,
        *,
        label: str,
        expected_keys: AbstractSet[str] | None = None,
        schema_label: str = "schema",
    ) -> dict[str, object]:
        """Return a JSON object, optionally enforcing its exact string keys."""

        if not isinstance(value, dict):
            self._fail(f"{label} must be an object, got {type(value).__name__}")
        if not all(isinstance(key, str) for key in value):
            self._fail(f"{label} keys must be strings")
        if expected_keys is not None and set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            unexpected = sorted(set(value) - expected_keys)
            self._fail(
                f"{label} fields do not match {schema_label}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return value

    def require_list(
        self,
        value: object,
        *,
        label: str,
        non_empty: bool = False,
    ) -> list[object]:
        """Return a JSON list, optionally requiring at least one value."""

        if not isinstance(value, list):
            self._fail(f"{label} must be a list, got {type(value).__name__}")
        if non_empty and not value:
            self._fail(f"{label} must be a non-empty list")
        return value

    def require_string(self, value: object, *, label: str) -> str:
        """Return a non-empty JSON string."""

        if not isinstance(value, str) or not value.strip():
            self._fail(f"{label} must be a non-empty string")
        return value

    def require_integer(
        self,
        value: object,
        *,
        label: str,
        minimum: int,
        maximum: int | None = None,
    ) -> int:
        """Return a non-boolean integer within the requested interval."""

        if maximum is not None and maximum < minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        if not isinstance(value, int) or isinstance(value, bool):
            self._fail(f"{label} must be an integer")
        if value < minimum or (maximum is not None and value > maximum):
            interval = (
                f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
            )
            self._fail(f"{label} must be {interval}, got {value}")
        return value

    def duplicate_object_hook(
        self,
        *,
        label: str,
    ) -> Callable[[list[tuple[str, object]]], dict[str, object]]:
        """Return a ``json.load`` hook that rejects duplicate object keys."""

        def reject_duplicates(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    self._fail(f"{label} contains duplicate key {key!r}")
                result[key] = value
            return result

        return reject_duplicates

    def _fail(self, message: str) -> NoReturn:
        raise self._error_factory(message)


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
    "JsonValueValidator",
    "require_positive_integer",
    "require_positive_real",
]
