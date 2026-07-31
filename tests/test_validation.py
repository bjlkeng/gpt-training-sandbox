"""Tests for shared structured-value validation helpers."""

from __future__ import annotations

import json

import pytest

from scratch_llm._validation import (
    JsonValueValidator,
    require_finite_non_negative_real,
    require_optional_real,
    require_real,
)


class DomainValidationError(ValueError):
    """Test-only domain error used to verify error preservation."""


VALIDATOR = JsonValueValidator(DomainValidationError)


def test_json_object_validation_preserves_domain_errors_and_exact_keys() -> None:
    value = {"format": "example", "version": 1}

    assert (
        VALIDATOR.require_object(
            value,
            label="artifact",
            expected_keys=frozenset({"format", "version"}),
            schema_label="version 1",
        )
        is value
    )

    with pytest.raises(
        DomainValidationError,
        match=r"artifact fields do not match version 1; "
        r"missing=\['version'\], unexpected=\['extra'\]",
    ):
        VALIDATOR.require_object(
            {"format": "example", "extra": True},
            label="artifact",
            expected_keys=frozenset({"format", "version"}),
            schema_label="version 1",
        )
    with pytest.raises(DomainValidationError, match="keys must be strings"):
        VALIDATOR.require_object({1: "value"}, label="artifact")
    with pytest.raises(DomainValidationError, match="must be an object"):
        VALIDATOR.require_object([], label="artifact")


def test_json_list_string_and_integer_validation() -> None:
    values = [1, 2]

    assert VALIDATOR.require_list(values, label="values") is values
    assert VALIDATOR.require_string("value", label="name") == "value"
    assert (
        VALIDATOR.require_integer(
            2,
            label="count",
            minimum=1,
            maximum=3,
        )
        == 2
    )

    with pytest.raises(DomainValidationError, match="non-empty list"):
        VALIDATOR.require_list([], label="values", non_empty=True)
    with pytest.raises(DomainValidationError, match="non-empty string"):
        VALIDATOR.require_string(" ", label="name")
    with pytest.raises(DomainValidationError, match="must be an integer"):
        VALIDATOR.require_integer(True, label="count", minimum=0)
    with pytest.raises(DomainValidationError, match=r"must be \[1, 3\]"):
        VALIDATOR.require_integer(4, label="count", minimum=1, maximum=3)


def test_duplicate_key_hook_preserves_the_domain_and_document_label() -> None:
    hook = VALIDATOR.duplicate_object_hook(label="artifact.json")

    with pytest.raises(
        DomainValidationError,
        match=r"artifact\.json contains duplicate key 'format'",
    ):
        json.loads(
            '{"format": "first", "format": "second"}',
            object_pairs_hook=hook,
        )


def test_real_validation_accepts_json_numbers_but_rejects_bools() -> None:
    assert require_real(2, name="score") == 2.0
    assert require_real(2.5, name="score") == 2.5

    for value in (True, "2.5"):
        with pytest.raises(TypeError, match="score must be a number"):
            require_real(value, name="score")


def test_optional_real_validation_preserves_none() -> None:
    assert require_optional_real(None, name="score") is None
    assert require_optional_real(2, name="score") == 2.0

    with pytest.raises(TypeError, match="score must be a number"):
        require_optional_real(False, name="score")


def test_finite_non_negative_real_validation_rejects_invalid_values() -> None:
    assert require_finite_non_negative_real(0, name="score") == 0.0
    assert require_finite_non_negative_real(2.5, name="score") == 2.5

    for value in (-1, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="score must be finite and non-negative"):
            require_finite_non_negative_real(value, name="score")
