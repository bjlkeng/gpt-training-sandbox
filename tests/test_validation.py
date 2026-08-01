"""Tests for shared structured-value validation helpers."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from scratch_llm._validation import (
    JsonValueValidator,
    require_finite_positive_real,
    require_finite_real,
    require_finite_unit_interval,
    require_finite_non_negative_real,
    require_integer,
    require_non_empty_string,
    require_non_negative_integer,
    require_non_negative_real,
    require_optional_non_negative_integer,
    require_optional_positive_integer,
    require_optional_real,
    require_positive_integer,
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
    assert VALIDATOR.require_object(
        MappingProxyType(value),
        label="artifact",
        expected_keys=frozenset({"format", "version"}),
        schema_label="version 1",
    ) == dict(value)

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
    assert VALIDATOR.require_string("", label="text", non_empty=False) == ""
    assert VALIDATOR.require_string(" ", label="text", non_empty=False) == " "
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
    with pytest.raises(
        DomainValidationError,
        match="text must be a string, got int",
    ):
        VALIDATOR.require_string(1, label="text", non_empty=False)
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


def test_integer_validation_composes_plain_bounded_and_optional_values() -> None:
    assert require_integer(-2, name="count") == -2
    assert require_non_negative_integer(0, name="count") == 0
    assert require_positive_integer(2, name="count") == 2
    assert require_optional_non_negative_integer(None, name="count") is None
    assert require_optional_non_negative_integer(0, name="count") == 0
    assert require_optional_positive_integer(None, name="count") is None
    assert require_optional_positive_integer(2, name="count") == 2

    for validator in (
        require_integer,
        require_non_negative_integer,
        require_positive_integer,
    ):
        with pytest.raises(TypeError, match="count must be an integer"):
            validator(True, name="count")
    with pytest.raises(ValueError, match="count must be non-negative"):
        require_non_negative_integer(-1, name="count")
    with pytest.raises(ValueError, match="count must be positive"):
        require_positive_integer(0, name="count")


def test_real_validation_composes_finiteness_and_bounds() -> None:
    assert require_finite_real(-2.5, name="score") == -2.5
    assert require_non_negative_real(0, name="score") == 0.0
    assert require_finite_positive_real(2, name="score") == 2.0
    assert require_finite_unit_interval(0.5, name="ratio") == 0.5

    with pytest.raises(ValueError, match="score must be finite"):
        require_finite_real(float("inf"), name="score")
    with pytest.raises(ValueError, match="score must be non-negative"):
        require_non_negative_real(-1, name="score")
    for value in (0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and greater than zero"):
            require_finite_positive_real(value, name="score")
    with pytest.raises(ValueError, match=r"ratio must be in \[0, 1\]"):
        require_finite_unit_interval(1.1, name="ratio")


def test_non_empty_string_validation_distinguishes_type_and_content() -> None:
    assert require_non_empty_string("value", name="label") == "value"

    with pytest.raises(TypeError, match="label must be a string"):
        require_non_empty_string(1, name="label")
    with pytest.raises(ValueError, match="label must be a non-empty string"):
        require_non_empty_string(" ", name="label")
