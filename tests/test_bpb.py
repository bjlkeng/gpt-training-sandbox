"""Protocol-neutral BPB arithmetic, evaluation, and result-contract tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math
import random
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from scratch_llm.bpb import (
    BPBAccumulation,
    BPBAccumulator,
    BaseValidationResult,
    accumulate_bpb,
    evaluate_bpb_batches,
)
from tests.fixtures.bpb_conformance import BPB_CONFORMANCE_FIXTURE


def test_hand_computed_multibyte_example_matches_exact_formula() -> None:
    log_two = math.log(2)
    losses = torch.tensor(
        [log_two, 2 * log_two, 500.0, 3 * log_two],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1, 3, 2])
    token_bytes = torch.tensor([1, 2, 3, 0])

    result = accumulate_bpb(losses, targets, token_bytes)

    assert result == BPBAccumulation(
        processed_model_tokens=4,
        counted_target_tokens=3,
        counted_target_bytes=6,
        total_nats=6 * log_two,
    )
    assert result.bpb == pytest.approx(1.0, abs=1e-15)
    assert result.to_dict() == {
        "bpb": result.bpb,
        "counted_target_bytes": 6,
        "counted_target_tokens": 3,
        "processed_model_tokens": 4,
        "total_nats": 6 * log_two,
    }


def test_shared_fixture_excludes_special_masked_carried_and_negative_targets() -> None:
    fixture = BPB_CONFORMANCE_FIXTURE
    losses, targets, token_bytes, supervision_mask = fixture.tensors()

    result = accumulate_bpb(
        losses,
        targets,
        token_bytes,
        supervision_mask=supervision_mask,
    )

    assert len(fixture.documents[1].encode("utf-8")) > fixture.context_length
    assert fixture.position_kinds == (
        "ordinary",
        "ordinary_multibyte",
        "special",
        "carried_context",
        "padding",
        "negative_target",
    )
    assert result.processed_model_tokens == 6
    assert result.counted_target_tokens == 2
    assert result.counted_target_bytes == 3
    assert result.total_nats == pytest.approx(1.5)
    assert result.bpb == pytest.approx(1.5 / math.log(2) / 3)


@pytest.mark.parametrize(
    ("token_bytes", "message"),
    [
        (torch.tensor([[1, 2]]), "one-dimensional"),
        (torch.tensor([1.0, 2.0]), "integer dtype"),
        (torch.tensor([1, -1]), "non-negative"),
        (torch.tensor([True, False]), "integer dtype"),
        (torch.tensor([], dtype=torch.long), "must not be empty"),
    ],
)
def test_token_byte_table_validation_is_strict(
    token_bytes: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        BPBAccumulator(token_bytes)


@pytest.mark.parametrize(
    ("losses", "targets", "mask", "message"),
    [
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[0, 1]]),
            None,
            "shapes must match",
        ),
        (
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            None,
            "floating-point",
        ),
        (
            torch.tensor([1.0, float("nan")]),
            torch.tensor([0, 1]),
            None,
            "finite",
        ),
        (
            torch.tensor([1.0, float("inf")]),
            torch.tensor([0, 1]),
            None,
            "finite",
        ),
        (
            torch.tensor([1.0, -0.5]),
            torch.tensor([0, 1]),
            None,
            "non-negative",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.0, 1.0]),
            None,
            "integer dtype",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0, 2]),
            None,
            "target ID.*out of range",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0, 1]),
            torch.tensor([[True, False]]),
            "mask shape",
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            "mask.*torch.bool",
        ),
    ],
)
def test_invalid_chunk_fails_without_partially_updating_accumulator(
    losses: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    message: str,
) -> None:
    accumulator = BPBAccumulator(torch.tensor([1, 2]))
    accumulator.update(torch.tensor([0.25]), torch.tensor([0]))
    before = accumulator.snapshot()

    with pytest.raises((TypeError, ValueError), match=message):
        accumulator.update(losses, targets, supervision_mask=mask)

    assert accumulator.snapshot() == before


def test_zero_counted_bytes_fails_only_when_finalizing() -> None:
    accumulator = BPBAccumulator(torch.tensor([1, 0]))
    accumulator.update(
        torch.tensor([1.0, 2.0]),
        torch.tensor([-1, 1]),
    )
    assert accumulator.processed_model_tokens == 2

    with pytest.raises(ValueError, match="zero counted target bytes"):
        accumulator.finalize()

    with pytest.raises(ValueError, match="zero counted target bytes"):
        accumulate_bpb(
            torch.tensor([1.0]),
            torch.tensor([1]),
            torch.tensor([1, 0]),
        )


def test_chunked_accumulation_matches_one_shot() -> None:
    losses = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 0])
    token_bytes = torch.tensor([1, 2, 0])
    mask = torch.tensor([True, True, True, False])
    expected = accumulate_bpb(
        losses,
        targets,
        token_bytes,
        supervision_mask=mask,
    )
    accumulator = BPBAccumulator(token_bytes)

    accumulator.update(
        losses[:2],
        targets[:2],
        supervision_mask=mask[:2],
    )
    accumulator.update(
        losses[2:],
        targets[2:],
        supervision_mask=mask[2:],
    )

    assert accumulator.finalize() == expected


class _SideEffectProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        assert loss_reduction == "none"
        assert inputs.shape == targets.shape
        random.random()
        np.random.random()
        torch.rand(3)
        return inputs.to(torch.float64) * self.scale


class _PolicyGuardedBatches:
    def __init__(self, values: list[tuple[torch.Tensor, ...]]) -> None:
        self.values = values
        self.policy = {"strategy": "protocol-owned", "shuffle": False}
        self.iterations = 0

    def __iter__(self) -> Any:
        self.iterations += 1
        random.random()
        np.random.random()
        torch.rand(1)
        return iter(self.values)

    def __len__(self) -> int:
        raise AssertionError("BPB evaluation must not inspect loader length")

    def state_dict(self) -> dict[str, object]:
        raise AssertionError("BPB evaluation must not inspect loader state")


def test_model_evaluation_restores_modes_rng_and_training_state() -> None:
    model = _SideEffectProbeModel()
    model.train()
    model.dropout.eval()
    model.scale.grad = torch.tensor(7.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    optimizer_state = deepcopy(optimizer.state_dict())
    scheduler_state = deepcopy(scheduler.state_dict())
    parameter_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    gradient_state = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    module_modes = [(module, module.training) for module in model.modules()]
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    python_rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state().clone()
    numpy_rng_state = np.random.get_state()
    batches = _PolicyGuardedBatches(
        [
            (
                torch.tensor([[0.5, 1.0]]),
                torch.tensor([[0, 1]]),
                torch.tensor([[True, True]]),
            ),
            (
                torch.tensor([[1.5, 50.0]]),
                torch.tensor([[0, 1]]),
                torch.tensor([[True, False]]),
            ),
        ]
    )
    policy_before = deepcopy(batches.policy)

    result = evaluate_bpb_batches(
        model,
        batches,
        torch.tensor([1, 2]),
        device="cpu",
    )

    assert result.processed_model_tokens == 4
    assert result.counted_target_tokens == 3
    assert result.counted_target_bytes == 4
    assert result.total_nats == pytest.approx(3.0)
    assert [(module, module.training) for module in model.modules()] == module_modes
    assert random.getstate() == python_rng_state
    assert torch.equal(torch.random.get_rng_state(), torch_rng_state)
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_rng_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_rng_state[1])
    assert current_numpy_state[2:] == numpy_rng_state[2:]
    assert batches.policy == policy_before
    assert batches.iterations == 1
    assert optimizer.state_dict() == optimizer_state
    assert scheduler.state_dict() == scheduler_state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, parameter_state[name])
    for name, parameter in model.named_parameters():
        if name in gradient_state:
            torch.testing.assert_close(parameter.grad, gradient_state[name])
        else:
            assert parameter.grad is None


def _validation_result() -> BaseValidationResult:
    accumulation = BPBAccumulation(
        processed_model_tokens=12,
        counted_target_tokens=3,
        counted_target_bytes=6,
        total_nats=4.5,
    )
    return BaseValidationResult.from_accumulation(
        accumulation,
        protocol_id="full_documents",
        protocol_version=1,
        reference_commit=None,
        reference_config={
            "packing": {"context_length": 8, "continuations": True},
            "order": ["shard-0", "shard-1"],
        },
        checkpoint_identity="sha256:" + "1" * 64,
        tokenizer_identity="sha256:" + "2" * 64,
        validation_manifest_identity="sha256:" + "3" * 64,
        source_documents=2,
        source_tokens=5,
        source_bytes=10,
        unique_source_tokens=3,
        unique_source_bytes=6,
    )


def test_validation_result_is_immutable_consistent_and_canonical_json() -> None:
    source_config = {
        "packing": {"context_length": 8, "continuations": True},
        "order": ["shard-0", "shard-1"],
    }
    accumulation = BPBAccumulation(
        processed_model_tokens=12,
        counted_target_tokens=3,
        counted_target_bytes=6,
        total_nats=4.5,
    )
    result = BaseValidationResult.from_accumulation(
        accumulation,
        protocol_id="full_documents",
        protocol_version=1,
        reference_commit=None,
        reference_config=source_config,
        checkpoint_identity="sha256:" + "1" * 64,
        tokenizer_identity="sha256:" + "2" * 64,
        validation_manifest_identity="sha256:" + "3" * 64,
        source_documents=2,
        source_tokens=5,
        source_bytes=10,
        unique_source_tokens=3,
        unique_source_bytes=6,
    )
    source_config["packing"]["context_length"] = 99
    source_config["order"].append("mutated")

    assert isinstance(result.reference_config, MappingProxyType)
    assert result.source_token_retention == pytest.approx(3 / 5)
    assert result.source_byte_retention == pytest.approx(6 / 10)
    assert result.bpb == pytest.approx(4.5 / math.log(2) / 6)
    assert result.reference_config["packing"]["context_length"] == 8  # type: ignore[index]
    with pytest.raises(TypeError):
        result.reference_config["new"] = True  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.total_nats = 0.0  # type: ignore[misc]

    payload = result.to_dict()
    assert tuple(payload) == (
        "protocol_id",
        "protocol_version",
        "reference_commit",
        "reference_config",
        "checkpoint_identity",
        "tokenizer_identity",
        "validation_manifest_identity",
        "source_documents",
        "source_tokens",
        "source_bytes",
        "processed_model_tokens",
        "counted_target_tokens",
        "counted_target_bytes",
        "unique_source_tokens",
        "unique_source_bytes",
        "source_token_retention",
        "source_byte_retention",
        "total_nats",
        "bpb",
    )
    serialized = result.to_json()
    assert serialized == result.to_json()
    assert serialized.endswith("\n")
    assert json.loads(serialized) == payload
    assert "NaN" not in serialized
    assert "Infinity" not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"checkpoint_identity": ""}, "checkpoint_identity"),
        ({"source_documents": -1}, "source_documents"),
        ({"unique_source_tokens": 6}, "unique_source_tokens.*source_tokens"),
        ({"counted_target_tokens": 2}, "counted_target_tokens.*unique"),
        ({"processed_model_tokens": 2}, "processed_model_tokens"),
        ({"source_token_retention": 0.1}, "source_token_retention"),
        ({"bpb": 99.0}, "bpb"),
        ({"total_nats": float("nan")}, "total_nats.*finite"),
    ],
)
def test_validation_result_rejects_inconsistent_or_non_finite_fields(
    mutation: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_validation_result(), **mutation)


def test_validation_result_rejects_non_json_reference_config() -> None:
    with pytest.raises(ValueError, match="reference_config.*finite"):
        BaseValidationResult.from_accumulation(
            BPBAccumulation(
                processed_model_tokens=1,
                counted_target_tokens=1,
                counted_target_bytes=1,
                total_nats=1.0,
            ),
            protocol_id="test",
            protocol_version=1,
            reference_commit=None,
            reference_config={"threshold": float("inf")},
            checkpoint_identity="checkpoint",
            tokenizer_identity="tokenizer",
            validation_manifest_identity="manifest",
            source_documents=1,
            source_tokens=1,
            source_bytes=1,
            unique_source_tokens=1,
            unique_source_bytes=1,
        )
