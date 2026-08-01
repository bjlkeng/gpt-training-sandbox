"""Assistant-only SFT BPB protocol and trainer callback tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import random
from typing import Callable

import numpy as np
import pytest
import torch
from torch import nn

from scratch_llm.chat.conversation import (
    AssistantMessage,
    Conversation,
    PythonOutputPart,
    PythonPart,
    TextPart,
    UserMessage,
)
from scratch_llm.chat.loader import (
    InMemorySFTSource,
    SFTBatchInfo,
    SFTConversationLoader,
    SFTMixtureEntry,
)
from scratch_llm.chat.rendering import CHAT_RENDERER_ID
from scratch_llm.evaluation.sft_bpb import (
    SFT_ASSISTANT_BPB_PROTOCOL_ID,
    SFT_ASSISTANT_BPB_PROTOCOL_VERSION,
    SFTAssistantBPBCallback,
    SFTAssistantBPBResult,
    SFTValidationError,
    evaluate_sft_assistant_bpb,
)
from scratch_llm.tokenization.artifacts import build_token_byte_lengths
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from tests.fixtures.bpb_conformance import BPB_CONFORMANCE_FIXTURE


class _LossFromInputsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        assert loss_reduction == "none"
        assert inputs.shape == targets.shape
        return inputs.to(torch.float64) * self.scale


class _ConstantLossModel(nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        assert loss_reduction == "none"
        return torch.ones_like(targets, dtype=torch.float64)


class _FiniteSFTLoader:
    def __init__(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        conversations_per_batch: list[int] | None = None,
        repeat: bool = False,
        global_step: int = 0,
        before_batch: Callable[[], None] | None = None,
    ) -> None:
        self.batches = batches
        self.repeat = repeat
        self.global_step = global_step
        self.before_batch = before_batch
        self._position = 0
        self._conversations_per_batch = conversations_per_batch or [1] * len(batches)
        self._last_batch_info: SFTBatchInfo | None = None

    @property
    def last_batch_info(self) -> SFTBatchInfo:
        assert self._last_batch_info is not None
        return self._last_batch_info

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._position == len(self.batches):
            raise StopIteration
        if self.before_batch is not None:
            self.before_batch()
        batch = self.batches[self._position]
        conversation_count = self._conversations_per_batch[self._position]
        self._position += 1
        self.global_step += 1
        self._last_batch_info = SFTBatchInfo(
            epoch=0,
            epoch_step=self._position,
            row_item_identities=(
                tuple(
                    f"conversation-{self._position}-{index}"
                    for index in range(conversation_count)
                ),
            ),
            content_lengths=(batch[0].numel(),),
        )
        return batch


def _evaluate(
    model: nn.Module,
    loader: object,
    token_bytes: torch.Tensor,
    *,
    max_batches: int | None = None,
) -> SFTAssistantBPBResult:
    return evaluate_sft_assistant_bpb(
        model,
        loader,
        token_bytes,
        checkpoint_identity="checkpoint:test",
        tokenizer_identity="tokenizer:test",
        validation_mixture_identity="mixture:test",
        device="cpu",
        max_batches=max_batches,
    )


def test_hand_computed_ascii_and_multibyte_targets_match_exact_bpb() -> None:
    log_two = math.log(2)
    loader = _FiniteSFTLoader(
        [
            (
                torch.tensor(
                    [[log_two, 2 * log_two, 100.0, 200.0]],
                    dtype=torch.float64,
                ),
                torch.tensor([[0, 1, 2, -1]]),
            )
        ]
    )

    result = _evaluate(
        _LossFromInputsModel(),
        loader,
        torch.tensor([1, 2, 0]),
    )

    assert result.protocol_id == SFT_ASSISTANT_BPB_PROTOCOL_ID
    assert result.protocol_version == SFT_ASSISTANT_BPB_PROTOCOL_VERSION
    assert result.renderer_identity == CHAT_RENDERER_ID
    assert result.evaluated_batches == 1
    assert result.source_conversations == 1
    assert result.processed_model_tokens == 4
    assert result.supervised_target_tokens == 2
    assert result.supervised_target_bytes == 3
    assert result.total_nats == pytest.approx(3 * log_two)
    assert result.bpb == pytest.approx(1.0, abs=1e-15)


def test_sft_protocol_reuses_shared_bpb_conformance_fixture() -> None:
    fixture = BPB_CONFORMANCE_FIXTURE
    losses, targets, token_bytes, supervision_mask = fixture.tensors()
    sft_targets = targets.clone()
    sft_targets[~supervision_mask] = -1
    loader = _FiniteSFTLoader([(losses.unsqueeze(0), sft_targets.unsqueeze(0))])

    result = _evaluate(_LossFromInputsModel(), loader, token_bytes)

    assert result.processed_model_tokens == len(fixture.targets)
    assert result.supervised_target_tokens == 2
    assert result.supervised_target_bytes == 3
    assert result.total_nats == pytest.approx(1.5)
    assert result.bpb == pytest.approx(1.5 / math.log(2) / 3)


def test_real_rendering_counts_only_assistant_text_and_python_text_bytes() -> None:
    tokenizer = ByteTokenizer()
    conversation = Conversation(
        messages=(
            UserMessage(content="ignore this user text"),
            AssistantMessage(
                content=(
                    TextPart(text="A"),
                    PythonPart(text="xy"),
                    PythonOutputPart(text="ignore this tool output"),
                    TextPart(text="é"),
                )
            ),
        )
    )
    source = InMemorySFTSource(
        [conversation],
        source_identity="fixture:rendered",
        shuffle=False,
    )
    loader = SFTConversationLoader(
        [SFTMixtureEntry(source)],
        tokenizer=tokenizer,
        batch_size=1,
        max_seq_len=64,
        packing_buffer_size=1,
        seed=7,
        repeat=False,
    )

    result = evaluate_sft_assistant_bpb(
        _ConstantLossModel(),
        loader,
        build_token_byte_lengths(tokenizer),
        checkpoint_identity="checkpoint:rendered",
        tokenizer_identity=tokenizer.get_identity(),
        validation_mixture_identity="mixture:rendered",
        device="cpu",
    )

    expected_bytes = len("Axyé".encode("utf-8"))
    assert result.source_conversations == 1
    assert result.processed_model_tokens == 64
    assert result.supervised_target_tokens == expected_bytes
    assert result.supervised_target_bytes == expected_bytes
    assert result.total_nats == pytest.approx(float(expected_bytes))
    assert result.bpb == pytest.approx(1 / math.log(2))


def test_finite_budget_consumes_only_complete_batches_and_reports_coverage() -> None:
    loader = _FiniteSFTLoader(
        [
            (torch.tensor([[1.0, 2.0]]), torch.tensor([[0, 0]])),
            (torch.tensor([[9.0, 9.0]]), torch.tensor([[0, 0]])),
        ],
        conversations_per_batch=[2, 3],
    )

    result = _evaluate(
        _LossFromInputsModel(),
        loader,
        torch.tensor([1]),
        max_batches=1,
    )

    assert result.batch_budget == 1
    assert result.evaluated_batches == 1
    assert result.source_conversations == 2
    assert result.processed_model_tokens == 2
    assert result.total_nats == pytest.approx(3.0)
    assert loader.global_step == 1


@pytest.mark.parametrize("max_batches", [0, -1, True, 1.5])
def test_invalid_finite_budget_fails_before_consuming_data(max_batches: object) -> None:
    loader = _FiniteSFTLoader([(torch.tensor([[1.0]]), torch.tensor([[0]]))])

    with pytest.raises((TypeError, ValueError), match="max_batches"):
        _evaluate(  # type: ignore[arg-type]
            _LossFromInputsModel(),
            loader,
            torch.tensor([1]),
            max_batches=max_batches,
        )

    assert loader.global_step == 0


def test_empty_and_zero_byte_validation_fail_clearly() -> None:
    with pytest.raises(SFTValidationError, match="no batches"):
        _evaluate(_LossFromInputsModel(), _FiniteSFTLoader([]), torch.tensor([1]))

    zero_byte_loader = _FiniteSFTLoader(
        [(torch.tensor([[3.0, 4.0]]), torch.tensor([[0, -1]]))]
    )
    with pytest.raises(
        SFTValidationError, match="zero counted assistant-content bytes"
    ):
        _evaluate(
            _LossFromInputsModel(),
            zero_byte_loader,
            torch.tensor([0]),
        )


@pytest.mark.parametrize(
    ("loader", "message"),
    [
        (
            _FiniteSFTLoader(
                [(torch.tensor([[1.0]]), torch.tensor([[0]]))],
                repeat=True,
            ),
            "finite",
        ),
        (
            _FiniteSFTLoader(
                [(torch.tensor([[1.0]]), torch.tensor([[0]]))],
                global_step=1,
            ),
            "fresh",
        ),
    ],
)
def test_validation_requires_a_fresh_finite_loader(
    loader: _FiniteSFTLoader,
    message: str,
) -> None:
    with pytest.raises(SFTValidationError, match=message):
        _evaluate(_LossFromInputsModel(), loader, torch.tensor([1]))


def test_result_is_immutable_strictly_consistent_and_canonical_json() -> None:
    result = _evaluate(
        _LossFromInputsModel(),
        _FiniteSFTLoader([(torch.tensor([[0.5, 1.0]]), torch.tensor([[0, 0]]))]),
        torch.tensor([2]),
    )

    with pytest.raises(FrozenInstanceError):
        result.bpb = 9.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="bpb does not match"):
        replace(result, bpb=result.bpb + 1.0)
    with pytest.raises(ValueError, match="protocol_id"):
        replace(result, protocol_id="unknown")
    with pytest.raises(ValueError, match="finite"):
        replace(result, total_nats=float("nan"))

    assert json.loads(result.to_json()) == result.to_dict()
    assert result.to_json().endswith("\n")
    assert "NaN" not in result.to_json()


class _FailingRandomModel(_LossFromInputsModel):
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_reduction: str,
    ) -> torch.Tensor:
        random.random()
        np.random.random()
        torch.rand(1)
        self.dropout.train()
        raise RuntimeError("model failed")


def test_failure_restores_every_module_mode_and_cpu_rng_state() -> None:
    model = _FailingRandomModel()
    model.train()
    model.dropout.eval()
    module_modes = [(module, module.training) for module in model.modules()]
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    loader = _FiniteSFTLoader(
        [(torch.tensor([[1.0]]), torch.tensor([[0]]))],
        before_batch=lambda: (random.random(), np.random.random(), torch.rand(1)),
    )

    with pytest.raises(RuntimeError, match="model failed"):
        _evaluate(model, loader, torch.tensor([1]))

    assert [(module, module.training) for module in model.modules()] == module_modes
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_trainer_callback_uses_fresh_loader_and_names_the_requested_step(
    tmp_path: Path,
) -> None:
    created_loaders: list[_FiniteSFTLoader] = []

    def loader_factory() -> _FiniteSFTLoader:
        loader = _FiniteSFTLoader([(torch.tensor([[0.5]]), torch.tensor([[0]]))])
        created_loaders.append(loader)
        return loader

    callback = SFTAssistantBPBCallback(
        model=_LossFromInputsModel(),
        validation_loader_factory=loader_factory,
        token_bytes=torch.tensor([1]),
        checkpoint_identity_prefix="run:memory",
        tokenizer_identity="tokenizer:callback",
        validation_mixture_identity="mixture:callback",
        device="cpu",
        max_batches=None,
    )
    before = tuple(tmp_path.iterdir())

    result = callback(7)

    assert result.checkpoint_identity == "run:memory#step:7"
    assert len(created_loaders) == 1
    assert created_loaders[0].global_step == 1
    assert tuple(tmp_path.iterdir()) == before
    with pytest.raises(ValueError, match="step"):
        callback(-1)
