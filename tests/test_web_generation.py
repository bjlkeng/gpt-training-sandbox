"""Single-lease cancellable generation tests for the web session service."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from pathlib import Path
import threading

import pytest

from scratch_llm.chat import AssistantMessage, TokenEvent, UserMessage
from scratch_llm.config import GenerationConfig
from scratch_llm.web.service import (
    ChatSessionService,
    GenerationOverrides,
    GenerationTerminal,
    WebServiceError,
)
from tests.test_web_service import FakeEngine, _checkpoint


def _start_event() -> TokenEvent:
    return TokenEvent(
        type="start",
        token_ids=(),
        text_delta="",
        prompt_token_count=4,
        generated_token_count=0,
        sampled_token_count=0,
        elapsed_seconds=0,
        completion_reason=None,
        stop_token_id=None,
    )


def _token_event(token_id: int, text: str, count: int) -> TokenEvent:
    return TokenEvent(
        type="token",
        token_ids=(token_id,),
        text_delta=text,
        prompt_token_count=4,
        generated_token_count=count,
        sampled_token_count=count,
        elapsed_seconds=count / 10,
        completion_reason=None,
        stop_token_id=None,
    )


def _complete_event(
    generated: int,
    *,
    stop_token_id: int | None = None,
) -> TokenEvent:
    reason = "stop_token" if stop_token_id is not None else "max_new_tokens"
    return TokenEvent(
        type="complete",
        token_ids=(),
        text_delta="",
        prompt_token_count=4,
        generated_token_count=generated,
        sampled_token_count=generated + (stop_token_id is not None),
        elapsed_seconds=0.5,
        completion_reason=reason,
        stop_token_id=stop_token_id,
    )


class ScriptedEngine(FakeEngine):
    def __init__(
        self,
        path: Path,
        device: str,
        script: Sequence[TokenEvent | BaseException],
        *,
        block_before: int | None = None,
        entered_block: threading.Event | None = None,
        release_block: threading.Event | None = None,
    ) -> None:
        super().__init__(path, device)
        self.script = list(script)
        self.block_before = block_before
        self.entered_block = entered_block
        self.release_block = release_block
        self.default_generation_config = GenerationConfig(
            temperature=0.7,
            top_k=11,
            max_new_tokens=9,
            seed=3,
        )
        self.appended_messages: list[str] = []
        self.settings: list[GenerationConfig] = []
        self.rollback_calls = 0

    def append_user_message(self, text: str) -> None:
        self.appended_messages.append(text)
        self.messages = (*self.messages, UserMessage(text))
        self.chat_status = "awaiting_assistant"

    def generate_stream(
        self,
        config: GenerationConfig | None = None,
    ) -> Iterator[TokenEvent]:
        assert config is not None
        self.settings.append(config)
        self.chat_status = "generating"
        text_parts: list[str] = []
        completed = False
        try:
            for index, item in enumerate(self.script):
                if index == self.block_before:
                    if self.entered_block is not None:
                        self.entered_block.set()
                    if self.release_block is not None:
                        self.release_block.wait(timeout=5)
                if isinstance(item, BaseException):
                    raise item
                text_parts.append(item.text_delta)
                if item.type == "complete":
                    self.messages = (
                        *self.messages,
                        AssistantMessage("".join(text_parts)),
                    )
                    self.chat_status = "completed"
                    completed = True
                yield item
        finally:
            if not completed:
                self.chat_status = "cancelled"

    def rollback_last_turn(self, *, include_completed: bool = False) -> None:
        self.rollback_calls += 1
        if self.messages and isinstance(self.messages[-1], UserMessage):
            self.messages = self.messages[:-1]
        elif (
            include_completed
            and self.messages
            and isinstance(self.messages[-1], AssistantMessage)
        ):
            self.messages = self.messages[:-2]
        self.chat_status = (
            "completed"
            if self.messages and isinstance(self.messages[-1], AssistantMessage)
            else "idle"
        )


def _service_with_engine(
    tmp_path: Path,
    engine: ScriptedEngine,
) -> ChatSessionService:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    service = ChatSessionService(root, engine_factory=lambda _path, _device: engine)
    asyncio.run(service.load_checkpoint("model.pt"))
    return service


async def _collect(lease) -> list[TokenEvent | GenerationTerminal]:
    return [item async for item in lease]


@pytest.mark.parametrize("stop_token_id", [None, 264, 256])
def test_completed_generation_forwards_events_losslessly_and_commits_once(
    tmp_path: Path,
    stop_token_id: int | None,
) -> None:
    script = [
        _start_event(),
        _token_event(ord("A"), "A", 1),
        _complete_event(1, stop_token_id=stop_token_id),
    ]
    engine = ScriptedEngine(tmp_path / "model.pt", "cpu", script)
    service = _service_with_engine(tmp_path, engine)

    async def scenario() -> list[TokenEvent | GenerationTerminal]:
        lease = await service.start_generation(
            "hello",
            GenerationOverrides(temperature=0.25, max_new_tokens=4),
        )
        return await _collect(lease)

    items = asyncio.run(scenario())

    assert items[:-1] == script[:-1]
    terminal = items[-1]
    assert isinstance(terminal, GenerationTerminal)
    assert terminal.outcome == "completed"
    assert terminal.completion_event == script[-1]
    assert terminal.state.status == "ready"
    assert engine.appended_messages == ["hello"]
    assert engine.messages[-1] == AssistantMessage("A")
    assert engine.settings == [
        GenerationConfig(
            temperature=0.25,
            top_k=11,
            max_new_tokens=4,
            seed=3,
        )
    ]
    assert service.status == "ready"


def test_cancellation_rolls_back_partial_turn_and_allows_immediate_reuse(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    engine = ScriptedEngine(
        tmp_path / "model.pt",
        "cpu",
        [
            _start_event(),
            _token_event(ord("A"), "A", 1),
            _token_event(ord("B"), "B", 2),
            _complete_event(2),
        ],
        block_before=2,
        entered_block=entered,
        release_block=release,
    )
    engine.messages = (UserMessage("earlier"), AssistantMessage("answer"))
    prior_messages = engine.messages
    service = _service_with_engine(tmp_path, engine)

    async def scenario() -> None:
        lease = await service.start_generation("cancel me", GenerationOverrides())
        assert await anext(lease) == _start_event()
        assert await anext(lease) == _token_event(ord("A"), "A", 1)
        assert await asyncio.to_thread(entered.wait, 2)
        lease.cancel()
        release.set()
        terminal = await anext(lease)
        assert isinstance(terminal, GenerationTerminal)
        assert terminal.outcome == "cancelled"
        with pytest.raises(StopAsyncIteration):
            await anext(lease)

        engine.script = [_start_event(), _complete_event(0)]
        engine.block_before = None
        next_lease = await service.start_generation("next", GenerationOverrides())
        next_items = await _collect(next_lease)
        assert isinstance(next_items[-1], GenerationTerminal)
        assert next_items[-1].outcome == "completed"

    asyncio.run(scenario())

    assert engine.rollback_calls == 1
    assert engine.messages[:2] == prior_messages
    assert engine.appended_messages == ["cancel me", "next"]
    assert service.status == "ready"


def test_single_lease_is_immediate_busy_and_mutations_do_not_queue(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    engine = ScriptedEngine(
        tmp_path / "model.pt",
        "cpu",
        [_start_event(), _complete_event(0)],
        block_before=1,
        entered_block=entered,
        release_block=release,
    )
    service = _service_with_engine(tmp_path, engine)

    async def scenario() -> None:
        lease = await service.start_generation("owner", GenerationOverrides())
        assert await anext(lease) == _start_event()
        assert await asyncio.to_thread(entered.wait, 2)
        with pytest.raises(WebServiceError) as concurrent:
            await service.start_generation("second", GenerationOverrides())
        assert concurrent.value.code == "busy"
        with pytest.raises(WebServiceError) as reset:
            service.reset()
        assert reset.value.code == "busy"
        with pytest.raises(WebServiceError) as load:
            await service.load_checkpoint("model.pt")
        assert load.value.code == "busy"
        lease.cancel()
        release.set()
        await lease.wait()

    asyncio.run(scenario())
    assert engine.appended_messages == ["owner"]


def test_generation_failure_is_sanitized_rolled_back_and_reusable(
    tmp_path: Path,
) -> None:
    engine = ScriptedEngine(
        tmp_path / "model.pt",
        "cpu",
        [_start_event(), RuntimeError("private model failure /secret/path")],
    )
    service = _service_with_engine(tmp_path, engine)

    async def scenario() -> GenerationTerminal:
        lease = await service.start_generation("fail", GenerationOverrides())
        assert await anext(lease) == _start_event()
        terminal = await anext(lease)
        assert isinstance(terminal, GenerationTerminal)
        engine.script = [_start_event(), _complete_event(0)]
        retry = await service.start_generation("retry", GenerationOverrides())
        assert (await _collect(retry))[-1].outcome == "completed"  # type: ignore[union-attr]
        return terminal

    terminal = asyncio.run(scenario())

    assert terminal.outcome == "failed"
    assert terminal.error_code == "generation_failed"
    assert terminal.error_message == "generation failed"
    assert "/secret/path" not in repr(terminal)
    assert engine.rollback_calls == 1
    assert service.status == "ready"


def test_shutdown_cancels_generation_before_releasing_engine(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    engine = ScriptedEngine(
        tmp_path / "model.pt",
        "cpu",
        [_start_event(), _complete_event(0)],
        block_before=1,
        entered_block=entered,
        release_block=release,
    )
    service = _service_with_engine(tmp_path, engine)

    async def scenario() -> None:
        lease = await service.start_generation("shutdown", GenerationOverrides())
        assert await anext(lease) == _start_event()
        assert await asyncio.to_thread(entered.wait, 2)
        shutdown = asyncio.create_task(service.shutdown())
        await asyncio.sleep(0)
        assert lease.cancel_requested is True
        release.set()
        await shutdown
        assert (await lease.wait()).outcome == "cancelled"

    asyncio.run(scenario())
    assert service.status == "unloaded"
    assert engine.close_calls == 1


def test_generation_request_is_bounded_and_unloaded_errors_are_stable(
    tmp_path: Path,
) -> None:
    service = ChatSessionService(tmp_path)

    async def unloaded() -> None:
        with pytest.raises(WebServiceError) as raised:
            await service.start_generation("hello", GenerationOverrides())
        assert raised.value.code == "checkpoint_not_loaded"

    asyncio.run(unloaded())

    engine = ScriptedEngine(tmp_path / "model.pt", "cpu", [_start_event()])
    loaded = _service_with_engine(tmp_path, engine)

    async def invalid() -> None:
        with pytest.raises(WebServiceError) as message:
            await loaded.start_generation("x" * 16_385, GenerationOverrides())
        assert message.value.code == "request_too_large"
        with pytest.raises(WebServiceError) as settings:
            await loaded.start_generation(
                "hello",
                GenerationOverrides(max_new_tokens=100_000),
            )
        assert settings.value.code == "invalid_generation_request"

    asyncio.run(invalid())
    assert engine.appended_messages == []
