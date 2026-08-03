"""Opt-in raw-content tracking across terminal and local-web chat sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from io import StringIO
from pathlib import Path
import threading
import time

import pytest

from scripts.chat import run_terminal_chat
from scratch_llm.chat import (
    AssistantMessage,
    ChatEventTracker,
    Conversation,
    TokenEvent,
    UserMessage,
    write_conversation_jsonl,
)
from scratch_llm.config import GenerationConfig
from scratch_llm.tracking import JsonlTracker, NullTracker, RunSummary, RunTracker
from scratch_llm.web.service import (
    ChatSessionService,
    GenerationOverrides,
    GenerationTerminal,
)
from tests.test_web_generation import (
    ScriptedEngine,
    _complete_event,
    _start_event,
    _token_event,
)
from tests.test_web_service import _checkpoint


PROMPT_SECRET = "PROMPT_SECRET_71a9"
RESPONSE_SECRET = "RESPONSE_SECRET_f84c"


class _RecordingTracker(NullTracker):
    def __init__(self, *, delay: float = 0) -> None:
        self.logs: list[dict[str, object]] = []
        self.artifacts: list[tuple[str, str, str]] = []
        self.delay = delay
        self.active_calls = 0
        self.max_active_calls = 0

    def log(self, metrics: dict[str, object], step: int | None = None) -> None:
        assert step is None
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.logs.append(dict(metrics))
        finally:
            self.active_calls -= 1

    def log_artifact(self, path: str, name: str, type: str) -> None:
        self.artifacts.append((path, name, type))


def _identity_factory() -> Callable[[str], str]:
    counters: dict[str, int] = {}

    def create(kind: str) -> str:
        counters[kind] = counters.get(kind, 0) + 1
        return f"{kind}-{counters[kind]}"

    return create


def _complete(text: str = RESPONSE_SECRET) -> Iterator[TokenEvent]:
    yield _start_event()
    yield _token_event(ord("A"), text, 1)
    yield _complete_event(1)


class _TerminalEngine:
    def __init__(self, responses: tuple[str, ...] = (RESPONSE_SECRET,)) -> None:
        self.default_generation_config = GenerationConfig()
        self.responses = iter(responses)
        self.messages: tuple[UserMessage | AssistantMessage, ...] = ()

    def append_user_message(self, text: str) -> None:
        self.messages = (*self.messages, UserMessage(text))

    def generate_stream(self, _settings: GenerationConfig) -> Iterator[TokenEvent]:
        response = next(self.responses)

        def generate() -> Iterator[TokenEvent]:
            yield _start_event()
            yield _token_event(ord("A"), response, 1)
            self.messages = (*self.messages, AssistantMessage(response))
            yield _complete_event(1)

        return generate()

    def get_last_completed_message(self, role: str) -> str:
        index = -2 if role == "user" else -1
        return self.messages[index].content  # type: ignore[return-value]

    def reset(self) -> None:
        self.messages = ()

    def save_transcript(self, path: Path) -> Path:
        return write_conversation_jsonl(Conversation(messages=self.messages), path)


class _PrivacyWebEngine(ScriptedEngine):
    def get_last_completed_message(self, role: str) -> str:
        index = -2 if role == "user" else -1
        return self.messages[index].content  # type: ignore[return-value]


def _event_tracker(
    sink: NullTracker,
    *,
    log_prompts: bool,
    log_responses: bool,
    warnings: list[str] | None = None,
) -> ChatEventTracker:
    return ChatEventTracker(
        sink,
        run_id="run-privacy-matrix",
        log_prompts=log_prompts,
        log_responses=log_responses,
        identity_factory=_identity_factory(),
        warning_sink=(warnings if warnings is not None else []).append,
    )


@pytest.mark.parametrize("log_prompts", [False, True])
@pytest.mark.parametrize("log_responses", [False, True])
def test_policy_constructs_only_opted_in_content_and_keeps_all_sinks_safe(
    tmp_path: Path,
    log_prompts: bool,
    log_responses: bool,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    local = JsonlTracker(metrics_path)
    summary = RunSummary(
        summary_path,
        run={"name": "matrix", "output_dir": str(tmp_path), "stage": "chat"},
    )
    local.attach_summary(summary)
    remote = _RecordingTracker()
    run_tracker = RunTracker(summary, local, remote)
    warnings: list[str] = []
    tracking = _event_tracker(
        run_tracker,
        log_prompts=log_prompts,
        log_responses=log_responses,
        warnings=warnings,
    )
    prompt_calls = 0
    response_calls = 0

    def prompt() -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return PROMPT_SECRET

    def response() -> str:
        nonlocal response_calls
        response_calls += 1
        return RESPONSE_SECRET

    session = tracking.start_session("cli")
    session.record_completed_turn(
        _complete_event(2),
        prompt_factory=prompt,
        response_factory=response,
    )
    run_tracker.finish()

    assert prompt_calls == int(log_prompts)
    assert response_calls == int(log_responses)
    assert len(warnings) == int(log_prompts or log_responses)
    assert len(remote.logs) == 1
    record = remote.logs[0]
    assert record["chat/run_id"] == "run-privacy-matrix"
    assert record["chat/session_id"].startswith("session-")  # type: ignore[union-attr]
    assert record["chat/turn_id"].startswith("turn-")  # type: ignore[union-attr]
    assert record["chat/transport"] == "cli"
    assert record["chat/generated_tokens"] == 2
    assert (record.get("chat/prompt") == PROMPT_SECRET) is log_prompts
    assert (record.get("chat/response") == RESPONSE_SECRET) is log_responses
    assert remote.artifacts == []
    combined = "\n".join(
        (
            metrics_path.read_text(encoding="utf-8"),
            summary_path.read_text(encoding="utf-8"),
            repr(remote.logs),
            repr(remote.artifacts),
        )
    )
    assert (PROMPT_SECRET in combined) is log_prompts
    assert (RESPONSE_SECRET in combined) is log_responses


@pytest.mark.parametrize("log_prompts", [False, True])
@pytest.mark.parametrize("log_responses", [False, True])
def test_cli_and_web_share_the_four_mode_policy_and_preserve_explicit_export(
    tmp_path: Path,
    log_prompts: bool,
    log_responses: bool,
) -> None:
    cli_sink = _RecordingTracker()
    cli_tracking = _event_tracker(
        cli_sink,
        log_prompts=log_prompts,
        log_responses=log_responses,
    )
    transcript = tmp_path / "terminal.jsonl"
    run_terminal_chat(
        _TerminalEngine(),  # type: ignore[arg-type]
        GenerationConfig(),
        prompt=PROMPT_SECRET,
        transcript_path=transcript,
        chat_tracking=cli_tracking,
    )

    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    web_sink = _RecordingTracker()
    web_tracking = _event_tracker(
        web_sink,
        log_prompts=log_prompts,
        log_responses=log_responses,
    )
    engine = _PrivacyWebEngine(
        root / "model.pt",
        "cpu",
        list(_complete()),
    )
    service = ChatSessionService(
        root,
        engine_factory=lambda _path, _device: engine,
        chat_tracking=web_tracking,
    )

    async def web_turn() -> GenerationTerminal:
        await service.load_checkpoint("model.pt")
        lease = await service.start_generation(PROMPT_SECRET, GenerationOverrides())
        items = [item async for item in lease]
        terminal = items[-1]
        assert isinstance(terminal, GenerationTerminal)
        return terminal

    terminal = asyncio.run(web_turn())

    for record, transport in ((cli_sink.logs[0], "cli"), (web_sink.logs[0], "web")):
        assert record["chat/transport"] == transport
        assert (record.get("chat/prompt") == PROMPT_SECRET) is log_prompts
        assert (record.get("chat/response") == RESPONSE_SECRET) is log_responses
    assert terminal.outcome == "completed"
    assert terminal.aggregate.generated_tokens == 1
    assert web_sink.logs[0]["chat/session_id"] == terminal.aggregate.session_id
    assert web_sink.logs[0]["chat/turn_id"] == terminal.aggregate.turn_id
    assert PROMPT_SECRET in transcript.read_text(encoding="utf-8")
    assert RESPONSE_SECRET in transcript.read_text(encoding="utf-8")
    assert PROMPT_SECRET in service.export_transcript().decode("utf-8")
    assert RESPONSE_SECRET in service.export_transcript().decode("utf-8")


def test_reset_cancel_failure_and_concurrent_sessions_never_cross_or_leak(
    tmp_path: Path,
) -> None:
    sink = _RecordingTracker(delay=0.01)
    tracking = _event_tracker(sink, log_prompts=True, log_responses=True)
    first = tracking.start_session("cli")
    second = tracking.start_session("web")
    threads = [
        threading.Thread(
            target=session.record_completed_turn,
            args=(_complete_event(1),),
            kwargs={
                "prompt_factory": lambda value=prompt: value,
                "response_factory": lambda value=response: value,
            },
        )
        for session, prompt, response in (
            (first, "PROMPT_ONE", "RESPONSE_ONE"),
            (second, "PROMPT_TWO", "RESPONSE_TWO"),
        )
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert sink.max_active_calls == 1
    assert {
        (record["chat/transport"], record["chat/prompt"], record["chat/response"])
        for record in sink.logs
    } == {
        ("cli", "PROMPT_ONE", "RESPONSE_ONE"),
        ("web", "PROMPT_TWO", "RESPONSE_TWO"),
    }
    first_session = first.session_id
    first.reset()
    assert first.session_id != first_session

    root = tmp_path / "cancel-catalog"
    _checkpoint(root, "model.pt")
    entered = threading.Event()
    release = threading.Event()
    engine = _PrivacyWebEngine(
        root / "model.pt",
        "cpu",
        list(_complete("PARTIAL_RESPONSE_SECRET")),
        block_before=2,
        entered_block=entered,
        release_block=release,
    )
    cancel_sink = _RecordingTracker()
    service = ChatSessionService(
        root,
        engine_factory=lambda _path, _device: engine,
        chat_tracking=_event_tracker(
            cancel_sink,
            log_prompts=True,
            log_responses=True,
        ),
    )

    async def cancel() -> None:
        await service.load_checkpoint("model.pt")
        lease = await service.start_generation(
            "CANCEL_PROMPT_SECRET", GenerationOverrides()
        )
        await anext(lease)
        await anext(lease)
        assert await asyncio.to_thread(entered.wait, 2)
        lease.cancel()
        release.set()
        terminal = await anext(lease)
        assert isinstance(terminal, GenerationTerminal)
        assert terminal.outcome == "cancelled"

    asyncio.run(cancel())
    assert cancel_sink.logs == []


def test_adapter_resets_rotate_identity_and_failures_never_log_partial_content(
    tmp_path: Path,
) -> None:
    reset_sink = _RecordingTracker()
    run_terminal_chat(
        _TerminalEngine(("FIRST_RESPONSE", "SECOND_RESPONSE")),  # type: ignore[arg-type]
        GenerationConfig(),
        prompt=None,
        input_stream=StringIO("FIRST_PROMPT\n/reset\nSECOND_PROMPT\n/quit\n"),
        output_stream=StringIO(),
        chat_tracking=_event_tracker(
            reset_sink,
            log_prompts=True,
            log_responses=True,
        ),
    )
    assert [record["chat/prompt"] for record in reset_sink.logs] == [
        "FIRST_PROMPT",
        "SECOND_PROMPT",
    ]
    assert [record["chat/response"] for record in reset_sink.logs] == [
        "FIRST_RESPONSE",
        "SECOND_RESPONSE",
    ]
    assert (
        reset_sink.logs[0]["chat/session_id"] != reset_sink.logs[1]["chat/session_id"]
    )
    assert all(record["chat/session_turn_count"] == 1 for record in reset_sink.logs)

    class _FailingTerminalEngine(_TerminalEngine):
        def generate_stream(
            self,
            _settings: GenerationConfig,
        ) -> Iterator[TokenEvent]:
            yield _start_event()
            yield _token_event(ord("A"), "PARTIAL_TERMINAL_SECRET", 1)
            raise RuntimeError("generation failed")

    terminal_failure_sink = _RecordingTracker()
    with pytest.raises(RuntimeError, match="generation failed"):
        run_terminal_chat(
            _FailingTerminalEngine(),  # type: ignore[arg-type]
            GenerationConfig(),
            prompt="FAILED_TERMINAL_PROMPT",
            output_stream=StringIO(),
            chat_tracking=_event_tracker(
                terminal_failure_sink,
                log_prompts=True,
                log_responses=True,
            ),
        )
    assert terminal_failure_sink.logs == []

    root = tmp_path / "failure-catalog"
    _checkpoint(root, "model.pt")
    web_failure_sink = _RecordingTracker()
    failure_engine = _PrivacyWebEngine(
        root / "model.pt",
        "cpu",
        [
            _start_event(),
            _token_event(ord("A"), "PARTIAL_WEB_SECRET", 1),
            RuntimeError("private failure"),
        ],
    )
    service = ChatSessionService(
        root,
        engine_factory=lambda _path, _device: failure_engine,
        chat_tracking=_event_tracker(
            web_failure_sink,
            log_prompts=True,
            log_responses=True,
        ),
    )

    async def fail_web() -> GenerationTerminal:
        await service.load_checkpoint("model.pt")
        lease = await service.start_generation(
            "FAILED_WEB_PROMPT",
            GenerationOverrides(),
        )
        items = [item async for item in lease]
        terminal = items[-1]
        assert isinstance(terminal, GenerationTerminal)
        return terminal

    assert asyncio.run(fail_web()).outcome == "failed"
    assert web_failure_sink.logs == []
