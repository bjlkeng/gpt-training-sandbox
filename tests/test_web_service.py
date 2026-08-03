"""Checkpoint-session service tests with no real model or accelerator."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
import threading
import time

import pytest

from scratch_llm.chat import (
    CHAT_RENDERER_ID,
    AssistantMessage,
    ChatState,
    Conversation,
    conversation_to_jsonl_bytes,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from scratch_llm.web.service import (
    MAX_CATALOG_ID_BYTES,
    MAX_TEXT_BYTES,
    MAX_TOKEN_IDS,
    ChatSessionService,
    WebServiceError,
)


class FakeEngine:
    """A ChatEngine-shaped session fake backed by the real byte tokenizer."""

    def __init__(
        self,
        path: Path,
        device: str,
        *,
        max_context_tokens: int = 64,
    ) -> None:
        self.path = path
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.tokenizer = ByteTokenizer()
        self.chat_status = "idle"
        self.messages = ()
        self.reset_calls = 0
        self.tokenize_calls = 0
        self.detokenize_calls = 0
        self.close_calls = 0
        self.pending_prompt_token_ids: tuple[int, ...] = ()

    def get_state(self) -> ChatState:
        return ChatState(
            checkpoint_path=str(self.path),
            checkpoint_step=7,
            training_stage="sft",
            device=self.device,
            tokenizer_identity=self.tokenizer.get_identity(),
            renderer_id=CHAT_RENDERER_ID,
            status=self.chat_status,  # type: ignore[arg-type]
            messages=self.messages,
            prompt_token_count=0,
            generated_token_count=0,
            sampled_token_count=0,
            dropped_turn_count=0,
            truncated_user_token_count=0,
            generation_seconds=None,
            completion_reason=None,
            stop_token_id=None,
        )

    def reset(self) -> None:
        self.reset_calls += 1
        self.chat_status = "idle"
        self.messages = ()

    def get_pending_prompt_token_ids(self) -> tuple[int, ...]:
        return self.pending_prompt_token_ids

    def export_transcript_bytes(self) -> bytes:
        if (
            self.chat_status != "completed"
            or not self.messages
            or not isinstance(self.messages[-1], AssistantMessage)
        ):
            raise RuntimeError("completed conversation required")
        return conversation_to_jsonl_bytes(Conversation(messages=self.messages))

    def tokenize(self, text: str) -> tuple[int, ...]:
        self.tokenize_calls += 1
        return tuple(self.tokenizer.encode(text))

    def detokenize(self, token_ids: Sequence[int]) -> str:
        self.detokenize_calls += 1
        return self.tokenizer.decode(token_ids)

    def close(self) -> None:
        self.close_calls += 1


class RecordingEngineFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.created: list[FakeEngine] = []
        self.fail_names: set[str] = set()

    def __call__(self, path: Path, device: str) -> FakeEngine:
        self.calls.append((path, device))
        if path.name in self.fail_names:
            raise RuntimeError(f"secret loader detail for {path}")
        engine = FakeEngine(path, device)
        self.created.append(engine)
        return engine


def _checkpoint(directory: Path, relative_path: str) -> Path:
    path = directory / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def _load(service: ChatSessionService, checkpoint_id: str):
    return asyncio.run(service.load_checkpoint(checkpoint_id))


def test_catalog_is_sorted_readable_contained_and_path_sanitized(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    _checkpoint(root, "z.pt")
    _checkpoint(root, "nested/a.pt")
    _checkpoint(root, "ignore.bin")
    (root / "directory.pt").mkdir()
    _checkpoint(root, "bad\nname.pt")
    unreadable = _checkpoint(root, "unreadable.pt")
    unreadable.chmod(0)
    outside = _checkpoint(tmp_path, "outside.pt")
    (root / "escape.pt").symlink_to(outside)
    (root / "broken.pt").symlink_to(tmp_path / "missing.pt")
    service = ChatSessionService(root, engine_factory=RecordingEngineFactory())

    try:
        catalog = service.list_checkpoints()
    finally:
        unreadable.chmod(0o600)

    assert [entry.checkpoint_id for entry in catalog] == ["nested/a.pt", "z.pt"]
    assert [entry.name for entry in catalog] == ["a.pt", "z.pt"]
    serialized = repr([entry.to_dict() for entry in catalog])
    assert str(root) not in serialized
    assert str(outside) not in serialized


@pytest.mark.parametrize(
    "checkpoint_id",
    [
        "../outside.pt",
        "/tmp/outside.pt",
        "escape.pt",
        "missing.pt",
        "a.pt\x00",
        "x" * (MAX_CATALOG_ID_BYTES + 1),
    ],
)
def test_load_accepts_only_an_exact_catalog_identity(
    tmp_path: Path,
    checkpoint_id: str,
) -> None:
    root = tmp_path / "catalog"
    root.mkdir()
    _checkpoint(root, "valid.pt")
    outside = _checkpoint(tmp_path, "outside.pt")
    (root / "escape.pt").symlink_to(outside)
    factory = RecordingEngineFactory()
    service = ChatSessionService(root, engine_factory=factory)

    with pytest.raises(WebServiceError) as raised:
        _load(service, checkpoint_id)

    assert raised.value.code == "checkpoint_not_found"
    assert raised.value.status_code == 404
    assert factory.calls == []
    assert str(root) not in str(raised.value)
    assert str(outside) not in str(raised.value)


def test_successful_load_atomically_swaps_and_releases_the_previous_engine(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    first_path = _checkpoint(root, "first.pt")
    second_path = _checkpoint(root, "nested/second.pt")
    factory = RecordingEngineFactory()
    service = ChatSessionService(root, device="cpu", engine_factory=factory)

    first_state = _load(service, "first.pt")
    first_engine = factory.created[0]
    second_state = _load(service, "nested/second.pt")

    assert factory.calls == [
        (first_path.resolve(), "cpu"),
        (second_path.resolve(), "cpu"),
    ]
    assert first_state.checkpoint_id == "first.pt"
    assert second_state.to_dict() == {
        "status": "ready",
        "checkpoint_id": "nested/second.pt",
        "checkpoint_step": 7,
        "training_stage": "sft",
        "device": "cpu",
        "tokenizer_identity": ByteTokenizer().get_identity(),
        "renderer_id": CHAT_RENDERER_ID,
        "context": {
            "prompt_tokens": 0,
            "max_tokens": 64,
            "dropped_turns": 0,
            "truncated_user_tokens": 0,
        },
    }
    assert first_engine.close_calls == 1
    assert factory.created[1].close_calls == 0
    assert service.status == "ready"


def test_failed_load_preserves_the_previous_conversation_and_sanitizes_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "working.pt")
    _checkpoint(root, "broken.pt")
    factory = RecordingEngineFactory()
    factory.fail_names.add("broken.pt")
    service = ChatSessionService(root, engine_factory=factory)
    _load(service, "working.pt")
    previous = factory.created[0]
    previous.messages = (object(),)

    with pytest.raises(WebServiceError) as raised:
        _load(service, "broken.pt")

    assert raised.value.code == "checkpoint_load_failed"
    assert "secret loader detail" not in str(raised.value)
    assert str(root) not in str(raised.value)
    assert service.status == "ready"
    assert previous.close_calls == 0
    assert service.tokenize("still usable") == tuple(b"still usable")

    reset_state = service.reset()

    assert reset_state.status == "ready"
    assert reset_state.checkpoint_id == "working.pt"
    assert previous.reset_calls == 1


def test_cancelled_load_restores_prior_session_and_releases_late_result(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "working.pt")
    _checkpoint(root, "slow.pt")
    started = threading.Event()
    release = threading.Event()
    created: list[FakeEngine] = []

    def factory(path: Path, device: str) -> FakeEngine:
        if path.name == "slow.pt":
            started.set()
            release.wait(timeout=5)
        engine = FakeEngine(path, device)
        created.append(engine)
        return engine

    async def scenario() -> None:
        service = ChatSessionService(root, engine_factory=factory)
        await service.load_checkpoint("working.pt")
        previous = created[0]
        task = asyncio.create_task(service.load_checkpoint("slow.pt"))
        assert await asyncio.to_thread(started.wait, 2)
        assert service.status == "loading"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.status == "ready"
        assert service.get_state().checkpoint_id == "working.pt"
        assert previous.close_calls == 0
        release.set()
        deadline = time.monotonic() + 2
        while len(created) < 2 or created[1].close_calls == 0:
            if time.monotonic() >= deadline:
                pytest.fail("late replacement engine was not released")
            await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_shutdown_serializes_with_loading_and_releases_the_winner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "working.pt")
    _checkpoint(root, "slow.pt")
    started = threading.Event()
    release = threading.Event()
    created: list[FakeEngine] = []

    def factory(path: Path, device: str) -> FakeEngine:
        if path.name == "slow.pt":
            started.set()
            release.wait(timeout=5)
        engine = FakeEngine(path, device)
        created.append(engine)
        return engine

    async def scenario() -> None:
        service = ChatSessionService(root, engine_factory=factory)
        await service.load_checkpoint("working.pt")
        load_task = asyncio.create_task(service.load_checkpoint("slow.pt"))
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(WebServiceError) as busy:
            service.reset()
        assert busy.value.code == "busy"
        shutdown_task = asyncio.create_task(service.shutdown())
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        release.set()
        assert (await load_task).checkpoint_id == "slow.pt"
        await shutdown_task
        assert service.status == "unloaded"
        assert service.get_state().checkpoint_id is None
        assert [engine.close_calls for engine in created] == [1, 1]

    asyncio.run(scenario())


def test_startup_loads_only_the_explicit_catalog_identity(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "initial.pt")
    factory = RecordingEngineFactory()
    service = ChatSessionService(
        root,
        initial_checkpoint_id="initial.pt",
        engine_factory=factory,
    )

    async def scenario() -> None:
        await service.startup()
        assert service.get_state().checkpoint_id == "initial.pt"
        await service.startup()
        assert len(factory.created) == 1
        await service.shutdown()

    asyncio.run(scenario())
    assert factory.created[0].close_calls == 1


def test_reset_and_tokenizer_operations_have_stable_unloaded_and_busy_errors(
    tmp_path: Path,
) -> None:
    service = ChatSessionService(tmp_path, engine_factory=RecordingEngineFactory())

    for operation in (
        service.reset,
        lambda: service.tokenize("hello"),
        lambda: service.detokenize([1]),
    ):
        with pytest.raises(WebServiceError) as raised:
            operation()
        assert raised.value.code == "checkpoint_not_loaded"
        assert raised.value.status_code == 409


def test_tokenizer_round_trip_limits_and_special_token_policy(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    factory = RecordingEngineFactory()
    service = ChatSessionService(root, engine_factory=factory)
    _load(service, "model.pt")
    engine = factory.created[0]
    literal_special = "héllo <|bos|> 🚀"

    token_ids = service.tokenize(literal_special)
    decoded = service.detokenize(token_ids)

    assert decoded == literal_special
    assert token_ids == tuple(literal_special.encode("utf-8"))
    assert ByteTokenizer().get_bos_token_id() not in token_ids
    assert service.detokenize([ByteTokenizer().get_bos_token_id()]) == "<|bos|>"

    with pytest.raises(WebServiceError) as too_much_text:
        service.tokenize("x" * (MAX_TEXT_BYTES + 1))
    assert too_much_text.value.code == "request_too_large"
    with pytest.raises(WebServiceError) as too_many_ids:
        service.detokenize([0] * (MAX_TOKEN_IDS + 1))
    assert too_many_ids.value.code == "request_too_large"
    before_calls = engine.detokenize_calls
    with pytest.raises(WebServiceError) as malformed:
        service.detokenize([True])  # type: ignore[list-item]
    assert malformed.value.code == "invalid_token_ids"
    assert engine.detokenize_calls == before_calls


def test_engine_generation_state_is_reported_as_service_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    factory = RecordingEngineFactory()
    service = ChatSessionService(root, engine_factory=factory)
    _load(service, "model.pt")
    factory.created[0].chat_status = "generating"

    assert service.get_state().status == "generating"
    with pytest.raises(WebServiceError) as raised:
        service.reset()
    assert raised.value.code == "busy"
