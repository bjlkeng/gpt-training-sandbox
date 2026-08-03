"""Metrics, export, and renderer contracts for the local web harness."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
import torch

from scratch_llm.chat import (
    CHAT_RENDERER_ID,
    AssistantMessage,
    Conversation,
    UserMessage,
    conversation_to_jsonl_bytes,
    read_conversations,
)
from scratch_llm.config import GenerationConfig, WebConfig
from scratch_llm.diagnostics.accelerator_memory import AcceleratorMemorySnapshot
from scratch_llm.web.app import create_app
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
from tests.test_web_service import RecordingEngineFactory, _checkpoint


def _identity_factory(*suffixes: str) -> Callable[[str], str]:
    values = iter(suffixes)
    return lambda kind: f"{kind}-{next(values)}"


def _memory_snapshot(peak_mib: int) -> AcceleratorMemorySnapshot:
    mib = 1024**2
    return AcceleratorMemorySnapshot(
        device=torch.device("cuda:0"),
        available=True,
        allocated_bytes=mib,
        reserved_bytes=peak_mib * mib,
        peak_allocated_bytes=peak_mib * mib,
        peak_reserved_bytes=peak_mib * mib,
        capacity_bytes=24 * 1024 * mib,
    )


def test_completed_turn_exposes_fake_clock_metrics_debug_and_safe_aggregate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    engine = ScriptedEngine(
        root / "model.pt",
        "cpu",
        [
            _start_event(),
            _token_event(ord("A"), "RESPONSE_SECRET_A", 1),
            _token_event(ord("B"), "RESPONSE_SECRET_B", 2),
            _complete_event(2, stop_token_id=264),
        ],
    )
    engine.pending_prompt_token_ids = (256, 1, 2, 3)
    reset_devices: list[str] = []
    service = ChatSessionService(
        root,
        engine_factory=lambda _path, _device: engine,
        identity_factory=_identity_factory("initial", "loaded", "turn-1", "reset"),
        reset_memory_peak=lambda device: reset_devices.append(device) is None,
        collect_memory=lambda _device: _memory_snapshot(12),
    )

    async def scenario() -> GenerationTerminal:
        await service.load_checkpoint("model.pt")
        lease = await service.start_generation(
            "PROMPT_SECRET",
            GenerationOverrides(),
            include_debug=True,
        )
        items = [item async for item in lease]
        terminal = items[-1]
        assert isinstance(terminal, GenerationTerminal)
        return terminal

    terminal = asyncio.run(scenario())

    assert reset_devices == ["cpu"]
    assert terminal.metrics is not None
    assert terminal.metrics.to_dict() == {
        "generated_tokens": 2,
        "sampled_tokens": 3,
        "generation_seconds": 0.5,
        "prefill_latency_seconds": 0.1,
        "decode_latency_per_sampled_token_seconds": 0.2,
        "tokens_per_second": 6.0,
        "peak_memory_mib": 12.0,
    }
    assert terminal.debug is not None
    assert terminal.debug.to_dict() == {
        "prompt_token_ids": [256, 1, 2, 3],
        "generated_token_ids": [ord("A"), ord("B")],
        "completion_reason": "stop_token",
        "stop_token_id": 264,
    }
    assert terminal.aggregate.to_dict() == {
        "session_id": "session-loaded",
        "turn_id": "turn-turn-1",
        "turn_count": 1,
        "generated_tokens": 2,
        "tokens_per_second": 6.0,
        "avg_decode_ms_per_token": 200.0,
        "peak_memory_mib": 12.0,
    }
    assert "PROMPT_SECRET" not in repr(terminal.aggregate)
    assert "RESPONSE_SECRET" not in repr(terminal.aggregate)

    prior_session_id = terminal.aggregate.session_id
    service.reset()
    reset_aggregate = service.get_session_aggregate()
    assert reset_aggregate.session_id != prior_session_id
    assert reset_aggregate.turn_id is None
    assert reset_aggregate.turn_count == 0


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ([_start_event(), _complete_event(0)], None),
        (
            [_start_event(), _complete_event(0, stop_token_id=264)],
            {
                "generated_tokens": 0,
                "sampled_tokens": 1,
                "generation_seconds": 0.5,
                "prefill_latency_seconds": 0.5,
                "decode_latency_per_sampled_token_seconds": None,
                "tokens_per_second": 2.0,
                "peak_memory_mib": None,
            },
        ),
    ],
)
def test_empty_and_first_stop_metric_semantics(
    tmp_path: Path,
    script: list[object],
    expected: dict[str, object] | None,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    engine = ScriptedEngine(root / "model.pt", "cpu", script)  # type: ignore[arg-type]
    service = ChatSessionService(
        root,
        engine_factory=lambda _path, _device: engine,
        reset_memory_peak=lambda _device: False,
    )

    async def scenario() -> GenerationTerminal:
        await service.load_checkpoint("model.pt")
        lease = await service.start_generation("hello", GenerationOverrides())
        terminal = [item async for item in lease][-1]
        assert isinstance(terminal, GenerationTerminal)
        return terminal

    terminal = asyncio.run(scenario())

    if expected is None:
        assert terminal.metrics is not None
        assert terminal.metrics.tokens_per_second is None
        assert terminal.metrics.prefill_latency_seconds is None
        assert terminal.metrics.decode_latency_per_sampled_token_seconds is None
        assert terminal.aggregate.tokens_per_second is None
    else:
        assert terminal.metrics is not None
        assert terminal.metrics.to_dict() == expected
    assert terminal.debug is None


def test_transcript_download_is_canonical_unicode_and_has_no_path_input(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    factory = RecordingEngineFactory()
    service = ChatSessionService(root, engine_factory=factory)
    app = create_app(
        web_config=WebConfig(checkpoint_dir=str(root)),
        generation_config=GenerationConfig(),
        service_factory=lambda: service,
    )
    conversation = Conversation(
        messages=(UserMessage("Café ☕"), AssistantMessage("第二 🚀"))
    )
    forbidden_path = tmp_path / "client-chosen.jsonl"

    with TestClient(app) as client:
        client.post("/api/load_checkpoint", json={"checkpoint_id": "model.pt"})
        engine = factory.created[0]
        engine.messages = conversation.messages
        engine.chat_status = "completed"
        response = client.get(
            "/api/transcript",
            params={"path": str(forbidden_path)},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"
    assert response.headers["content-disposition"] == (
        'attachment; filename="scratch-llm-transcript.jsonl"'
    )
    assert response.content == conversation_to_jsonl_bytes(conversation)
    downloaded = tmp_path / "downloaded.jsonl"
    downloaded.write_bytes(response.content)
    assert read_conversations(downloaded) == (conversation,)
    assert not forbidden_path.exists()


def test_renderer_catalog_is_server_owned_and_rejects_unsupported_choices(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    factory = RecordingEngineFactory()
    service = ChatSessionService(root, engine_factory=factory)
    app = create_app(
        web_config=WebConfig(checkpoint_dir=str(root)),
        generation_config=GenerationConfig(),
        service_factory=lambda: service,
    )

    with TestClient(app) as client:
        catalog = client.get("/api/renderers")
        unsupported = client.post(
            "/api/select_renderer",
            json={"renderer_id": "client_template_v999"},
        )
        client.post("/api/load_checkpoint", json={"checkpoint_id": "model.pt"})
        engine = factory.created[0]
        engine.messages = (UserMessage("prior"), AssistantMessage("history"))
        selected = client.post(
            "/api/select_renderer",
            json={"renderer_id": CHAT_RENDERER_ID},
        )

    assert catalog.json() == {
        "api_version": "v1",
        "active_renderer_id": None,
        "renderers": [
            {"id": CHAT_RENDERER_ID, "name": CHAT_RENDERER_ID},
        ],
    }
    assert unsupported.status_code == 422
    assert unsupported.json()["error"]["code"] == "unsupported_renderer"
    assert selected.status_code == 200
    assert selected.json()["history_reset"] is False
    assert selected.json()["state"]["renderer_id"] == CHAT_RENDERER_ID
    assert engine.messages == (UserMessage("prior"), AssistantMessage("history"))
