"""FastAPI WebSocket generation protocol tests."""

from __future__ import annotations

from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient

from scratch_llm.config import GenerationConfig, WebConfig
from scratch_llm.web.app import create_app
from scratch_llm.web.service import ChatSessionService
from tests.test_web_generation import (
    ScriptedEngine,
    _complete_event,
    _start_event,
    _token_event,
)
from tests.test_web_service import _checkpoint


def _client(
    root: Path, engine: ScriptedEngine
) -> tuple[TestClient, ChatSessionService]:
    service = ChatSessionService(root, engine_factory=lambda _path, _device: engine)
    app = create_app(
        web_config=WebConfig(checkpoint_dir=str(root)),
        generation_config=GenerationConfig(),
        service_factory=lambda: service,
    )
    return TestClient(app), service


def _generate(
    message: str = "hello",
    *,
    debug: bool = False,
) -> dict[str, object]:
    return {
        "protocol_version": "v1",
        "type": "generate",
        "message": message,
        "debug": debug,
        "settings": {
            "temperature": 0.25,
            "top_k": 7,
            "max_new_tokens": 4,
            "seed": 5,
        },
    }


def test_websocket_streams_lossless_events_then_one_done(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    script = [
        _start_event(),
        _token_event(ord("A"), "A", 1),
        _complete_event(1, stop_token_id=264),
    ]
    engine = ScriptedEngine(tmp_path / "model.pt", "cpu", script)
    engine.pending_prompt_token_ids = (256, 104, 105)
    client, _service = _client(root, engine)

    with client:
        assert (
            client.post(
                "/api/load_checkpoint", json={"checkpoint_id": "model.pt"}
            ).status_code
            == 200
        )
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(_generate(debug=True))
            start = websocket.receive_json()
            token = websocket.receive_json()
            done = websocket.receive_json()

    assert start == {
        "protocol_version": "v1",
        "type": "start",
        "event": script[0].to_dict(),
    }
    assert token == {
        "protocol_version": "v1",
        "type": "token",
        "event": script[1].to_dict(),
    }
    assert done["protocol_version"] == "v1"
    assert done["type"] == "done"
    assert done["event"] == script[2].to_dict()
    assert done["state"]["status"] == "ready"
    assert done["metrics"] == {
        "generated_tokens": 1,
        "sampled_tokens": 2,
        "generation_seconds": 0.5,
        "prefill_latency_seconds": 0.1,
        "decode_latency_per_sampled_token_seconds": 0.4,
        "tokens_per_second": 4.0,
        "peak_memory_mib": None,
    }
    assert done["debug"] == {
        "prompt_token_ids": [256, 104, 105],
        "generated_token_ids": [ord("A")],
        "completion_reason": "stop_token",
        "stop_token_id": 264,
    }
    assert done["aggregate"]["session_id"].startswith("session-")
    assert done["aggregate"]["turn_id"].startswith("turn-")
    assert done["aggregate"]["turn_count"] == 1
    assert engine.messages[-1].content == "A"  # type: ignore[union-attr]


def test_stop_is_cooperative_and_concurrent_client_gets_busy(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    entered = threading.Event()
    release = threading.Event()
    engine = ScriptedEngine(
        tmp_path / "model.pt",
        "cpu",
        [_start_event(), _token_event(ord("A"), "A", 1), _complete_event(1)],
        block_before=1,
        entered_block=entered,
        release_block=release,
    )
    client, service = _client(root, engine)

    with client:
        client.post("/api/load_checkpoint", json={"checkpoint_id": "model.pt"})
        with client.websocket_connect("/ws/generate") as owner:
            owner.send_json(_generate("owner"))
            assert owner.receive_json()["type"] == "start"
            assert entered.wait(timeout=2)
            with client.websocket_connect("/ws/generate") as concurrent:
                concurrent.send_json(_generate("second"))
                busy = concurrent.receive_json()
            assert busy["type"] == "busy"
            assert busy["error"]["code"] == "busy"
            owner.send_json({"protocol_version": "v1", "type": "stop"})
            release.set()
            cancelled = owner.receive_json()

    assert cancelled["type"] == "cancelled"
    assert cancelled["state"]["status"] == "ready"
    assert engine.appended_messages == ["owner"]
    assert service.status == "unloaded"


def test_disconnect_cancels_and_releases_lease_for_next_request(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    entered = threading.Event()
    release = threading.Event()
    engine = ScriptedEngine(
        tmp_path / "model.pt",
        "cpu",
        [_start_event(), _token_event(ord("A"), "A", 1), _complete_event(1)],
        block_before=1,
        entered_block=entered,
        release_block=release,
    )
    client, service = _client(root, engine)

    with client:
        client.post("/api/load_checkpoint", json={"checkpoint_id": "model.pt"})
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(_generate("disconnect"))
            assert websocket.receive_json()["type"] == "start"
            assert entered.wait(timeout=2)
        release.set()
        deadline = time.monotonic() + 2
        while service.status != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        engine.script = [_start_event(), _complete_event(0)]
        engine.block_before = None
        with client.websocket_connect("/ws/generate") as retry:
            retry.send_json(_generate("retry"))
            assert retry.receive_json()["type"] == "start"
            assert retry.receive_json()["type"] == "done"


def test_malformed_and_failed_requests_use_sanitized_protocol_errors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    _checkpoint(root, "model.pt")
    engine = ScriptedEngine(
        tmp_path / "model.pt",
        "cpu",
        [_start_event(), RuntimeError("private /secret/model failure")],
    )
    client, _service = _client(root, engine)

    with client:
        with client.websocket_connect("/ws/generate") as unloaded:
            unloaded.send_json(_generate())
            unloaded_error = unloaded.receive_json()
        client.post("/api/load_checkpoint", json={"checkpoint_id": "model.pt"})
        with client.websocket_connect("/ws/generate") as malformed:
            malformed.send_json(
                {
                    "protocol_version": "v1",
                    "type": "generate",
                    "message": ["raw", "sentinel"],
                }
            )
            malformed_error = malformed.receive_json()
        with client.websocket_connect("/ws/generate") as failed:
            failed.send_json(_generate("fail"))
            assert failed.receive_json()["type"] == "start"
            failure = failed.receive_json()

    assert unloaded_error["type"] == "error"
    assert unloaded_error["error"]["code"] == "checkpoint_not_loaded"
    assert malformed_error == {
        "protocol_version": "v1",
        "type": "error",
        "error": {
            "code": "invalid_request",
            "message": "generation request is invalid",
        },
    }
    assert "sentinel" not in repr(malformed_error)
    assert failure["type"] == "error"
    assert failure["error"]["code"] == "generation_failed"
    assert "/secret/model" not in repr(failure)
