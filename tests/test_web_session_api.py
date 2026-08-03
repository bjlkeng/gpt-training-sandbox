"""HTTP contract tests for checkpoint and tokenizer sessions."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scratch_llm.config import GenerationConfig, WebConfig
from scratch_llm.web.app import create_app
from scratch_llm.web.service import ChatSessionService, MAX_TOKEN_IDS
from tests.test_web_service import RecordingEngineFactory, _checkpoint


def _client(
    checkpoint_dir: Path,
    factory: RecordingEngineFactory,
) -> TestClient:
    service = ChatSessionService(checkpoint_dir, engine_factory=factory)
    app = create_app(
        web_config=WebConfig(checkpoint_dir=str(checkpoint_dir)),
        generation_config=GenerationConfig(),
        service_factory=lambda: service,
    )
    return TestClient(app)


def test_catalog_load_reset_and_tokenizer_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "private" / "checkpoints"
    _checkpoint(root, "nested/model.pt")
    factory = RecordingEngineFactory()

    with _client(root, factory) as client:
        catalog = client.get("/api/checkpoints")
        loaded = client.post(
            "/api/load_checkpoint",
            json={"checkpoint_id": "nested/model.pt"},
        )
        encoded = client.post("/api/tokenize", json={"text": "café 🚀"})
        decoded = client.post(
            "/api/detokenize",
            json={"token_ids": encoded.json()["token_ids"]},
        )
        reset = client.post("/api/reset")

    assert catalog.status_code == 200
    assert catalog.json() == {
        "api_version": "v1",
        "active_checkpoint_id": None,
        "checkpoints": [
            {"id": "nested/model.pt", "name": "model.pt"},
        ],
    }
    assert loaded.status_code == 200
    assert loaded.json()["state"]["checkpoint_id"] == "nested/model.pt"
    assert str(root) not in loaded.text
    assert encoded.status_code == 200
    assert encoded.json() == {
        "api_version": "v1",
        "token_ids": list("café 🚀".encode("utf-8")),
        "token_count": len("café 🚀".encode("utf-8")),
    }
    assert decoded.json() == {
        "api_version": "v1",
        "text": "café 🚀",
        "token_count": len("café 🚀".encode("utf-8")),
    }
    assert reset.status_code == 200
    assert reset.json()["state"]["context"]["prompt_tokens"] == 0
    assert factory.created[0].close_calls == 1


def test_api_errors_are_typed_bounded_and_do_not_echo_private_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private" / "checkpoints"
    _checkpoint(root, "broken.pt")
    factory = RecordingEngineFactory()
    factory.fail_names.add("broken.pt")

    with _client(root, factory) as client:
        unloaded = client.post("/api/reset")
        traversal = client.post(
            "/api/load_checkpoint",
            json={"checkpoint_id": f"../{tmp_path.name}/secret.pt"},
        )
        broken = client.post(
            "/api/load_checkpoint",
            json={"checkpoint_id": "broken.pt"},
        )
        malformed = client.post(
            "/api/detokenize",
            json={"token_ids": [True]},
        )
        oversized = client.post(
            "/api/detokenize",
            json={"token_ids": [0] * (MAX_TOKEN_IDS + 1)},
        )

    assert unloaded.status_code == 409
    assert unloaded.json()["error"]["code"] == "checkpoint_not_loaded"
    assert traversal.status_code == 404
    assert traversal.json()["error"]["code"] == "checkpoint_not_found"
    assert tmp_path.name not in traversal.text
    assert broken.status_code == 422
    assert broken.json() == {
        "api_version": "v1",
        "error": {
            "code": "checkpoint_load_failed",
            "message": "checkpoint could not be loaded",
        },
    }
    assert str(root) not in broken.text
    assert "secret loader detail" not in broken.text
    assert malformed.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_request"
    assert "true" not in malformed.text.lower()
    assert oversized.status_code == 422
    assert oversized.json()["error"]["code"] == "invalid_request"
