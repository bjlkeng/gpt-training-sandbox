"""Focused tests for the optional local web application boundary."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from scratch_llm.config import GenerationConfig, WebConfig
from scratch_llm.web.app import WebService, create_app


@dataclass
class RecordingService:
    """Small lifecycle fake that cannot load a checkpoint or touch a device."""

    name: str
    events: list[str] = field(default_factory=list)
    fail_startup: bool = False

    async def startup(self) -> None:
        self.events.append(f"{self.name}:startup")
        if self.fail_startup:
            raise RuntimeError("startup failed")

    async def shutdown(self) -> None:
        self.events.append(f"{self.name}:shutdown")


def _app(
    *,
    service_factory: object | None = None,
    checkpoint_dir: str = "/private/checkpoints",
):
    keyword_arguments: dict[str, object] = {
        "web_config": WebConfig(
            host="localhost",
            port=4321,
            checkpoint_dir=checkpoint_dir,
        ),
        "generation_config": GenerationConfig(
            temperature=0.25,
            top_k=7,
            top_p=0.9,
            max_new_tokens=31,
            seed=17,
        ),
    }
    if service_factory is not None:
        keyword_arguments["service_factory"] = service_factory
    return create_app(**keyword_arguments)


def test_health_and_config_are_stable_versioned_sanitized_contracts() -> None:
    app = _app(checkpoint_dir="/private/checkpoints/API_KEY=sentinel")

    with TestClient(app) as client:
        health = client.get("/api/health")
        public_config = client.get("/api/config")

    assert health.status_code == 200
    assert health.json() == {
        "api_version": "v1",
        "status": "ok",
        "ready": True,
    }
    assert public_config.status_code == 200
    assert public_config.json() == {
        "api_version": "v1",
        "runtime": {
            "host": "localhost",
            "port": 4321,
            "remote_bind_allowed": False,
        },
        "generation": {
            "temperature": 0.25,
            "top_k": 7,
            "top_p": 0.9,
            "max_new_tokens": 31,
            "seed": 17,
        },
        "capabilities": {
            "health": True,
            "config": True,
            "checkpoint_sessions": True,
            "tokenizer": True,
            "streaming": True,
            "cancellation": True,
            "metrics": True,
            "transcript_export": True,
            "inspection": True,
        },
    }
    serialized = public_config.text
    assert "/private/checkpoints" not in serialized
    assert "API_KEY" not in serialized


def test_factory_is_lazy_and_app_instances_own_independent_services() -> None:
    created: list[RecordingService] = []

    def service_factory() -> WebService:
        service = RecordingService(name=f"service-{len(created)}")
        created.append(service)
        return service

    first = _app(service_factory=service_factory)
    second = _app(service_factory=service_factory)

    assert created == []
    assert first is not second
    assert first.state is not second.state

    with TestClient(first) as client:
        assert len(created) == 1
        assert first.state.service is created[0]
        assert not hasattr(second.state, "service")
        assert client.get("/api/health").json()["ready"] is True

    assert created[0].events == ["service-0:startup", "service-0:shutdown"]
    assert first.state.ready is False
    assert not hasattr(first.state, "service")

    with TestClient(second):
        assert len(created) == 2
        assert second.state.service is created[1]

    assert created[1].events == ["service-1:startup", "service-1:shutdown"]


def test_failed_startup_still_runs_owned_service_cleanup() -> None:
    service = RecordingService(name="broken", fail_startup=True)
    app = _app(service_factory=lambda: service)

    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(app):
            pass

    assert service.events == ["broken:startup", "broken:shutdown"]
    assert app.state.ready is False
    assert not hasattr(app.state, "service")


def test_constructing_an_app_does_not_construct_its_service() -> None:
    def fail_if_called() -> WebService:
        raise AssertionError("service construction belongs to application startup")

    app = _app(service_factory=fail_if_called)

    assert app.state.ready is False
