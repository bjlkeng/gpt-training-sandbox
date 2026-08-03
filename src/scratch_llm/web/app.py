"""FastAPI boundary for the local chat application."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal, Protocol

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from scratch_llm.config import GenerationConfig, WebConfig


API_VERSION: Literal["v1"] = "v1"


class WebService(Protocol):
    """Application-owned service lifecycle used by web transport adapters."""

    async def startup(self) -> None:
        """Acquire resources needed while the application is serving."""

    async def shutdown(self) -> None:
        """Release resources acquired by ``startup``."""


class _IdleWebService:
    """Dependency-free placeholder until a checkpoint session is requested."""

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HealthResponse(_ResponseModel):
    """Versioned application-lifecycle readiness."""

    api_version: Literal["v1"] = API_VERSION
    status: Literal["ok"] = "ok"
    ready: bool


class PublicRuntimeConfig(_ResponseModel):
    """Non-secret socket policy visible to the local client."""

    host: str
    port: int
    remote_bind_allowed: bool


class PublicGenerationConfig(_ResponseModel):
    """Supported generation defaults visible to clients."""

    temperature: float
    top_k: int | None
    top_p: float | None
    max_new_tokens: int
    seed: int | None


class PublicCapabilities(_ResponseModel):
    """HTTP capabilities available in this API slice."""

    health: Literal[True] = True
    config: Literal[True] = True


class PublicConfigResponse(_ResponseModel):
    """Sanitized, versioned runtime and generation capabilities."""

    api_version: Literal["v1"] = API_VERSION
    runtime: PublicRuntimeConfig
    generation: PublicGenerationConfig
    capabilities: PublicCapabilities = PublicCapabilities()


def _idle_service_factory() -> WebService:
    return _IdleWebService()


def _public_config(
    web_config: WebConfig,
    generation_config: GenerationConfig,
) -> PublicConfigResponse:
    """Copy only explicitly public scalar configuration into the API model."""

    return PublicConfigResponse(
        runtime=PublicRuntimeConfig(
            host=web_config.host,
            port=web_config.port,
            remote_bind_allowed=web_config.allow_remote_bind,
        ),
        generation=PublicGenerationConfig(**generation_config.to_dict()),
    )


def create_app(
    *,
    web_config: WebConfig,
    generation_config: GenerationConfig,
    service_factory: Callable[[], WebService] = _idle_service_factory,
) -> FastAPI:
    """Return a fresh lazy application with an independently owned service."""

    if not isinstance(web_config, WebConfig):
        raise TypeError("web_config must be a WebConfig")
    if not isinstance(generation_config, GenerationConfig):
        raise TypeError("generation_config must be a GenerationConfig")
    if not callable(service_factory):
        raise TypeError("service_factory must be callable")
    web_config.validate()
    generation_config.validate()
    public_config = _public_config(web_config, generation_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = service_factory()
        app.state.service = service
        try:
            await service.startup()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                await service.shutdown()
            finally:
                del app.state.service

    app = FastAPI(
        title="scratch-llm local chat",
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.state.ready = False

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(ready=bool(request.app.state.ready))

    @app.get("/api/config", response_model=PublicConfigResponse)
    async def config() -> PublicConfigResponse:
        return public_config

    return app


__all__ = [
    "API_VERSION",
    "HealthResponse",
    "PublicCapabilities",
    "PublicConfigResponse",
    "PublicGenerationConfig",
    "PublicRuntimeConfig",
    "WebService",
    "create_app",
]
