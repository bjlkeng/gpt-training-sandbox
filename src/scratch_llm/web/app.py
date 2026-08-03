"""FastAPI boundary for the local chat application."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from contextlib import asynccontextmanager
from typing import Literal, cast

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scratch_llm.config import GenerationConfig, WebConfig
from scratch_llm.web.service import (
    MAX_CATALOG_ID_BYTES,
    MAX_GENERATION_TOKENS,
    MAX_SEED,
    MAX_TEXT_BYTES,
    MAX_TEMPERATURE,
    MAX_TOP_K,
    MAX_TOKEN_IDS,
    MIN_SEED,
    ChatSessionService,
    GenerationLease,
    GenerationOverrides,
    GenerationTerminal,
    PublicSessionState,
    WebService,
    WebServiceError,
    WebSessionService,
)


API_VERSION: Literal["v1"] = "v1"


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
    checkpoint_sessions: Literal[True] = True
    tokenizer: Literal[True] = True
    streaming: Literal[True] = True
    cancellation: Literal[True] = True


class PublicConfigResponse(_ResponseModel):
    """Sanitized, versioned runtime and generation capabilities."""

    api_version: Literal["v1"] = API_VERSION
    runtime: PublicRuntimeConfig
    generation: PublicGenerationConfig
    capabilities: PublicCapabilities = PublicCapabilities()


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LoadCheckpointRequest(_RequestModel):
    checkpoint_id: str = Field(min_length=1, max_length=MAX_CATALOG_ID_BYTES)


class TokenizeRequest(_RequestModel):
    text: str = Field(max_length=MAX_TEXT_BYTES)


class DetokenizeRequest(_RequestModel):
    token_ids: list[int] = Field(max_length=MAX_TOKEN_IDS)


class GenerationSettingsRequest(_RequestModel):
    temperature: float | None = Field(default=None, ge=0, le=MAX_TEMPERATURE)
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    max_new_tokens: int | None = Field(
        default=None,
        ge=1,
        le=MAX_GENERATION_TOKENS,
    )
    seed: int | None = Field(default=None, ge=MIN_SEED, le=MAX_SEED)


class GenerateRequest(_RequestModel):
    protocol_version: Literal["v1"]
    type: Literal["generate"]
    message: str = Field(max_length=MAX_TEXT_BYTES)
    settings: GenerationSettingsRequest = Field(
        default_factory=GenerationSettingsRequest
    )


class StopRequest(_RequestModel):
    protocol_version: Literal["v1"]
    type: Literal["stop"]


class CheckpointResponse(_ResponseModel):
    id: str
    name: str


class CheckpointCatalogResponse(_ResponseModel):
    api_version: Literal["v1"] = API_VERSION
    active_checkpoint_id: str | None
    checkpoints: tuple[CheckpointResponse, ...]


class ContextStateResponse(_ResponseModel):
    prompt_tokens: int
    max_tokens: int
    dropped_turns: int
    truncated_user_tokens: int


class SessionStateResponse(_ResponseModel):
    status: Literal["unloaded", "loading", "ready", "generating", "failed"]
    checkpoint_id: str | None
    checkpoint_step: int | None
    training_stage: Literal["sft"] | None
    device: str
    tokenizer_identity: str | None
    renderer_id: str | None
    context: ContextStateResponse | None


class SessionResponse(_ResponseModel):
    api_version: Literal["v1"] = API_VERSION
    state: SessionStateResponse


class TokenizeResponse(_ResponseModel):
    api_version: Literal["v1"] = API_VERSION
    token_ids: tuple[int, ...]
    token_count: int


class DetokenizeResponse(_ResponseModel):
    api_version: Literal["v1"] = API_VERSION
    text: str
    token_count: int


class ErrorDetail(_ResponseModel):
    code: str
    message: str


class ErrorResponse(_ResponseModel):
    api_version: Literal["v1"] = API_VERSION
    error: ErrorDetail


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
    service_factory: Callable[[], WebService] | None = None,
) -> FastAPI:
    """Return a fresh lazy application with an independently owned service."""

    if not isinstance(web_config, WebConfig):
        raise TypeError("web_config must be a WebConfig")
    if not isinstance(generation_config, GenerationConfig):
        raise TypeError("generation_config must be a GenerationConfig")
    if service_factory is not None and not callable(service_factory):
        raise TypeError("service_factory must be callable or None")
    web_config.validate()
    generation_config.validate()
    public_config = _public_config(web_config, generation_config)
    active_service_factory = service_factory or (
        lambda: ChatSessionService(web_config.checkpoint_dir)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = active_service_factory()
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

    @app.exception_handler(WebServiceError)
    async def web_service_error(
        _request: Request,
        error: WebServiceError,
    ) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(code=error.code, message=error.public_message)
        )
        return JSONResponse(
            status_code=error.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        payload = ErrorResponse(
            error=ErrorDetail(
                code="invalid_request",
                message="request payload is invalid",
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(ready=bool(request.app.state.ready))

    @app.get("/api/config", response_model=PublicConfigResponse)
    async def config() -> PublicConfigResponse:
        return public_config

    @app.get("/api/checkpoints", response_model=CheckpointCatalogResponse)
    async def checkpoints(request: Request) -> CheckpointCatalogResponse:
        service = _session_service(request)
        return CheckpointCatalogResponse(
            active_checkpoint_id=service.active_checkpoint_id,
            checkpoints=tuple(
                CheckpointResponse(**entry.to_dict())
                for entry in service.list_checkpoints()
            ),
        )

    @app.post("/api/load_checkpoint", response_model=SessionResponse)
    async def load_checkpoint(
        payload: LoadCheckpointRequest,
        request: Request,
    ) -> SessionResponse:
        state = await _session_service(request).load_checkpoint(payload.checkpoint_id)
        return _session_response(state)

    @app.post("/api/reset", response_model=SessionResponse)
    async def reset(request: Request) -> SessionResponse:
        return _session_response(_session_service(request).reset())

    @app.post("/api/tokenize", response_model=TokenizeResponse)
    async def tokenize(
        payload: TokenizeRequest,
        request: Request,
    ) -> TokenizeResponse:
        token_ids = _session_service(request).tokenize(payload.text)
        return TokenizeResponse(
            token_ids=token_ids,
            token_count=len(token_ids),
        )

    @app.post("/api/detokenize", response_model=DetokenizeResponse)
    async def detokenize(
        payload: DetokenizeRequest,
        request: Request,
    ) -> DetokenizeResponse:
        text = _session_service(request).detokenize(payload.token_ids)
        return DetokenizeResponse(
            text=text,
            token_count=len(payload.token_ids),
        )

    @app.websocket("/ws/generate")
    async def generate(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            raw_request = await websocket.receive_json()
            request = GenerateRequest.model_validate(raw_request)
        except WebSocketDisconnect:
            return
        except (TypeError, ValueError, ValidationError):
            await websocket.send_json(
                _socket_error_payload(
                    code="invalid_request",
                    message="generation request is invalid",
                )
            )
            await websocket.close(code=1008)
            return

        settings = request.settings
        try:
            lease = await _session_service_from_websocket(websocket).start_generation(
                request.message,
                GenerationOverrides(
                    temperature=settings.temperature,
                    top_k=settings.top_k,
                    max_new_tokens=settings.max_new_tokens,
                    seed=settings.seed,
                ),
            )
        except WebServiceError as error:
            await websocket.send_json(
                _socket_error_payload(
                    code=error.code,
                    message=error.public_message,
                    busy=error.code == "busy",
                )
            )
            await websocket.close(code=1008)
            return

        stop_listener = asyncio.create_task(_listen_for_stop(websocket, lease))
        try:
            async for item in lease:
                if isinstance(item, GenerationTerminal):
                    await websocket.send_json(_terminal_payload(item))
                else:
                    await websocket.send_json(
                        {
                            "protocol_version": API_VERSION,
                            "type": item.type,
                            "event": item.to_dict(),
                        }
                    )
        except (RuntimeError, WebSocketDisconnect):
            lease.cancel()
        finally:
            stop_listener.cancel()
            with suppress(asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
                await stop_listener
            if not lease.done:
                lease.cancel()
            await lease.wait()
            with suppress(RuntimeError, WebSocketDisconnect):
                await websocket.close(code=1000)

    return app


def _session_service(request: Request) -> WebSessionService:
    return cast(WebSessionService, request.app.state.service)


def _session_service_from_websocket(websocket: WebSocket) -> WebSessionService:
    return cast(WebSessionService, websocket.app.state.service)


def _session_response(state: PublicSessionState) -> SessionResponse:
    return SessionResponse(
        state=SessionStateResponse.model_validate(state.to_dict()),
    )


async def _listen_for_stop(
    websocket: WebSocket,
    lease: GenerationLease,
) -> None:
    try:
        while True:
            raw_request = await websocket.receive_json()
            StopRequest.model_validate(raw_request)
            lease.cancel()
            return
    except (TypeError, ValueError, ValidationError, WebSocketDisconnect):
        lease.cancel()


def _socket_error_payload(
    *,
    code: str,
    message: str,
    busy: bool = False,
) -> dict[str, object]:
    return {
        "protocol_version": API_VERSION,
        "type": "busy" if busy else "error",
        "error": {"code": code, "message": message},
    }


def _terminal_payload(terminal: GenerationTerminal) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_version": API_VERSION,
        "type": {
            "completed": "done",
            "cancelled": "cancelled",
            "failed": "error",
        }[terminal.outcome],
        "state": terminal.state.to_dict(),
    }
    if terminal.completion_event is not None:
        payload["event"] = terminal.completion_event.to_dict()
    if terminal.error_code is not None and terminal.error_message is not None:
        payload["error"] = {
            "code": terminal.error_code,
            "message": terminal.error_message,
        }
    return payload


__all__ = [
    "API_VERSION",
    "CheckpointCatalogResponse",
    "CheckpointResponse",
    "ContextStateResponse",
    "DetokenizeRequest",
    "DetokenizeResponse",
    "ErrorDetail",
    "ErrorResponse",
    "GenerateRequest",
    "GenerationSettingsRequest",
    "HealthResponse",
    "LoadCheckpointRequest",
    "PublicCapabilities",
    "PublicConfigResponse",
    "PublicGenerationConfig",
    "PublicRuntimeConfig",
    "SessionResponse",
    "SessionStateResponse",
    "StopRequest",
    "TokenizeRequest",
    "TokenizeResponse",
    "WebService",
    "create_app",
]
