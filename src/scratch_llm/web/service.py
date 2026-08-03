"""Transport-neutral ownership of one local web chat checkpoint session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Protocol, TypeAlias
import unicodedata

from scratch_llm._validation import require_non_negative_integer
from scratch_llm.chat import ChatEngine, ChatState


MAX_CATALOG_ID_BYTES = 512
MAX_TEXT_BYTES = 16_384
MAX_TOKEN_IDS = 16_384

ServiceStatus: TypeAlias = Literal[
    "unloaded",
    "loading",
    "ready",
    "generating",
    "failed",
]


class WebService(Protocol):
    """Application-owned service lifecycle used by the FastAPI boundary."""

    async def startup(self) -> None:
        """Acquire resources needed while the application is serving."""

    async def shutdown(self) -> None:
        """Release resources acquired by ``startup``."""


class WebSessionService(WebService, Protocol):
    """Transport-facing session operations supplied to the FastAPI adapter."""

    @property
    def active_checkpoint_id(self) -> str | None:
        """Return the sanitized active catalog identity, if loaded."""

    def list_checkpoints(self) -> tuple[CheckpointCatalogEntry, ...]:
        """Return the sanitized checkpoint catalog."""

    async def load_checkpoint(self, checkpoint_id: str) -> PublicSessionState:
        """Atomically load one catalog checkpoint."""

    def reset(self) -> PublicSessionState:
        """Reset the active chat conversation."""

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize ordinary text with the active checkpoint."""

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Decode IDs with the active checkpoint."""


class SessionEngine(Protocol):
    """Narrow shared-ChatEngine surface consumed by the web session."""

    @property
    def max_context_tokens(self) -> int:
        """Return the checkpoint model context limit."""

    def get_state(self) -> ChatState:
        """Return an immutable engine snapshot."""

    def reset(self) -> None:
        """Clear the current conversation."""

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Encode ordinary text with the active checkpoint tokenizer."""

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Decode IDs with the active checkpoint tokenizer."""

    def close(self) -> None:
        """Release resources owned by this engine."""


EngineFactory: TypeAlias = Callable[[Path, str], SessionEngine]


class WebServiceError(RuntimeError):
    """Stable public failure raised by the checkpoint-session boundary."""

    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        self.code = code
        self.public_message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CheckpointCatalogEntry:
    """One sanitized client-selectable checkpoint identity."""

    checkpoint_id: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.checkpoint_id, "name": self.name}


@dataclass(frozen=True, slots=True)
class PublicContextState:
    """Non-content context-window state visible to local clients."""

    prompt_tokens: int
    max_tokens: int
    dropped_turns: int
    truncated_user_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "dropped_turns": self.dropped_turns,
            "truncated_user_tokens": self.truncated_user_tokens,
        }


@dataclass(frozen=True, slots=True)
class PublicSessionState:
    """Immutable session state with no host paths, content, model, or tensors."""

    status: ServiceStatus
    checkpoint_id: str | None
    checkpoint_step: int | None
    training_stage: Literal["sft"] | None
    device: str
    tokenizer_identity: str | None
    renderer_id: str | None
    context: PublicContextState | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_step": self.checkpoint_step,
            "training_stage": self.training_stage,
            "device": self.device,
            "tokenizer_identity": self.tokenizer_identity,
            "renderer_id": self.renderer_id,
            "context": None if self.context is None else self.context.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _CatalogRecord:
    entry: CheckpointCatalogEntry
    resolved_path: Path


def _create_engine(path: Path, device: str) -> SessionEngine:
    return ChatEngine(path, device=device)


class ChatSessionService:
    """Own and atomically mutate one active shared ``ChatEngine``."""

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        device: str = "cpu",
        initial_checkpoint_id: str | None = None,
        engine_factory: EngineFactory = _create_engine,
    ) -> None:
        if isinstance(checkpoint_dir, bytes):
            raise TypeError("checkpoint_dir must be a string or path-like value")
        try:
            self._checkpoint_dir = Path(checkpoint_dir)
        except TypeError as error:
            raise TypeError(
                "checkpoint_dir must be a string or path-like value"
            ) from error
        if not str(self._checkpoint_dir):
            raise ValueError("checkpoint_dir must not be empty")
        if not isinstance(device, str) or not device.strip():
            raise TypeError("device must be a non-empty string")
        if initial_checkpoint_id is not None and not isinstance(
            initial_checkpoint_id, str
        ):
            raise TypeError("initial_checkpoint_id must be a string or None")
        if not callable(engine_factory):
            raise TypeError("engine_factory must be callable")
        self._device = device
        self._initial_checkpoint_id = initial_checkpoint_id
        self._engine_factory = engine_factory
        self._engine: SessionEngine | None = None
        self._active_checkpoint_id: str | None = None
        self._status: ServiceStatus = "unloaded"
        self._mutation_lock = asyncio.Lock()
        self._load_task: asyncio.Task[SessionEngine] | None = None

    @property
    def status(self) -> ServiceStatus:
        """Return the current lifecycle state, including engine generation."""

        if self._status in {"loading", "failed"}:
            return self._status
        if self._engine is None:
            return "unloaded"
        try:
            if self._engine.get_state().is_generating:
                return "generating"
        except Exception:
            return "failed"
        return "ready"

    @property
    def active_checkpoint_id(self) -> str | None:
        return self._active_checkpoint_id

    async def startup(self) -> None:
        """Load the explicitly configured initial catalog identity, if any."""

        if self._initial_checkpoint_id is not None and self._engine is None:
            await self.load_checkpoint(self._initial_checkpoint_id)

    async def shutdown(self) -> None:
        """Drop and release the service-owned engine deterministically."""

        async with self._mutation_lock:
            engine = self._engine
            self._engine = None
            self._active_checkpoint_id = None
            self._status = "unloaded"
            if engine is not None:
                self._release_engine(engine)

    def list_checkpoints(self) -> tuple[CheckpointCatalogEntry, ...]:
        """Return a deterministic catalog containing no absolute host paths."""

        return tuple(record.entry for record in self._catalog_records())

    async def load_checkpoint(self, checkpoint_id: str) -> PublicSessionState:
        """Construct and validate a replacement before atomically swapping it."""

        if self.status in {"loading", "generating"}:
            raise _busy_error()
        async with self._mutation_lock:
            if self.status in {"loading", "generating"}:
                raise _busy_error()
            return await self._load_checkpoint(checkpoint_id)

    async def _load_checkpoint(self, checkpoint_id: str) -> PublicSessionState:
        record = self._find_catalog_record(checkpoint_id)
        prior_status: ServiceStatus = (
            "ready" if self._engine is not None else "unloaded"
        )
        self._status = "loading"
        load_task = asyncio.create_task(
            asyncio.to_thread(
                self._engine_factory,
                record.resolved_path,
                self._device,
            )
        )
        self._load_task = load_task
        replacement: SessionEngine | None = None
        try:
            replacement = await asyncio.shield(load_task)
            self._validate_replacement(replacement)
        except asyncio.CancelledError:
            self._status = prior_status
            load_task.add_done_callback(self._release_cancelled_result)
            raise
        except Exception as error:
            self._status = "failed"
            if replacement is not None:
                self._release_engine(replacement)
            raise WebServiceError(
                "checkpoint_load_failed",
                "checkpoint could not be loaded",
                status_code=422,
            ) from error
        finally:
            if self._load_task is load_task:
                self._load_task = None

        previous = self._engine
        self._engine = replacement
        self._active_checkpoint_id = record.entry.checkpoint_id
        self._status = "ready"
        if previous is not None and previous is not replacement:
            self._release_engine(previous)
        return self.get_state()

    def reset(self) -> PublicSessionState:
        """Delegate conversation reset to the active shared engine."""

        engine = self._require_available_engine()
        try:
            engine.reset()
        except Exception as error:
            raise WebServiceError(
                "session_operation_failed",
                "session reset failed",
                status_code=500,
            ) from error
        self._status = "ready"
        return self.get_state()

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Encode bounded ordinary text with no implicit special tokens."""

        engine = self._require_available_engine()
        if not isinstance(text, str):
            raise WebServiceError(
                "invalid_text",
                "text must be a string",
                status_code=422,
            )
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise WebServiceError(
                "request_too_large",
                "text exceeds the tokenizer request limit",
                status_code=413,
            )
        try:
            token_ids = _normalize_token_ids(engine.tokenize(text))
        except WebServiceError:
            raise
        except Exception as error:
            raise WebServiceError(
                "invalid_text",
                "text could not be tokenized",
                status_code=422,
            ) from error
        if len(token_ids) > MAX_TOKEN_IDS:
            raise WebServiceError(
                "request_too_large",
                "tokenized text exceeds the response limit",
                status_code=413,
            )
        return token_ids

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Decode a bounded validated ID sequence with the active tokenizer."""

        engine = self._require_available_engine()
        normalized = _normalize_token_ids(token_ids)
        if len(normalized) > MAX_TOKEN_IDS:
            raise WebServiceError(
                "request_too_large",
                "token IDs exceed the tokenizer request limit",
                status_code=413,
            )
        try:
            text = engine.detokenize(normalized)
        except Exception as error:
            raise WebServiceError(
                "invalid_token_ids",
                "token IDs are invalid for the active tokenizer",
                status_code=422,
            ) from error
        if not isinstance(text, str):
            raise WebServiceError(
                "session_operation_failed",
                "tokenizer returned an invalid result",
                status_code=500,
            )
        return text

    def get_state(self) -> PublicSessionState:
        """Return an immutable public view without transcript content or paths."""

        engine = self._engine
        if engine is None:
            status: ServiceStatus = "failed" if self._status == "failed" else "unloaded"
            return PublicSessionState(
                status=status,
                checkpoint_id=None,
                checkpoint_step=None,
                training_stage=None,
                device=self._device,
                tokenizer_identity=None,
                renderer_id=None,
                context=None,
            )
        try:
            state = engine.get_state()
            context_limit = engine.max_context_tokens
        except Exception as error:
            raise WebServiceError(
                "session_unavailable",
                "checkpoint session is unavailable",
                status_code=503,
            ) from error
        return PublicSessionState(
            status=self.status,
            checkpoint_id=self._active_checkpoint_id,
            checkpoint_step=state.checkpoint_step,
            training_stage=state.training_stage,
            device=state.device,
            tokenizer_identity=state.tokenizer_identity,
            renderer_id=state.renderer_id,
            context=PublicContextState(
                prompt_tokens=state.prompt_token_count,
                max_tokens=context_limit,
                dropped_turns=state.dropped_turn_count,
                truncated_user_tokens=state.truncated_user_token_count,
            ),
        )

    def _require_available_engine(self) -> SessionEngine:
        status = self.status
        if status in {"loading", "generating"}:
            raise _busy_error()
        if self._engine is None:
            raise WebServiceError(
                "checkpoint_not_loaded",
                "load a checkpoint first",
                status_code=409,
            )
        return self._engine

    def _find_catalog_record(self, checkpoint_id: object) -> _CatalogRecord:
        if not _is_safe_catalog_id(checkpoint_id):
            raise _checkpoint_not_found()
        for record in self._catalog_records():
            if record.entry.checkpoint_id == checkpoint_id:
                return record
        raise _checkpoint_not_found()

    def _catalog_records(self) -> tuple[_CatalogRecord, ...]:
        try:
            root = self._checkpoint_dir.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return ()
        if not root.is_dir():
            return ()

        records: list[_CatalogRecord] = []
        try:
            walker = os.walk(root, topdown=True, onerror=lambda _error: None)
            for directory, child_directories, filenames in walker:
                child_directories.sort()
                filenames.sort()
                parent = Path(directory)
                for filename in filenames:
                    candidate = parent / filename
                    if candidate.suffix != ".pt":
                        continue
                    try:
                        relative = candidate.relative_to(root).as_posix()
                        if not _is_safe_catalog_id(relative):
                            continue
                        resolved = candidate.resolve(strict=True)
                        resolved.relative_to(root)
                        if not resolved.is_file():
                            continue
                        with resolved.open("rb"):
                            pass
                    except (OSError, RuntimeError, ValueError):
                        continue
                    records.append(
                        _CatalogRecord(
                            entry=CheckpointCatalogEntry(
                                checkpoint_id=relative,
                                name=candidate.name,
                            ),
                            resolved_path=resolved,
                        )
                    )
        except OSError:
            return ()
        return tuple(sorted(records, key=lambda record: record.entry.checkpoint_id))

    @staticmethod
    def _validate_replacement(engine: SessionEngine) -> None:
        state = engine.get_state()
        if state.training_stage != "sft":
            raise ValueError("replacement engine is not an SFT session")
        limit = engine.max_context_tokens
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("replacement engine has no valid context limit")

    @staticmethod
    def _release_engine(engine: SessionEngine) -> None:
        try:
            engine.close()
        except Exception:
            pass

    def _release_cancelled_result(
        self,
        task: asyncio.Task[SessionEngine],
    ) -> None:
        if task.cancelled():
            return
        try:
            engine = task.result()
        except BaseException:
            return
        self._release_engine(engine)


def _normalize_token_ids(token_ids: object) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise WebServiceError(
            "invalid_token_ids",
            "token IDs must be a sequence of integers",
            status_code=422,
        )
    normalized: list[int] = []
    for token_id in token_ids:
        try:
            normalized.append(require_non_negative_integer(token_id, name="token ID"))
        except (TypeError, ValueError):
            raise WebServiceError(
                "invalid_token_ids",
                "token IDs must be non-negative integers",
                status_code=422,
            ) from None
    return tuple(normalized)


def _is_safe_catalog_id(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        encoded = value.encode("utf-8")
        path = Path(value)
    except (UnicodeError, ValueError):
        return False
    if len(encoded) > MAX_CATALOG_ID_BYTES:
        return False
    if any(unicodedata.category(character).startswith("C") for character in value):
        return False
    return (
        not path.is_absolute()
        and path.suffix == ".pt"
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.as_posix() == value
    )


def _busy_error() -> WebServiceError:
    return WebServiceError(
        "busy",
        "checkpoint session is busy",
        status_code=409,
    )


def _checkpoint_not_found() -> WebServiceError:
    return WebServiceError(
        "checkpoint_not_found",
        "checkpoint is not available in the configured catalog",
        status_code=404,
    )


__all__ = [
    "MAX_CATALOG_ID_BYTES",
    "MAX_TEXT_BYTES",
    "MAX_TOKEN_IDS",
    "ChatSessionService",
    "CheckpointCatalogEntry",
    "EngineFactory",
    "PublicContextState",
    "PublicSessionState",
    "ServiceStatus",
    "SessionEngine",
    "WebService",
    "WebServiceError",
    "WebSessionService",
]
