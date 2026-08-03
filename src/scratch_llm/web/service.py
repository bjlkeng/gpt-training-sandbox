"""Transport-neutral ownership of one local web chat checkpoint session."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Literal, Protocol, TypeAlias
import unicodedata

from scratch_llm._validation import require_non_negative_integer
from scratch_llm.chat import (
    SUPPORTED_CHAT_RENDERER_IDS,
    ChatEngine,
    ChatEventTracker,
    ChatState,
    ChatTrackingSession,
    TokenEvent,
    close_token_stream,
)
from scratch_llm.config import GenerationConfig, apply_generation_overrides
from scratch_llm.diagnostics.accelerator_memory import (
    AcceleratorMemorySnapshot,
    collect_accelerator_memory,
    reset_accelerator_memory_peak,
)
from scratch_llm.web.inspection import (
    GenerationDebug,
    GenerationMetrics,
    IdentityFactory,
    SessionAggregate,
    SessionMetricsBoundary,
    finalize_generation_metrics,
    new_public_identity,
)


MAX_CATALOG_ID_BYTES = 512
MAX_GENERATION_TOKENS = 4_096
MAX_TEMPERATURE = 10.0
MAX_TEXT_BYTES = 16_384
MAX_TOP_K = 100_000
MAX_TOKEN_IDS = 16_384
MIN_SEED = -(2**63)
MAX_SEED = 2**63 - 1

ServiceStatus: TypeAlias = Literal[
    "unloaded",
    "loading",
    "ready",
    "generating",
    "failed",
]
MemoryCollector: TypeAlias = Callable[[str], AcceleratorMemorySnapshot]
MemoryPeakReset: TypeAlias = Callable[[str], bool]


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

    def list_renderers(self) -> tuple[str, ...]:
        """Return server-supported prompt renderer identities."""

    async def load_checkpoint(self, checkpoint_id: str) -> PublicSessionState:
        """Atomically load one catalog checkpoint."""

    def reset(self) -> PublicSessionState:
        """Reset the active chat conversation."""

    def get_state(self) -> PublicSessionState:
        """Return the current sanitized session state."""

    def select_renderer(self, renderer_id: str) -> RendererSelection:
        """Acknowledge one server-supported renderer selection."""

    def export_transcript(self) -> bytes:
        """Return canonical transcript bytes without accepting a path."""

    def get_session_aggregate(self) -> SessionAggregate:
        """Return aggregate values with no prompt or response content."""

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize ordinary text with the active checkpoint."""

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Decode IDs with the active checkpoint."""

    async def start_generation(
        self,
        message: str,
        overrides: GenerationOverrides,
        *,
        include_debug: bool = False,
    ) -> GenerationLease:
        """Acquire the sole generation lease without queueing."""


class SessionEngine(Protocol):
    """Narrow shared-ChatEngine surface consumed by the web session."""

    @property
    def max_context_tokens(self) -> int:
        """Return the checkpoint model context limit."""

    @property
    def default_generation_config(self) -> GenerationConfig:
        """Return detached checkpoint generation defaults."""

    def get_state(self) -> ChatState:
        """Return an immutable engine snapshot."""

    def reset(self) -> None:
        """Clear the current conversation."""

    def append_user_message(self, text: str) -> None:
        """Append and render one pending user turn."""

    def generate_stream(
        self,
        config: GenerationConfig | None = None,
    ) -> Iterator[TokenEvent]:
        """Return the shared synchronous token iterator."""

    def rollback_last_turn(self, *, include_completed: bool = False) -> None:
        """Discard the service-owned user transaction after interruption."""

    def get_pending_prompt_token_ids(self) -> tuple[int, ...]:
        """Return exact renderer-owned prompt IDs for explicit debugging."""

    def get_last_completed_message(
        self,
        role: Literal["user", "assistant"],
    ) -> str:
        """Return only the requested side of the latest committed turn."""

    def export_transcript_bytes(self) -> bytes:
        """Return one canonical completed conversation record."""

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
class GenerationOverrides:
    """Optional client overrides applied to checkpoint generation defaults."""

    temperature: float | None = None
    top_k: int | None = None
    max_new_tokens: int | None = None
    seed: int | None = None


GenerationOutcome: TypeAlias = Literal["completed", "cancelled", "failed"]


@dataclass(frozen=True, slots=True)
class RendererSelection:
    """Acknowledged renderer state and whether history was reset."""

    state: PublicSessionState
    history_reset: bool


@dataclass(frozen=True, slots=True)
class GenerationTerminal:
    """Exactly one terminal outcome for a service-owned generation lease."""

    outcome: GenerationOutcome
    state: PublicSessionState
    aggregate: SessionAggregate
    completion_event: TokenEvent | None = None
    metrics: GenerationMetrics | None = None
    debug: GenerationDebug | None = None
    error_code: str | None = None
    error_message: str | None = None


GenerationStreamItem: TypeAlias = TokenEvent | GenerationTerminal


class GenerationLease(AsyncIterator[GenerationStreamItem]):
    """Async event bridge around one cooperative worker-thread generation."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        session_id: str,
        turn_id: str,
    ) -> None:
        self._loop = loop
        self.session_id = session_id
        self.turn_id = turn_id
        self._queue: asyncio.Queue[GenerationStreamItem] = asyncio.Queue()
        self._done: asyncio.Future[GenerationTerminal] = loop.create_future()
        self._cancel = threading.Event()
        self._terminal_consumed = False
        self._worker: asyncio.Task[None] | None = None

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def done(self) -> bool:
        return self._done.done()

    def cancel(self) -> None:
        """Request cooperative cancellation without waiting for the lease."""

        self._cancel.set()

    async def wait(self) -> GenerationTerminal:
        """Wait for cleanup and return the terminal outcome."""

        return await asyncio.shield(self._done)

    def __aiter__(self) -> GenerationLease:
        return self

    async def __anext__(self) -> GenerationStreamItem:
        if self._terminal_consumed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if isinstance(item, GenerationTerminal):
            self._terminal_consumed = True
        return item

    def _start(self, worker: Callable[[], None]) -> None:
        self._worker = asyncio.create_task(asyncio.to_thread(worker))

    def _emit_from_worker(self, event: TokenEvent) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def _deliver_terminal(self, terminal: GenerationTerminal) -> None:
        self._queue.put_nowait(terminal)
        if not self._done.done():
            self._done.set_result(terminal)


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
        identity_factory: IdentityFactory = new_public_identity,
        chat_tracking: ChatEventTracker | None = None,
        reset_memory_peak: MemoryPeakReset = reset_accelerator_memory_peak,
        collect_memory: MemoryCollector = collect_accelerator_memory,
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
        if chat_tracking is not None and not isinstance(
            chat_tracking, ChatEventTracker
        ):
            raise TypeError("chat_tracking must be a ChatEventTracker or None")
        if not callable(reset_memory_peak):
            raise TypeError("reset_memory_peak must be callable")
        if not callable(collect_memory):
            raise TypeError("collect_memory must be callable")
        self._device = device
        self._initial_checkpoint_id = initial_checkpoint_id
        self._engine_factory = engine_factory
        self._session_metrics = SessionMetricsBoundary(identity_factory)
        self._chat_tracking_session: ChatTrackingSession | None = (
            None
            if chat_tracking is None
            else chat_tracking.start_session(
                "web",
                session_id=self._session_metrics.session_id,
            )
        )
        self._reset_memory_peak = reset_memory_peak
        self._collect_memory = collect_memory
        self._engine: SessionEngine | None = None
        self._active_checkpoint_id: str | None = None
        self._status: ServiceStatus = "unloaded"
        self._mutation_lock = asyncio.Lock()
        self._load_task: asyncio.Task[SessionEngine] | None = None
        self._generation_lease: GenerationLease | None = None

    @property
    def status(self) -> ServiceStatus:
        """Return the current lifecycle state, including engine generation."""

        if self._status in {"loading", "generating", "failed"}:
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

        lease = self._generation_lease
        if lease is not None:
            lease.cancel()
            await lease.wait()
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

    def list_renderers(self) -> tuple[str, ...]:
        """Return the renderer identities implemented by the shared chat layer."""

        return SUPPORTED_CHAT_RENDERER_IDS

    def select_renderer(self, renderer_id: str) -> RendererSelection:
        """Validate a server-owned renderer selection without templating here."""

        if renderer_id not in SUPPORTED_CHAT_RENDERER_IDS:
            raise WebServiceError(
                "unsupported_renderer",
                "prompt renderer is not supported",
                status_code=422,
            )
        self._require_available_engine()
        state = self.get_state()
        if state.renderer_id is None:
            raise WebServiceError(
                "checkpoint_not_loaded",
                "load a checkpoint first",
                status_code=409,
            )
        if state.renderer_id != renderer_id:
            raise WebServiceError(
                "unsupported_renderer",
                "prompt renderer is not supported by the loaded checkpoint",
                status_code=422,
            )
        return RendererSelection(state=state, history_reset=False)

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
            self._status = prior_status if self._engine is not None else "failed"
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
        self._session_metrics.start_new_session()
        self._reset_chat_tracking_session()
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
        self._session_metrics.start_new_session()
        self._reset_chat_tracking_session()
        return self.get_state()

    def _reset_chat_tracking_session(self) -> None:
        if self._chat_tracking_session is not None:
            self._chat_tracking_session.reset(
                session_id=self._session_metrics.session_id
            )

    def export_transcript(self) -> bytes:
        """Return one canonical download without exposing a write path."""

        engine = self._require_available_engine()
        try:
            payload = engine.export_transcript_bytes()
        except Exception as error:
            raise WebServiceError(
                "transcript_unavailable",
                "a completed conversation is required for transcript export",
                status_code=409,
            ) from error
        if not isinstance(payload, bytes):
            raise WebServiceError(
                "session_operation_failed",
                "transcript export failed",
                status_code=500,
            )
        return payload

    def get_session_aggregate(
        self,
        *,
        turn_id: str | None = None,
    ) -> SessionAggregate:
        """Return cumulative non-content values for the current session."""

        return self._session_metrics.snapshot(turn_id=turn_id)

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

    async def start_generation(
        self,
        message: str,
        overrides: GenerationOverrides,
        *,
        include_debug: bool = False,
    ) -> GenerationLease:
        """Validate and acquire the sole generation lease without queueing."""

        _validate_generation_message(message)
        if not isinstance(overrides, GenerationOverrides):
            raise _invalid_generation_request()
        if not isinstance(include_debug, bool):
            raise _invalid_generation_request()
        engine = self._require_available_engine()
        settings = _resolve_generation_config(engine, overrides)
        if self._generation_lease is not None or self.status != "ready":
            raise _busy_error()
        loop = asyncio.get_running_loop()
        turn_id = self._session_metrics.new_turn_id()
        lease = GenerationLease(
            loop,
            session_id=self._session_metrics.session_id,
            turn_id=turn_id,
        )
        self._generation_lease = lease
        self._status = "generating"
        lease._start(
            lambda: self._run_generation(
                lease,
                engine,
                message,
                settings,
                include_debug,
            )
        )
        return lease

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

    def _try_reset_memory_peak(self) -> bool:
        try:
            return self._reset_memory_peak(self._device)
        except Exception:
            return False

    def _try_collect_peak_memory(self, peak_was_reset: bool) -> float | None:
        if not peak_was_reset:
            return None
        try:
            snapshot = self._collect_memory(self._device)
        except Exception:
            return None
        if (
            not isinstance(snapshot, AcceleratorMemorySnapshot)
            or not snapshot.available
        ):
            return None
        return snapshot.peak_allocated_mib

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
        if state.renderer_id not in SUPPORTED_CHAT_RENDERER_IDS:
            raise ValueError("replacement engine uses an unsupported renderer")
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

    def _run_generation(
        self,
        lease: GenerationLease,
        engine: SessionEngine,
        message: str,
        settings: GenerationConfig,
        include_debug: bool,
    ) -> None:
        iterator: Iterator[TokenEvent] | None = None
        appended = False
        outcome: GenerationOutcome = "failed"
        completion_event: TokenEvent | None = None
        first_sample_seconds: float | None = None
        prompt_token_ids: tuple[int, ...] = ()
        generated_token_ids: list[int] = []
        memory_peak_reset = self._try_reset_memory_peak()
        try:
            if lease.cancel_requested:
                outcome = "cancelled"
                return
            engine.append_user_message(message)
            appended = True
            if include_debug:
                prompt_token_ids = engine.get_pending_prompt_token_ids()
            if lease.cancel_requested:
                outcome = "cancelled"
                return
            iterator = engine.generate_stream(settings)
            while True:
                if lease.cancel_requested:
                    outcome = "cancelled"
                    break
                try:
                    event = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "shared ChatEngine ended without a completion event"
                    ) from error
                if not isinstance(event, TokenEvent):
                    raise RuntimeError("shared ChatEngine emitted an invalid event")
                if lease.cancel_requested:
                    outcome = "cancelled"
                    break
                if event.type == "complete":
                    completion_event = event
                    outcome = "completed"
                    break
                if event.type == "token":
                    if first_sample_seconds is None:
                        first_sample_seconds = event.elapsed_seconds
                    if include_debug:
                        generated_token_ids.extend(event.token_ids)
                lease._emit_from_worker(event)
        except BaseException:
            outcome = "failed"
        finally:
            if iterator is not None:
                try:
                    close_token_stream(iterator)
                except BaseException:
                    if outcome == "completed":
                        outcome = "failed"
                        completion_event = None
            if appended and outcome != "completed":
                try:
                    engine.rollback_last_turn(include_completed=True)
                except BaseException:
                    outcome = "failed"
            metrics = (
                finalize_generation_metrics(
                    completion_event,
                    first_sample_seconds=first_sample_seconds,
                    peak_memory_mib=self._try_collect_peak_memory(memory_peak_reset),
                )
                if outcome == "completed" and completion_event is not None
                else None
            )
            debug = (
                GenerationDebug(
                    prompt_token_ids=prompt_token_ids,
                    generated_token_ids=tuple(generated_token_ids),
                    completion_reason=(
                        None
                        if completion_event is None
                        else completion_event.completion_reason
                    ),
                    stop_token_id=(
                        None
                        if completion_event is None
                        else completion_event.stop_token_id
                    ),
                )
                if include_debug
                else None
            )
            lease._loop.call_soon_threadsafe(
                self._finish_generation,
                lease,
                outcome,
                completion_event,
                metrics,
                debug,
            )

    def _finish_generation(
        self,
        lease: GenerationLease,
        outcome: GenerationOutcome,
        completion_event: TokenEvent | None,
        metrics: GenerationMetrics | None,
        debug: GenerationDebug | None,
    ) -> None:
        if self._generation_lease is lease:
            self._generation_lease = None
            self._status = "ready" if self._engine is not None else "unloaded"
        if (
            outcome == "completed"
            and completion_event is not None
            and self._chat_tracking_session is not None
            and self._engine is not None
        ):
            try:
                engine = self._engine
                self._chat_tracking_session.record_completed_turn(
                    completion_event,
                    prompt_factory=lambda: engine.get_last_completed_message("user"),
                    response_factory=lambda: engine.get_last_completed_message(
                        "assistant"
                    ),
                    turn_id=lease.turn_id,
                )
            except Exception:
                outcome = "failed"
                completion_event = None
                metrics = None
                try:
                    self._engine.rollback_last_turn(include_completed=True)
                except Exception:
                    pass
        try:
            state = self.get_state()
        except WebServiceError:
            state = PublicSessionState(
                status="failed",
                checkpoint_id=self._active_checkpoint_id,
                checkpoint_step=None,
                training_stage=None,
                device=self._device,
                tokenizer_identity=None,
                renderer_id=None,
                context=None,
            )
            outcome = "failed"
            completion_event = None
            metrics = None
        if outcome == "completed" and metrics is not None:
            self._session_metrics.record(metrics)
        aggregate = self.get_session_aggregate(turn_id=lease.turn_id)
        terminal = GenerationTerminal(
            outcome=outcome,
            state=state,
            aggregate=aggregate,
            completion_event=completion_event,
            metrics=metrics,
            debug=debug,
            error_code="generation_failed" if outcome == "failed" else None,
            error_message="generation failed" if outcome == "failed" else None,
        )
        lease._deliver_terminal(terminal)


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


def _validate_generation_message(message: object) -> None:
    if not isinstance(message, str):
        raise _invalid_generation_request()
    if len(message.encode("utf-8")) > MAX_TEXT_BYTES:
        raise WebServiceError(
            "request_too_large",
            "user message exceeds the generation request limit",
            status_code=413,
        )


def _resolve_generation_config(
    engine: SessionEngine,
    overrides: GenerationOverrides,
) -> GenerationConfig:
    try:
        settings = apply_generation_overrides(
            engine.default_generation_config,
            {
                "temperature": overrides.temperature,
                "top_k": overrides.top_k,
                "max_new_tokens": overrides.max_new_tokens,
                "seed": overrides.seed,
            },
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise _invalid_generation_request() from error
    if (
        settings.temperature > MAX_TEMPERATURE
        or settings.max_new_tokens > MAX_GENERATION_TOKENS
        or (settings.top_k is not None and settings.top_k > MAX_TOP_K)
        or (settings.seed is not None and not MIN_SEED <= settings.seed <= MAX_SEED)
        or settings.top_p is not None
    ):
        raise _invalid_generation_request()
    return settings


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


def _invalid_generation_request() -> WebServiceError:
    return WebServiceError(
        "invalid_generation_request",
        "generation request is invalid",
        status_code=422,
    )


def _checkpoint_not_found() -> WebServiceError:
    return WebServiceError(
        "checkpoint_not_found",
        "checkpoint is not available in the configured catalog",
        status_code=404,
    )


__all__ = [
    "MAX_CATALOG_ID_BYTES",
    "MAX_GENERATION_TOKENS",
    "MAX_SEED",
    "MAX_TEMPERATURE",
    "MAX_TEXT_BYTES",
    "MAX_TOP_K",
    "MAX_TOKEN_IDS",
    "MIN_SEED",
    "ChatSessionService",
    "CheckpointCatalogEntry",
    "EngineFactory",
    "GenerationDebug",
    "GenerationLease",
    "GenerationMetrics",
    "GenerationOutcome",
    "GenerationOverrides",
    "GenerationStreamItem",
    "GenerationTerminal",
    "PublicContextState",
    "PublicSessionState",
    "RendererSelection",
    "SessionAggregate",
    "ServiceStatus",
    "SessionEngine",
    "WebService",
    "WebServiceError",
    "WebSessionService",
]
