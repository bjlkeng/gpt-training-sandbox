"""Privacy-gated structured tracking for completed chat turns."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Literal, TypeAlias
import unicodedata
from uuid import uuid4
import warnings

from scratch_llm.chat.engine import TokenEvent
from scratch_llm.tracking import Tracker


ChatTransport: TypeAlias = Literal["cli", "web"]
ContentFactory: TypeAlias = Callable[[], str]
IdentityFactory: TypeAlias = Callable[[str], str]
WarningSink: TypeAlias = Callable[[str], None]
_MAX_PUBLIC_ID_BYTES = 256


class ChatPrivacyWarning(UserWarning):
    """Raw chat content was explicitly enabled for tracking."""


def new_public_identity(kind: str) -> str:
    """Return an opaque local run, session, or turn identity."""

    return f"{kind}-{uuid4().hex}"


def create_public_identity(factory: Callable[[str], object], kind: str) -> str:
    """Create and validate one bounded, printable opaque identity."""

    if not callable(factory):
        raise TypeError("identity_factory must be callable")
    if not isinstance(kind, str) or not kind:
        raise ValueError("identity kind must be a non-empty string")
    try:
        identity = factory(kind)
    except Exception as error:
        raise RuntimeError(f"failed to create {kind} identity") from error
    return _validate_supplied_identity(identity, kind)


def _validate_supplied_identity(identity: object, kind: str) -> str:
    if (
        not isinstance(identity, str)
        or not identity
        or len(identity.encode("utf-8")) > _MAX_PUBLIC_ID_BYTES
        or any(
            unicodedata.category(character).startswith("C") for character in identity
        )
    ):
        raise ValueError(f"{kind} identity is invalid")
    return identity


def _emit_privacy_warning(message: str) -> None:
    warnings.warn(message, ChatPrivacyWarning, stacklevel=3)


class ChatEventTracker:
    """Construct raw fields only behind their matching explicit opt-in."""

    def __init__(
        self,
        tracker: Tracker,
        *,
        run_id: str,
        log_prompts: bool = False,
        log_responses: bool = False,
        identity_factory: IdentityFactory = new_public_identity,
        warning_sink: WarningSink = _emit_privacy_warning,
    ) -> None:
        if not isinstance(tracker, Tracker):
            raise TypeError("tracker must implement the Tracker contract")
        if not isinstance(log_prompts, bool):
            raise TypeError("log_prompts must be a boolean")
        if not isinstance(log_responses, bool):
            raise TypeError("log_responses must be a boolean")
        if not callable(identity_factory):
            raise TypeError("identity_factory must be callable")
        if not callable(warning_sink):
            raise TypeError("warning_sink must be callable")
        self._tracker = tracker
        self._run_id = _validate_supplied_identity(run_id, "run")
        self._log_prompts = log_prompts
        self._log_responses = log_responses
        self._identity_factory = identity_factory
        self._lock = threading.Lock()
        if log_prompts or log_responses:
            enabled = " and ".join(
                label
                for label, selected in (
                    ("raw prompts", log_prompts),
                    ("raw responses", log_responses),
                )
                if selected
            )
            warning_sink(
                "Privacy warning: explicit chat tracking will record "
                f"{enabled} in local JSONL and every enabled tracker backend."
            )

    def start_session(
        self,
        transport: ChatTransport,
        *,
        session_id: str | None = None,
    ) -> ChatTrackingSession:
        """Create independently identified state for one CLI or web session."""

        if transport not in {"cli", "web"}:
            raise ValueError("transport must be 'cli' or 'web'")
        identity = (
            self._new_identity("session")
            if session_id is None
            else _validate_supplied_identity(session_id, "session")
        )
        return ChatTrackingSession(self, transport=transport, session_id=identity)

    def _new_identity(self, kind: str) -> str:
        return create_public_identity(self._identity_factory, kind)

    def _record_completed_turn(
        self,
        session: ChatTrackingSession,
        completion: TokenEvent,
        *,
        prompt_factory: ContentFactory,
        response_factory: ContentFactory,
        turn_id: str | None,
    ) -> str:
        if not isinstance(completion, TokenEvent) or completion.type != "complete":
            raise ValueError("completed chat tracking requires a complete TokenEvent")
        if not callable(prompt_factory):
            raise TypeError("prompt_factory must be callable")
        if not callable(response_factory):
            raise TypeError("response_factory must be callable")
        with self._lock:
            active_turn_id = (
                self._new_identity("turn")
                if turn_id is None
                else _validate_supplied_identity(turn_id, "turn")
            )
            next_turn_count = session._turn_count + 1
            next_generated_tokens = (
                session._generated_tokens + completion.generated_token_count
            )
            record: dict[str, object] = {
                "chat/run_id": self._run_id,
                "chat/session_id": session._session_id,
                "chat/turn_id": active_turn_id,
                "chat/transport": session._transport,
                "chat/session_turn_count": next_turn_count,
                "chat/session_generated_tokens": next_generated_tokens,
                "chat/prompt_tokens": completion.prompt_token_count,
                "chat/generated_tokens": completion.generated_token_count,
                "chat/sampled_tokens": completion.sampled_token_count,
                "chat/generation_seconds": completion.elapsed_seconds,
                "chat/completion_reason": completion.completion_reason,
                "chat/stop_token_id": completion.stop_token_id,
            }
            if self._log_prompts:
                record["chat/prompt"] = _content(prompt_factory, "prompt")
            if self._log_responses:
                record["chat/response"] = _content(response_factory, "response")
            self._tracker.log(record)
            session._turn_count = next_turn_count
            session._generated_tokens = next_generated_tokens
            return active_turn_id


class ChatTrackingSession:
    """One resettable session identity routed through a shared privacy gate."""

    def __init__(
        self,
        owner: ChatEventTracker,
        *,
        transport: ChatTransport,
        session_id: str,
    ) -> None:
        self._owner = owner
        self._transport = transport
        self._session_id = session_id
        self._turn_count = 0
        self._generated_tokens = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def reset(self, *, session_id: str | None = None) -> str:
        """Rotate identity and counters without emitting a content record."""

        with self._owner._lock:
            self._session_id = (
                self._owner._new_identity("session")
                if session_id is None
                else _validate_supplied_identity(session_id, "session")
            )
            self._turn_count = 0
            self._generated_tokens = 0
            return self._session_id

    def record_completed_turn(
        self,
        completion: TokenEvent,
        *,
        prompt_factory: ContentFactory,
        response_factory: ContentFactory,
        turn_id: str | None = None,
    ) -> str:
        """Record one committed turn, conditionally resolving each raw side."""

        return self._owner._record_completed_turn(
            self,
            completion,
            prompt_factory=prompt_factory,
            response_factory=response_factory,
            turn_id=turn_id,
        )


def _content(factory: ContentFactory, label: str) -> str:
    try:
        content = factory()
    except Exception as error:
        raise RuntimeError(f"could not obtain completed {label} content") from error
    if not isinstance(content, str):
        raise TypeError(f"completed {label} content must be a string")
    return content


__all__ = [
    "ChatEventTracker",
    "ChatPrivacyWarning",
    "ChatTrackingSession",
    "ChatTransport",
    "ContentFactory",
    "IdentityFactory",
    "WarningSink",
    "create_public_identity",
    "new_public_identity",
]
