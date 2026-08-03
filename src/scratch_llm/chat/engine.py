"""Transport-neutral chat inference state and streaming lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
import codecs
from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
import time
from typing import Literal, TypeAlias

import torch
from torch import nn

from scratch_llm._validation import (
    require_non_negative_integer,
)
from scratch_llm.chat.conversation import (
    AssistantMessage,
    Conversation,
    UserMessage,
)
from scratch_llm.chat.rendering import (
    CHAT_RENDERER_ID,
    CompletionPrompt,
    render_completion_prompt,
)
from scratch_llm.config import GenerationConfig
from scratch_llm.generation import (
    CompletionReason,
    GeneratedToken,
    GenerationComplete,
    stream_generate_sequence,
)
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.utils import get_device


ChatStatus: TypeAlias = Literal[
    "idle",
    "awaiting_assistant",
    "generating",
    "completed",
    "cancelled",
    "failed",
]
TokenEventType: TypeAlias = Literal["start", "token", "complete"]
_HistoryMessage: TypeAlias = UserMessage | AssistantMessage
_CheckpointLoader: TypeAlias = Callable[..., object]


def _load_model_checkpoint(path: Path, *, device: str) -> object:
    """Import the checkpoint layer lazily to keep chat-domain imports acyclic."""

    from scratch_llm.training.checkpoint import load_model_checkpoint

    return load_model_checkpoint(path, device=device)


class ChatEngineError(RuntimeError):
    """A checkpoint or conversation cannot complete a chat transition."""


@dataclass(frozen=True, slots=True)
class ChatState:
    """Immutable snapshot of one engine's transcript and generation lifecycle."""

    checkpoint_path: str
    checkpoint_step: int
    training_stage: Literal["sft"]
    device: str
    tokenizer_identity: str
    renderer_id: str
    status: ChatStatus
    messages: tuple[_HistoryMessage, ...]
    prompt_token_count: int
    generated_token_count: int
    sampled_token_count: int
    generation_seconds: float | None
    completion_reason: CompletionReason | None
    stop_token_id: int | None

    @property
    def is_generating(self) -> bool:
        """Return whether a generation transaction currently owns the engine."""

        return self.status == "generating"

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible snapshot."""

        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "training_stage": self.training_stage,
            "device": self.device,
            "tokenizer_identity": self.tokenizer_identity,
            "renderer_id": self.renderer_id,
            "status": self.status,
            "is_generating": self.is_generating,
            "messages": [_message_payload(message) for message in self.messages],
            "prompt_token_count": self.prompt_token_count,
            "generated_token_count": self.generated_token_count,
            "sampled_token_count": self.sampled_token_count,
            "generation_seconds": self.generation_seconds,
            "completion_reason": self.completion_reason,
            "stop_token_id": self.stop_token_id,
        }


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One immutable, JSON-compatible chat streaming event."""

    type: TokenEventType
    token_ids: tuple[int, ...]
    text_delta: str
    prompt_token_count: int
    generated_token_count: int
    sampled_token_count: int
    elapsed_seconds: float
    completion_reason: CompletionReason | None
    stop_token_id: int | None

    def __post_init__(self) -> None:
        if self.type not in {"start", "token", "complete"}:
            raise ValueError("type must be 'start', 'token', or 'complete'")
        if not isinstance(self.token_ids, tuple):
            raise TypeError("token_ids must be a tuple")
        for position, token_id in enumerate(self.token_ids):
            require_non_negative_integer(token_id, name=f"token_ids[{position}]")
        if not isinstance(self.text_delta, str):
            raise TypeError("text_delta must be a string")
        require_non_negative_integer(
            self.prompt_token_count,
            name="prompt_token_count",
        )
        require_non_negative_integer(
            self.generated_token_count,
            name="generated_token_count",
        )
        require_non_negative_integer(
            self.sampled_token_count,
            name="sampled_token_count",
        )
        if not isinstance(self.elapsed_seconds, (int, float)) or isinstance(
            self.elapsed_seconds, bool
        ):
            raise TypeError("elapsed_seconds must be a number")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.type == "start":
            if self.token_ids or self.text_delta:
                raise ValueError("start events cannot contain visible output")
            if self.generated_token_count or self.sampled_token_count:
                raise ValueError("start event token counts must be zero")
        elif self.type == "token":
            if len(self.token_ids) != 1:
                raise ValueError("token events must contain exactly one token ID")
            if self.generated_token_count != self.sampled_token_count:
                raise ValueError("visible token event counts must match")
        elif self.token_ids:
            raise ValueError("complete events cannot contain visible token IDs")

        if self.type == "complete":
            if self.completion_reason is None:
                raise ValueError("complete events require a completion_reason")
            expected_sampled = self.generated_token_count + (
                1 if self.completion_reason == "stop_token" else 0
            )
            if self.sampled_token_count != expected_sampled:
                raise ValueError("complete event token counts are inconsistent")
            if (self.stop_token_id is None) == (self.completion_reason == "stop_token"):
                raise ValueError("complete event stop metadata is inconsistent")
        elif self.completion_reason is not None or self.stop_token_id is not None:
            raise ValueError("only complete events may contain stop metadata")

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible event object."""

        return {
            "type": self.type,
            "token_ids": list(self.token_ids),
            "text_delta": self.text_delta,
            "prompt_token_count": self.prompt_token_count,
            "generated_token_count": self.generated_token_count,
            "sampled_token_count": self.sampled_token_count,
            "elapsed_seconds": self.elapsed_seconds,
            "completion_reason": self.completion_reason,
            "stop_token_id": self.stop_token_id,
        }


class ChatEngine:
    """Own one loaded SFT model and one transactional chat conversation."""

    def __init__(
        self,
        checkpoint_path: str | PathLike[str],
        device: str | torch.device = "cpu",
        *,
        checkpoint_loader: _CheckpointLoader = _load_model_checkpoint,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(checkpoint_path, bytes):
            raise TypeError("checkpoint_path must be a string or path-like value")
        try:
            path = Path(checkpoint_path)
        except TypeError as error:
            raise TypeError(
                "checkpoint_path must be a string or path-like value"
            ) from error
        if not str(path):
            raise ValueError("checkpoint_path must not be empty")
        if not callable(checkpoint_loader):
            raise TypeError("checkpoint_loader must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        resolved_device = get_device(device)
        try:
            checkpoint = checkpoint_loader(path, device=str(resolved_device))
        except Exception as error:
            raise ChatEngineError(
                f"could not load SFT checkpoint {path}: {error}"
            ) from error

        training_stage = getattr(checkpoint, "training_stage", None)
        if training_stage != "sft":
            raise ChatEngineError(
                "ChatEngine requires an SFT checkpoint; "
                f"checkpoint {path} records training stage {training_stage!r}"
            )
        model = getattr(checkpoint, "model", None)
        if not isinstance(model, nn.Module):
            raise ChatEngineError("SFT checkpoint did not restore a Torch model")
        tokenizer = getattr(checkpoint, "tokenizer", None)
        if not isinstance(tokenizer, Tokenizer):
            raise ChatEngineError("SFT checkpoint did not restore a tokenizer")
        step = getattr(checkpoint, "step", None)
        try:
            checkpoint_step = require_non_negative_integer(
                step,
                name="checkpoint step",
            )
        except (TypeError, ValueError) as error:
            raise ChatEngineError(
                f"SFT checkpoint has invalid metadata: {error}"
            ) from error
        max_seq_len = getattr(model, "max_seq_len", None)
        if (
            not isinstance(max_seq_len, int)
            or isinstance(max_seq_len, bool)
            or max_seq_len <= 0
        ):
            raise ChatEngineError(
                "SFT checkpoint model must expose a positive max_seq_len"
            )
        checkpoint_config = getattr(checkpoint, "config", None)
        default_generation = getattr(checkpoint_config, "generation", None)
        if not isinstance(default_generation, GenerationConfig):
            raise ChatEngineError(
                "SFT checkpoint is missing canonical GenerationConfig defaults"
            )
        try:
            default_generation.validate()
            tokenizer_identity = tokenizer.get_identity()
            assistant_end_token_id = tokenizer.encode_special("<|assistant_end|>")
            model.to(resolved_device)
            model.eval()
        except Exception as error:
            raise ChatEngineError(
                f"SFT checkpoint {path} is incompatible with chat inference: {error}"
            ) from error

        self._checkpoint_path = str(path)
        self._checkpoint_step = checkpoint_step
        self._device = resolved_device
        self._model = model
        self._tokenizer = tokenizer
        self._tokenizer_identity = tokenizer_identity
        self._assistant_end_token_id = assistant_end_token_id
        self._default_generation = _copy_generation_config(default_generation)
        self._clock = clock
        self._messages: tuple[_HistoryMessage, ...] = ()
        self._pending_prompt: CompletionPrompt | None = None
        self._status: ChatStatus = "idle"
        self._active = False
        self._generation_started_at: float | None = None
        self._prompt_token_count = 0
        self._generated_token_count = 0
        self._sampled_token_count = 0
        self._generation_seconds: float | None = None
        self._completion_reason: CompletionReason | None = None
        self._stop_token_id: int | None = None

    @property
    def default_generation_config(self) -> GenerationConfig:
        """Return a detached copy of the checkpoint's generation defaults."""

        return _copy_generation_config(self._default_generation)

    def reset(self) -> None:
        """Clear conversation state without reloading the checkpoint."""

        self._require_inactive()
        self._messages = ()
        self._pending_prompt = None
        self._status = "idle"
        self._generation_started_at = None
        self._prompt_token_count = 0
        self._generated_token_count = 0
        self._sampled_token_count = 0
        self._generation_seconds = None
        self._completion_reason = None
        self._stop_token_id = None

    def append_user_message(self, text: str) -> None:
        """Append one validated user turn and prepare its canonical prompt."""

        self._require_inactive()
        if self._messages and not isinstance(self._messages[-1], AssistantMessage):
            raise ChatEngineError(
                "cannot append a user message before the pending assistant response"
            )
        user_message = UserMessage(text)
        candidate_messages = (*self._messages, user_message)
        conversation = Conversation(messages=candidate_messages)
        prompt = render_completion_prompt(conversation, self._tokenizer)

        self._messages = candidate_messages
        self._pending_prompt = prompt
        self._status = "awaiting_assistant"
        self._generation_started_at = None
        self._prompt_token_count = len(prompt.token_ids)
        self._generated_token_count = 0
        self._sampled_token_count = 0
        self._generation_seconds = None
        self._completion_reason = None
        self._stop_token_id = None

    def generate_stream(
        self,
        config: GenerationConfig | None = None,
    ) -> Iterator[TokenEvent]:
        """Return a lazy token stream for the pending user turn."""

        self._require_inactive()
        if (
            self._pending_prompt is None
            or not self._messages
            or not isinstance(self._messages[-1], UserMessage)
        ):
            raise ChatEngineError("append a user message before generating a response")
        settings = (
            self.default_generation_config
            if config is None
            else _copy_generation_config(config)
        )
        if settings.top_p is not None:
            raise ChatEngineError(
                "top_p sampling is not implemented by the shared generator"
            )
        prompt = self._pending_prompt
        prompt_tensor = torch.tensor(
            [prompt.token_ids],
            dtype=torch.long,
            device=self._device,
        )
        return self._run_generation(prompt, prompt_tensor, settings)

    def get_state(self) -> ChatState:
        """Return an immutable snapshot with no live model or tensor values."""

        generation_seconds = self._generation_seconds
        if self._active and self._generation_started_at is not None:
            generation_seconds = self._elapsed_since(self._generation_started_at)
        return ChatState(
            checkpoint_path=self._checkpoint_path,
            checkpoint_step=self._checkpoint_step,
            training_stage="sft",
            device=str(self._device),
            tokenizer_identity=self._tokenizer_identity,
            renderer_id=CHAT_RENDERER_ID,
            status=self._status,
            messages=self._messages,
            prompt_token_count=self._prompt_token_count,
            generated_token_count=self._generated_token_count,
            sampled_token_count=self._sampled_token_count,
            generation_seconds=generation_seconds,
            completion_reason=self._completion_reason,
            stop_token_id=self._stop_token_id,
        )

    def _run_generation(
        self,
        prompt: CompletionPrompt,
        prompt_tensor: torch.Tensor,
        settings: GenerationConfig,
    ) -> Generator[TokenEvent, None, None]:
        self._require_inactive()
        if self._pending_prompt is not prompt:
            raise ChatEngineError(
                "conversation changed after the generation stream was created"
            )
        started_at = self._read_clock()
        self._active = True
        self._status = "generating"
        self._generation_started_at = started_at
        self._generated_token_count = 0
        self._sampled_token_count = 0
        self._generation_seconds = 0.0
        self._completion_reason = None
        self._stop_token_id = None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        generated_token_ids: list[int] = []
        text_parts: list[str] = []
        generation_stream = stream_generate_sequence(
            self._model,
            prompt_tensor,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            seed=settings.seed,
            stop_token_ids={self._assistant_end_token_id},
        )
        interrupted_status: Literal["cancelled", "failed"] | None = None
        try:
            yield self._event(type="start", elapsed_seconds=0.0)
            for event in generation_stream:
                if isinstance(event, GeneratedToken):
                    token_bytes = self._tokenizer.decode_single_token_bytes(
                        event.token_id
                    )
                    text_delta = decoder.decode(token_bytes, final=False)
                    generated_token_ids.append(event.token_id)
                    text_parts.append(text_delta)
                    elapsed = self._elapsed_since(started_at)
                    self._generated_token_count = event.generated_token_count
                    self._sampled_token_count = event.sampled_token_count
                    self._generation_seconds = elapsed
                    yield self._event(
                        type="token",
                        token_ids=(event.token_id,),
                        text_delta=text_delta,
                        elapsed_seconds=elapsed,
                    )
                    continue

                if not isinstance(event, GenerationComplete):  # pragma: no cover
                    raise ChatEngineError("shared generator emitted an unknown event")
                generation_stream.close()
                trailing_text = decoder.decode(b"", final=True)
                text_parts.append(trailing_text)
                sequence = event.sequence
                if sequence.prompt_token_ids != prompt.token_ids:
                    raise ChatEngineError("shared generator changed the chat prompt")
                if sequence.generated_token_ids != tuple(generated_token_ids):
                    raise ChatEngineError(
                        "shared generator completion does not match streamed tokens"
                    )
                assistant_text = "".join(text_parts)
                if self._tokenizer.decode(generated_token_ids) != assistant_text:
                    raise ChatEngineError(
                        "incremental tokenizer decoding was not lossless"
                    )
                elapsed = self._elapsed_since(started_at)
                completed_messages = (
                    *self._messages,
                    AssistantMessage(assistant_text),
                )
                Conversation(messages=completed_messages)
                completion_event = self._event(
                    type="complete",
                    text_delta=trailing_text,
                    generated_token_count=len(generated_token_ids),
                    sampled_token_count=sequence.sampled_token_count,
                    elapsed_seconds=elapsed,
                    completion_reason=sequence.completion_reason,
                    stop_token_id=sequence.stop_token_id,
                )
                self._messages = completed_messages
                self._pending_prompt = None
                self._status = "completed"
                self._active = False
                self._generation_started_at = None
                self._generated_token_count = len(generated_token_ids)
                self._sampled_token_count = sequence.sampled_token_count
                self._generation_seconds = elapsed
                self._completion_reason = sequence.completion_reason
                self._stop_token_id = sequence.stop_token_id
                yield completion_event
                return
            raise ChatEngineError("shared generator ended without completion metadata")
        except GeneratorExit:
            interrupted_status = "cancelled"
            raise
        except BaseException:
            interrupted_status = "failed"
            raise
        finally:
            generation_stream.close()
            if self._active:
                self._finish_interrupted(
                    interrupted_status or "cancelled",
                    started_at=started_at,
                )

    def _event(
        self,
        *,
        type: TokenEventType,
        token_ids: tuple[int, ...] = (),
        text_delta: str = "",
        generated_token_count: int | None = None,
        sampled_token_count: int | None = None,
        elapsed_seconds: float,
        completion_reason: CompletionReason | None = None,
        stop_token_id: int | None = None,
    ) -> TokenEvent:
        return TokenEvent(
            type=type,
            token_ids=token_ids,
            text_delta=text_delta,
            prompt_token_count=self._prompt_token_count,
            generated_token_count=(
                self._generated_token_count
                if generated_token_count is None
                else generated_token_count
            ),
            sampled_token_count=(
                self._sampled_token_count
                if sampled_token_count is None
                else sampled_token_count
            ),
            elapsed_seconds=elapsed_seconds,
            completion_reason=completion_reason,
            stop_token_id=stop_token_id,
        )

    def _finish_interrupted(
        self,
        status: Literal["cancelled", "failed"],
        *,
        started_at: float,
    ) -> None:
        self._status = status
        self._active = False
        self._generation_started_at = None
        self._generation_seconds = self._elapsed_since(started_at)
        self._completion_reason = None
        self._stop_token_id = None

    def _require_inactive(self) -> None:
        if self._active:
            raise ChatEngineError("a generation transaction is already active")

    def _read_clock(self) -> float:
        value = self._clock()
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ChatEngineError("clock must return a finite number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ChatEngineError("clock must return a finite number")
        return normalized

    def _elapsed_since(self, started_at: float) -> float:
        elapsed = self._read_clock() - started_at
        if elapsed < 0:
            raise ChatEngineError("clock moved backwards during generation")
        return elapsed


def _copy_generation_config(config: GenerationConfig) -> GenerationConfig:
    if not isinstance(config, GenerationConfig):
        raise TypeError(
            f"config must be a GenerationConfig, got {type(config).__name__}"
        )
    config.validate()
    return GenerationConfig(**config.to_dict())


def _message_payload(message: _HistoryMessage) -> dict[str, str]:
    content = message.content
    if not isinstance(content, str):
        raise AssertionError("ChatEngine assistant messages must contain text")
    return {"role": message.role, "content": content}


__all__ = [
    "ChatEngine",
    "ChatEngineError",
    "ChatState",
    "ChatStatus",
    "TokenEvent",
    "TokenEventType",
]
