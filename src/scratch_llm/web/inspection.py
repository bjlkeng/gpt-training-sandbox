"""Privacy-safe web generation metrics and explicit debug values."""

from __future__ import annotations

from dataclasses import dataclass

from scratch_llm.chat import (
    IdentityFactory,
    TokenEvent,
    create_public_identity,
    new_public_identity,
)


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Finalized server-owned turn metrics.

    Sampled tokens include a terminal stop token; generated tokens do not.
    Prefill is time to the first sample (including an immediate stop). Decode
    latency averages samples after the first and is absent for zero/one sample.
    Throughput is sampled tokens divided by total time and is absent when the
    count or duration is zero. Peak memory is absent outside measurable CUDA.
    Cancelled and failed attempts expose no ``GenerationMetrics`` at all.
    """

    generated_tokens: int
    sampled_tokens: int
    generation_seconds: float
    prefill_latency_seconds: float | None
    decode_latency_per_sampled_token_seconds: float | None
    tokens_per_second: float | None
    peak_memory_mib: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "generated_tokens": self.generated_tokens,
            "sampled_tokens": self.sampled_tokens,
            "generation_seconds": self.generation_seconds,
            "prefill_latency_seconds": self.prefill_latency_seconds,
            "decode_latency_per_sampled_token_seconds": (
                self.decode_latency_per_sampled_token_seconds
            ),
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_mib": self.peak_memory_mib,
        }


@dataclass(frozen=True, slots=True)
class GenerationDebug:
    """Raw token metadata retained only for an explicit local debug request."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    completion_reason: str | None
    stop_token_id: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "completion_reason": self.completion_reason,
            "stop_token_id": self.stop_token_id,
        }


@dataclass(frozen=True, slots=True)
class SessionAggregate:
    """Privacy-safe cumulative session values for downstream tracking."""

    session_id: str
    turn_id: str | None
    turn_count: int
    generated_tokens: int
    tokens_per_second: float | None
    avg_decode_ms_per_token: float | None
    peak_memory_mib: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "turn_count": self.turn_count,
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": self.tokens_per_second,
            "avg_decode_ms_per_token": self.avg_decode_ms_per_token,
            "peak_memory_mib": self.peak_memory_mib,
        }


class SessionMetricsBoundary:
    """Own IDs and cumulative scalar values without raw chat content."""

    def __init__(self, identity_factory: IdentityFactory) -> None:
        if not callable(identity_factory):
            raise TypeError("identity_factory must be callable")
        self._identity_factory = identity_factory
        self.start_new_session()

    @property
    def session_id(self) -> str:
        return self._session_id

    def start_new_session(self) -> None:
        self._session_id = self._create_identity("session")
        self._completed_turn_count = 0
        self._generated_token_count = 0
        self._total_sampled_tokens = 0
        self._total_generation_seconds = 0.0
        self._total_decode_seconds = 0.0
        self._decoded_sample_count = 0
        self._peak_memory_mib: float | None = None

    def new_turn_id(self) -> str:
        return self._create_identity("turn")

    def record(self, metrics: GenerationMetrics) -> None:
        self._completed_turn_count += 1
        self._generated_token_count += metrics.generated_tokens
        self._total_sampled_tokens += metrics.sampled_tokens
        self._total_generation_seconds += metrics.generation_seconds
        decode_sample_count = max(metrics.sampled_tokens - 1, 0)
        if (
            decode_sample_count
            and metrics.decode_latency_per_sampled_token_seconds is not None
        ):
            self._decoded_sample_count += decode_sample_count
            self._total_decode_seconds += (
                metrics.decode_latency_per_sampled_token_seconds * decode_sample_count
            )
        if metrics.peak_memory_mib is not None:
            self._peak_memory_mib = (
                metrics.peak_memory_mib
                if self._peak_memory_mib is None
                else max(self._peak_memory_mib, metrics.peak_memory_mib)
            )

    def snapshot(self, *, turn_id: str | None = None) -> SessionAggregate:
        tokens_per_second = (
            self._total_sampled_tokens / self._total_generation_seconds
            if self._total_sampled_tokens > 0 and self._total_generation_seconds > 0
            else None
        )
        avg_decode_ms = (
            1000 * self._total_decode_seconds / self._decoded_sample_count
            if self._decoded_sample_count > 0
            else None
        )
        return SessionAggregate(
            session_id=self._session_id,
            turn_id=turn_id,
            turn_count=self._completed_turn_count,
            generated_tokens=self._generated_token_count,
            tokens_per_second=tokens_per_second,
            avg_decode_ms_per_token=avg_decode_ms,
            peak_memory_mib=self._peak_memory_mib,
        )

    def _create_identity(self, kind: str) -> str:
        return create_public_identity(self._identity_factory, kind)


def finalize_generation_metrics(
    completion: TokenEvent,
    *,
    first_sample_seconds: float | None,
    peak_memory_mib: float | None,
) -> GenerationMetrics:
    """Derive one turn's values from shared monotonic event timestamps."""

    sampled_tokens = completion.sampled_token_count
    generation_seconds = completion.elapsed_seconds
    if first_sample_seconds is None and sampled_tokens > 0:
        first_sample_seconds = generation_seconds
    decode_samples = max(sampled_tokens - 1, 0)
    decode_seconds_per_sample = (
        max(generation_seconds - first_sample_seconds, 0.0) / decode_samples
        if first_sample_seconds is not None and decode_samples > 0
        else None
    )
    tokens_per_second = (
        sampled_tokens / generation_seconds
        if sampled_tokens > 0 and generation_seconds > 0
        else None
    )
    return GenerationMetrics(
        generated_tokens=completion.generated_token_count,
        sampled_tokens=sampled_tokens,
        generation_seconds=generation_seconds,
        prefill_latency_seconds=first_sample_seconds,
        decode_latency_per_sampled_token_seconds=decode_seconds_per_sample,
        tokens_per_second=tokens_per_second,
        peak_memory_mib=peak_memory_mib,
    )


__all__ = [
    "GenerationDebug",
    "GenerationMetrics",
    "IdentityFactory",
    "SessionAggregate",
    "SessionMetricsBoundary",
    "finalize_generation_metrics",
    "new_public_identity",
]
