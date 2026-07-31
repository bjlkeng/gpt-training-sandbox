"""Deterministic planning and resilient downloads for ClimbMix parquet shards."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from os import replace as atomic_replace
from pathlib import Path
import re
import time
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scratch_llm._validation import (
    require_non_negative_integer,
    require_non_negative_real,
    require_positive_integer,
    require_positive_real,
)


CLIMBMIX_BASE_URL = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
)
DEFAULT_CLIMBMIX_DATA_DIR = Path("data/parquet/base_data_climbmix")
# The fixed final shard is validation and must never enter a training prefix.
CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX = 6542
_CONTENT_RANGE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
_UNSATISFIED_CONTENT_RANGE = re.compile(r"^bytes \*/([0-9]+)$")


class ClimbMixDownloadError(RuntimeError):
    """A shard could not be made ready without violating download invariants."""


class _RetryableDownloadError(ClimbMixDownloadError):
    """A retry may continue from the retained temporary file."""


class _RestartDownloadError(ClimbMixDownloadError):
    """The temporary file cannot be trusted and must be discarded."""


@dataclass(frozen=True)
class ClimbMixDownloadTarget:
    """One remote shard and its same-directory local publication paths."""

    index: int
    url: str
    destination: Path

    @property
    def temporary_path(self) -> Path:
        """Return the restartable path used before atomic publication."""

        return Path(f"{self.destination}.part")


@dataclass(frozen=True)
class ClimbMixDownloadResult:
    """Outcome for one destination that is ready for downstream readers."""

    target: ClimbMixDownloadTarget
    status: Literal["downloaded", "skipped"]
    bytes_ready: int
    attempts: int


@dataclass(frozen=True)
class ClimbMixDownloadSummary:
    """Aggregate counts and bytes for one deterministic target plan."""

    planned_shards: int
    ready_shards: int
    downloaded_shards: int
    skipped_shards: int
    total_bytes: int
    results: tuple[ClimbMixDownloadResult, ...]


class DownloadResponse(Protocol):
    """Small streaming response contract used by the real and fake transports."""

    status: int
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes:
        """Read at most ``size`` response bytes."""


class ResponseOpener(Protocol):
    """Open one HTTP response, optionally from a byte offset."""

    def __call__(
        self,
        url: str,
        *,
        offset: int | None,
        timeout: float,
    ) -> AbstractContextManager[DownloadResponse]:
        """Return a managed streaming response."""


def plan_climbmix_downloads(
    *,
    num_train_shards: int,
    include_val: bool,
    data_dir: str | Path = DEFAULT_CLIMBMIX_DATA_DIR,
    base_url: str = CLIMBMIX_BASE_URL,
    validation_shard_index: int = CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
) -> tuple[ClimbMixDownloadTarget, ...]:
    """Plan a train prefix and optional fixed validation shard without overlap."""

    num_train_shards = require_non_negative_integer(
        num_train_shards,
        name="num_train_shards",
    )
    validation_shard_index = require_non_negative_integer(
        validation_shard_index,
        name="validation_shard_index",
    )
    if not isinstance(include_val, bool):
        raise TypeError(
            f"include_val must be a boolean, got {type(include_val).__name__}"
        )
    if num_train_shards > validation_shard_index:
        raise ValueError(
            f"train prefix [0, {num_train_shards}) would overlap fixed validation "
            f"shard index {validation_shard_index}"
        )
    if num_train_shards == 0 and not include_val:
        raise ValueError("download selection must include at least one shard")
    if not isinstance(base_url, str):
        raise TypeError(
            f"base_url must be a non-empty string, got {type(base_url).__name__}"
        )
    normalized_base_url = base_url.rstrip("/")
    if not normalized_base_url:
        raise ValueError("base_url must be a non-empty string")

    directory = Path(data_dir)
    indices = list(range(num_train_shards))
    if include_val:
        indices.append(validation_shard_index)

    targets = []
    for index in indices:
        filename = f"shard_{index:05d}.parquet"
        targets.append(
            ClimbMixDownloadTarget(
                index=index,
                url=f"{normalized_base_url}/{filename}",
                destination=directory / filename,
            )
        )
    return tuple(targets)


def download_climbmix_target(
    target: ClimbMixDownloadTarget,
    *,
    opener: ResponseOpener | None = None,
    timeout: float = 30.0,
    max_attempts: int = 5,
    backoff_base: float = 1.0,
    chunk_size: int = 1024 * 1024,
    sleep: Callable[[float], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ClimbMixDownloadResult:
    """Download one target to a restartable temporary file and publish it."""

    if opener is None:
        opener = open_http_response
    if sleep is None:
        sleep = time.sleep
    max_attempts = require_positive_integer(max_attempts, name="max_attempts")
    chunk_size = require_positive_integer(chunk_size, name="chunk_size")
    timeout = require_positive_real(timeout, name="timeout")
    backoff_base = require_non_negative_real(
        backoff_base,
        name="backoff_base",
    )

    destination = target.destination
    if destination.exists():
        if not destination.is_file():
            raise ClimbMixDownloadError(
                f"{destination.name}: destination is not a regular file: {destination}"
            )
        existing_size = destination.stat().st_size
        if existing_size > 0:
            _notify(progress, f"{destination.name}: already complete; skipping")
            return ClimbMixDownloadResult(
                target=target,
                status="skipped",
                bytes_ready=existing_size,
                attempts=0,
            )

    temporary_path = target.temporary_path
    if temporary_path.exists() and not temporary_path.is_file():
        raise ClimbMixDownloadError(
            f"{destination.name}: temporary path is not a regular file: "
            f"{temporary_path}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ClimbMixDownloadError(
            f"{destination.name}: could not create destination directory "
            f"{destination.parent}: {error}"
        ) from error

    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        resume_offset = _partial_size(temporary_path) or None
        try:
            with opener(
                target.url,
                offset=resume_offset,
                timeout=timeout,
            ) as response:
                file_mode, expected_size = _response_download_plan(
                    response,
                    requested_offset=resume_offset,
                )
                if resume_offset is not None and file_mode == "wb":
                    _notify(
                        progress,
                        f"{destination.name}: server ignored Range; restarting "
                        "safely from byte 0",
                    )
                if file_mode is not None:
                    with temporary_path.open(file_mode) as temporary_file:
                        while chunk := response.read(chunk_size):
                            temporary_file.write(chunk)

            actual_size = _partial_size(temporary_path)
            if actual_size < expected_size:
                raise _RetryableDownloadError(
                    f"truncated response: expected {expected_size} bytes, "
                    f"received {actual_size}"
                )
            if actual_size > expected_size:
                raise _RestartDownloadError(
                    f"response exceeded declared size {expected_size}: "
                    f"received {actual_size} bytes"
                )
        except _RestartDownloadError as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
        except (_RetryableDownloadError, OSError, URLError, EOFError) as error:
            last_error = error
        else:
            try:
                atomic_replace(temporary_path, destination)
            except OSError as error:
                raise ClimbMixDownloadError(
                    f"{destination.name}: validated {actual_size}-byte temporary "
                    f"file could not be atomically published; {_partial_state(target)}: "
                    f"{error}"
                ) from error
            _notify(
                progress,
                f"{destination.name}: ready ({actual_size} bytes)",
            )
            return ClimbMixDownloadResult(
                target=target,
                status="downloaded",
                bytes_ready=actual_size,
                attempts=attempt,
            )

        if attempt < max_attempts:
            delay = backoff_base * (2 ** (attempt - 1))
            _notify(
                progress,
                f"{destination.name}: attempt {attempt}/{max_attempts} failed "
                f"({last_error}); {_partial_state(target)}; retrying in "
                f"{delay:g}s",
            )
            sleep(delay)

    attempt_word = "attempt" if max_attempts == 1 else "attempts"
    raise ClimbMixDownloadError(
        f"{destination.name} failed after {max_attempts} {attempt_word}: "
        f"{last_error}; {_partial_state(target)}"
    ) from last_error


def download_climbmix_targets(
    targets: Iterable[ClimbMixDownloadTarget],
    *,
    opener: ResponseOpener | None = None,
    timeout: float = 30.0,
    max_attempts: int = 5,
    backoff_base: float = 1.0,
    chunk_size: int = 1024 * 1024,
    sleep: Callable[[float], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ClimbMixDownloadSummary:
    """Make every planned target ready and return machine-checkable totals."""

    planned_targets = tuple(targets)
    results = tuple(
        download_climbmix_target(
            target,
            opener=opener,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff_base=backoff_base,
            chunk_size=chunk_size,
            sleep=sleep,
            progress=progress,
        )
        for target in planned_targets
    )
    downloaded_shards = sum(result.status == "downloaded" for result in results)
    skipped_shards = sum(result.status == "skipped" for result in results)
    return ClimbMixDownloadSummary(
        planned_shards=len(planned_targets),
        ready_shards=len(results),
        downloaded_shards=downloaded_shards,
        skipped_shards=skipped_shards,
        total_bytes=sum(result.bytes_ready for result in results),
        results=results,
    )


def _response_download_plan(
    response: DownloadResponse,
    *,
    requested_offset: int | None,
) -> tuple[Literal["ab", "wb"] | None, int]:
    if response.status == 416 and requested_offset is not None:
        raw_content_range = response.headers.get("Content-Range")
        match = (
            _UNSATISFIED_CONTENT_RANGE.fullmatch(raw_content_range)
            if raw_content_range is not None
            else None
        )
        if match is not None and int(match.group(1)) == requested_offset:
            # A range starting exactly at EOF proves the retained file has the
            # remote representation's declared total size.
            return None, requested_offset
        raise _RestartDownloadError(
            f"incompatible unsatisfied Content-Range {raw_content_range!r} "
            f"for requested offset {requested_offset}"
        )

    content_length = _response_content_length(response.headers)
    if requested_offset is None:
        if response.status != 200:
            raise _RestartDownloadError(
                f"expected HTTP 200 for a fresh download, got {response.status}"
            )
        return "wb", content_length

    if response.status == 200:
        # The server ignored Range. Its body is a full response, so restart safely.
        return "wb", content_length
    if response.status != 206:
        raise _RestartDownloadError(
            f"expected HTTP 206 while resuming at byte {requested_offset}, "
            f"got {response.status}"
        )

    raw_content_range = response.headers.get("Content-Range")
    match = (
        _CONTENT_RANGE.fullmatch(raw_content_range)
        if raw_content_range is not None
        else None
    )
    if match is None:
        raise _RestartDownloadError(
            "resumed response must contain a valid Content-Range header; "
            f"got {raw_content_range!r}"
        )
    start, end, total = (int(value) for value in match.groups())
    if (
        start != requested_offset
        or end < start
        or end + 1 != total
        or content_length != end - start + 1
    ):
        raise _RestartDownloadError(
            f"incompatible Content-Range {raw_content_range!r} for requested "
            f"offset {requested_offset} and Content-Length {content_length}"
        )
    return "ab", total


@contextmanager
def open_http_response(
    url: str,
    *,
    offset: int | None,
    timeout: float,
) -> Iterator[DownloadResponse]:
    """Open a streaming GET using only the Python standard library."""

    headers = {"Accept-Encoding": "identity"}
    if offset is not None:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers, method="GET")
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        if error.code != 416:
            raise
        response = error
    try:
        yield response  # type: ignore[misc]
    finally:
        response.close()


def _response_content_length(headers: Mapping[str, str]) -> int:
    raw_length = headers.get("Content-Length")
    try:
        content_length = int(raw_length) if raw_length is not None else 0
    except ValueError as error:
        raise _RestartDownloadError(
            f"response Content-Length must be a positive integer, got {raw_length!r}"
        ) from error
    if content_length <= 0:
        raise _RestartDownloadError(
            f"response Content-Length must be a positive integer, got {raw_length!r}"
        )
    return content_length


def _partial_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _partial_state(target: ClimbMixDownloadTarget) -> str:
    partial_size = _partial_size(target.temporary_path)
    if partial_size:
        return (
            f"restartable partial retained at {target.temporary_path} "
            f"({partial_size} bytes)"
        )
    return f"next attempt restarts at byte 0 using {target.temporary_path}"


def _notify(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
