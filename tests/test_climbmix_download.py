"""Tests for resilient, offline ClimbMix shard downloads."""

from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import pytest

import scratch_llm.climbmix as climbmix
from scripts.download_climbmix import main as download_main
from scratch_llm.climbmix import (
    CLIMBMIX_BASE_URL,
    CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    ClimbMixDownloadError,
    ClimbMixDownloadTarget,
    download_climbmix_target,
    download_climbmix_targets,
    plan_climbmix_downloads,
)


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body))}
        self._body = BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


class InterruptingResponse(FakeResponse):
    def __init__(self, first_chunk: bytes, *, expected_size: int) -> None:
        super().__init__(
            first_chunk,
            headers={"Content-Length": str(expected_size)},
        )
        self._interrupted = False

    def read(self, size: int = -1) -> bytes:
        if not self._interrupted:
            self._interrupted = True
            return super().read(size)
        raise OSError("connection dropped during response body")


class ScriptedOpener:
    def __init__(self, *events: FakeResponse | BaseException) -> None:
        self.events = list(events)
        self.calls: list[tuple[str, int | None, float]] = []

    @contextmanager
    def __call__(
        self,
        url: str,
        *,
        offset: int | None,
        timeout: float,
    ) -> Any:
        self.calls.append((url, offset, timeout))
        if not self.events:
            raise AssertionError("unexpected HTTP request")
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        yield event


def _target(tmp_path: Path) -> ClimbMixDownloadTarget:
    return plan_climbmix_downloads(
        num_train_shards=1,
        include_val=False,
        data_dir=tmp_path,
    )[0]


def test_plan_climbmix_downloads_selects_train_prefix_and_fixed_validation(
    tmp_path: Path,
) -> None:
    targets = plan_climbmix_downloads(
        num_train_shards=2,
        include_val=True,
        data_dir=tmp_path,
    )

    assert [target.index for target in targets] == [
        0,
        1,
        CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    ]
    assert [target.destination.name for target in targets] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_06542.parquet",
    ]
    assert [target.url for target in targets] == [
        f"{CLIMBMIX_BASE_URL}/shard_00000.parquet",
        f"{CLIMBMIX_BASE_URL}/shard_00001.parquet",
        f"{CLIMBMIX_BASE_URL}/shard_06542.parquet",
    ]
    assert all(
        target.temporary_path == target.destination.with_suffix(".parquet.part")
        for target in targets
    )


@pytest.mark.parametrize("num_train_shards", [-1, True, 1.5, "2"])
def test_plan_climbmix_downloads_rejects_invalid_train_counts(
    tmp_path: Path,
    num_train_shards: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="num_train_shards"):
        plan_climbmix_downloads(
            num_train_shards=num_train_shards,  # type: ignore[arg-type]
            include_val=True,
            data_dir=tmp_path,
        )


def test_plan_climbmix_downloads_rejects_validation_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"train prefix.*overlap.*validation.*6542"):
        plan_climbmix_downloads(
            num_train_shards=CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX + 1,
            include_val=True,
            data_dir=tmp_path,
        )


def test_plan_climbmix_downloads_rejects_an_empty_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one shard"):
        plan_climbmix_downloads(
            num_train_shards=0,
            include_val=False,
            data_dir=tmp_path,
        )


def test_download_climbmix_target_publishes_fresh_content_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)
    body = b"parquet payload"
    opener = ScriptedOpener(FakeResponse(body))
    replacements: list[tuple[Path, Path]] = []
    real_replace = climbmix.atomic_replace

    def record_replace(source: Path, destination: Path) -> None:
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(climbmix, "atomic_replace", record_replace)

    result = download_climbmix_target(
        target,
        opener=opener,
        backoff_base=0,
    )

    assert result.status == "downloaded"
    assert result.bytes_ready == len(body)
    assert result.attempts == 1
    assert target.destination.read_bytes() == body
    assert not target.temporary_path.exists()
    assert replacements == [(target.temporary_path, target.destination)]
    assert opener.calls == [(target.url, None, 30.0)]


def test_download_climbmix_target_skips_a_published_nonempty_file(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    target.destination.write_bytes(b"already complete")
    opener = ScriptedOpener()

    result = download_climbmix_target(target, opener=opener)

    assert result.status == "skipped"
    assert result.bytes_ready == len(b"already complete")
    assert result.attempts == 0
    assert opener.calls == []


def test_download_climbmix_target_resumes_a_compatible_range(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    prefix = b"partial "
    suffix = b"payload"
    target.temporary_path.write_bytes(prefix)
    opener = ScriptedOpener(
        FakeResponse(
            suffix,
            status=206,
            headers={
                "Content-Length": str(len(suffix)),
                "Content-Range": (
                    f"bytes {len(prefix)}-{len(prefix) + len(suffix) - 1}/"
                    f"{len(prefix) + len(suffix)}"
                ),
            },
        )
    )

    result = download_climbmix_target(
        target,
        opener=opener,
        backoff_base=0,
    )

    assert result.status == "downloaded"
    assert result.bytes_ready == len(prefix + suffix)
    assert result.attempts == 1
    assert target.destination.read_bytes() == prefix + suffix
    assert opener.calls == [(target.url, len(prefix), 30.0)]


def test_download_climbmix_target_publishes_a_complete_retained_temporary_file(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    complete_body = b"complete but not yet renamed"
    target.temporary_path.write_bytes(complete_body)
    opener = ScriptedOpener(
        FakeResponse(
            b"",
            status=416,
            headers={
                "Content-Length": "0",
                "Content-Range": f"bytes */{len(complete_body)}",
            },
        )
    )

    result = download_climbmix_target(
        target,
        opener=opener,
        backoff_base=0,
    )

    assert result.status == "downloaded"
    assert result.bytes_ready == len(complete_body)
    assert target.destination.read_bytes() == complete_body
    assert opener.calls == [(target.url, len(complete_body), 30.0)]


def test_download_climbmix_target_restarts_when_range_is_ignored(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    target.temporary_path.write_bytes(b"stale prefix")
    complete_body = b"complete replacement"
    opener = ScriptedOpener(FakeResponse(complete_body, status=200))

    result = download_climbmix_target(
        target,
        opener=opener,
        backoff_base=0,
    )

    assert result.status == "downloaded"
    assert target.destination.read_bytes() == complete_body
    assert opener.calls == [(target.url, len(b"stale prefix"), 30.0)]


def test_download_climbmix_target_retries_transient_open_failures(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    body = b"eventually available"
    opener = ScriptedOpener(
        OSError("temporary connection failure"),
        FakeResponse(body),
    )
    delays: list[float] = []

    result = download_climbmix_target(
        target,
        opener=opener,
        max_attempts=2,
        backoff_base=0.25,
        sleep=delays.append,
    )

    assert result.status == "downloaded"
    assert result.attempts == 2
    assert target.destination.read_bytes() == body
    assert [offset for _, offset, _ in opener.calls] == [None, None]
    assert delays == [0.25]


def test_download_climbmix_target_resumes_after_a_truncated_response(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    prefix = b"first half"
    suffix = b" second half"
    total_size = len(prefix + suffix)
    opener = ScriptedOpener(
        FakeResponse(
            prefix,
            headers={"Content-Length": str(total_size)},
        ),
        FakeResponse(
            suffix,
            status=206,
            headers={
                "Content-Length": str(len(suffix)),
                "Content-Range": (f"bytes {len(prefix)}-{total_size - 1}/{total_size}"),
            },
        ),
    )

    result = download_climbmix_target(
        target,
        opener=opener,
        max_attempts=2,
        backoff_base=0,
        sleep=lambda _: None,
    )

    assert result.status == "downloaded"
    assert result.attempts == 2
    assert target.destination.read_bytes() == prefix + suffix
    assert [offset for _, offset, _ in opener.calls] == [None, len(prefix)]


def test_download_climbmix_target_discards_an_incompatible_range_before_retry(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    stale_prefix = b"stale"
    target.temporary_path.write_bytes(stale_prefix)
    complete_body = b"clean full response"
    opener = ScriptedOpener(
        FakeResponse(
            b"wrong continuation",
            status=206,
            headers={
                "Content-Length": str(len(b"wrong continuation")),
                "Content-Range": "bytes 99-116/117",
            },
        ),
        FakeResponse(complete_body),
    )

    result = download_climbmix_target(
        target,
        opener=opener,
        max_attempts=2,
        backoff_base=0,
        sleep=lambda _: None,
    )

    assert result.status == "downloaded"
    assert target.destination.read_bytes() == complete_body
    assert [offset for _, offset, _ in opener.calls] == [
        len(stale_prefix),
        None,
    ]


def test_download_climbmix_target_retains_interrupted_state_without_publishing(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    partial = b"restart me"
    opener = ScriptedOpener(
        InterruptingResponse(partial, expected_size=len(partial) + 10)
    )

    with pytest.raises(
        ClimbMixDownloadError,
        match=(
            r"shard_00000\.parquet.*failed after 1 attempt.*"
            r"restartable partial retained.*10 bytes"
        ),
    ):
        download_climbmix_target(
            target,
            opener=opener,
            max_attempts=1,
            backoff_base=0,
            sleep=lambda _: None,
        )

    assert not target.destination.exists()
    assert target.temporary_path.read_bytes() == partial


def test_download_climbmix_target_never_publishes_a_zero_byte_response(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    opener = ScriptedOpener(FakeResponse(b""))

    with pytest.raises(ClimbMixDownloadError, match=r"Content-Length.*positive"):
        download_climbmix_target(
            target,
            opener=opener,
            max_attempts=1,
            backoff_base=0,
            sleep=lambda _: None,
        )

    assert not target.destination.exists()
    assert not target.temporary_path.exists()


def test_download_climbmix_targets_summarizes_ready_counts_and_bytes(
    tmp_path: Path,
) -> None:
    targets = plan_climbmix_downloads(
        num_train_shards=2,
        include_val=False,
        data_dir=tmp_path,
    )
    targets[0].destination.write_bytes(b"existing")
    opener = ScriptedOpener(FakeResponse(b"new payload"))

    summary = download_climbmix_targets(
        targets,
        opener=opener,
        backoff_base=0,
        sleep=lambda _: None,
    )

    assert summary.planned_shards == 2
    assert summary.ready_shards == 2
    assert summary.downloaded_shards == 1
    assert summary.skipped_shards == 1
    assert summary.total_bytes == len(b"existingnew payload")


def test_download_command_prints_json_summary_and_human_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "shard_00000.parquet").write_bytes(b"existing")
    validation_body = b"validation payload"
    opener = ScriptedOpener(FakeResponse(validation_body))

    exit_code = download_main(
        [
            "--num-train-shards",
            "1",
            "--include-val",
            "--data-dir",
            str(tmp_path),
            "--base-url",
            "https://example.test/climbmix/",
            "--backoff-base",
            "0",
        ],
        opener=opener,
        sleep=lambda _: None,
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary == {
        "data_dir": str(tmp_path),
        "downloaded_shards": 1,
        "planned_shards": 2,
        "ready_shards": 2,
        "skipped_shards": 1,
        "total_bytes": len(b"existing") + len(validation_body),
    }
    assert "Planned 2 shards" in captured.err
    assert "shard_00000.parquet: already complete; skipping" in captured.err
    assert "shard_06542.parquet: ready" in captured.err
    assert opener.calls == [
        (
            "https://example.test/climbmix/shard_06542.parquet",
            None,
            30.0,
        )
    ]


def test_download_command_rejects_an_overlapping_train_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        download_main(
            [
                "--num-train-shards",
                str(CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX + 1),
                "--include-val",
            ],
            opener=ScriptedOpener(),
        )

    assert raised.value.code == 2
    assert "overlap fixed validation shard index 6542" in capsys.readouterr().err
