"""Real-browser acceptance for the actual loopback web command."""

from __future__ import annotations

from pathlib import Path
import socket

import pytest

from scripts.web_smoke import browser_runtime_paths, run_smoke
from scratch_llm.chat import read_conversations


@pytest.mark.skipif(
    browser_runtime_paths() is None,
    reason="Firefox and geckodriver are not installed",
)
def test_actual_web_command_completes_controlled_loopback_browser_flow(
    tmp_path: Path,
) -> None:
    screenshots = tmp_path / "screenshots"

    result = run_smoke(tmp_path / "artifacts", screenshot_dir=screenshots)

    assert result.external_requests == ()
    assert result.desktop_screenshot == screenshots / "local-web-chat-desktop.png"
    assert result.narrow_screenshot == screenshots / "local-web-chat-narrow.png"
    for screenshot in (result.desktop_screenshot, result.narrow_screenshot):
        assert screenshot is not None
        assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    conversations = read_conversations(result.transcript_path)
    assert [
        [(item.role, item.content) for item in chat.messages] for chat in conversations
    ] == [[("user", "Controlled smoke prompt"), ("assistant", "AAA")]]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(1)
        assert client.connect_ex(("127.0.0.1", result.port)) != 0
