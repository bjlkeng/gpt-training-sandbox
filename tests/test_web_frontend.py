"""Plain local frontend asset and behavior contract tests."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient

from scratch_llm.config import GenerationConfig, WebConfig
from scratch_llm.web.app import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_plain_frontend_and_local_assets_have_no_remote_dependencies() -> None:
    app = create_app(
        web_config=WebConfig(),
        generation_config=GenerationConfig(
            temperature=0.25,
            top_k=7,
            max_new_tokens=31,
        ),
    )

    with TestClient(app) as client:
        page = client.get("/")
        styles = client.get("/assets/styles.css")
        script = client.get("/assets/app.js")

    assert page.status_code == styles.status_code == script.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert styles.headers["content-type"].startswith("text/css")
    assert "javascript" in script.headers["content-type"]
    assert page.headers["content-security-policy"] == (
        "default-src 'self'; connect-src 'self' ws: wss:; "
        "img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )
    assert '<link rel="stylesheet" href="/assets/styles.css">' in page.text
    assert '<script type="module" src="/assets/app.js"></script>' in page.text
    assert 'maxlength="16384"' in page.text
    assert 'max="10.0"' in page.text
    assert 'value="0.25"' in page.text
    assert 'max="100000"' in page.text
    assert 'value="7"' in page.text
    assert 'max="4096"' in page.text
    assert 'value="31"' in page.text
    assert "{{" not in page.text
    combined = page.text + styles.text + script.text
    for remote_marker in ("https://", "http://", "//cdn", "@import url"):
        assert remote_marker not in combined


def test_frontend_has_accessible_controls_and_no_client_inference_logic() -> None:
    asset_root = files("scratch_llm.web").joinpath("static")
    html = asset_root.joinpath("index.html").read_text(encoding="utf-8")
    javascript = asset_root.joinpath("app.js").read_text(encoding="utf-8")

    for control_id in (
        "message-input",
        "temperature",
        "top-k",
        "max-new-tokens",
        "send-button",
        "stop-button",
        "reset-button",
        "context-status",
        "connection-status",
        "chat-log",
    ):
        assert f'id="{control_id}"' in html
    assert 'aria-live="polite"' in html
    assert "<label" in html
    assert "innerHTML" not in javascript
    assert "createTextNode" in javascript
    assert "textContent" in javascript
    for forbidden_logic in (
        "encode_special",
        "assistant_end",
        "render_completion_prompt",
        "max_seq_len -",
        "tokenizer.encode",
        "model.forward",
    ):
        assert forbidden_logic not in javascript


def test_frontend_behavior_suite_runs_without_browser_or_network() -> None:
    result = subprocess.run(
        ["node", "--test", "tests/js/test_web_app.mjs"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "fail 0" in result.stdout
