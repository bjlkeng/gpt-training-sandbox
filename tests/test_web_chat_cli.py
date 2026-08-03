"""Command-boundary tests for the optional local web server."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import scripts.web_chat as web_chat_script
from scratch_llm.config import GenerationConfig, WebConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _blocked_web_environment(tmp_path: Path) -> dict[str, str]:
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        """
import importlib.abc
import sys

BLOCKED = {
    "fastapi",
    "pydantic",
    "selenium",
    "starlette",
    "uvicorn",
    "websockets",
}

class BlockOptionalWebImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(
                f"No module named {root!r}",
                name=root,
            )
        return None

sys.meta_path.insert(0, BlockOptionalWebImports())
""".lstrip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(PROJECT_ROOT)))
    return environment


def _run_without_web_extra(
    tmp_path: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.web_chat", *arguments],
        cwd=PROJECT_ROOT,
        env=_blocked_web_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )


def test_help_and_core_imports_do_not_require_web_extra(tmp_path: Path) -> None:
    help_result = _run_without_web_extra(tmp_path, "--help")
    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import scratch_llm; import scratch_llm.chat; "
                "import scripts.chat; import scripts.web_chat; import scripts.web_smoke"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=_blocked_web_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0, help_result.stderr
    assert "--allow-remote-bind" in help_result.stdout
    assert import_result.returncode == 0, import_result.stderr


def test_live_execution_without_web_extra_has_one_actionable_error(
    tmp_path: Path,
) -> None:
    result = _run_without_web_extra(
        tmp_path,
        "--checkpoint",
        str(tmp_path / "model.pt"),
    )

    assert result.returncode == 2
    assert "uv sync --extra web" in result.stderr
    assert "Traceback" not in result.stderr


def test_launcher_reuses_validated_configs_and_uvicorn_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    sentinel_app = object()

    def create_app(**keyword_arguments: object) -> object:
        calls["create_app"] = keyword_arguments
        return sentinel_app

    def run_server(app: object, **keyword_arguments: object) -> None:
        calls["run_server"] = (app, keyword_arguments)

    def create_service(*arguments: object, **keyword_arguments: object) -> object:
        calls["create_service"] = (arguments, keyword_arguments)
        return object()

    monkeypatch.setattr(
        web_chat_script,
        "_load_web_runtime",
        lambda: (create_app, create_service, run_server),
    )

    exit_code = web_chat_script.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoints" / "model.pt"),
            "--temperature",
            "0.5",
            "--top-k",
            "9",
            "--max-new-tokens",
            "22",
        ]
    )

    assert exit_code == 0
    app_arguments = calls["create_app"]
    assert isinstance(app_arguments, dict)
    web_config = app_arguments["web_config"]
    generation_config = app_arguments["generation_config"]
    service_factory = app_arguments["service_factory"]
    assert isinstance(web_config, WebConfig)
    assert web_config.host == "127.0.0.1"
    assert web_config.port == 8000
    assert web_config.allow_remote_bind is False
    assert web_config.checkpoint_dir == str(tmp_path / "checkpoints")
    assert generation_config == GenerationConfig(
        temperature=0.5,
        top_k=9,
        max_new_tokens=22,
    )
    assert callable(service_factory)
    service_factory()
    assert calls["create_service"] == (
        (str(tmp_path / "checkpoints"),),
        {"device": "cpu", "initial_checkpoint_id": "model.pt"},
    )
    assert calls["run_server"] == (
        sentinel_app,
        {"host": "127.0.0.1", "port": 8000},
    )


@pytest.mark.parametrize("port", [0, 65_536])
def test_launcher_rejects_invalid_ports_before_loading_web_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    port: int,
) -> None:
    monkeypatch.setattr(
        web_chat_script,
        "_load_web_runtime",
        lambda: pytest.fail("optional dependencies loaded before validation"),
    )

    with pytest.raises(SystemExit) as raised:
        web_chat_script.main(
            [
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--port",
                str(port),
            ]
        )

    assert raised.value.code == 2
    assert "web.port" in capsys.readouterr().err


def test_non_loopback_bind_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    loaded = False

    def load_runtime() -> tuple[object, object, object]:
        nonlocal loaded
        loaded = True
        return object(), object(), object()

    monkeypatch.setattr(web_chat_script, "_load_web_runtime", load_runtime)

    with pytest.raises(SystemExit) as raised:
        web_chat_script.main(
            [
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--host",
                "0.0.0.0",
            ]
        )

    assert raised.value.code == 2
    assert "--allow-remote-bind" in capsys.readouterr().err
    assert loaded is False


def test_explicit_remote_bind_opt_in_reaches_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    sentinel_app = object()

    monkeypatch.setattr(
        web_chat_script,
        "_load_web_runtime",
        lambda: (
            lambda **_kwargs: sentinel_app,
            lambda *_args, **_kwargs: object(),
            lambda app, **kwargs: calls.append((app, kwargs)),
        ),
    )

    result = web_chat_script.main(
        [
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--host",
            "0.0.0.0",
            "--allow-remote-bind",
        ]
    )

    assert result == 0
    assert calls == [(sentinel_app, {"host": "0.0.0.0", "port": 8000})]
