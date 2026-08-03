"""Run the deterministic, loopback-only local web chat browser smoke."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, cast
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from scripts._checkpoint_fixtures import create_tiny_sft_checkpoint
from scratch_llm.chat import read_conversations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT_SECONDS = 30.0
TRANSCRIPT_FILENAME = "scratch-llm-transcript.jsonl"
DESKTOP_SCREENSHOT = "local-web-chat-desktop.png"
NARROW_SCREENSHOT = "local-web-chat-narrow.png"


class WebSmokeError(RuntimeError):
    """A stable, actionable failure from the live browser acceptance path."""


@dataclass(frozen=True, slots=True)
class WebSmokeResult:
    """Durable outputs and assertions from one successful smoke run."""

    transcript_path: Path
    desktop_screenshot: Path | None
    narrow_screenshot: Path | None
    server_log: Path
    driver_log: Path
    external_requests: tuple[str, ...]
    port: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "ok",
            "transcript_records": len(read_conversations(self.transcript_path)),
            "desktop_screenshot": (
                None
                if self.desktop_screenshot is None
                else str(self.desktop_screenshot)
            ),
            "narrow_screenshot": (
                None if self.narrow_screenshot is None else str(self.narrow_screenshot)
            ),
            "external_requests": list(self.external_requests),
        }


class _AuditProxy(ThreadingHTTPServer):
    external_requests: list[str]


class _AuditProxyHandler(BaseHTTPRequestHandler):
    """Reject and record any browser request that was not bypassed as local."""

    def _reject(self) -> None:
        server = cast(_AuditProxy, self.server)
        server.external_requests.append(f"{self.command} {self.path}")
        self.send_response(502)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    do_CONNECT = _reject
    do_DELETE = _reject
    do_GET = _reject
    do_HEAD = _reject
    do_OPTIONS = _reject
    do_PATCH = _reject
    do_POST = _reject
    do_PUT = _reject

    def log_message(self, _format: str, *_arguments: object) -> None:
        return


@contextmanager
def _audit_proxy() -> Iterator[_AuditProxy]:
    server = _AuditProxy(("127.0.0.1", 0), _AuditProxyHandler)
    server.external_requests = []
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def browser_runtime_paths() -> tuple[str, str] | None:
    """Return explicit local Firefox and geckodriver paths, when installed."""

    firefox = shutil.which("firefox")
    geckodriver = shutil.which("geckodriver")
    if firefox is None or geckodriver is None:
        return None
    return firefox, geckodriver


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(
    url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    server_log: Path,
) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = server_log.read_text(encoding="utf-8", errors="replace")
            raise WebSmokeError(
                f"web command exited with {process.returncode} before readiness; "
                f"server log: {server_log}\n{detail[-2_000:]}"
            )
        try:
            with opener.open(url, timeout=1) as response:
                payload = json.loads(response.read())
            if payload == {"api_version": "v1", "status": "ok", "ready": True}:
                return
            last_error = f"unexpected health payload: {payload!r}"
        except (OSError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.05)
    raise WebSmokeError(
        f"web command was not ready within {timeout_seconds:g}s ({last_error}); "
        f"server log: {server_log}"
    )


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WebSmokeError(message)


def _configure_firefox(
    options: Any,
    *,
    download_dir: Path,
    proxy_port: int,
) -> None:
    options.add_argument("-headless")
    preferences: dict[str, object] = {
        "app.normandy.api_url": "",
        "app.normandy.enabled": False,
        "app.shield.optoutstudies.enabled": False,
        "app.update.disabledForTesting": True,
        "browser.download.alwaysOpenPanel": False,
        "browser.download.dir": str(download_dir),
        "browser.download.folderList": 2,
        "browser.download.manager.showWhenStarting": False,
        "browser.helperApps.neverAsk.saveToDisk": "application/x-ndjson",
        "browser.safebrowsing.downloads.enabled": False,
        "browser.safebrowsing.malware.enabled": False,
        "browser.safebrowsing.phishing.enabled": False,
        "browser.shell.checkDefaultBrowser": False,
        "browser.startup.homepage_override.mstone": "ignore",
        "browser.startup.page": 0,
        "browser.uitour.enabled": False,
        "datareporting.healthreport.uploadEnabled": False,
        "datareporting.policy.dataSubmissionEnabled": False,
        "dom.push.enabled": False,
        "extensions.getAddons.cache.enabled": False,
        "extensions.installDistroAddons": False,
        "extensions.systemAddon.update.enabled": False,
        "extensions.update.enabled": False,
        "geo.provider.testing": True,
        "geo.wifi.scan": False,
        "media.gmp-manager.updateEnabled": False,
        "network.captive-portal-service.enabled": False,
        "network.connectivity-service.enabled": False,
        "network.dns.disablePrefetch": True,
        "network.prefetch-next": False,
        "network.proxy.http": "127.0.0.1",
        "network.proxy.http_port": proxy_port,
        "network.proxy.no_proxies_on": "127.0.0.1, localhost",
        "network.proxy.share_proxy_settings": True,
        "network.proxy.ssl": "127.0.0.1",
        "network.proxy.ssl_port": proxy_port,
        "network.proxy.type": 1,
        "network.trr.mode": 5,
        "pdfjs.disabled": True,
        "security.certerrors.mitm.priming.enabled": False,
        "security.remote_settings.crlite_filters.enabled": False,
        "security.remote_settings.intermediates.enabled": False,
        "services.settings.server": "data:,#remote-settings-dummy/v1",
        "signon.rememberSignons": False,
        "startup.homepage_welcome_url": "about:blank",
        "startup.homepage_welcome_url.additional": "",
        "toolkit.telemetry.enabled": False,
    }
    for name, value in preferences.items():
        options.set_preference(name, value)


def _fill(control: Any, value: str) -> None:
    control.clear()
    control.send_keys(value)


def _save_screenshot(driver: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved = bool(driver.save_full_page_screenshot(str(destination)))
    _require(saved and destination.stat().st_size > 0, f"failed to save {destination}")
    return destination


def _browser_flow(
    *,
    base_url: str,
    run_dir: Path,
    screenshot_dir: Path | None,
    timeout_seconds: float,
    proxy: _AuditProxy,
    geckodriver_path: str,
) -> WebSmokeResult:
    # Keep Selenium optional at import time so core commands remain importable.
    from selenium.webdriver import Firefox
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.ui import Select, WebDriverWait

    downloads = run_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    driver_log = run_dir / "geckodriver.log"
    options = Options()
    _configure_firefox(
        options,
        download_dir=downloads,
        proxy_port=int(proxy.server_address[1]),
    )
    driver_environment = os.environ.copy()
    driver_environment["MOZ_REMOTE_SETTINGS_DEVTOOLS"] = "1"
    service = Service(
        executable_path=geckodriver_path,
        log_output=str(driver_log),
        env=driver_environment,
    )
    driver: Any | None = None
    desktop: Path | None = None
    narrow: Path | None = None
    try:
        driver = Firefox(options=options, service=service)
        driver.set_page_load_timeout(timeout_seconds)
        driver.set_script_timeout(timeout_seconds)
        driver.set_window_size(1_440, 1_000)
        wait = WebDriverWait(driver, timeout_seconds, poll_frequency=0.05)
        driver.get(base_url)

        def status() -> str:
            return str(driver.find_element(By.ID, "connection-status").text)

        wait.until(lambda _driver: status() == "Checkpoint ready")
        checkpoint = driver.find_element(By.ID, "checkpoint-select")
        _require(
            Select(checkpoint).first_selected_option.get_attribute("value")
            == "alpha.pt",
            "initial checkpoint was not selected",
        )

        _fill(driver.find_element(By.ID, "temperature"), "0")
        _fill(driver.find_element(By.ID, "top-k"), "1")
        _fill(driver.find_element(By.ID, "max-new-tokens"), "3")
        driver.find_element(By.CSS_SELECTOR, ".debug-panel summary").click()
        driver.find_element(By.ID, "debug-enabled").click()
        _fill(driver.find_element(By.ID, "message-input"), "Controlled smoke prompt")
        driver.find_element(By.ID, "send-button").click()

        wait.until(lambda _driver: status() == "Complete")
        assistant = driver.find_element(By.CSS_SELECTOR, ".message.assistant p")
        _require(assistant.text == "AAA", "streamed deterministic text was not AAA")
        generated = driver.find_element(By.ID, "generated-token-metric").text
        throughput = driver.find_element(By.ID, "throughput-metric").text
        _require(generated == "3", "generated-token metric was not 3")
        _require(throughput.endswith(" tok/s"), "throughput metric was not rendered")
        debug = json.loads(driver.find_element(By.ID, "debug-output").text)
        _require(debug["generated_token_ids"] == [65, 65, 65], "debug IDs differ")
        _require(bool(debug["prompt_token_ids"]), "debug prompt IDs are empty")
        _require(debug["context"]["prompt_tokens"] > 0, "debug context is empty")

        driver.find_element(By.ID, "export-button").click()
        wait.until(lambda _driver: status() == "Transcript downloaded.")
        export = cast(
            dict[str, object],
            driver.execute_async_script(
                """
                const done = arguments[0];
                fetch('/api/transcript', {headers: {accept: 'application/x-ndjson'}})
                  .then(async response => done({
                    ok: response.ok,
                    body: await response.text(),
                    disposition: response.headers.get('content-disposition'),
                  }))
                  .catch(error => done({ok: false, error: String(error)}));
                """
            ),
        )
        _require(export.get("ok") is True, f"browser export failed: {export!r}")
        _require(
            export.get("disposition")
            == 'attachment; filename="scratch-llm-transcript.jsonl"',
            "browser export did not use the fixed filename",
        )
        body = export.get("body")
        _require(isinstance(body, str), "browser export was not text")
        transcript = run_dir / TRANSCRIPT_FILENAME
        transcript.write_text(cast(str, body), encoding="utf-8")
        conversations = read_conversations(transcript)
        _require(len(conversations) == 1, "export did not contain one conversation")
        messages = [(item.role, item.content) for item in conversations[0].messages]
        _require(
            messages
            == [
                ("user", "Controlled smoke prompt"),
                ("assistant", "AAA"),
            ],
            "exported transcript differs from the controlled conversation",
        )

        if screenshot_dir is not None:
            desktop = _save_screenshot(driver, screenshot_dir / DESKTOP_SCREENSHOT)
            driver.set_window_size(430, 900)
            narrow = _save_screenshot(driver, screenshot_dir / NARROW_SCREENSHOT)
            driver.set_window_size(1_440, 1_000)

        _fill(driver.find_element(By.ID, "max-new-tokens"), "400")
        _fill(driver.find_element(By.ID, "message-input"), "Controlled stop prompt")
        driver.find_element(By.ID, "send-button").click()
        wait.until(lambda _driver: status() == "Generating…")
        driver.find_element(By.ID, "stop-button").click()
        wait.until(
            lambda _driver: status() == "Generation stopped; this turn was not saved."
        )
        _require(
            len(driver.find_elements(By.CSS_SELECTOR, ".message")) == 2,
            "stopped turn remained visible",
        )

        checkpoint = driver.find_element(By.ID, "checkpoint-select")
        Select(checkpoint).select_by_value("beta.pt")
        driver.find_element(By.ID, "load-checkpoint-button").click()
        wait.until(lambda _driver: status() == "Checkpoint loaded.")
        _require(
            not driver.find_elements(By.CSS_SELECTOR, ".message"),
            "checkpoint switch did not clear the prior conversation",
        )
        _fill(driver.find_element(By.ID, "max-new-tokens"), "1")
        _fill(driver.find_element(By.ID, "message-input"), "Controlled reset prompt")
        driver.find_element(By.ID, "send-button").click()
        wait.until(lambda _driver: status() == "Complete")
        wait.until(
            lambda _driver: (
                driver.find_element(By.CSS_SELECTOR, ".message.assistant p").text == "B"
            )
        )
        driver.find_element(By.ID, "reset-button").click()
        wait.until(lambda _driver: status() == "Conversation reset.")
        _require(
            not driver.find_elements(By.CSS_SELECTOR, ".message"),
            "reset did not clear the conversation",
        )

        resource_urls = cast(
            list[str],
            driver.execute_script(
                "return performance.getEntriesByType('resource').map(e => e.name);"
            ),
        )
        remote_resources = [
            url
            for url in resource_urls
            if urlsplit(url).hostname not in {"127.0.0.1", "localhost"}
        ]
        _require(
            not remote_resources, f"page loaded remote resources: {remote_resources}"
        )
        return WebSmokeResult(
            transcript_path=transcript,
            desktop_screenshot=desktop,
            narrow_screenshot=narrow,
            server_log=run_dir / "server.log",
            driver_log=driver_log,
            external_requests=tuple(proxy.external_requests),
            port=urlsplit(base_url).port or 0,
        )
    except Exception as error:
        if driver is not None:
            failure_screenshot = run_dir / "failure.png"
            page_source = run_dir / "failure-page.html"
            try:
                driver.save_screenshot(str(failure_screenshot))
                page_source.write_text(driver.page_source, encoding="utf-8")
            except Exception:
                pass
        if isinstance(error, WebSmokeError):
            raise
        raise WebSmokeError(
            f"browser flow failed: {type(error).__name__}: {error}; "
            f"diagnostics: {run_dir}"
        ) from error
    finally:
        if driver is not None:
            driver.quit()


def run_smoke(
    run_dir: Path,
    *,
    screenshot_dir: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> WebSmokeResult:
    """Run the actual local command and browser flow with controlled fixtures."""

    runtime = browser_runtime_paths()
    if runtime is None:
        raise WebSmokeError(
            "Firefox and geckodriver are required for the browser smoke"
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    _firefox_path, geckodriver_path = runtime
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    fixture_options = {
        "seq_len": 512,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 64,
        "max_new_tokens": 400,
    }
    create_tiny_sft_checkpoint(
        checkpoints / "alpha.pt",
        preferred_token_id=65,
        **fixture_options,
    )
    create_tiny_sft_checkpoint(
        checkpoints / "beta.pt",
        preferred_token_id=66,
        **fixture_options,
    )

    port = _available_port()
    base_url = f"http://127.0.0.1:{port}"
    server_log = run_dir / "server.log"
    command = [
        sys.executable,
        "-m",
        "scripts.web_chat",
        "--checkpoint",
        str(checkpoints / "alpha.pt"),
        "--device",
        "cpu",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    with server_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_server(
                f"{base_url}/api/health",
                process,
                timeout_seconds=timeout_seconds,
                server_log=server_log,
            )
            with _audit_proxy() as proxy:
                result = _browser_flow(
                    base_url=base_url,
                    run_dir=run_dir,
                    screenshot_dir=screenshot_dir,
                    timeout_seconds=timeout_seconds,
                    proxy=proxy,
                    geckodriver_path=geckodriver_path,
                )
                _require(
                    not proxy.external_requests,
                    f"browser attempted external requests: {proxy.external_requests}",
                )
                return replace(
                    result,
                    external_requests=tuple(proxy.external_requests),
                )
        finally:
            _stop_process_tree(process)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic local web chat browser smoke."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("runs/web-smoke"),
        help="Keep diagnostics here (default: %(default)s).",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        help="Also regenerate controlled desktop and narrow screenshots here.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Bound each server and browser wait (default: %(default)ss).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = run_smoke(
            arguments.artifacts_dir,
            screenshot_dir=arguments.screenshots_dir,
            timeout_seconds=arguments.timeout_seconds,
        )
        print(json.dumps(result.to_dict(), sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"web smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DESKTOP_SCREENSHOT",
    "NARROW_SCREENSHOT",
    "WebSmokeError",
    "WebSmokeResult",
    "browser_runtime_paths",
    "build_parser",
    "main",
    "run_smoke",
]
