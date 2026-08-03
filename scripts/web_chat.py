"""Serve the local-only browser chat interface."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import importlib
from typing import TypeAlias, cast

from scratch_llm.config import ConfigValidationError, GenerationConfig, WebConfig
from scripts._common import (
    add_generation_arguments,
    checkpoint_parser,
    resolve_generation_arguments,
)


COMMAND = "web_chat"
WEB_INSTALL_COMMAND = "uv sync --extra web"

_CreateApp: TypeAlias = Callable[..., object]
_CreateService: TypeAlias = Callable[..., object]
_RunServer: TypeAlias = Callable[..., None]


class WebDependencyError(RuntimeError):
    """The optional local-web runtime is not installed completely."""


def build_parser() -> argparse.ArgumentParser:
    """Return the local web chat command parser."""

    parser = checkpoint_parser(COMMAND, "Serve the local browser chat interface.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind; defaults to loopback.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to bind.",
    )
    parser.add_argument(
        "--allow-remote-bind",
        action="store_true",
        help=(
            "SECURITY: explicitly permit a non-loopback host; traffic is not "
            "authenticated or encrypted."
        ),
    )
    add_generation_arguments(parser)
    return parser


def _load_web_runtime() -> tuple[_CreateApp, _CreateService, _RunServer]:
    """Import optional server dependencies only for live execution."""

    try:
        app_module = importlib.import_module("scratch_llm.web.app")
        service_module = importlib.import_module("scratch_llm.web.service")
        uvicorn_module = importlib.import_module("uvicorn")
    except ModuleNotFoundError as error:
        missing = error.name or "unknown"
        raise WebDependencyError(
            f"optional web dependency {missing!r} is unavailable; "
            f"install it with: {WEB_INSTALL_COMMAND}"
        ) from None
    return (
        cast(_CreateApp, getattr(app_module, "create_app")),
        cast(_CreateService, getattr(service_module, "ChatSessionService")),
        cast(_RunServer, getattr(uvicorn_module, "run")),
    )


def _resolve_configs(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> tuple[WebConfig, GenerationConfig]:
    """Validate launch and public generation settings before web imports."""

    try:
        web_config = WebConfig(
            host=arguments.host,
            port=arguments.port,
            checkpoint_dir=str(arguments.checkpoint.parent),
            allow_remote_bind=arguments.allow_remote_bind,
        )
        generation_config = resolve_generation_arguments(
            GenerationConfig(),
            arguments,
        )
    except ConfigValidationError as error:
        message = str(error)
        if error.path == "web.host" and not arguments.allow_remote_bind:
            message += "; pass --allow-remote-bind to acknowledge remote exposure"
        parser.error(message)
    return web_config, generation_config


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local web chat command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    web_config, generation_config = _resolve_configs(parser, arguments)
    try:
        create_app, create_service, run_server = _load_web_runtime()
    except WebDependencyError as error:
        parser.error(str(error))
    app = create_app(
        web_config=web_config,
        generation_config=generation_config,
        service_factory=lambda: create_service(
            web_config.checkpoint_dir,
            device=arguments.device,
            initial_checkpoint_id=arguments.checkpoint.name,
        ),
    )
    run_server(app, host=web_config.host, port=web_config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
