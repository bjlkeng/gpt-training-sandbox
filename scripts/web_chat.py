"""Serve the local-only browser chat interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts._common import checkpoint_parser, run_checkpoint_stub


COMMAND = "web_chat"


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local web chat command."""

    return run_checkpoint_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
