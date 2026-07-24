"""Evaluate a supervised-finetuned chat model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts._common import config_parser, run_config_stub


COMMAND = "eval_chat"


def build_parser() -> argparse.ArgumentParser:
    """Return the chat-evaluation command parser."""

    parser = config_parser(COMMAND, "Evaluate a chat model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to evaluate once chat evaluation is implemented.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chat-evaluation command."""

    return run_config_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
