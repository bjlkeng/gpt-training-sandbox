"""Chat with a supervised-finetuned checkpoint in the terminal."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts._common import checkpoint_parser, run_checkpoint_stub


COMMAND = "chat"


def build_parser() -> argparse.ArgumentParser:
    """Return the terminal chat command parser."""

    parser = checkpoint_parser(COMMAND, "Chat with a model in the terminal.")
    parser.add_argument(
        "-p",
        "--prompt",
        help="Run one prompt non-interactively instead of opening a chat loop.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the terminal chat command."""

    return run_checkpoint_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
