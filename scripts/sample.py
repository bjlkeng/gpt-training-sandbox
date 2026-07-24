"""Generate text from a pretrained checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts._common import checkpoint_parser, run_checkpoint_stub


COMMAND = "sample"


def build_parser() -> argparse.ArgumentParser:
    """Return the Sprint-compatible base sampling command parser."""

    parser = checkpoint_parser(COMMAND, "Sample text from a base-model checkpoint.")
    parser.add_argument(
        "-p",
        "--prompt",
        action="append",
        help="Optional prompt; repeat to sample from multiple prompts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the base sampling command."""

    return run_checkpoint_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
