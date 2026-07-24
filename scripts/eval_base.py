"""Evaluate a pretrained base model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts._common import config_parser, run_config_stub


COMMAND = "eval_base"


def build_parser() -> argparse.ArgumentParser:
    """Return the base-model evaluation command parser."""

    parser = config_parser(COMMAND, "Evaluate a pretrained base model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to evaluate once base evaluation is implemented.",
    )
    parser.add_argument(
        "--eval",
        default="bpb,sample",
        metavar="MODES",
        help="Comma-separated evaluation modes.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        help="Optional maximum examples per CORE task.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the base-model evaluation command."""

    return run_config_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
