"""Pretrain a decoder-only language model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts._common import config_parser, run_config_stub


COMMAND = "pretrain"


def build_parser() -> argparse.ArgumentParser:
    """Return the pretraining command parser."""

    return config_parser(COMMAND, "Pretrain a decoder-only language model.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pretraining command."""

    return run_config_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
