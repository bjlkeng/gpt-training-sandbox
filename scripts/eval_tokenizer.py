"""Evaluate a trained tokenizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts._common import config_parser, run_config_stub


COMMAND = "eval_tokenizer"


def build_parser() -> argparse.ArgumentParser:
    """Return the tokenizer-evaluation command parser."""

    return config_parser(COMMAND, "Evaluate tokenizer quality and throughput.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tokenizer-evaluation command."""

    return run_config_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
