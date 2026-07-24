"""Train the project tokenizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts._common import config_parser, run_config_stub


COMMAND = "train_tokenizer"


def build_parser() -> argparse.ArgumentParser:
    """Return the tokenizer-training command parser."""

    return config_parser(COMMAND, "Train a regex byte-BPE tokenizer.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tokenizer-training command."""

    return run_config_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
