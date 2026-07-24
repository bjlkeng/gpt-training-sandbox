"""Supervised-finetune a base model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scripts._common import config_parser, run_config_stub


COMMAND = "train_sft"


def build_parser() -> argparse.ArgumentParser:
    """Return the supervised-finetuning command parser."""

    return config_parser(COMMAND, "Supervised-finetune a base model.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the supervised-finetuning command."""

    return run_config_stub(build_parser(), command=COMMAND, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
