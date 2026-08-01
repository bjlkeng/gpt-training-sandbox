"""Prepare tracked tokenized shards from canonical local parquet inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scratch_llm.data_preparation import (
    DataPreparationError,
    prepare_tracked_tokenized_parquet_shards,
)
from scratch_llm.pretraining import PretrainingError, load_production_tokenizer
from scratch_llm.tokenized_data import TokenizedDataError
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "prepare_data"


def build_parser() -> argparse.ArgumentParser:
    """Return the tracked data-preparation parser."""

    parser = config_parser(
        COMMAND,
        "Tokenize the configured ClimbMix parquet selection into local shards.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Parquet rows encoded per streaming batch (default: 1024).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the configured tokenized dataset after full validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve configuration and prepare or describe the tokenized dataset."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if config.data.profile != "nanochat_climbmix":
        parser.error("prepare_data requires data.profile=nanochat_climbmix")
    if config.tokenizer.type != "regex_byte_bpe":
        parser.error("prepare_data requires tokenizer.type=regex_byte_bpe")

    paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
    if arguments.dry_run:
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
            print(f"Parquet input: {config.data.parquet_dir}")
            print(f"Tokenized output: {config.data.tokenized_dir}")
            print(f"Train shards: {config.data.num_pretrain_train_shards}")
            print(f"Encoding batch size: {arguments.batch_size}")
        return 0

    with tracker:
        try:
            tokenizer = load_production_tokenizer(config)
            result = prepare_tracked_tokenized_parquet_shards(
                config.data.parquet_dir,
                config.data.tokenized_dir,
                tokenizer=tokenizer,
                tracker=tracker,
                run_dir=paths.run_dir,
                num_train_shards=config.data.num_pretrain_train_shards,
                batch_size=arguments.batch_size,
                text_column=config.data.text_column,
                overwrite=arguments.overwrite,
            )
        except (
            DataPreparationError,
            OSError,
            PretrainingError,
            RuntimeError,
            TokenizedDataError,
            TypeError,
            ValueError,
        ) as error:
            parser.error(str(error))

    print(f"Run directory: {paths.run_dir}")
    print(f"Tokenized output: {config.data.tokenized_dir}")
    print(f"Manifest: {result.tokenized_manifest_path}")
    print(f"Reused tokenized data: {result.reused_tokenized_data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
