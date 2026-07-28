"""Train the project tokenizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from scratch_llm.bpe_optimized import (
    benchmark_bpe_trainers,
    write_bpe_training_benchmark,
)
from scratch_llm.tokenizer_training import (
    collect_bounded_parquet_training_texts,
    train_tokenizer_from_parquet,
)
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "train_tokenizer"


def build_parser() -> argparse.ArgumentParser:
    """Return the tokenizer-training command parser."""

    parser = config_parser(COMMAND, "Train a regex byte-BPE tokenizer.")
    parser.add_argument(
        "--algorithm",
        choices=("optimized", "reference"),
        default="optimized",
        help=(
            "Pair-counting implementation; optimized is the scalable default "
            "and reference is the readable fallback."
        ),
    )
    parser.add_argument(
        "--benchmark-trainers",
        action="store_true",
        help=(
            "Before training, compare reference and optimized implementations "
            "on an explicitly bounded corpus."
        ),
    )
    parser.add_argument(
        "--benchmark-vocab-size",
        type=int,
        metavar="N",
        help=(
            "Vocabulary size for the bounded comparison "
            "(default: min(configured vocab, 512))."
        ),
    )
    parser.add_argument(
        "--benchmark-max-documents",
        type=int,
        default=64,
        metavar="N",
        help="Maximum benchmark documents (default: 64).",
    )
    parser.add_argument(
        "--benchmark-max-characters",
        type=int,
        default=100_000,
        metavar="N",
        help="Maximum benchmark Unicode characters (default: 100000).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tokenizer-training command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = resolve_config_arguments(parser, arguments)

    if arguments.dry_run:
        paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
        with tracker:
            print(f"Run directory: {paths.run_dir}")
            print(f"Resolved config: {paths.config_path}")
            print("Resolved values:")
            print(config.to_yaml(), end="")
        return 0

    paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
    benchmark_path = None
    with tracker:
        try:
            if arguments.benchmark_trainers:
                benchmark_documents = collect_bounded_parquet_training_texts(
                    config,
                    max_documents=arguments.benchmark_max_documents,
                    max_characters=arguments.benchmark_max_characters,
                )
                benchmark = benchmark_bpe_trainers(
                    benchmark_documents,
                    vocab_size=(
                        arguments.benchmark_vocab_size
                        if arguments.benchmark_vocab_size is not None
                        else min(config.tokenizer.vocab_size, 512)
                    ),
                    max_documents=arguments.benchmark_max_documents,
                    max_characters=arguments.benchmark_max_characters,
                )
                if not benchmark.equivalent:
                    raise RuntimeError(
                        "optimized BPE trainer disagreed with the reference "
                        "during the bounded benchmark"
                    )
                benchmark_path = write_bpe_training_benchmark(
                    benchmark,
                    paths.metrics_dir / "bpe_training_benchmark.json",
                )
            result = train_tokenizer_from_parquet(
                config,
                paths,
                algorithm=arguments.algorithm,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(str(error))

    print(f"Algorithm: {result.algorithm}")
    print(f"Vocabulary size: {result.training_result.vocab_size}")
    print(f"Documents: {result.training_result.document_count}")
    print(f"Characters: {result.training_result.character_count}")
    print(f"Tokenizer artifacts: {result.artifact_dir}")
    print(f"Training report: {result.report_path}")
    if benchmark_path is not None:
        print(f"Trainer benchmark: {benchmark_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
