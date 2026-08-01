"""Evaluate a trained tokenizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.tokenization.bpe import RegexBPETokenizer
from scratch_llm.tokenization.tokenizer import ByteTokenizer, Tokenizer
from scratch_llm.evaluation.tokenizer import (
    collect_evaluation_corpora,
    evaluate_tokenizer,
    write_tokenizer_evaluation_reports,
)
from scratch_llm.evaluation.tokenizer_tracking import track_tokenizer_evaluation
from scripts._common import (
    config_parser,
    prepare_tracked_run,
    resolve_config_arguments,
)


COMMAND = "eval_tokenizer"


def build_parser() -> argparse.ArgumentParser:
    """Return the tokenizer-evaluation command parser."""

    parser = config_parser(COMMAND, "Evaluate tokenizer quality and throughput.")
    parser.add_argument(
        "--tokenizer-artifacts",
        type=Path,
        help=(
            "Saved regex byte-BPE artifact directory; required when "
            "tokenizer.type is regex_byte_bpe."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="ClimbMix-style parquet directory (default: config data.parquet_dir).",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        help=(
            "Training-shard prefix to sample "
            "(default: config data.num_tokenizer_train_shards)."
        ),
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=32,
        metavar="N",
        help="Evaluate at most N documents from each ClimbMix split (default: 32).",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=100_000,
        metavar="N",
        help=(
            "Evaluate at most N Unicode characters from each ClimbMix split "
            "(default: 100000)."
        ),
    )
    parser.add_argument(
        "--document-char-cap",
        type=int,
        metavar="N",
        help=(
            "Retain at most N characters per ClimbMix document "
            "(default: config data.doc_cap_chars)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="Rows streamed from PyArrow at once (default: 1024).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Opt in to GPT-2 and cl100k token-count comparisons; requires the "
            "tokenizer-comparison extra."
        ),
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        metavar="N",
        help="Untimed encode and decode warmup iterations (default: 1).",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=3,
        metavar="N",
        help="Timed encode and decode iterations (default: 3).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tokenizer-evaluation command."""

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

    if (
        config.tokenizer.type == "regex_byte_bpe"
        and arguments.tokenizer_artifacts is None
    ):
        parser.error(
            "--tokenizer-artifacts is required when tokenizer.type is regex_byte_bpe"
        )
    if config.tokenizer.type == "byte" and arguments.tokenizer_artifacts is not None:
        parser.error("--tokenizer-artifacts cannot be used when tokenizer.type is byte")

    paths, tracker = prepare_tracked_run(parser, config, command=COMMAND)
    with tracker:
        try:
            tokenizer = _load_tokenizer(
                config.tokenizer.type,
                arguments.tokenizer_artifacts,
                expected_vocab_size=config.tokenizer.vocab_size,
            )
            data_dir = (
                arguments.data_dir
                if arguments.data_dir is not None
                else Path(config.data.parquet_dir)
            )
            corpora = collect_evaluation_corpora(
                data_dir,
                num_train_shards=(
                    arguments.num_train_shards
                    if arguments.num_train_shards is not None
                    else config.data.num_tokenizer_train_shards
                ),
                max_documents=arguments.max_documents,
                max_characters=arguments.max_characters,
                document_char_cap=(
                    arguments.document_char_cap
                    if arguments.document_char_cap is not None
                    else config.data.doc_cap_chars
                ),
                batch_size=arguments.batch_size,
                text_column=config.data.text_column,
            )
            result = evaluate_tokenizer(
                tokenizer,
                corpora,
                compare=arguments.compare,
                benchmark_warmup_iterations=arguments.benchmark_warmup,
                benchmark_iterations=arguments.benchmark_iterations,
            )
            json_path, markdown_path = write_tokenizer_evaluation_reports(
                result,
                paths.metrics_dir,
            )
            track_tokenizer_evaluation(
                result,
                json_path,
                tracker=tracker,
                run_dir=paths.run_dir,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(str(error))

    tokenizer_label = (
        str(arguments.tokenizer_artifacts)
        if arguments.tokenizer_artifacts is not None
        else "built-in byte tokenizer"
    )
    aggregate = result.to_dict()["aggregate"]
    print(f"Tokenizer: {tokenizer_label}")
    print(f"Vocabulary size: {result.vocab_size}")
    print(
        "Aggregate: "
        f"{aggregate['bytes']} bytes, {aggregate['tokens']} tokens, "
        f"{aggregate['bytes_per_token']:.3f} bytes/token"
    )
    print(f"Round trip: {'pass' if aggregate['round_trip'] else 'fail'}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


def _load_tokenizer(
    tokenizer_type: str,
    artifact_path: Path | None,
    *,
    expected_vocab_size: int,
) -> Tokenizer:
    if tokenizer_type == "byte":
        tokenizer: Tokenizer = ByteTokenizer()
    elif tokenizer_type == "regex_byte_bpe":
        if artifact_path is None:
            raise ValueError("regex byte-BPE evaluation requires artifact path")
        tokenizer = RegexBPETokenizer.load(artifact_path)
    else:
        raise ValueError(f"unsupported tokenizer type {tokenizer_type!r}")
    if tokenizer.get_vocab_size() != expected_vocab_size:
        raise ValueError(
            "loaded tokenizer vocabulary size "
            f"{tokenizer.get_vocab_size()} does not match configured "
            f"tokenizer.vocab_size {expected_vocab_size}"
        )
    return tokenizer


if __name__ == "__main__":
    raise SystemExit(main())
