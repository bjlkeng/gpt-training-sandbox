"""Report deterministic statistics for selected raw ClimbMix parquet data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from scratch_llm.data.climbmix import DEFAULT_CLIMBMIX_DATA_DIR
from scratch_llm.data.statistics import (
    RawDataSplitStatistics,
    compute_raw_data_statistics,
    write_raw_data_statistics,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the raw-data statistics command-line interface."""

    parser = argparse.ArgumentParser(
        prog="python -m scripts.data_stats",
        description="Count selected raw ClimbMix parquet documents and text.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_CLIMBMIX_DATA_DIR,
        help=f"Parquet shard directory (default: {DEFAULT_CLIMBMIX_DATA_DIR}).",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        metavar="N",
        help="Select at most N canonical training shards (default: all local).",
    )
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--include-val",
        dest="include_validation",
        action="store_true",
        help="Include the fixed final validation shard (the default).",
    )
    validation_group.add_argument(
        "--no-val",
        dest="include_validation",
        action="store_false",
        help="Report only the selected training shards.",
    )
    parser.set_defaults(include_validation=True)
    parser.add_argument(
        "--max-documents",
        "--doc-cap",
        dest="max_documents",
        type=int,
        metavar="N",
        help="Count at most N documents per selected split.",
    )
    parser.add_argument(
        "--max-characters",
        "--max-chars",
        dest="max_characters",
        type=int,
        metavar="N",
        help="Count at most N characters per selected split.",
    )
    parser.add_argument(
        "--document-char-cap",
        "--doc-cap-chars",
        dest="document_char_cap",
        type=int,
        metavar="N",
        help="Count at most N characters from each document.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Parquet string column to count (default: text).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="Rows streamed from PyArrow at once (default: 1024).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/data_stats.json"),
        help="Atomic JSON report path (default: metrics/data_stats.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compute, print, and atomically persist raw-data statistics."""

    arguments = build_parser().parse_args(argv)
    try:
        result = compute_raw_data_statistics(
            arguments.data_dir,
            num_train_shards=arguments.num_train_shards,
            include_validation=arguments.include_validation,
            max_documents=arguments.max_documents,
            max_characters=arguments.max_characters,
            document_char_cap=arguments.document_char_cap,
            batch_size=arguments.batch_size,
            text_column=arguments.text_column,
        )
        report_path = write_raw_data_statistics(result, arguments.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(_format_split("train", result.train))
    print(_format_split("validation", result.validation))
    payload = result.to_dict()
    total = payload["total"]
    print(
        "total: "
        f"{_plural(total['selected_shard_count'], 'shard')}, "
        f"{_plural(total['documents'], 'document')}, "
        f"{_plural(total['characters'], 'character')}, "
        f"{_plural(total['utf8_bytes'], 'UTF-8 byte')}"
    )
    print(f"bounded: {'yes' if result.bounded else 'no'}")
    print(f"report: {report_path}")
    return 0


def _format_split(name: str, result: RawDataSplitStatistics) -> str:
    return (
        f"{name}: {_plural(len(result.selected_shards), 'shard')}, "
        f"{_plural(result.documents, 'document')}, "
        f"{_plural(result.characters, 'character')}, "
        f"{_plural(result.utf8_bytes, 'UTF-8 byte')}"
    )


def _plural(value: object, singular: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {singular}{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
