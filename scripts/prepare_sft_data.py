"""Prepare and validate one normalized SFT parquet source."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.data.sft_sources import (
    SFTConversationDataset,
    SFTDatasetError,
    get_sft_dataset_spec,
    preview_examples_identity,
)
from scratch_llm.data.hub import (
    HubParquetError,
    prepare_hub_parquet_cache,
    publish_local_parquet_cache,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the bounded SFT source preparation parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Download or inject one supported SFT parquet source, verify its "
            "cache, and normalize a bounded deterministic preview."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("smoltalk", "mmlu", "gsm8k"),
        help="Normalized SFT dataset adapter.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Dataset split; allowed values depend on --dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/parquet/sft"),
        help="Root for verified versioned parquet caches.",
    )
    parser.add_argument(
        "--local-parquet",
        type=Path,
        action="append",
        default=[],
        help=(
            "Inject a local parquet shard through the normal staging checks; "
            "repeat for multiple shards."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the deterministic bounded preview (default: 42).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of normalized preview conversations (default: 3).",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=1024,
        help="Maximum source rows retained by seeded streaming (default: 1024).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved source without discovery, download, or writes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare one source or print its network-free dry-run plan."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        spec = get_sft_dataset_spec(arguments.dataset, arguments.split)
    except SFTDatasetError as error:
        parser.error(str(error))

    if arguments.limit <= 0:
        parser.error("--limit must be a positive integer")
    if arguments.shuffle_buffer_size <= 0:
        parser.error("--shuffle-buffer-size must be a positive integer")
    if not 0 <= arguments.seed <= 2**32 - 1:
        parser.error("--seed must be in range [0, 4294967295]")

    print(f"Dataset: {spec.dataset}/{spec.subset}/{spec.split}")
    print(f"Repository: {spec.repository}")
    print(f"Contract identity: {spec.source_identity}")
    print(f"Cache target: {arguments.cache_dir / spec.cache_key}")
    if arguments.dry_run:
        print("Dry run: no discovery, download, or cache mutation")
        return 0

    try:
        if arguments.local_parquet:
            cache = publish_local_parquet_cache(
                spec,
                arguments.cache_dir,
                tuple(arguments.local_parquet),
            )
        else:
            cache = prepare_hub_parquet_cache(spec, arguments.cache_dir)
        source = SFTConversationDataset(
            cache,
            shuffle_buffer_size=arguments.shuffle_buffer_size,
        )
        examples = tuple(
            source.iter_examples(seed=arguments.seed, stop=arguments.limit)
        )
    except (HubParquetError, OSError, SFTDatasetError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Cache directory: {cache.directory}")
    print(f"Source identity: {source.source_identity}")
    print(f"Rows: {len(source)}")
    print(f"Validated conversations: {len(examples)}")
    print(f"Preview identity: {preview_examples_identity(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
