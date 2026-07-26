"""Download a bounded ClimbMix train prefix and optional validation shard."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import sys

from scratch_llm.climbmix import (
    CLIMBMIX_BASE_URL,
    DEFAULT_CLIMBMIX_DATA_DIR,
    ClimbMixDownloadError,
    ResponseOpener,
    download_climbmix_targets,
    plan_climbmix_downloads,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        prog="python -m scripts.download_climbmix",
        description="Download a partial ClimbMix parquet dataset safely.",
    )
    parser.add_argument(
        "--num-train-shards",
        type=int,
        required=True,
        metavar="N",
        help="Number of leading training shards to download.",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also download the fixed final validation shard.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_CLIMBMIX_DATA_DIR,
        help=f"Destination directory (default: {DEFAULT_CLIMBMIX_DATA_DIR}).",
    )
    parser.add_argument(
        "--base-url",
        default=CLIMBMIX_BASE_URL,
        help="Base URL containing canonical shard filenames.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum attempts per shard (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="HTTP timeout per request (default: 30).",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Initial retry delay before exponential backoff (default: 1).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    opener: ResponseOpener | None = None,
    sleep: Callable[[float], None] | None = None,
) -> int:
    """Run the ClimbMix downloader."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    if arguments.backoff_base < 0:
        parser.error("--backoff-base must be non-negative")

    try:
        targets = plan_climbmix_downloads(
            num_train_shards=arguments.num_train_shards,
            include_val=arguments.include_val,
            data_dir=arguments.data_dir,
            base_url=arguments.base_url,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Planned {len(targets)} shards in {arguments.data_dir}",
        file=sys.stderr,
    )
    try:
        summary = download_climbmix_targets(
            targets,
            opener=opener,
            timeout=arguments.timeout,
            max_attempts=arguments.max_attempts,
            backoff_base=arguments.backoff_base,
            sleep=sleep,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except ClimbMixDownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "data_dir": str(arguments.data_dir),
                "downloaded_shards": summary.downloaded_shards,
                "planned_shards": summary.planned_shards,
                "ready_shards": summary.ready_shards,
                "skipped_shards": summary.skipped_shards,
                "total_bytes": summary.total_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
