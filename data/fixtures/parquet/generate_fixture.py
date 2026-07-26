"""Regenerate the repository's tiny synthetic parquet shards."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]


FIXTURE_DIR = Path(__file__).resolve().parent
FIXTURE_ROWS = {
    0: [
        "First synthetic training document.",
        "Unicode train text: café ☕",
        "",
    ],
    1: [
        "Second shard, first document.",
        "你好 from the tiny corpus.",
        "Last training document 🚀",
    ],
    6542: [
        "Fixed validation document.",
        "",
        "Validation Unicode: Καλημέρα.",
    ],
}


def main() -> None:
    """Write each fixed shard with two deterministic row groups."""

    for shard_index, rows in FIXTURE_ROWS.items():
        table = pa.table({"text": pa.array(rows, type=pa.string())})
        destination = FIXTURE_DIR / f"shard_{shard_index:05d}.parquet"
        pq.write_table(
            table,
            destination,
            row_group_size=2,
            compression="NONE",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        )


if __name__ == "__main__":
    main()
