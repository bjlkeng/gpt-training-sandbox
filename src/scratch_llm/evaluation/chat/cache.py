"""Shared read-only row loading for verified chat-evaluation parquet caches."""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scratch_llm.data.hub import CachedHubParquetDataset


class ChatEvaluationCacheError(ValueError):
    """A verified chat-evaluation cache cannot be materialized."""


def read_cached_parquet_rows(
    cache: CachedHubParquetDataset,
    *,
    additional_columns: tuple[str, ...] = (),
) -> tuple[Mapping[str, object], ...]:
    """Read selected columns from one fully verified cache in source order."""

    if not isinstance(cache, CachedHubParquetDataset):
        raise TypeError("cache must be a CachedHubParquetDataset")
    if not isinstance(additional_columns, tuple) or any(
        not isinstance(column, str) or not column for column in additional_columns
    ):
        raise TypeError("additional_columns must be a tuple of non-empty strings")
    if len(set(additional_columns)) != len(additional_columns):
        raise ValueError("additional_columns must be unique")
    columns = tuple(dict.fromkeys((*cache.spec.required_columns, *additional_columns)))
    try:
        tables = tuple(
            pq.read_table(path, columns=list(columns)) for path in cache.shard_paths
        )
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        rows = table.to_pylist()
    except (OSError, pa.ArrowException) as error:
        raise ChatEvaluationCacheError(
            f"could not read cached chat-evaluation parquet: {error}"
        ) from error
    if len(rows) != cache.row_count:
        raise ChatEvaluationCacheError(
            f"read {len(rows)} rows but manifest records {cache.row_count}"
        )
    if not all(isinstance(row, Mapping) for row in rows):
        raise ChatEvaluationCacheError("cached parquet contains a non-object row")
    return tuple(rows)


__all__ = ["ChatEvaluationCacheError", "read_cached_parquet_rows"]
