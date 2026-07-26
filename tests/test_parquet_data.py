"""Tests for deterministic ClimbMix-style parquet discovery and iteration."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from scratch_llm.data import (
    CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    list_parquet_files,
    parquets_iter_batched,
    select_parquet_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "parquet"
PARQUET_FIXTURE_README = PARQUET_FIXTURE_DIR / "README.md"
TINY_TEXT_PATH = PROJECT_ROOT / "data" / "fixtures" / "tiny.txt"


def _write_parquet(
    path: Path,
    values: list[object],
    *,
    column: str = "text",
    row_group_size: int = 2,
) -> None:
    pq.write_table(
        pa.table({column: values}),
        path,
        row_group_size=row_group_size,
    )


def test_list_parquet_files_filters_and_sorts_canonical_shards(
    tmp_path: Path,
) -> None:
    expected_names = [
        "shard_1.parquet",
        "shard_2.parquet",
        "shard_10.parquet",
    ]
    for name in reversed(expected_names):
        (tmp_path / name).touch()

    (tmp_path / "notes.parquet").touch()
    (tmp_path / "shard_3.json").touch()
    (tmp_path / "shard_-1.parquet").touch()
    (tmp_path / "shard_4.parquet.tmp").touch()
    (tmp_path / "shard_5.parquet").mkdir()

    discovered = list_parquet_files(tmp_path)

    assert [path.name for path in discovered] == expected_names


def test_list_parquet_files_rejects_duplicate_numeric_indices(
    tmp_path: Path,
) -> None:
    (tmp_path / "shard_1.parquet").touch()
    (tmp_path / "shard_00001.parquet").touch()

    with pytest.raises(
        ValueError,
        match=r"duplicate parquet shard index 1.*shard_00001\.parquet.*"
        r"shard_1\.parquet",
    ):
        list_parquet_files(tmp_path)


def test_list_parquet_files_reports_missing_or_non_directory_paths(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match=r"parquet data directory.*missing"):
        list_parquet_files(missing)

    regular_file = tmp_path / "file"
    regular_file.touch()
    with pytest.raises(NotADirectoryError, match=r"parquet data path.*file"):
        list_parquet_files(regular_file)


def test_select_parquet_files_uses_train_prefix_and_fixed_validation_shard(
    tmp_path: Path,
) -> None:
    files = [
        tmp_path / "shard_06542.parquet",
        tmp_path / "shard_00001.parquet",
        tmp_path / "shard_00000.parquet",
    ]

    train = select_parquet_files(files, "train", num_train_shards=1)
    all_available_train = select_parquet_files(
        files,
        "train",
        num_train_shards=10,
    )
    validation = select_parquet_files(files, "val")

    assert [path.name for path in train] == ["shard_00000.parquet"]
    assert [path.name for path in all_available_train] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
    ]
    assert [path.name for path in validation] == ["shard_06542.parquet"]
    assert CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX == 6542


def test_select_parquet_files_handles_partial_local_datasets_without_leakage(
    tmp_path: Path,
) -> None:
    train_only = [
        tmp_path / "shard_00002.parquet",
        tmp_path / "shard_00000.parquet",
    ]

    selected = select_parquet_files(
        train_only,
        "train",
        num_train_shards=3,
    )

    assert [path.name for path in selected] == [
        "shard_00000.parquet",
        "shard_00002.parquet",
    ]
    with pytest.raises(
        FileNotFoundError,
        match=r"fixed validation parquet shard.*6542.*not found",
    ):
        select_parquet_files(train_only, "val")


@pytest.mark.parametrize("split", ["validation", "test", "", 1, None])
def test_select_parquet_files_rejects_unknown_splits(
    tmp_path: Path,
    split: object,
) -> None:
    with pytest.raises(ValueError, match=r"split must be 'train' or 'val'"):
        select_parquet_files(
            [tmp_path / "shard_00000.parquet"],
            split,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("num_train_shards", [-1, True, 1.5, "1"])
def test_select_parquet_files_rejects_invalid_train_prefix_sizes(
    tmp_path: Path,
    num_train_shards: object,
) -> None:
    with pytest.raises(
        (TypeError, ValueError),
        match="num_train_shards",
    ):
        select_parquet_files(
            [tmp_path / "shard_00000.parquet"],
            "train",
            num_train_shards=num_train_shards,  # type: ignore[arg-type]
        )


def test_select_parquet_files_rejects_shards_after_configured_final_index(
    tmp_path: Path,
) -> None:
    files = [
        tmp_path / "shard_06542.parquet",
        tmp_path / "shard_06543.parquet",
    ]

    with pytest.raises(
        ValueError,
        match=r"shard index 6543 exceeds.*validation.*6542",
    ):
        select_parquet_files(files, "train")


def test_parquets_iter_batched_streams_bounded_file_strides(
    tmp_path: Path,
) -> None:
    _write_parquet(tmp_path / "shard_00000.parquet", ["0a", "0b", "0c"])
    _write_parquet(tmp_path / "shard_00001.parquet", ["1a", "1b"])
    _write_parquet(tmp_path / "shard_00002.parquet", ["2a", "2b", "2c"])
    _write_parquet(tmp_path / "shard_06542.parquet", ["validation"])

    even_file_batches = list(
        parquets_iter_batched(
            "train",
            start=0,
            step=2,
            data_dir=tmp_path,
            num_train_shards=3,
            batch_size=2,
        )
    )
    odd_file_batches = list(
        parquets_iter_batched(
            "train",
            start=1,
            step=2,
            data_dir=tmp_path,
            num_train_shards=3,
            batch_size=2,
        )
    )

    assert even_file_batches == [["0a", "0b"], ["0c"], ["2a", "2b"], ["2c"]]
    assert odd_file_batches == [["1a", "1b"]]
    assert all(
        isinstance(batch, list)
        and 0 < len(batch) <= 2
        and all(isinstance(text, str) for text in batch)
        for batch in even_file_batches + odd_file_batches
    )


def test_parquets_iter_batched_selects_only_fixed_validation_shard(
    tmp_path: Path,
) -> None:
    _write_parquet(tmp_path / "shard_00000.parquet", ["train"])
    _write_parquet(
        tmp_path / "shard_00007.parquet",
        ["val one", "val two"],
        row_group_size=1,
    )

    batches = list(
        parquets_iter_batched(
            "val",
            data_dir=tmp_path,
            validation_shard_index=7,
            batch_size=8,
        )
    )

    assert batches == [["val one", "val two"]]


def test_parquets_iter_batched_supports_a_configured_text_column(
    tmp_path: Path,
) -> None:
    _write_parquet(
        tmp_path / "shard_00000.parquet",
        ["first", "second"],
        column="body",
    )

    batches = list(
        parquets_iter_batched(
            "train",
            data_dir=tmp_path,
            text_column="body",
            batch_size=1,
        )
    )

    assert batches == [["first"], ["second"]]


def test_parquets_iter_batched_rejects_missing_or_non_string_columns(
    tmp_path: Path,
) -> None:
    missing_column = tmp_path / "missing"
    missing_column.mkdir()
    _write_parquet(
        missing_column / "shard_00000.parquet",
        ["value"],
        column="body",
    )

    with pytest.raises(
        ValueError,
        match=r"text column 'text'.*not found.*shard_00000\.parquet",
    ):
        list(parquets_iter_batched("train", data_dir=missing_column))

    wrong_type = tmp_path / "wrong-type"
    wrong_type.mkdir()
    _write_parquet(wrong_type / "shard_00000.parquet", [1, 2, 3])

    with pytest.raises(
        TypeError,
        match=r"text column 'text'.*string.*int64.*shard_00000\.parquet",
    ):
        list(parquets_iter_batched("train", data_dir=wrong_type))


def test_parquets_iter_batched_rejects_null_text_values(tmp_path: Path) -> None:
    _write_parquet(tmp_path / "shard_00000.parquet", ["valid", None])

    with pytest.raises(
        ValueError,
        match=r"text column 'text'.*null.*shard_00000\.parquet",
    ):
        list(parquets_iter_batched("train", data_dir=tmp_path))


@pytest.mark.parametrize("split", ["validation", "test", "", 1, None])
def test_parquets_iter_batched_rejects_unknown_splits(
    tmp_path: Path,
    split: object,
) -> None:
    with pytest.raises(ValueError, match=r"split must be 'train' or 'val'"):
        list(
            parquets_iter_batched(
                split,  # type: ignore[arg-type]
                data_dir=tmp_path,
            )
        )


@pytest.mark.parametrize(
    ("start", "step", "error_type", "message"),
    [
        (-1, 1, ValueError, "start"),
        (True, 1, TypeError, "start"),
        (0, 0, ValueError, "step"),
        (0, True, TypeError, "step"),
        (0, 1.5, TypeError, "step"),
        (2, 2, ValueError, r"start.*less than step"),
    ],
)
def test_parquets_iter_batched_validates_file_stride(
    tmp_path: Path,
    start: object,
    step: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        list(
            parquets_iter_batched(
                "train",
                start=start,  # type: ignore[arg-type]
                step=step,  # type: ignore[arg-type]
                data_dir=tmp_path,
            )
        )


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_parquets_iter_batched_requires_a_positive_integer_batch_size(
    tmp_path: Path,
    batch_size: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="batch_size"):
        list(
            parquets_iter_batched(
                "train",
                data_dir=tmp_path,
                batch_size=batch_size,  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("text_column", ["", "  ", 1, None])
def test_parquets_iter_batched_requires_a_nonempty_text_column(
    tmp_path: Path,
    text_column: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="text_column"):
        list(
            parquets_iter_batched(
                "train",
                data_dir=tmp_path,
                text_column=text_column,  # type: ignore[arg-type]
            )
        )


def test_checked_in_parquet_fixture_covers_split_and_text_edge_cases() -> None:
    files = list_parquet_files(PARQUET_FIXTURE_DIR)

    assert [path.name for path in files] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_06542.parquet",
    ]
    assert all(pq.ParquetFile(path).num_row_groups >= 2 for path in files)

    train_texts = [
        text
        for batch in parquets_iter_batched(
            "train",
            data_dir=PARQUET_FIXTURE_DIR,
            num_train_shards=2,
            batch_size=2,
        )
        for text in batch
    ]
    validation_texts = [
        text
        for batch in parquets_iter_batched(
            "val",
            data_dir=PARQUET_FIXTURE_DIR,
            batch_size=2,
        )
        for text in batch
    ]

    assert train_texts == [
        "First synthetic training document.",
        "Unicode train text: café ☕",
        "",
        "Second shard, first document.",
        "你好 from the tiny corpus.",
        "Last training document 🚀",
    ]
    assert validation_texts == [
        "Fixed validation document.",
        "",
        "Validation Unicode: Καλημέρα.",
    ]
    assert {text for text in train_texts if text}.isdisjoint(
        text for text in validation_texts if text
    )


def test_parquet_fixture_has_provenance_and_coexists_with_tiny_text() -> None:
    provenance = PARQUET_FIXTURE_README.read_text(encoding="utf-8")

    assert "synthetic" in provenance.lower()
    assert "authored for this repository" in provenance.lower()
    assert "generate_fixture.py" in provenance
    assert "shard_06542.parquet" in provenance
    assert TINY_TEXT_PATH.read_text(encoding="utf-8").strip()
