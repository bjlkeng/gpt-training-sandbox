"""Frozen, minimal oracle for nanochat's pinned BOS best-fit evaluator."""

from __future__ import annotations

from typing import Final


REFERENCE_COMMIT: Final = "41865401f73ff1c5321ae53297bceb2b78d4c8b4"
REFERENCE_FILE_SHA256: Final = {
    "nanochat/dataloader.py": (
        "5cc72d7207931f112d685ba8e04c112e1a4ab7756dbbb29b95bdb4908a21864d"
    ),
    "nanochat/loss_eval.py": (
        "00faad1e0ae8912022f79ee4bf583c4f9b4c058e4523c5674144648c49229fd6"
    ),
    "scripts/base_train.py": (
        "d806cfa36d51f246186bd24e8693cc09ddcf96545ab5c5355a3450d1eddfd8ac"
    ),
}

# Two tokenizer batches repeat in source order. The frozen rows exercise:
# - a five-token largest fit followed by a one-token shortest crop,
# - equal-length largest-fit ties selecting the first document,
# - refill of a whole document batch when the buffer drops below its threshold,
# - a second validation cycle producing the same model rows.
BOS_TOKEN_ID: Final = 99
DOCUMENT_BATCHES: Final = (
    (
        (0, (10, 11)),
        (1, (20, 21)),
    ),
    (
        (2, (30, 31, 32, 33)),
        (3, (40,)),
        (4, (50, 51, 52, 53, 54, 55)),
    ),
)
EXPECTED_INPUT_BATCH: Final = (
    (99, 30, 31, 32, 33),
    (99, 10, 11, 99, 20),
)
EXPECTED_TARGET_BATCH: Final = (
    (30, 31, 32, 33, 99),
    (10, 11, 99, 20, 21),
)


__all__ = [
    "BOS_TOKEN_ID",
    "DOCUMENT_BATCHES",
    "EXPECTED_INPUT_BATCH",
    "EXPECTED_TARGET_BATCH",
    "REFERENCE_COMMIT",
    "REFERENCE_FILE_SHA256",
]
