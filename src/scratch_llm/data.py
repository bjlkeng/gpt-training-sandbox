"""Small deterministic datasets for the educational training pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as integer_index

import torch
from torch import Tensor
from torch.utils.data import Dataset

from scratch_llm.tokenizer import VOCAB_SIZE


class NextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose every contiguous fixed-length next-token window in a token stream."""

    def __init__(
        self,
        token_ids: Sequence[int],
        seq_len: int,
        *,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self.seq_len = _require_positive_integer(seq_len, name="seq_len")
        self.vocab_size = _require_positive_integer(vocab_size, name="vocab_size")
        validated_ids = [
            _validate_token_id(
                token_id,
                position=position,
                vocab_size=self.vocab_size,
            )
            for position, token_id in enumerate(token_ids)
        ]
        self._token_ids = torch.tensor(validated_ids, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of complete ``seq_len + 1`` windows."""

        return max(self._token_ids.numel() - self.seq_len, 0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return input IDs and their one-token-shifted targets."""

        if isinstance(index, bool):
            raise TypeError("dataset index must be an integer, got bool")
        try:
            index = integer_index(index)
        except TypeError as error:
            raise TypeError(
                f"dataset index must be an integer, got {type(index).__name__}"
            ) from error

        dataset_length = len(self)
        if index < 0:
            index += dataset_length
        if not 0 <= index < dataset_length:
            raise IndexError("dataset index out of range")

        stop = index + self.seq_len
        inputs = self._token_ids[index:stop]
        targets = self._token_ids[index + 1 : stop + 1]
        return inputs, targets


def _require_positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_token_id(token_id: object, *, position: int, vocab_size: int) -> int:
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise TypeError(
            f"token ID at position {position} must be an integer, "
            f"got {type(token_id).__name__}"
        )
    if not 0 <= token_id < vocab_size:
        raise ValueError(
            f"token ID at position {position} must be in range "
            f"[0, {vocab_size}); got {token_id}"
        )
    return token_id
