"""Small, deterministic tokenizers used by the educational pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import Final, Mapping


BYTE_VOCAB_SIZE: Final = 256
SPECIAL_TOKENS: Final = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)
SPECIAL_TOKEN_IDS: Final[Mapping[str, int]] = MappingProxyType(
    {token: BYTE_VOCAB_SIZE + offset for offset, token in enumerate(SPECIAL_TOKENS)}
)
_SPECIAL_TOKENS_BY_ID: Final[Mapping[int, str]] = MappingProxyType(
    {token_id: token for token, token_id in SPECIAL_TOKEN_IDS.items()}
)
VOCAB_SIZE: Final = BYTE_VOCAB_SIZE + len(SPECIAL_TOKENS)


class ByteTokenizer:
    """Encode text as UTF-8 bytes plus a small, fixed control vocabulary."""

    def encode(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]:
        """Return one token ID for each byte in ``text``'s UTF-8 encoding."""

        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        token_ids = list(text.encode("utf-8"))
        if prepend is not None:
            token_ids.insert(
                0, self._resolve_special_token(prepend, argument="prepend")
            )
        if append is not None:
            token_ids.append(self._resolve_special_token(append, argument="append"))
        return token_ids

    def __call__(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
    ) -> list[int]:
        """Alias for :meth:`encode`."""

        return self.encode(text, prepend=prepend, append=append)

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode IDs to text, replacing malformed generated UTF-8 sequences."""

        encoded = bytearray()
        for position, token_id in enumerate(token_ids):
            token_id = self._validate_token_id(token_id, position=position)
            if token_id < BYTE_VOCAB_SIZE:
                encoded.append(token_id)
            else:
                encoded.extend(_SPECIAL_TOKENS_BY_ID[token_id].encode("utf-8"))
        return encoded.decode("utf-8", errors="replace")

    def encode_special(self, token: str) -> int:
        """Return the locked ID for a supported special token."""

        if not isinstance(token, str):
            raise TypeError(
                f"special token must be a string, got {type(token).__name__}"
            )
        try:
            return SPECIAL_TOKEN_IDS[token]
        except KeyError as error:
            raise ValueError(f"unsupported special token {token!r}") from error

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        """Return the raw bytes represented by one token ID."""

        token_id = self._validate_token_id(token_id)
        if token_id < BYTE_VOCAB_SIZE:
            return bytes([token_id])
        return _SPECIAL_TOKENS_BY_ID[token_id].encode("utf-8")

    def get_vocab_size(self) -> int:
        """Return the complete byte-plus-control vocabulary size."""

        return VOCAB_SIZE

    def get_bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID."""

        return SPECIAL_TOKEN_IDS["<|bos|>"]

    def get_special_tokens(self) -> set[str]:
        """Return a copy of the supported special-token names."""

        return set(SPECIAL_TOKENS)

    def _resolve_special_token(self, token: str | int, *, argument: str) -> int:
        if isinstance(token, str):
            return self.encode_special(token)
        if not isinstance(token, int) or isinstance(token, bool):
            raise TypeError(
                f"{argument} special token must be a supported token string "
                f"or integer ID, got {type(token).__name__}"
            )
        if token not in _SPECIAL_TOKENS_BY_ID:
            raise ValueError(
                f"{argument} special token ID must be in range "
                f"[{BYTE_VOCAB_SIZE}, {VOCAB_SIZE}); got {token}"
            )
        return token

    @staticmethod
    def _validate_token_id(token_id: object, *, position: int | None = None) -> int:
        label = "token ID" if position is None else f"token ID at position {position}"
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(
                f"{label} must be an integer, got {type(token_id).__name__}"
            )
        if not 0 <= token_id < VOCAB_SIZE:
            raise ValueError(
                f"{label} must be in range [0, {VOCAB_SIZE}); got {token_id}"
            )
        return token_id
