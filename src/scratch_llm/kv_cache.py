"""Bounded, external, transactional per-layer key/value storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class KVCacheError(RuntimeError):
    """A cache request would violate shape, ownership, or transaction state."""


@dataclass(frozen=True)
class KVCacheMetadata:
    layer_count: int
    batch_size: int
    kv_head_count: int
    head_dimension: int
    capacity: int
    device: torch.device
    dtype: torch.dtype
    layer_shape: tuple[int, int, int, int]
    bytes_per_token: int
    allocated_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "batch_size": self.batch_size,
            "bytes_per_token": self.bytes_per_token,
            "capacity": self.capacity,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "head_dimension": self.head_dimension,
            "kv_head_count": self.kv_head_count,
            "layer_count": self.layer_count,
            "layer_shape": list(self.layer_shape),
        }


class KVCache:
    """Preallocated K/V tensors whose logical position advances atomically."""

    def __init__(
        self,
        *,
        layer_count: int,
        batch_size: int,
        kv_head_count: int,
        head_dimension: int,
        capacity: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        for name, value in (
            ("layer_count", layer_count),
            ("batch_size", batch_size),
            ("kv_head_count", kv_head_count),
            ("head_dimension", head_dimension),
            ("capacity", capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise KVCacheError(f"{name} must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise KVCacheError("dtype must be a floating-point torch dtype")
        resolved_device = torch.device(device)
        shape = (
            layer_count,
            batch_size,
            kv_head_count,
            capacity,
            head_dimension,
        )
        self._keys = torch.empty(shape, device=resolved_device, dtype=dtype)
        self._values = torch.empty(shape, device=resolved_device, dtype=dtype)
        self._position = 0
        self._active_transaction: KVCacheTransaction | None = None
        layer_shape = (batch_size, kv_head_count, capacity, head_dimension)
        bytes_per_token = (
            2
            * layer_count
            * batch_size
            * kv_head_count
            * head_dimension
            * self._keys.element_size()
        )
        self.metadata = KVCacheMetadata(
            layer_count=layer_count,
            batch_size=batch_size,
            kv_head_count=kv_head_count,
            head_dimension=head_dimension,
            capacity=capacity,
            device=resolved_device,
            dtype=dtype,
            layer_shape=layer_shape,
            bytes_per_token=bytes_per_token,
            allocated_bytes=bytes_per_token * capacity,
        )

    @property
    def position(self) -> int:
        return self._position

    @property
    def layer_shape(self) -> tuple[int, int, int, int]:
        return self.metadata.layer_shape

    @property
    def bytes_per_token(self) -> int:
        return self.metadata.bytes_per_token

    @property
    def allocated_bytes(self) -> int:
        return self.metadata.allocated_bytes

    def layer_keys(self, layer_index: int) -> torch.Tensor:
        self._validate_layer_index(layer_index)
        return self._keys[layer_index, :, :, : self._position, :]

    def layer_values(self, layer_index: int) -> torch.Tensor:
        self._validate_layer_index(layer_index)
        return self._values[layer_index, :, :, : self._position, :]

    def reset(self) -> None:
        if self._active_transaction is not None:
            raise KVCacheError("cannot reset during an active transaction")
        self._position = 0

    def begin(
        self,
        *,
        token_count: int,
        batch_size: int,
        kv_head_count: int,
        head_dimension: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> KVCacheTransaction:
        if self._active_transaction is not None:
            raise KVCacheError("another cache transaction is already active")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
        ):
            raise KVCacheError("token_count must be a positive integer")
        if self._position > 0 and token_count != 1:
            raise KVCacheError("decode accepts exactly one token after prefill")
        expected = {
            "batch_size": self.metadata.batch_size,
            "kv_head_count": self.metadata.kv_head_count,
            "head_dimension": self.metadata.head_dimension,
            "device": self.metadata.device,
            "dtype": self.metadata.dtype,
        }
        actual = {
            "batch_size": batch_size,
            "kv_head_count": kv_head_count,
            "head_dimension": head_dimension,
            "device": torch.device(device),
            "dtype": dtype,
        }
        for name, expected_value in expected.items():
            if actual[name] != expected_value:
                raise KVCacheError(
                    f"cache {name} mismatch: expected {expected_value}, "
                    f"got {actual[name]}"
                )
        end = self._position + token_count
        if end > self.metadata.capacity:
            raise KVCacheError(
                f"cache position {self._position} plus {token_count} tokens "
                f"exceeds capacity {self.metadata.capacity}"
            )
        transaction = KVCacheTransaction(
            self,
            start=self._position,
            token_count=token_count,
        )
        self._active_transaction = transaction
        return transaction

    def _validate_layer_index(self, layer_index: int) -> None:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.metadata.layer_count
        ):
            raise KVCacheError(
                f"layer_index must be in [0, {self.metadata.layer_count})"
            )


class KVCacheTransaction:
    """One model forward's uncommitted per-layer cache append."""

    def __init__(self, cache: KVCache, *, start: int, token_count: int) -> None:
        self._cache = cache
        self.start = start
        self.token_count = token_count
        self.end = start + token_count
        self._written_layers: set[int] = set()
        self._closed = False

    def write(
        self,
        layer_index: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            self._ensure_active()
            self._cache._validate_layer_index(layer_index)
            if layer_index in self._written_layers:
                raise KVCacheError(f"duplicate layer write {layer_index}")
            expected_shape = (
                self._cache.metadata.batch_size,
                self._cache.metadata.kv_head_count,
                self.token_count,
                self._cache.metadata.head_dimension,
            )
            for name, tensor in (("key", keys), ("value", values)):
                if not isinstance(tensor, torch.Tensor):
                    raise KVCacheError(f"{name} must be a Tensor")
                if tuple(tensor.shape) != expected_shape:
                    raise KVCacheError(
                        f"{name} shape must be {expected_shape}, got "
                        f"{tuple(tensor.shape)}"
                    )
                if tensor.device != self._cache.metadata.device:
                    raise KVCacheError(
                        f"{name} device must be {self._cache.metadata.device}, "
                        f"got {tensor.device}"
                    )
                if tensor.dtype != self._cache.metadata.dtype:
                    raise KVCacheError(
                        f"{name} dtype must be {self._cache.metadata.dtype}, "
                        f"got {tensor.dtype}"
                    )
        except Exception:
            self.rollback()
            raise

        with torch.no_grad():
            self._cache._keys[
                layer_index,
                :,
                :,
                self.start : self.end,
                :,
            ].copy_(keys)
            self._cache._values[
                layer_index,
                :,
                :,
                self.start : self.end,
                :,
            ].copy_(values)
        self._written_layers.add(layer_index)
        return (
            self._cache._keys[layer_index, :, :, : self.end, :],
            self._cache._values[layer_index, :, :, : self.end, :],
        )

    def commit(self) -> None:
        self._ensure_active()
        missing = sorted(
            set(range(self._cache.metadata.layer_count)) - self._written_layers
        )
        if missing:
            self.rollback()
            missing_text = ", ".join(str(index) for index in missing)
            raise KVCacheError(f"missing layer writes: {missing_text}")
        self._cache._position = self.end
        self._cache._active_transaction = None
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            return
        if self._cache._active_transaction is self:
            self._cache._active_transaction = None
        self._closed = True

    def _ensure_active(self) -> None:
        if self._closed or self._cache._active_transaction is not self:
            raise KVCacheError("cache transaction is no longer active")
