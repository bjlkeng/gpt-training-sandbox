"""Transactional model-level KV-cache contract and attention parity."""

from __future__ import annotations

import pytest
import torch

from scratch_llm.config import GPTConfig
from scratch_llm.kv_cache import KVCache, KVCacheError
from scratch_llm.model import GPT


def _config(*, backend: str = "manual", layers: int = 2) -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        seq_len=6,
        n_layer=layers,
        n_head=2,
        n_embd=8,
        mlp_ratio=2,
        dropout=0.0,
        attention_backend=backend,  # type: ignore[arg-type]
    )


def _cache(*, layers: int = 2, capacity: int = 6) -> KVCache:
    return KVCache(
        layer_count=layers,
        batch_size=1,
        kv_head_count=2,
        head_dimension=4,
        capacity=capacity,
        device="cpu",
        dtype=torch.float32,
    )


def test_cache_metadata_byte_accounting_commit_and_reset_visibility() -> None:
    cache = _cache()
    transaction = cache.begin(
        token_count=2,
        batch_size=1,
        kv_head_count=2,
        head_dimension=4,
        device="cpu",
        dtype=torch.float32,
    )
    keys = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    values = keys + 100

    for layer_index in range(2):
        visible_keys, visible_values = transaction.write(
            layer_index,
            keys + layer_index * 1000,
            values + layer_index * 1000,
        )
        assert visible_keys.shape == visible_values.shape == (1, 2, 2, 4)
    transaction.commit()

    assert cache.position == 2
    assert cache.layer_shape == (1, 2, 6, 4)
    assert cache.bytes_per_token == 2 * 2 * 1 * 2 * 4 * 4
    assert cache.allocated_bytes == cache.bytes_per_token * 6
    assert cache.metadata.to_dict() == {
        "allocated_bytes": 768,
        "batch_size": 1,
        "bytes_per_token": 128,
        "capacity": 6,
        "device": "cpu",
        "dtype": "float32",
        "head_dimension": 4,
        "kv_head_count": 2,
        "layer_count": 2,
        "layer_shape": [1, 2, 6, 4],
        "layer_window_sizes": [None, None],
    }
    torch.testing.assert_close(cache.layer_keys(1), keys + 1000)
    torch.testing.assert_close(cache.layer_values(1), values + 1000)

    cache.reset()

    assert cache.position == 0
    assert cache.layer_keys(0).shape == (1, 2, 0, 4)
    assert cache.layer_values(1).shape == (1, 2, 0, 4)


def test_prefill_writes_exact_projected_keys_and_values_for_every_layer() -> None:
    torch.manual_seed(131)
    model = GPT(_config()).eval()
    cache = model.create_kv_cache(batch_size=1, capacity=6)
    projected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    for layer_index, block in enumerate(model.blocks):
        attention = block.attn

        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
            heads: int = attention.n_head,
            head_dim: int = attention.head_dim,
        ) -> None:
            _q, key, value = output.chunk(3, dim=-1)
            projected[index] = (
                key.view(1, 3, heads, head_dim).transpose(1, 2),
                value.view(1, 3, heads, head_dim).transpose(1, 2),
            )

        handles.append(attention.qkv.register_forward_hook(capture))
    tokens = torch.tensor([[1, 2, 3]])
    try:
        with torch.inference_mode():
            cached_logits = model(tokens, kv_cache=cache)
    finally:
        for handle in handles:
            handle.remove()

    assert cache.position == 3
    for layer_index in range(2):
        torch.testing.assert_close(
            cache.layer_keys(layer_index), projected[layer_index][0]
        )
        torch.testing.assert_close(
            cache.layer_values(layer_index), projected[layer_index][1]
        )
    with torch.inference_mode():
        ordinary_logits = model(tokens)
    torch.testing.assert_close(cached_logits, ordinary_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_prefill_then_single_token_decode_matches_full_forward(backend: str) -> None:
    torch.manual_seed(137)
    model = GPT(_config(backend=backend)).eval()
    cache = model.create_kv_cache(batch_size=1, capacity=6)
    prompt = torch.tensor([[1, 2, 3]])
    next_token = torch.tensor([[4]])

    with torch.inference_mode():
        prompt_logits = model(prompt, kv_cache=cache)
        decoded_logits = model(next_token, kv_cache=cache)
        ordinary_logits = model(torch.tensor([[1, 2, 3, 4]]))

    assert prompt_logits.shape == (1, 3, 32)
    assert decoded_logits.shape == (1, 1, 32)
    assert cache.position == 4
    torch.testing.assert_close(
        decoded_logits[:, -1],
        ordinary_logits[:, -1],
        rtol=2e-5,
        atol=1e-6,
    )


def test_reset_allows_safe_reuse_without_exposing_stale_values() -> None:
    torch.manual_seed(139)
    model = GPT(_config()).eval()
    cache = model.create_kv_cache(batch_size=1, capacity=6)
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[6, 5]])

    with torch.inference_mode():
        model(first, kv_cache=cache)
        cache.reset()
        reused = model(second, kv_cache=cache)
        ordinary = model(second)

    assert cache.position == 2
    torch.testing.assert_close(reused, ordinary)


def test_overflow_and_multi_token_decode_fail_before_partial_writes() -> None:
    model = GPT(_config()).eval()
    cache = model.create_kv_cache(batch_size=1, capacity=3)
    with torch.inference_mode():
        model(torch.tensor([[1, 2]]), kv_cache=cache)
        committed = cache.layer_keys(0).clone()
        with pytest.raises(KVCacheError, match="decode accepts exactly one token"):
            model(torch.tensor([[3, 4]]), kv_cache=cache)
        assert cache.position == 2
        torch.testing.assert_close(cache.layer_keys(0), committed)
        model(torch.tensor([[3]]), kv_cache=cache)
        with pytest.raises(KVCacheError, match="capacity 3"):
            model(torch.tensor([[4]]), kv_cache=cache)

    assert cache.position == 3


def test_missing_and_duplicate_layer_writes_roll_back_without_advancement() -> None:
    cache = _cache()
    key = torch.zeros((1, 2, 1, 4))
    transaction = cache.begin(
        token_count=1,
        batch_size=1,
        kv_head_count=2,
        head_dimension=4,
        device="cpu",
        dtype=torch.float32,
    )
    transaction.write(0, key, key)
    with pytest.raises(KVCacheError, match="missing layer writes: 1"):
        transaction.commit()
    assert cache.position == 0

    duplicate = cache.begin(
        token_count=1,
        batch_size=1,
        kv_head_count=2,
        head_dimension=4,
        device="cpu",
        dtype=torch.float32,
    )
    duplicate.write(0, key, key)
    with pytest.raises(KVCacheError, match="duplicate layer write 0"):
        duplicate.write(0, key, key)
    assert cache.position == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 2, "batch_size"),
        ("kv_head_count", 1, "kv_head_count"),
        ("head_dimension", 8, "head_dimension"),
        ("device", torch.device("meta"), "device"),
        ("dtype", torch.float16, "dtype"),
    ],
)
def test_transaction_metadata_mismatch_is_rejected_before_writes(
    field: str,
    value: object,
    message: str,
) -> None:
    cache = _cache()
    arguments: dict[str, object] = {
        "token_count": 1,
        "batch_size": 1,
        "kv_head_count": 2,
        "head_dimension": 4,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    arguments[field] = value

    with pytest.raises(KVCacheError, match=message):
        cache.begin(**arguments)  # type: ignore[arg-type]

    assert cache.position == 0


def test_cache_tensor_shape_and_dtype_mismatch_abort_transaction() -> None:
    cache = _cache()
    transaction = cache.begin(
        token_count=1,
        batch_size=1,
        kv_head_count=2,
        head_dimension=4,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(KVCacheError, match="key shape"):
        transaction.write(0, torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 1, 4))

    replacement = cache.begin(
        token_count=1,
        batch_size=1,
        kv_head_count=2,
        head_dimension=4,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(KVCacheError, match="value dtype"):
        replacement.write(
            0,
            torch.zeros(1, 2, 1, 4),
            torch.zeros(1, 2, 1, 4, dtype=torch.float16),
        )
    assert cache.position == 0


def test_model_rejects_layer_count_mismatch_before_any_write() -> None:
    model = GPT(_config(layers=2)).eval()
    cache = _cache(layers=1)

    with torch.inference_mode(), pytest.raises(KVCacheError, match="layer_count"):
        model(torch.tensor([[1, 2]]), kv_cache=cache)

    assert cache.position == 0
    assert cache.layer_keys(0).shape[-2] == 0


def test_cache_is_inference_only_external_and_absent_from_state_dict() -> None:
    model = GPT(_config())
    keys = set(model.state_dict())
    cache = model.create_kv_cache(batch_size=1, capacity=6)

    with pytest.raises(KVCacheError, match="inference-only"):
        model(torch.tensor([[1, 2]]), kv_cache=cache)

    assert set(model.state_dict()) == keys
    assert not any("cache" in key for key in model.state_dict())
    assert cache not in tuple(model.modules())
