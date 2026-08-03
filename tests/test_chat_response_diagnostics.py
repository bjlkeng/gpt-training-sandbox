"""Tests for pure diagnostics over completed fixed SFT samples."""

from __future__ import annotations

import json

import pytest

from scratch_llm.chat.rendering import CHAT_RENDERER_ID
from scratch_llm.evaluation.chat.diagnostics import (
    compute_fixed_sft_diagnostics,
)
from scratch_llm.evaluation.sft_sampling import (
    FIXED_SFT_PROMPTS,
    FixedSFTSample,
    FixedSFTSamplesResult,
    FixedSFTSamplingConfig,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer


def _fixed_samples(
    texts: tuple[str, str, str, str, str],
    *,
    stop_kinds: tuple[str, str, str, str, str] = (
        "assistant_end",
        "assistant_end",
        "assistant_end",
        "assistant_end",
        "assistant_end",
    ),
) -> FixedSFTSamplesResult:
    tokenizer = ByteTokenizer()
    config = FixedSFTSamplingConfig(seed=42)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    samples = []
    for index, (prompt, text, stop_kind) in enumerate(
        zip(FIXED_SFT_PROMPTS, texts, stop_kinds, strict=True)
    ):
        generated_ids = tuple(tokenizer.encode(text))
        if stop_kind == "assistant_end":
            completion_reason = "stop_token"
            stop_token_id = assistant_end
            sampled_count = len(generated_ids) + 1
        elif stop_kind == "bos":
            completion_reason = "stop_token"
            stop_token_id = bos
            sampled_count = len(generated_ids) + 1
        elif stop_kind == "max_new_tokens":
            completion_reason = "max_new_tokens"
            stop_token_id = None
            sampled_count = len(generated_ids)
        else:  # pragma: no cover - test fixture contract.
            raise AssertionError(f"unsupported stop kind {stop_kind!r}")
        samples.append(
            FixedSFTSample(
                prompt_index=index,
                prompt=prompt,
                prompt_token_ids=(index + 1,),
                seed=config.seed + index,
                generated_token_ids=generated_ids,
                sampled_token_count=sampled_count,
                elapsed_seconds=1.0,
                completion_reason=completion_reason,  # type: ignore[arg-type]
                stop_token_id=stop_token_id,
                text=text,
            )
        )
    return FixedSFTSamplesResult(
        checkpoint_identity="checkpoint",
        tokenizer_identity=tokenizer.get_identity(),
        renderer_identity=CHAT_RENDERER_ID,
        assistant_end_token_id=assistant_end,
        bos_token_id=bos,
        config=config,
        samples=tuple(samples),
    )


def test_fixed_sft_diagnostics_compute_stop_length_empty_and_identity_metrics() -> None:
    texts = (
        "",
        "```python\ndef reverse(value):\n    return value[::-1]\n```",
        "\u2003",
        "é",
        '{"name":"Ada","age":37,"city":"NYC"}',
    )
    samples = _fixed_samples(
        texts,
        stop_kinds=(
            "assistant_end",
            "assistant_end",
            "bos",
            "max_new_tokens",
            "assistant_end",
        ),
    )

    result = compute_fixed_sft_diagnostics(samples)

    token_lengths = tuple(len(ByteTokenizer().encode(text)) for text in texts)
    assert result.sample_count == 5
    assert result.assistant_end_stop_count == 3
    assert result.assistant_end_stop_rate == 3 / 5
    assert result.bos_safety_stop_count == 1
    assert result.max_token_count == 1
    assert result.visible_token_min == min(token_lengths)
    assert result.visible_token_max == max(token_lengths)
    assert result.visible_token_mean == sum(token_lengths) / 5
    assert result.empty_response_count == 2
    assert result.empty_response_rate == 2 / 5
    assert result.json_prompt.parse_valid is True
    assert result.json_prompt.is_object is True
    assert result.json_prompt.has_exact_required_keys is True
    assert result.code_prompt.category == "balanced_python_fence"
    assert result.format_violation_count == 0
    payload = result.to_dict()
    assert payload["identities"] == {
        "checkpoint": "checkpoint",
        "generation": samples.generation_identity,
        "prompt_set": samples.prompt_set_identity,
        "renderer": CHAT_RENDERER_ID,
        "tokenizer": samples.tokenizer_identity,
    }
    assert payload["stops"] == {
        "assistant_end_count": 3,
        "assistant_end_rate": 3 / 5,
        "bos_safety_count": 1,
        "denominator": 5,
        "max_token_count": 1,
    }
    assert payload["visible_tokens"] == {
        "denominator": 5,
        "max": max(token_lengths),
        "mean": sum(token_lengths) / 5,
        "min": min(token_lengths),
    }
    assert payload["empty_responses"] == {
        "count": 2,
        "denominator": 5,
        "rate": 2 / 5,
    }
    assert payload["format_violations"] == {
        "count": 0,
        "denominator": 2,
        "rate": 0.0,
    }
    assert payload["interpretation"] == "format_only_not_answer_quality"


@pytest.mark.parametrize(
    ("response", "parse_valid", "is_object", "present", "exact"),
    [
        ('{"name":"Ada","age":37,"city":"NYC"}', True, True, 3, True),
        ('{"name":"Ada","age":37}', True, True, 2, False),
        (
            '{"name":"Ada","age":37,"city":"NYC","extra":true}',
            True,
            True,
            3,
            False,
        ),
        ('["name","age","city"]', True, False, 0, False),
        ('{"name":"Ada"} trailing', False, False, 0, False),
        (
            '{"name":"first","name":"second","age":37,"city":"NYC"}',
            False,
            False,
            0,
            False,
        ),
        ('{"name":', False, False, 0, False),
        ('{"name":"Ada","age":NaN,"city":"NYC"}', False, False, 0, False),
        (' \n {"name":"Ada","age":37,"city":"NYC"}\t', True, True, 3, True),
    ],
)
def test_json_diagnostics_parse_the_whole_response_strictly(
    response: str,
    parse_valid: bool,
    is_object: bool,
    present: int,
    exact: bool,
) -> None:
    samples = _fixed_samples(("ok", "plain code", "ok", "ok", response))

    diagnostic = compute_fixed_sft_diagnostics(samples).json_prompt

    assert diagnostic.parse_valid is parse_valid
    assert diagnostic.is_object is is_object
    assert (
        sum(
            (
                diagnostic.has_name_key,
                diagnostic.has_age_key,
                diagnostic.has_city_key,
            )
        )
        == present
    )
    assert diagnostic.has_exact_required_keys is exact


@pytest.mark.parametrize(
    ("response", "category", "fence_count", "violation"),
    [
        ("def reverse(value):\n    return value[::-1]", "plain_code", 0, False),
        (
            "```python\ndef reverse(value):\n    return value[::-1]\n```",
            "balanced_python_fence",
            2,
            False,
        ),
        (
            "Here is the function:\n```python\ndef reverse(value):\n"
            "    return value[::-1]\n```\nIt runs in linear time.",
            "balanced_python_fence",
            2,
            False,
        ),
        (
            "```\ndef reverse(value):\n    return value[::-1]\n```",
            "balanced_untyped_fence",
            2,
            False,
        ),
        ("```python\ndef reverse(value):", "malformed_fence", 1, True),
        (
            "```python\none\n```\n```python\ntwo\n```",
            "malformed_fence",
            4,
            True,
        ),
        ("```javascript\ncode\n```", "malformed_fence", 2, True),
        ("inline ```python code```", "malformed_fence", 2, True),
    ],
)
def test_code_fence_diagnostics_use_one_deterministic_category(
    response: str,
    category: str,
    fence_count: int,
    violation: bool,
) -> None:
    samples = _fixed_samples(("ok", response, "ok", "ok", "{}"))

    result = compute_fixed_sft_diagnostics(samples)

    assert result.code_prompt.category == category
    assert result.code_prompt.fence_count == fence_count
    assert result.code_prompt.is_format_violation is violation


def test_fixed_sft_diagnostics_serialize_stably_without_sample_content() -> None:
    samples = _fixed_samples(
        (
            "response one",
            "plain code",
            "response three",
            "response four",
            '{"name":"Ada","age":37,"city":"NYC"}',
        )
    )

    first = compute_fixed_sft_diagnostics(samples)
    repeated = compute_fixed_sft_diagnostics(samples)

    assert first == repeated
    assert first.to_dict() == repeated.to_dict()
    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert "response one" not in serialized
    assert list(FIXED_SFT_PROMPTS) not in first.to_dict().values()
