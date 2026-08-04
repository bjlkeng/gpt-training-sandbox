"""Pure format diagnostics for the existing fixed SFT sample suite."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Final, Literal, TypeAlias

from scratch_llm.evaluation.sft_sampling import FixedSFTSamplesResult


_CODE_PROMPT_INDEX: Final = 1
_JSON_PROMPT_INDEX: Final = 4
_REQUIRED_JSON_KEYS: Final = frozenset({"name", "age", "city"})
_FORMAT_PROMPT_COUNT: Final = 2

CodeFenceCategory: TypeAlias = Literal[
    "balanced_python_fence",
    "balanced_untyped_fence",
    "plain_code",
    "malformed_fence",
]


@dataclass(frozen=True, slots=True)
class JSONPromptDiagnostic:
    """Strict whole-response JSON shape diagnostics for the JSON prompt."""

    parse_valid: bool
    is_object: bool
    has_name_key: bool
    has_age_key: bool
    has_city_key: bool
    has_exact_required_keys: bool

    @property
    def is_format_violation(self) -> bool:
        """Return whether the response misses the requested JSON object shape."""

        return not (
            self.parse_valid and self.is_object and self.has_exact_required_keys
        )

    def to_dict(self) -> dict[str, bool]:
        """Return stable JSON-compatible JSON diagnostics."""

        return {
            "has_age_key": self.has_age_key,
            "has_city_key": self.has_city_key,
            "has_exact_required_keys": self.has_exact_required_keys,
            "has_name_key": self.has_name_key,
            "is_format_violation": self.is_format_violation,
            "is_object": self.is_object,
            "parse_valid": self.parse_valid,
        }


@dataclass(frozen=True, slots=True)
class CodePromptDiagnostic:
    """Deterministic Markdown-fence diagnostics for the code prompt."""

    category: CodeFenceCategory
    fence_count: int

    @property
    def is_format_violation(self) -> bool:
        """Return whether fence markers are present but malformed."""

        return self.category == "malformed_fence"

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible code diagnostics."""

        return {
            "category": self.category,
            "fence_count": self.fence_count,
            "is_format_violation": self.is_format_violation,
        }


@dataclass(frozen=True, slots=True)
class FixedSFTDiagnostics:
    """Content-free diagnostics and identities for one fixed SFT sample result."""

    checkpoint_identity: str
    tokenizer_identity: str
    renderer_identity: str
    prompt_set_identity: str
    generation_identity: str
    sample_count: int
    assistant_end_stop_count: int
    bos_safety_stop_count: int
    max_token_count: int
    visible_token_mean: float
    visible_token_min: int
    visible_token_max: int
    empty_response_count: int
    json_prompt: JSONPromptDiagnostic
    code_prompt: CodePromptDiagnostic

    @property
    def assistant_end_stop_rate(self) -> float:
        """Return assistant-end stops divided by all five fixed samples."""

        return self.assistant_end_stop_count / self.sample_count

    @property
    def empty_response_rate(self) -> float:
        """Return Unicode-whitespace-empty responses divided by all samples."""

        return self.empty_response_count / self.sample_count

    @property
    def format_violation_count(self) -> int:
        """Return violations across the one JSON and one code-format prompt."""

        return int(self.json_prompt.is_format_violation) + int(
            self.code_prompt.is_format_violation
        )

    @property
    def format_violation_rate(self) -> float:
        """Return format violations divided by the two format-specific prompts."""

        return self.format_violation_count / _FORMAT_PROMPT_COUNT

    def to_dict(self) -> dict[str, object]:
        """Return stable metrics without raw prompts, responses, or token IDs."""

        return {
            "code_prompt": self.code_prompt.to_dict(),
            "empty_responses": {
                "count": self.empty_response_count,
                "denominator": self.sample_count,
                "rate": self.empty_response_rate,
            },
            "format_violations": {
                "count": self.format_violation_count,
                "denominator": _FORMAT_PROMPT_COUNT,
                "rate": self.format_violation_rate,
            },
            "identities": {
                "checkpoint": self.checkpoint_identity,
                "generation": self.generation_identity,
                "prompt_set": self.prompt_set_identity,
                "renderer": self.renderer_identity,
                "tokenizer": self.tokenizer_identity,
            },
            "interpretation": "format_only_not_answer_quality",
            "json_prompt": self.json_prompt.to_dict(),
            "sample_count": self.sample_count,
            "stops": {
                "assistant_end_count": self.assistant_end_stop_count,
                "assistant_end_rate": self.assistant_end_stop_rate,
                "bos_safety_count": self.bos_safety_stop_count,
                "denominator": self.sample_count,
                "max_token_count": self.max_token_count,
            },
            "visible_tokens": {
                "denominator": self.sample_count,
                "max": self.visible_token_max,
                "mean": self.visible_token_mean,
                "min": self.visible_token_min,
            },
        }


def compute_fixed_sft_diagnostics(
    samples: FixedSFTSamplesResult,
) -> FixedSFTDiagnostics:
    """Compute format-only diagnostics from the canonical fixed SFT result."""

    if not isinstance(samples, FixedSFTSamplesResult):
        raise TypeError(
            f"samples must be a FixedSFTSamplesResult, got {type(samples).__name__}"
        )

    visible_lengths = tuple(sample.generated_token_count for sample in samples.samples)
    assistant_end_stop_count = sum(
        sample.completion_reason == "stop_token"
        and sample.stop_token_id == samples.assistant_end_token_id
        for sample in samples.samples
    )
    bos_safety_stop_count = sum(
        sample.completion_reason == "stop_token"
        and sample.stop_token_id == samples.bos_token_id
        for sample in samples.samples
    )

    return FixedSFTDiagnostics(
        checkpoint_identity=samples.checkpoint_identity,
        tokenizer_identity=samples.tokenizer_identity,
        renderer_identity=samples.renderer_identity,
        prompt_set_identity=samples.prompt_set_identity,
        generation_identity=samples.generation_identity,
        sample_count=len(samples.samples),
        assistant_end_stop_count=assistant_end_stop_count,
        bos_safety_stop_count=bos_safety_stop_count,
        max_token_count=sum(
            sample.completion_reason == "max_new_tokens" for sample in samples.samples
        ),
        visible_token_mean=sum(visible_lengths) / len(visible_lengths),
        visible_token_min=min(visible_lengths),
        visible_token_max=max(visible_lengths),
        empty_response_count=sum(not sample.text.strip() for sample in samples.samples),
        json_prompt=_diagnose_json(samples.samples[_JSON_PROMPT_INDEX].text),
        code_prompt=_diagnose_code(samples.samples[_CODE_PROMPT_INDEX].text),
    )


def _diagnose_json(response: str) -> JSONPromptDiagnostic:
    try:
        parsed = json.loads(
            response,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except ValueError:
        return JSONPromptDiagnostic(False, False, False, False, False, False)

    if not isinstance(parsed, dict):
        return JSONPromptDiagnostic(True, False, False, False, False, False)

    keys = set(parsed)
    return JSONPromptDiagnostic(
        parse_valid=True,
        is_object=True,
        has_name_key="name" in keys,
        has_age_key="age" in keys,
        has_city_key="city" in keys,
        has_exact_required_keys=keys == _REQUIRED_JSON_KEYS,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _diagnose_code(response: str) -> CodePromptDiagnostic:
    """Classify zero fences or one exact standalone Markdown fence pair."""

    fence_count = response.count("```")
    if fence_count == 0:
        return CodePromptDiagnostic("plain_code", 0)

    fence_lines = [line for line in response.splitlines() if line.startswith("```")]
    if fence_count != 2 or len(fence_lines) != 2 or fence_lines[1] != "```":
        return CodePromptDiagnostic("malformed_fence", fence_count)
    if fence_lines[0] == "```python":
        category: CodeFenceCategory = "balanced_python_fence"
    elif fence_lines[0] == "```":
        category = "balanced_untyped_fence"
    else:
        category = "malformed_fence"
    return CodePromptDiagnostic(category, fence_count)
