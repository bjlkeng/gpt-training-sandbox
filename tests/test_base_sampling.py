"""Fixed base-sampling result and Markdown artifact conformance tests."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path

import pytest
import torch

from scratch_llm.base_sampling import (
    FIXED_BASE_PROMPTS,
    FIXED_BASE_PROMPT_SET_IDENTITY,
    BaseSamplesResult,
    FixedBaseSamplingConfig,
    generate_and_write_fixed_base_samples,
    generate_fixed_base_samples,
    write_base_samples_markdown,
)
from scratch_llm.tokenizer import ByteTokenizer


EXPECTED_PROMPTS = (
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
)
SCRIPTED_TEXT = "```\n# not a heading\n<|bos|> café ✓"


class _ScriptedCompletionModel(torch.nn.Module):
    def __init__(
        self,
        token_ids: tuple[int, ...],
        *,
        bos_token_id: int,
        fail_after_completions: int | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = 128
        self.vocab_size = ByteTokenizer().get_vocab_size()
        self._script = (*token_ids, bos_token_id)
        self._position = 0
        self._completed = 0
        self._fail_after_completions = fail_after_completions
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if (
            self._fail_after_completions is not None
            and self._completed >= self._fail_after_completions
        ):
            raise RuntimeError("scripted generation failure")
        next_id = self._script[self._position]
        self._position += 1
        if self._position == len(self._script):
            self._position = 0
            self._completed += 1
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.vocab_size),
            -torch.inf,
            device=token_ids.device,
        )
        logits[:, -1, next_id] = self.anchor
        return logits


class _ConstantCompletionModel(torch.nn.Module):
    def __init__(self, token_id: int) -> None:
        super().__init__()
        self.max_seq_len = 128
        self.vocab_size = ByteTokenizer().get_vocab_size()
        self.token_id = token_id
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.vocab_size),
            -torch.inf,
            device=token_ids.device,
        )
        logits[:, -1, self.token_id] = self.anchor
        return logits


def _clock() -> Iterator[float]:
    current = 0.0
    while True:
        yield current
        current += 1.0


def _generate_result() -> BaseSamplesResult:
    tokenizer = ByteTokenizer()
    clock_values = _clock()
    return generate_fixed_base_samples(
        _ScriptedCompletionModel(
            tuple(tokenizer.encode(SCRIPTED_TEXT)),
            bos_token_id=tokenizer.get_bos_token_id(),
        ),
        tokenizer,
        checkpoint_identity="sha256:" + "1" * 64,
        config=FixedBaseSamplingConfig(
            max_new_tokens=128,
            temperature=0,
            top_k=None,
            seed=41,
        ),
        device="cpu",
        clock=lambda: next(clock_values),
    )


def test_prompt_suite_and_generation_config_are_frozen() -> None:
    config = FixedBaseSamplingConfig(
        max_new_tokens=128,
        temperature=0,
        top_k=None,
        seed=41,
    )

    assert FIXED_BASE_PROMPTS == EXPECTED_PROMPTS
    assert FIXED_BASE_PROMPT_SET_IDENTITY.startswith("sha256:")
    assert config.to_dict() == {
        "max_new_tokens": 128,
        "prompt_seed_strategy": "seed_plus_prompt_index",
        "seed": 41,
        "stop_tokens": "tokenizer_bos_only",
        "temperature": 0.0,
        "top_k": None,
    }


def test_all_fixed_prompts_record_stops_unicode_timing_and_identities() -> None:
    result = _generate_result()
    tokenizer = ByteTokenizer()
    expected_ids = tuple(tokenizer.encode(SCRIPTED_TEXT))

    assert result.checkpoint_identity == "sha256:" + "1" * 64
    assert result.tokenizer_identity == tokenizer.get_identity()
    assert result.prompt_set_identity == FIXED_BASE_PROMPT_SET_IDENTITY
    assert result.generation_identity.startswith("sha256:")
    assert result.prompts == EXPECTED_PROMPTS
    assert len(result.samples) == 7
    for index, sample in enumerate(result.samples):
        assert sample.prompt == EXPECTED_PROMPTS[index]
        assert sample.prompt_index == index
        assert sample.seed == 41 + index
        assert sample.generated_token_ids == expected_ids
        assert tokenizer.get_bos_token_id() not in sample.generated_token_ids
        assert sample.generated_token_count == len(expected_ids)
        assert sample.sampled_token_count == len(expected_ids) + 1
        assert sample.elapsed_seconds == 1
        assert sample.tokens_per_second == len(expected_ids) + 1
        assert sample.completion_reason == "stop_token"
        assert sample.stop_token_id == tokenizer.get_bos_token_id()
        assert sample.text == SCRIPTED_TEXT


def test_fixed_samples_use_exact_max_new_tokens_fallback() -> None:
    tokenizer = ByteTokenizer()
    clock_values = _clock()

    result = generate_fixed_base_samples(
        _ConstantCompletionModel(ord("A")),
        tokenizer,
        checkpoint_identity="checkpoint",
        config=FixedBaseSamplingConfig(
            max_new_tokens=2,
            temperature=0,
            top_k=None,
            seed=17,
        ),
        device="cpu",
        clock=lambda: next(clock_values),
    )

    assert all(
        sample.generated_token_ids == (ord("A"), ord("A")) for sample in result.samples
    )
    assert all(sample.sampled_token_count == 2 for sample in result.samples)
    assert all(
        sample.completion_reason == "max_new_tokens" for sample in result.samples
    )
    assert all(sample.stop_token_id is None for sample in result.samples)
    assert all(sample.text == "AA" for sample in result.samples)


def test_markdown_is_deterministic_safely_fenced_and_keeps_ordinary_bos_text(
    tmp_path: Path,
) -> None:
    result = _generate_result()
    metrics_dir = tmp_path / "metrics"

    path = write_base_samples_markdown(result, metrics_dir)
    first = path.read_bytes()
    write_base_samples_markdown(result, metrics_dir)
    markdown = first.decode("utf-8")

    assert path == metrics_dir / "base_samples.md"
    assert path.read_bytes() == first
    assert first.endswith(b"\n")
    assert f"Checkpoint identity: `{result.checkpoint_identity}`" in markdown
    assert f"Tokenizer identity: `{result.tokenizer_identity}`" in markdown
    assert f"Prompt-set identity: `{result.prompt_set_identity}`" in markdown
    assert f"Generation identity: `{result.generation_identity}`" in markdown
    assert markdown.index(EXPECTED_PROMPTS[0]) < markdown.index(EXPECTED_PROMPTS[-1])
    assert SCRIPTED_TEXT in markdown
    assert "<|bos|>" in markdown
    assert "````text\n```\n# not a heading" in markdown


def test_artifact_is_not_published_when_any_prompt_generation_fails(
    tmp_path: Path,
) -> None:
    tokenizer = ByteTokenizer()
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    destination = metrics_dir / "base_samples.md"
    destination.write_text("stable\n", encoding="utf-8")
    clock_values = _clock()

    with pytest.raises(RuntimeError, match="scripted generation failure"):
        generate_and_write_fixed_base_samples(
            _ScriptedCompletionModel(
                tuple(tokenizer.encode("answer")),
                bos_token_id=tokenizer.get_bos_token_id(),
                fail_after_completions=2,
            ),
            tokenizer,
            metrics_dir,
            checkpoint_identity="checkpoint",
            config=FixedBaseSamplingConfig(
                max_new_tokens=32,
                temperature=0,
                top_k=None,
                seed=5,
            ),
            device="cpu",
            clock=lambda: next(clock_values),
        )

    assert destination.read_text(encoding="utf-8") == "stable\n"
    assert not tuple(metrics_dir.glob(".base_samples.md.*.tmp"))


def test_atomic_write_failure_preserves_destination_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _generate_result()
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    destination = metrics_dir / "base_samples.md"
    destination.write_text("stable\n", encoding="utf-8")

    def fail_replace(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> None:
        raise OSError(f"cannot replace {source} with {target}")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        write_base_samples_markdown(result, metrics_dir)

    assert destination.read_text(encoding="utf-8") == "stable\n"
    assert not tuple(metrics_dir.glob(".base_samples.md.*.tmp"))


def test_readme_documents_fixed_base_sample_artifact_and_stop_rule() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )

    assert "`metrics/base_samples.md`" in readme
    assert "BOS-only stop set" in readme
    assert "seed plus the prompt index" in readme
    for prompt in EXPECTED_PROMPTS:
        assert prompt in readme
