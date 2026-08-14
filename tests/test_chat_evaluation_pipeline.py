"""Tests for thin single-device chat-evaluation orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import scratch_llm.evaluation.chat.pipeline as pipeline
from scratch_llm.chat.rendering import CHAT_RENDERER_ID
from scratch_llm.config import ProjectConfig, RunConfig
from scratch_llm.evaluation.chat.categorical import CategoricalTaskResult
from scratch_llm.evaluation.chat.diagnostics import (
    CodePromptDiagnostic,
    FixedSFTDiagnostics,
    JSONPromptDiagnostic,
)
from scratch_llm.evaluation.chat.generative import (
    GenerativeEvaluationConfig,
    GenerativeProblemResult,
    GenerativeSampleResult,
    GenerativeTaskResult,
)
from scratch_llm.evaluation.chat.pipeline import evaluate_checkpoint_chat_model
from scratch_llm.evaluation.chat.reporting import (
    ChatEvaluationError,
    ChatEvaluationSettings,
)
from scratch_llm.evaluation.sft_sampling import (
    FIXED_SFT_PROMPT_SET_IDENTITY,
    FixedSFTSamplingConfig,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer


_CHECKPOINT = "sha256:" + "1" * 64


def _config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(
            name="chat-evaluation-test",
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        )
    )


def _settings(
    tasks: tuple[str, ...],
    *,
    allow_code: bool = False,
    executor_identity: str | None = None,
) -> ChatEvaluationSettings:
    return ChatEvaluationSettings(
        task_names=tasks,  # type: ignore[arg-type]
        batch_size=2,
        max_problems=None,
        generation=GenerativeEvaluationConfig(
            num_samples=1,
            max_new_tokens=1,
            temperature=0,
            top_k=1,
            seed=7,
        ),
        fixed_sampling=FixedSFTSamplingConfig(
            max_new_tokens=1,
            temperature=0,
            top_k=1,
            seed=7,
        ),
        allow_generated_code_execution=allow_code,
        executor_identity=executor_identity,
    )


def _task_result(task_name: str, tokenizer_identity: str):
    common = {
        "task_name": task_name,
        "checkpoint_identity": _CHECKPOINT,
        "tokenizer_identity": tokenizer_identity,
        "source_identity": f"source:{task_name}",
        "dataset_identity": f"dataset:{task_name}",
        "order_identity": f"order:{task_name}",
        "run_kind": "full",
        "max_problems": None,
    }
    if task_name in {"ARC-Easy", "ARC-Challenge", "MMLU"}:
        return CategoricalTaskResult(
            **common,
            passed_count=1,
            evaluated_count=1,
            available_count=1,
            elapsed_seconds=1,
        )
    outcome = "test_failure" if task_name == "HumanEval" else None
    sample = GenerativeSampleResult(
        problem_index=0,
        sample_index=0,
        seed=7,
        passed=False,
        generated_token_count=1,
        sampled_token_count=1,
        completion_reason="max_new_tokens",
        stop_token_id=None,
        completion_identity=f"completion:{task_name}",
        score_outcome=outcome,
    )
    return GenerativeTaskResult(
        **common,
        available_count=1,
        assistant_end_token_id=263,
        bos_token_id=264,
        config=GenerativeEvaluationConfig(
            num_samples=1,
            max_new_tokens=1,
            temperature=0,
            top_k=1,
            seed=7,
        ),
        problems=(
            GenerativeProblemResult(
                problem_index=0,
                problem_identity=f"problem:{task_name}",
                source_row=0,
                passed=False,
                samples=(sample,),
            ),
        ),
        scoring_identity=("executor:v1" if task_name == "HumanEval" else None),
    )


def _diagnostics(tokenizer_identity: str) -> FixedSFTDiagnostics:
    return FixedSFTDiagnostics(
        checkpoint_identity=_CHECKPOINT,
        tokenizer_identity=tokenizer_identity,
        renderer_identity=CHAT_RENDERER_ID,
        prompt_set_identity=FIXED_SFT_PROMPT_SET_IDENTITY,
        generation_identity="fixed-generation:v1",
        sample_count=5,
        assistant_end_stop_count=0,
        bos_safety_stop_count=0,
        max_token_count=5,
        visible_token_mean=1,
        visible_token_min=1,
        visible_token_max=1,
        empty_response_count=0,
        json_prompt=JSONPromptDiagnostic(False, False, False, False, False, False),
        code_prompt=CodePromptDiagnostic("plain_code", 0),
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: ProjectConfig,
    model: nn.Module,
    calls: list[str],
    fail_task: str | None = None,
    failure: BaseException | None = None,
) -> ByteTokenizer:
    tokenizer = ByteTokenizer()
    monkeypatch.setattr(
        pipeline,
        "load_checkpoint_metadata",
        lambda path: (
            calls.append("metadata")
            or SimpleNamespace(config=config, step=4, training_stage="sft")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "file_identity",
        lambda path: calls.append("identity") or _CHECKPOINT,
    )
    monkeypatch.setattr(
        pipeline,
        "_load_task",
        lambda task_name, cache_root: (
            calls.append(f"load:{task_name}") or SimpleNamespace(name=task_name)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "load_model_checkpoint",
        lambda path, device: (
            calls.append("model")
            or SimpleNamespace(
                model=model,
                tokenizer=tokenizer,
                config=config,
                step=4,
                training_stage="sft",
            )
        ),
    )

    def run_task(*args, **kwargs):
        task_name = args[2].name
        calls.append(f"run:{task_name}")
        random.random()
        np.random.random()
        torch.rand(1)
        if task_name == fail_task:
            raise failure or RuntimeError("fixture task failed")
        return _task_result(task_name, tokenizer.get_identity())

    monkeypatch.setattr(pipeline, "_evaluate_task", run_task)
    monkeypatch.setattr(
        pipeline,
        "generate_fixed_sft_samples",
        lambda *args, **kwargs: (
            calls.append("fixed") or random.random() or SimpleNamespace()
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_fixed_sft_diagnostics",
        lambda samples: _diagnostics(tokenizer.get_identity()),
    )
    return tokenizer


def _rng_probe() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_full_orchestration_preflights_then_runs_once_in_canonical_order_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "sft.pt"
    checkpoint.write_bytes(b"fixture")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = nn.Sequential(nn.Dropout(), nn.Sequential(nn.Dropout()))
    model.train()
    model[0].eval()
    original_modes = tuple(module.training for module in model.modules())
    calls: list[str] = []
    _patch_runtime(monkeypatch, config=config, model=model, calls=calls)
    executor = SimpleNamespace(identity="executor:v1", execute=lambda program: None)
    settings = _settings(
        ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"),
        allow_code=True,
        executor_identity="executor:v1",
    )
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected_rng = _rng_probe()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    output = evaluate_checkpoint_chat_model(
        config,
        checkpoint_path=checkpoint,
        cache_root=cache_root,
        settings=settings,
        run_dir=run_dir,
        executor=executor,  # type: ignore[arg-type]
    )

    assert calls == [
        "metadata",
        "identity",
        "load:ARC-Easy",
        "load:ARC-Challenge",
        "load:MMLU",
        "load:GSM8K",
        "load:HumanEval",
        "model",
        "run:ARC-Easy",
        "run:ARC-Challenge",
        "run:MMLU",
        "run:GSM8K",
        "run:HumanEval",
        "fixed",
    ]
    assert output.completed.settings.full is True
    assert output.report_path == run_dir / "metrics" / "chat_eval.json"
    assert output.report_path.is_file()
    assert tuple(module.training for module in model.modules()) == original_modes
    assert _rng_probe() == pytest.approx(expected_rng)


def test_chat_tasks_and_fixed_samples_use_configured_sft_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.sft.dtype = "bfloat16"
    config.validate()
    checkpoint = tmp_path / "sft.pt"
    checkpoint.write_bytes(b"fixture")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[str] = []
    tokenizer = _patch_runtime(
        monkeypatch,
        config=config,
        model=nn.Linear(1, 1),
        calls=calls,
    )

    def run_task(*args: object, **_kwargs: object):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") is torch.bfloat16
        task = args[2]
        calls.append(f"run:{task.name}")
        return _task_result(task.name, tokenizer.get_identity())

    def fixed_samples(*_args: object, **_kwargs: object) -> SimpleNamespace:
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") is torch.bfloat16
        calls.append("fixed")
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, "_evaluate_task", run_task)
    monkeypatch.setattr(pipeline, "generate_fixed_sft_samples", fixed_samples)

    evaluate_checkpoint_chat_model(
        config,
        checkpoint_path=checkpoint,
        cache_root=cache_root,
        settings=_settings(("ARC-Easy",)),
        run_dir=run_dir,
    )

    assert calls[-2:] == ["run:ARC-Easy", "fixed"]


@pytest.mark.parametrize(
    "failure",
    (RuntimeError("fixture task failed"), KeyboardInterrupt()),
)
def test_task_failure_or_interruption_restores_state_and_leaves_no_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "sft.pt"
    checkpoint.write_bytes(b"fixture")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = nn.Sequential(nn.Dropout())
    model.train()
    original_modes = tuple(module.training for module in model.modules())
    calls: list[str] = []
    _patch_runtime(
        monkeypatch,
        config=config,
        model=model,
        calls=calls,
        fail_task="MMLU",
        failure=failure,
    )
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    expected_rng = _rng_probe()
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    with pytest.raises(type(failure)):
        evaluate_checkpoint_chat_model(
            config,
            checkpoint_path=checkpoint,
            cache_root=cache_root,
            settings=_settings(("ARC-Easy", "MMLU")),
            run_dir=run_dir,
        )

    assert "fixed" not in calls
    assert not (run_dir / "metrics" / "chat_eval.json").exists()
    assert tuple(module.training for module in model.modules()) == original_modes
    assert _rng_probe() == pytest.approx(expected_rng)


def test_code_refusal_and_checkpoint_preflight_happen_before_task_or_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "load_checkpoint_metadata",
        lambda path: (
            calls.append("metadata")
            or SimpleNamespace(config=config, step=1, training_stage="pretrain")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_task",
        lambda *args: calls.append("task") or None,
    )
    monkeypatch.setattr(
        pipeline,
        "load_model_checkpoint",
        lambda *args, **kwargs: calls.append("model") or None,
    )

    with pytest.raises(ChatEvaluationError, match="explicit.*opt-in"):
        evaluate_checkpoint_chat_model(
            config,
            checkpoint_path=checkpoint,
            cache_root=cache_root,
            settings=_settings(("HumanEval",)),
            run_dir=run_dir,
        )
    assert calls == []

    with pytest.raises(ChatEvaluationError, match="SFT checkpoint"):
        evaluate_checkpoint_chat_model(
            config,
            checkpoint_path=checkpoint,
            cache_root=cache_root,
            settings=_settings(("ARC-Easy",)),
            run_dir=run_dir,
        )
    assert calls == ["metadata"]


def test_preflight_rejects_invalid_paths_executor_and_checkpoint_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    settings = _settings(
        ("HumanEval",),
        allow_code=True,
        executor_identity="executor:v1",
    )
    different = replace(config, model=replace(config.model, n_embd=192))
    monkeypatch.setattr(
        pipeline,
        "load_checkpoint_metadata",
        lambda path: SimpleNamespace(
            config=different,
            step=1,
            training_stage="sft",
        ),
    )

    with pytest.raises(ChatEvaluationError, match="executor identity"):
        evaluate_checkpoint_chat_model(
            config,
            checkpoint_path=checkpoint,
            cache_root=cache_root,
            settings=settings,
            run_dir=run_dir,
            executor=SimpleNamespace(identity="other"),  # type: ignore[arg-type]
        )
    with pytest.raises(ChatEvaluationError, match="model"):
        evaluate_checkpoint_chat_model(
            config,
            checkpoint_path=checkpoint,
            cache_root=cache_root,
            settings=_settings(("ARC-Easy",)),
            run_dir=run_dir,
        )
    tokenizer_mismatch = replace(
        config,
        tokenizer=replace(config.tokenizer, artifact_dir="other-tokenizer"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_checkpoint_metadata",
        lambda path: SimpleNamespace(
            config=tokenizer_mismatch,
            step=1,
            training_stage="sft",
        ),
    )
    with pytest.raises(ChatEvaluationError, match="tokenizer"):
        evaluate_checkpoint_chat_model(
            config,
            checkpoint_path=checkpoint,
            cache_root=cache_root,
            settings=_settings(("ARC-Easy",)),
            run_dir=run_dir,
        )
    with pytest.raises(ChatEvaluationError, match="cache_root"):
        evaluate_checkpoint_chat_model(
            config,
            checkpoint_path=checkpoint,
            cache_root=tmp_path / "missing-cache",
            settings=_settings(("ARC-Easy",)),
            run_dir=run_dir,
        )
