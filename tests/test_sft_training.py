"""Bounded CPU tests for base-to-chat training and exact SFT resume."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

import scratch_llm.training.loop as training_loop
import scratch_llm.training.sft as sft_training
from scratch_llm.data.hub import publish_local_parquet_cache
from scratch_llm.data.sft_sources import get_sft_dataset_spec
from scratch_llm.config import (
    GPTConfig,
    ProjectConfig,
    RunConfig,
    SFTConfig,
    SFTSourceConfig,
    TokenizerConfig,
    TrainConfig,
    TrainDType,
)
from scratch_llm.evaluation.sft_sampling import (
    FixedSFTSamplingConfig,
    FixedSFTSamplesResult,
    generate_fixed_sft_samples,
)
from scratch_llm.identity import file_identity
from scratch_llm.model import GPT
from scratch_llm.run import prepare_run
from scratch_llm.tracking import build_tracker
from scratch_llm.training.checkpoint import (
    load_model_checkpoint,
    save_checkpoint,
)
from scratch_llm.training.optim import build_lr_scheduler, build_optimizer
from scratch_llm.training.sft import (
    SFTTrainingError,
    SFTTrainingOOMError,
    build_sft_conversation_sources,
    run_sft_training,
)
from scratch_llm.tokenization.tokenizer import ByteTokenizer, VOCAB_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = PROJECT_ROOT / "data" / "fixtures" / "chat" / "train.jsonl"
VALIDATION_JSONL = PROJECT_ROOT / "data" / "fixtures" / "chat" / "validation.jsonl"


class _FixedSampleModel(torch.nn.Module):
    def __init__(self, tokenizer: ByteTokenizer) -> None:
        super().__init__()
        self.max_seq_len = 128
        self.vocab_size = tokenizer.get_vocab_size()
        self.assistant_start = tokenizer.encode_special("<|assistant_start|>")
        self.assistant_end = tokenizer.encode_special("<|assistant_end|>")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*token_ids.shape, self.vocab_size),
            -100.0,
            dtype=torch.float32,
            device=token_ids.device,
        )
        for row, last_token in enumerate(token_ids[:, -1].tolist()):
            next_token = (
                ord("X") if last_token == self.assistant_start else self.assistant_end
            )
            logits[row, -1, next_token] = 100.0
        return logits


def _fixed_samples(checkpoint_identity: str) -> FixedSFTSamplesResult:
    tokenizer = ByteTokenizer()
    ticks = count()
    return generate_fixed_sft_samples(
        _FixedSampleModel(tokenizer),
        tokenizer,
        checkpoint_identity=checkpoint_identity,
        config=FixedSFTSamplingConfig(
            max_new_tokens=2,
            temperature=0.0,
            top_k=1,
            seed=11,
        ),
        device="cpu",
        clock=lambda: float(next(ticks)),
    )


def _config(tmp_path: Path, *, run_name: str) -> ProjectConfig:
    return ProjectConfig(
        run=RunConfig(
            name=run_name,
            seed=17,
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
        tokenizer=TokenizerConfig(type="byte", vocab_size=VOCAB_SIZE),
        model=GPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=64,
            n_layer=1,
            n_head=1,
            n_embd=8,
            mlp_ratio=2,
            dropout=0.0,
        ),
        train=TrainConfig(
            device_batch_size=1,
            total_batch_size_tokens=64,
            max_steps=4,
            learning_rate=0.2,
            weight_decay=0.1,
            warmup_steps=0,
            warmdown_ratio=0.0,
            eval_every=1,
            save_every=1,
            log_every=1,
        ),
        sft=SFTConfig(
            train_sources=[
                SFTSourceConfig(
                    path=str(TRAIN_JSONL),
                    shuffle=True,
                )
            ],
            validation_sources=[
                SFTSourceConfig(
                    path=str(VALIDATION_JSONL),
                    shuffle=False,
                )
            ],
            packing_buffer_size=4,
            device_batch_size=1,
            total_batch_size_tokens=64,
            max_steps=4,
            learning_rate=0.01,
            weight_decay=0.0,
            warmup_steps=0,
            warmdown_ratio=0.0,
            eval_every=2,
            eval_batches=1,
            save_every=2,
            log_every=1,
        ),
    )


def _base_checkpoint(tmp_path: Path, config: ProjectConfig) -> Path:
    path = tmp_path / "base.pt"
    config.sft.base_checkpoint = str(path)
    config.validate()
    torch.manual_seed(123)
    model = GPT(config.model)
    tokenizer = ByteTokenizer()
    optimizer = build_optimizer(model, config.train)
    scheduler = build_lr_scheduler(optimizer, config.train)
    return save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=0,
        tokenizer=tokenizer,
    )


def _run(
    config: ProjectConfig,
    *,
    base_checkpoint: Path | None = None,
    resume_from: Path | None = None,
):
    paths = prepare_run(config)
    tracker = build_tracker(config, paths, stage="train_sft")
    with tracker:
        return run_sft_training(
            config,
            paths=paths,
            tracker=tracker,
            base_checkpoint=base_checkpoint,
            resume_from=resume_from,
            sample_runner=_fixed_samples,
        )


def _metric_records(path: Path) -> list[dict[str, Any]]:
    return [
        record
        for line in path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line))["record_type"] == "metrics"
    ]


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, Mapping):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_nested_equal(actual[key], value)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_base_initialization_uses_fresh_sft_optimizer_and_writes_ranked_checkpoints(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, run_name="sft-fresh")
    base = _base_checkpoint(tmp_path, config)

    result = _run(config, base_checkpoint=base)

    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    loaded = load_model_checkpoint(result.checkpoint_path)
    records = _metric_records(result.metrics_path)
    assert result.initial_step == 0
    assert result.final_step == 4
    assert result.base_checkpoint_identity == file_identity(base)
    assert result.validation_state is not None
    assert result.validation_state.validation_step == 4
    assert (result.paths.checkpoints_dir / "best.pt").is_file()
    assert (result.paths.checkpoints_dir / "step_000002.pt").is_file()
    assert (result.paths.checkpoints_dir / "step_000004.pt").is_file()
    assert payload["training_stage"] == "sft"
    assert payload["base_checkpoint_identity"] == file_identity(base)
    assert payload["optimizer"]["param_groups"][0]["weight_decay"] == 0.0
    assert payload["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.01)
    assert loaded.model.training is False
    assert loaded.tokenizer.get_identity() == ByteTokenizer().get_identity()
    assert all(
        step.telemetry is not None
        and step.telemetry.processed_model_tokens == config.sft.total_batch_size_tokens
        and step.telemetry.supervised_target_tokens > 0
        for step in result.steps
    )
    assert [record["step"] for record in records] == [1, 2, 2, 3, 4, 4]
    training_records = [
        record for record in records if "sft/train_loss" in record["metrics"]
    ]
    validation_records = [
        record for record in records if "sft/val_bpb" in record["metrics"]
    ]
    assert [record["step"] for record in training_records] == [1, 2, 3, 4]
    assert [record["step"] for record in validation_records] == [2, 4]
    assert all(
        {
            "sft/train_loss",
            "sft/tok_per_sec",
            "sft/mfu",
            "sft/peak_memory_mib",
        }
        <= set(record["metrics"])
        for record in training_records
    )
    assert not any(
        metric.startswith("train/")
        for record in records
        for metric in record["metrics"]
    )
    assert result.evaluation_report_path.is_file()
    assert result.samples_path.is_file()
    assert result.samples is not None
    assert "Compute 2+3." not in result.samples_path.read_text(encoding="utf-8")
    artifact_records = [
        record
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line))["record_type"] == "artifact"
    ]
    assert [
        (record["path"], record["name"], record["type"])
        for record in artifact_records[-2:]
    ] == [
        ("metrics/sft_eval.json", "sft_eval", "evaluation"),
        ("metrics/sft_samples.md", "sft_samples", "evaluation"),
    ]
    assert not any(
        "chatcore" in metric.lower()
        for record in records
        for metric in record["metrics"]
    )


def test_fresh_sft_can_switch_a_manual_base_checkpoint_to_sdpa(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, run_name="sft-sdpa")
    base = _base_checkpoint(tmp_path, config)
    base_keys = set(load_model_checkpoint(base).model.state_dict())
    config.model.attention_backend = "sdpa"
    config.validate()

    result = _run(config, base_checkpoint=base)

    loaded = load_model_checkpoint(result.checkpoint_path)
    assert loaded.config.model.attention_backend == "sdpa"
    assert set(loaded.model.state_dict()) == base_keys


def test_gradient_accumulation_preserves_the_exact_sft_token_budget(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, run_name="sft-accumulation")
    config.sft.total_batch_size_tokens = 128
    config.sft.max_steps = 1
    config.sft.eval_every = 1
    config.sft.save_every = 1
    config.validate()
    base = _base_checkpoint(tmp_path, config)

    result = _run(config, base_checkpoint=base)

    telemetry = result.steps[0].telemetry
    assert config.sft.to_train_config(config.model.seq_len).grad_accum_steps == 2
    assert telemetry is not None
    assert telemetry.processed_model_tokens == 128
    assert telemetry.supervised_target_tokens > 0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_exact_sft_resume_matches_uninterrupted_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: TrainDType,
) -> None:
    config = _config(tmp_path, run_name="sft-uninterrupted")
    config.sft.dtype = dtype
    base = _base_checkpoint(tmp_path, config)
    fresh_clock = count()
    monkeypatch.setattr(training_loop, "perf_counter", lambda: float(next(fresh_clock)))
    uninterrupted = _run(config, base_checkpoint=base)
    resume_checkpoint = uninterrupted.paths.checkpoints_dir / "step_000002.pt"

    resumed_config = _config(tmp_path, run_name="sft-resumed")
    resumed_config.sft.dtype = dtype
    resumed_config.sft.base_checkpoint = str(base)
    resumed_clock = count()
    monkeypatch.setattr(
        training_loop,
        "perf_counter",
        lambda: float(next(resumed_clock)),
    )
    resumed = _run(resumed_config, resume_from=resume_checkpoint)

    uninterrupted_payload = torch.load(
        uninterrupted.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    resumed_payload = torch.load(
        resumed.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert [step.loss for step in resumed.steps] == [
        step.loss for step in uninterrupted.steps[2:]
    ]
    for field in ("model", "optimizer", "scheduler"):
        _assert_nested_equal(resumed_payload[field], uninterrupted_payload[field])
    assert resumed_payload["continuation"] == uninterrupted_payload["continuation"]
    assert resumed_payload["validation"] == uninterrupted_payload["validation"]
    assert resumed_payload["precision"] == uninterrupted_payload["precision"]
    assert resumed_payload["precision"]["dtype"] == dtype
    assert resumed.initial_step == 2
    assert resumed.final_step == 4
    assert resumed.base_checkpoint_identity == file_identity(base)
    assert [record["step"] for record in _metric_records(resumed.metrics_path)] == [
        3,
        4,
        4,
    ]


def test_resume_rejects_changed_continuation_config_before_writing_checkpoints(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, run_name="source")
    base = _base_checkpoint(tmp_path, config)
    source = _run(config, base_checkpoint=base)
    resume_checkpoint = source.paths.checkpoints_dir / "step_000002.pt"
    changed = _config(tmp_path, run_name="changed")
    changed.sft.base_checkpoint = str(base)
    changed.sft.learning_rate *= 2
    changed.validate()
    paths = prepare_run(changed)
    tracker = build_tracker(changed, paths, stage="train_sft")

    with tracker, pytest.raises(SFTTrainingError, match="resume config must match"):
        run_sft_training(
            changed,
            paths=paths,
            tracker=tracker,
            resume_from=resume_checkpoint,
        )

    assert not list(paths.checkpoints_dir.glob("*.pt"))


def test_injected_hub_cache_source_loads_offline_through_the_same_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet = tmp_path / "smoltalk.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ]
                }
            ]
        ),
        parquet,
    )
    spec = get_sft_dataset_spec("smoltalk", "train")
    cache_root = tmp_path / "cache"
    publish_local_parquet_cache(spec, cache_root, (parquet,))
    config = SFTConfig(
        train_sources=[
            SFTSourceConfig(
                kind="hub_cache",
                path=str(cache_root),
                dataset="smoltalk",
                split="train",
                shuffle=True,
            )
        ]
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be used"),
    )

    sources = build_sft_conversation_sources(config.train_sources, config=config)
    example = next(sources[0].iter_examples(seed=1))

    assert example.conversation.messages[-1].content == "Hi"
    assert sources[0].source_identity.startswith("sha256:")


def test_torch_oom_uses_shared_diagnostic_with_sft_cli_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_name="sft-oom")
    config.sft.device_batch_size = 2
    config.sft.total_batch_size_tokens = 128
    config.validate()
    base = _base_checkpoint(tmp_path, config)
    monkeypatch.setattr(
        sft_training,
        "run_training_steps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("fixture OOM")
        ),
    )

    with pytest.raises(SFTTrainingOOMError) as exc_info:
        _run(config, base_checkpoint=base)

    assert exc_info.value.diagnostic.attempt.device_batch_size == 2
    assert exc_info.value.diagnostic.attempt.total_batch_size_tokens == 128
    assert "--override sft.device_batch_size=" in str(exc_info.value)
