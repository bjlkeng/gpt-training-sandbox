"""Checkpoint, data, evaluation, and reporting composition for base models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import Tensor

from scratch_llm.evaluation.base import (
    BaseEvaluationContext,
    BaseEvaluationCoreRunner,
    BaseEvaluationError,
    BaseEvaluationMode,
    execute_base_evaluation_modes,
)
from scratch_llm.evaluation.base_tracking import (
    TrackedBaseEvaluation,
    report_completed_base_evaluation,
)
from scratch_llm.evaluation.sampling import (
    BaseSamplesResult,
    FixedBaseSamplingConfig,
    generate_fixed_base_samples,
)
from scratch_llm.training.best_checkpoint import PeriodicValidationResult
from scratch_llm.training.checkpoint import load_model_checkpoint
from scratch_llm.training.precision import PrecisionError, build_precision_policy
from scratch_llm.config import ProjectConfig
from scratch_llm.evaluation.core.bundle import CoreBundle, load_core_bundle
from scratch_llm.evaluation.core.results import CoreTaskResult
from scratch_llm.evaluation.core.pipeline import evaluate_core_bundle
from scratch_llm.evaluation.full_document_bpb import (
    FullDocumentProtocolConfig,
    evaluate_full_document_bpb,
)
from scratch_llm.identity import file_identity, project_config_identity
from scratch_llm.evaluation.nanochat_bpb import (
    NanochatCompatibilityConfig,
    evaluate_nanochat_compatible_bpb,
)
from scratch_llm.data.tokenized import (
    TokenizedShardReader,
    tokenized_manifest_identity,
)
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.tokenization.artifacts import build_token_byte_lengths
from scratch_llm.tracking import Tracker


def evaluate_checkpoint_base_model(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    modes: Sequence[BaseEvaluationMode],
    tracker: Tracker,
    run_dir: str | Path,
    max_per_task: int | None = None,
    core_bundle_path: str | Path | None = None,
    core_runner: BaseEvaluationCoreRunner | None = None,
    core_progress: Callable[[CoreTaskResult], None] | None = None,
) -> TrackedBaseEvaluation:
    """Run requested modes against one loaded checkpoint and publish on success."""

    if not isinstance(config, ProjectConfig):
        raise TypeError(f"config must be a ProjectConfig, got {type(config).__name__}")
    config.validate()
    try:
        precision = build_precision_policy(
            dtype=config.train.dtype,
            device=config.run.device,
        )
    except PrecisionError as error:
        raise BaseEvaluationError(
            f"invalid base evaluation precision policy: {error}"
        ) from error
    requested_modes = tuple(modes)
    if max_per_task is not None and "core" not in requested_modes:
        raise BaseEvaluationError("max_per_task requires the core evaluation mode")
    if core_bundle_path is not None and "core" not in requested_modes:
        raise BaseEvaluationError("core_bundle_path requires the core evaluation mode")
    if core_runner is not None and core_bundle_path is not None:
        raise BaseEvaluationError(
            "core_runner and core_bundle_path are mutually exclusive"
        )
    core_bundle: CoreBundle | None = None
    if "core" in requested_modes and core_runner is None:
        if core_bundle_path is None:
            raise BaseEvaluationError(
                "core_bundle_path is required for the core evaluation mode"
            )
        core_bundle = load_core_bundle(core_bundle_path)

    resolved_checkpoint_path = Path(checkpoint_path)
    checkpoint_identity = file_identity(resolved_checkpoint_path)
    checkpoint = load_model_checkpoint(
        resolved_checkpoint_path,
        device=config.run.device,
    )
    _validate_checkpoint_config(config, checkpoint.config)
    tokenizer = checkpoint.tokenizer
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("checkpoint tokenizer must implement Tokenizer")
    tokenizer_identity = tokenizer.get_identity()

    with ExitStack() as resources:
        reader: TokenizedShardReader | None = None
        token_bytes: Tensor | None = None
        manifest_identity: str | None = None
        if "bpb" in requested_modes:
            reader = resources.enter_context(
                TokenizedShardReader(
                    config.data.tokenized_dir,
                    tokenizer=tokenizer,
                )
            )
            manifest_identity = tokenized_manifest_identity(reader.manifest)
            token_bytes = load_evaluation_token_bytes(
                tokenizer,
                artifact_dir=config.tokenizer.artifact_dir,
            )

        context = BaseEvaluationContext(
            checkpoint_identity=checkpoint_identity,
            checkpoint_step=checkpoint.step,
            config_identity=project_config_identity(config),
            tokenizer_identity=tokenizer_identity,
            validation_manifest_identity=manifest_identity,
            run_kind="bounded" if max_per_task is not None else "full",
            max_per_task=max_per_task,
        )

        def run_bpb() -> PeriodicValidationResult:
            if reader is None or token_bytes is None:  # pragma: no cover - preflight.
                raise RuntimeError("BPB resources were not prepared")
            compatibility = evaluate_nanochat_compatible_bpb(
                checkpoint.model,
                tokenizer,
                reader,
                token_bytes,
                parquet_dir=config.data.parquet_dir,
                checkpoint_identity=checkpoint_identity,
                config=NanochatCompatibilityConfig(
                    device_batch_size=config.train.device_batch_size,
                    context_length=config.model.seq_len,
                    eval_tokens=config.train.eval_tokens,
                ),
                device=config.run.device,
            )
            full_document = evaluate_full_document_bpb(
                checkpoint.model,
                tokenizer,
                reader,
                token_bytes,
                checkpoint_identity=checkpoint_identity,
                config=FullDocumentProtocolConfig(
                    device_batch_size=config.train.device_batch_size,
                    context_length=config.model.seq_len,
                ),
                device=config.run.device,
            )
            return PeriodicValidationResult(
                compatibility=compatibility,
                full_document=full_document,
            )

        def run_samples() -> BaseSamplesResult:
            if config.generation.top_p is not None:
                raise BaseEvaluationError(
                    "top_p sampling is not implemented for fixed base samples"
                )
            return generate_fixed_base_samples(
                checkpoint.model,
                tokenizer,
                checkpoint_identity=checkpoint_identity,
                config=FixedBaseSamplingConfig(
                    max_new_tokens=config.generation.max_new_tokens,
                    temperature=config.generation.temperature,
                    top_k=config.generation.top_k,
                    seed=(
                        config.run.seed
                        if config.generation.seed is None
                        else config.generation.seed
                    ),
                ),
                device=config.run.device,
            )

        effective_core_runner = core_runner
        if core_bundle is not None:

            def run_core(limit: int | None):
                return evaluate_core_bundle(
                    checkpoint.model,
                    tokenizer,
                    core_bundle,
                    checkpoint_identity=checkpoint_identity,
                    max_per_task=limit,
                    device=config.run.device,
                    progress=core_progress,
                )

            effective_core_runner = run_core

        with precision.autocast():
            completed = execute_base_evaluation_modes(
                requested_modes,
                context=context,
                bpb_runner=run_bpb if "bpb" in requested_modes else None,
                sample_runner=run_samples if "sample" in requested_modes else None,
                core_runner=effective_core_runner,
            )

    return report_completed_base_evaluation(
        completed,
        tracker=tracker,
        run_dir=run_dir,
    )


def load_evaluation_token_bytes(
    tokenizer: Tokenizer,
    *,
    artifact_dir: str | None,
) -> Tensor:
    """Load the canonical token-byte table or derive it for the byte fixture."""

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement Tokenizer")
    if artifact_dir is None:
        return build_token_byte_lengths(tokenizer)
    path = Path(artifact_dir) / "token_bytes.pt"
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise BaseEvaluationError(
            f"could not load evaluation token bytes {path}: {error}"
        ) from error
    if type(value) is not Tensor:
        raise BaseEvaluationError(
            f"evaluation token bytes {path} must contain exactly one Tensor"
        )
    if value.device.type != "cpu" or value.ndim != 1:
        raise BaseEvaluationError(
            "evaluation token bytes must be a one-dimensional CPU Tensor"
        )
    if value.numel() != tokenizer.get_vocab_size():
        raise BaseEvaluationError(
            "evaluation token bytes size does not match tokenizer vocabulary"
        )
    return value


def _validate_checkpoint_config(
    requested: ProjectConfig,
    checkpoint: ProjectConfig,
) -> None:
    if not isinstance(checkpoint, ProjectConfig):
        raise TypeError("checkpoint config must be a ProjectConfig")
    differences: list[str] = []
    if checkpoint.model != requested.model:
        differences.append("model")
    if checkpoint.tokenizer != requested.tokenizer:
        differences.append("tokenizer")
    if differences:
        raise BaseEvaluationError(
            "checkpoint does not match the resolved evaluation config for: "
            + ", ".join(differences)
        )


__all__ = [
    "evaluate_checkpoint_base_model",
    "load_evaluation_token_bytes",
]
