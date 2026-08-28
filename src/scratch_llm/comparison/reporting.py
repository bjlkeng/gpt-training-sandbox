"""Protocol-aware payload construction and Markdown rendering for run comparisons."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, TypeGuard

from scratch_llm.comparison.model import (
    IDENTITY_FIELDS,
    RUN_COMPARISON_FORMAT,
    RUN_COMPARISON_FORMAT_VERSION,
    STEP_METRICS,
    RunSnapshot,
)
from scratch_llm.evaluation.base_tracking import (
    FULL_DOCUMENT_MINIMUM_TRAIN_METRIC,
    NANOCHAT_MINIMUM_TRAIN_METRIC,
)
from scratch_llm.identity import project_config_identity
from scratch_llm.evaluation.full_document_bpb import (
    FULL_DOCUMENT_PROTOCOL_ID,
    FULL_DOCUMENT_TRAIN_METRIC,
)
from scratch_llm.evaluation.nanochat_bpb import (
    NANOCHAT_COMPAT_PROTOCOL_ID,
    NANOCHAT_COMPAT_TRAIN_METRIC,
)
from scratch_llm.diagnostics.resource_estimation import (
    GPTModelSizeEstimate,
    estimate_gpt_model_size,
)


def build_comparison_payload(
    snapshots: tuple[RunSnapshot, ...],
) -> dict[str, Any]:
    """Build one deterministic JSON-compatible comparison payload."""

    run_payloads = [_run_payload(snapshot) for snapshot in snapshots]
    rankings = {
        protocol_id: _rank_protocol(snapshots, run_payloads, protocol_id)
        for protocol_id in (
            FULL_DOCUMENT_PROTOCOL_ID,
            NANOCHAT_COMPAT_PROTOCOL_ID,
        )
    }
    aligned_steps = sorted(
        {step for snapshot in snapshots for step in snapshot.training_metrics}
    )
    step_series = {
        str(step): {
            snapshot.name: dict(
                snapshot.training_metrics.get(
                    step,
                    {name: None for name in STEP_METRICS},
                )
            )
            for snapshot in snapshots
        }
        for step in aligned_steps
    }
    return {
        "aligned_steps": aligned_steps,
        "format": RUN_COMPARISON_FORMAT,
        "format_version": RUN_COMPARISON_FORMAT_VERSION,
        "identity_differences": _identity_differences(run_payloads),
        "numeric_deltas": _numeric_deltas(run_payloads),
        "rankings": rankings,
        "run_count": len(snapshots),
        "runs": run_payloads,
        "step_series": step_series,
    }


def _run_payload(snapshot: RunSnapshot) -> dict[str, Any]:
    evaluation = snapshot.base_evaluation
    results = {} if evaluation is None else evaluation["results"]
    compatibility = results.get(NANOCHAT_COMPAT_PROTOCOL_ID)
    full_document = results.get(FULL_DOCUMENT_PROTOCOL_ID)
    latest_step = max(snapshot.training_metrics, default=None)
    losses = [
        metrics["train/loss"]
        for metrics in snapshot.training_metrics.values()
        if _is_number(metrics.get("train/loss"))
    ]
    model = estimate_gpt_model_size(snapshot.config.model)
    configured_parameters = model.unique_parameters + (
        model.token_embedding_parameters if model.tie_weights else 0
    )
    blockers = _ranking_blockers(snapshot, compatibility, full_document)
    return {
        "evaluation": {
            NANOCHAT_COMPAT_PROTOCOL_ID: _protocol_summary(compatibility),
            FULL_DOCUMENT_PROTOCOL_ID: _protocol_summary(full_document),
        },
        "identities": _run_identities(snapshot, evaluation, model),
        "path": str(snapshot.path),
        "rankable": not blockers,
        "ranking_blockers": blockers,
        "run": snapshot.name,
        "status": snapshot.summary["status"],
        "training": {
            "best_compatibility_bpb": _minimum_metric(
                snapshot.training_metrics,
                NANOCHAT_COMPAT_TRAIN_METRIC,
                NANOCHAT_MINIMUM_TRAIN_METRIC,
            ),
            "best_full_document_bpb": _minimum_metric(
                snapshot.training_metrics,
                FULL_DOCUMENT_TRAIN_METRIC,
                FULL_DOCUMENT_MINIMUM_TRAIN_METRIC,
            ),
            "best_loss": min(losses) if losses else None,
            "configured_parameters": configured_parameters,
            "cumulative_processed_model_tokens": (
                None
                if latest_step is None
                else latest_step * snapshot.config.train.total_batch_size_tokens
            ),
            "latest_compatibility_bpb": _latest_metric(
                snapshot.training_metrics,
                NANOCHAT_COMPAT_TRAIN_METRIC,
            ),
            "latest_full_document_bpb": _latest_metric(
                snapshot.training_metrics,
                FULL_DOCUMENT_TRAIN_METRIC,
            ),
            "latest_loss": _latest_metric(
                snapshot.training_metrics,
                "train/loss",
            ),
            "latest_step": latest_step,
            "mfu": _latest_metric(snapshot.training_metrics, "train/mfu"),
            "peak_memory_mib": _latest_metric(
                snapshot.training_metrics,
                "train/peak_memory_mib",
            ),
            "tokens_per_second": _latest_metric(
                snapshot.training_metrics,
                "train/tok_per_sec",
            ),
            "total_training_flops": _latest_metric(
                snapshot.training_metrics,
                "total_training_flops",
            ),
            "total_training_time": _latest_metric(
                snapshot.training_metrics,
                "total_training_time",
            ),
            "unique_parameters": model.unique_parameters,
        },
    }


def _run_identities(
    snapshot: RunSnapshot,
    evaluation: Mapping[str, Any] | None,
    model: GPTModelSizeEstimate,
) -> dict[str, Any]:
    report_identities = {} if evaluation is None else evaluation.get("identities", {})
    checkpoint = report_identities.get("checkpoint")
    return {
        "checkpoint_identity": (
            checkpoint.get("identity") if isinstance(checkpoint, dict) else None
        ),
        "code_identity": None,
        "config_identity": report_identities.get("config")
        or project_config_identity(snapshot.config),
        "parameterization": {
            "n_head": model.head_count,
            "n_kv_head": model.n_kv_head,
            "sliding_window": snapshot.config.model.attention_window_identity(),
            "tie_weights": model.tie_weights,
            "unique_parameters": model.unique_parameters,
            "use_gqa": model.use_gqa,
            "value_embeddings": snapshot.config.model.value_embedding_identity(),
        },
        "hardware": {
            "device": snapshot.config.run.device,
            "mfu_peak_flops_basis": snapshot.config.train.mfu_peak_flops_basis,
            "mfu_peak_flops_per_second": (
                snapshot.config.train.mfu_peak_flops_per_second
            ),
        },
        "precision": {
            "activation_checkpointing": (
                snapshot.config.train.activation_checkpointing
            ),
            "compile": snapshot.config.train.compile,
            "dtype": snapshot.config.train.dtype,
        },
        "tokenizer_identity": report_identities.get("tokenizer"),
        "validation_manifest_identity": report_identities.get("validation_manifest"),
    }


def _ranking_blockers(
    snapshot: RunSnapshot,
    compatibility: object,
    full_document: object,
) -> list[str]:
    blockers: list[str] = []
    if snapshot.summary["status"] != "completed":
        blockers.append(f"run status is {snapshot.summary['status']}")
    if not snapshot.training_metrics:
        blockers.append("training metrics are missing")
    else:
        latest_step = max(snapshot.training_metrics)
        configured_steps = snapshot.config.train.max_steps
        if latest_step < configured_steps:
            blockers.append(
                f"training stopped at step {latest_step} of {configured_steps}"
            )
        elif latest_step > configured_steps:
            blockers.append(
                "training latest step "
                f"{latest_step} exceeds configured max_steps {configured_steps}"
            )
    if snapshot.base_evaluation is None:
        blockers.append("base_eval.json is missing")
    elif snapshot.base_evaluation.get(
        "completed_modes"
    ) != snapshot.base_evaluation.get("requested_modes"):
        blockers.append("base evaluation is incomplete")
    if compatibility is None:
        blockers.append(f"{NANOCHAT_COMPAT_PROTOCOL_ID} result is missing")
    if full_document is None:
        blockers.append(f"{FULL_DOCUMENT_PROTOCOL_ID} result is missing")
    return blockers


def _protocol_summary(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "bpb": value["bpb"],
        "checkpoint_identity": value["checkpoint_identity"],
        "protocol_id": value["protocol_id"],
        "protocol_version": value["protocol_version"],
        "reference_commit": value["reference_commit"],
        "reference_config": value["reference_config"],
        "source_byte_retention": value["source_byte_retention"],
        "tokenizer_identity": value["tokenizer_identity"],
        "validation_manifest_identity": value["validation_manifest_identity"],
    }


def _rank_protocol(
    snapshots: tuple[RunSnapshot, ...],
    run_payloads: list[dict[str, Any]],
    protocol_id: str,
) -> list[dict[str, Any]]:
    if any(not payload["rankable"] for payload in run_payloads):
        return []
    results: list[Mapping[str, Any]] = []
    for snapshot in snapshots:
        evaluation = snapshot.base_evaluation
        assert evaluation is not None
        result = evaluation["results"][protocol_id]
        assert isinstance(result, dict)
        results.append(result)
    keys = [
        _protocol_comparison_key(snapshot, result)
        for snapshot, result in zip(snapshots, results)
    ]
    if len(set(keys)) != 1:
        return []
    ordered = sorted(
        zip(snapshots, results),
        key=lambda item: (item[1]["bpb"], item[0].name),
    )
    ranking: list[dict[str, Any]] = []
    prior_bpb: float | None = None
    prior_rank = 0
    for position, (snapshot, result) in enumerate(ordered, start=1):
        bpb = float(result["bpb"])
        rank = prior_rank if prior_bpb == bpb else position
        ranking.append({"bpb": bpb, "rank": rank, "run": snapshot.name})
        prior_bpb = bpb
        prior_rank = rank
    return ranking


def _protocol_comparison_key(
    snapshot: RunSnapshot,
    result: Mapping[str, Any],
) -> str:
    evaluation = snapshot.base_evaluation
    assert evaluation is not None
    return _canonical(
        {
            "bounded": evaluation.get("bounded"),
            "protocol_id": result["protocol_id"],
            "protocol_version": result["protocol_version"],
            "reference_commit": result["reference_commit"],
            "reference_config": result["reference_config"],
            "run_kind": evaluation.get("run_kind"),
            "unique_parameters": estimate_gpt_model_size(
                snapshot.config.model
            ).unique_parameters,
            "tokenizer_identity": result["tokenizer_identity"],
            "validation_manifest_identity": result["validation_manifest_identity"],
        }
    )


def _identity_differences(
    run_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for field in IDENTITY_FIELDS:
        values = {
            payload["run"]: payload["identities"][field] for payload in run_payloads
        }
        if len({_canonical(value) for value in values.values()}) > 1:
            differences.append({"field": field, "values": values})
    return differences


def _numeric_deltas(run_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = run_payloads[0]
    fields = {
        "best_compatibility_bpb": ("training", "best_compatibility_bpb"),
        "best_full_document_bpb": ("training", "best_full_document_bpb"),
        "best_loss": ("training", "best_loss"),
        "latest_loss": ("training", "latest_loss"),
        "tokens_per_second": ("training", "tokens_per_second"),
        "mfu": ("training", "mfu"),
        "peak_memory_mib": ("training", "peak_memory_mib"),
    }
    deltas: list[dict[str, Any]] = []
    for label, path in fields.items():
        baseline_value = baseline[path[0]][path[1]]
        values: dict[str, float | None] = {}
        for payload in run_payloads:
            value = payload[path[0]][path[1]]
            values[payload["run"]] = (
                float(value) - float(baseline_value)
                if _is_number(value) and _is_number(baseline_value)
                else None
            )
        deltas.append(
            {
                "baseline": baseline["run"],
                "field": label,
                "values": values,
            }
        )
    return deltas


def render_comparison_markdown(payload: Mapping[str, Any]) -> str:
    """Render the identity-first comparison payload as stable Markdown."""

    runs = payload["runs"]
    lines = [
        "# Training run comparison",
        "",
        f"Compared runs: `{payload['run_count']}`",
        "",
        "## Identity differences",
        "",
    ]
    differences = payload["identity_differences"]
    if differences:
        lines.extend(["| Field | Values by run |", "| --- | --- |"])
        for difference in differences:
            lines.append(
                f"| {_escape_markdown(difference['field'])} | "
                f"{_escape_markdown(_canonical(difference['values']))} |"
            )
    else:
        lines.append("No identity differences detected.")
    unavailable_identities = [
        field
        for field in IDENTITY_FIELDS
        if all(run["identities"][field] is None for run in runs)
    ]
    if unavailable_identities:
        lines.extend(
            [
                "",
                "Unavailable across all runs: "
                + ", ".join(f"`{field}`" for field in unavailable_identities)
                + ".",
            ]
        )
    lines.extend(
        [
            "",
            "## Numeric deltas",
            "",
            "| Metric | Baseline | Deltas by run |",
            "| --- | --- | --- |",
        ]
    )
    for delta in payload["numeric_deltas"]:
        lines.append(
            f"| {_escape_markdown(delta['field'])} | "
            f"{_escape_markdown(delta['baseline'])} | "
            f"{_escape_markdown(_canonical(delta['values']))} |"
        )
    lines.extend(_training_summary_markdown(runs))
    lines.extend(_training_quality_markdown(runs))
    lines.extend(_performance_markdown(runs))
    lines.extend(_ranking_blockers_markdown(runs))
    lines.extend(
        _protocol_markdown(payload, NANOCHAT_COMPAT_PROTOCOL_ID, "Compatibility BPB")
    )
    lines.extend(
        _protocol_markdown(payload, FULL_DOCUMENT_PROTOCOL_ID, "Full-document BPB")
    )
    return "\n".join(lines) + "\n"


def _training_summary_markdown(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Training summary",
        "",
        "| Run | Status | Step | Configured parameters | Unique parameters | Processed tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        training = run["training"]
        lines.append(
            f"| {_escape_markdown(run['run'])} | "
            f"{_escape_markdown(run['status'])} | "
            f"{_display(training['latest_step'])} | "
            f"{_display(training['configured_parameters'])} | "
            f"{_display(training['unique_parameters'])} | "
            f"{_display(training['cumulative_processed_model_tokens'])} |"
        )
    return lines


def _training_quality_markdown(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Training quality",
        "",
        "| Run | Latest loss | Best loss | Latest compatibility BPB | Best compatibility BPB | Latest full-document BPB | Best full-document BPB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        training = run["training"]
        lines.append(
            f"| {_escape_markdown(run['run'])} | "
            f"{_display(training['latest_loss'])} | "
            f"{_display(training['best_loss'])} | "
            f"{_display(training['latest_compatibility_bpb'])} | "
            f"{_display(training['best_compatibility_bpb'])} | "
            f"{_display(training['latest_full_document_bpb'])} | "
            f"{_display(training['best_full_document_bpb'])} |"
        )
    return lines


def _performance_markdown(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Performance",
        "",
        "| Run | Tokens/sec | MFU | Peak MiB | Total FLOPs | Training seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        training = run["training"]
        lines.append(
            f"| {_escape_markdown(run['run'])} | "
            f"{_display(training['tokens_per_second'])} | "
            f"{_display(training['mfu'])} | "
            f"{_display(training['peak_memory_mib'])} | "
            f"{_display(training['total_training_flops'])} | "
            f"{_display(training['total_training_time'])} |"
        )
    return lines


def _ranking_blockers_markdown(runs: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Ranking blockers", ""]
    blocked_runs = [run for run in runs if run["ranking_blockers"]]
    if not blocked_runs:
        return [*lines, "No run-level ranking blockers detected."]
    for run in blocked_runs:
        lines.append(
            f"- {_escape_markdown(run['run'])}: "
            + "; ".join(_escape_markdown(reason) for reason in run["ranking_blockers"])
        )
    return lines


def _protocol_markdown(
    payload: Mapping[str, Any],
    protocol_id: str,
    title: str,
) -> list[str]:
    ranks = {entry["run"]: entry["rank"] for entry in payload["rankings"][protocol_id]}
    lines = [
        "",
        f"## {title}",
        "",
        "| Run | BPB | Source-byte retention | Rank |",
        "| --- | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        result = run["evaluation"][protocol_id]
        lines.append(
            f"| {_escape_markdown(run['run'])} | "
            f"{_display(None if result is None else result['bpb'])} | "
            f"{_display(None if result is None else result['source_byte_retention'])} | "
            f"{_display(ranks.get(run['run']))} |"
        )
    return lines


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _latest_metric(
    training_metrics: Mapping[int, Mapping[str, Any]],
    name: str,
) -> int | float | None:
    for step in sorted(training_metrics, reverse=True):
        value = training_metrics[step].get(name)
        if _is_number(value):
            return value
    return None


def _minimum_metric(
    training_metrics: Mapping[int, Mapping[str, Any]],
    *names: str,
) -> float | None:
    values: list[float] = []
    for metrics in training_metrics.values():
        for name in names:
            value = metrics.get(name)
            if _is_number(value):
                values.append(float(value))
    return min(values, default=None)


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = ["build_comparison_payload", "render_comparison_markdown"]
