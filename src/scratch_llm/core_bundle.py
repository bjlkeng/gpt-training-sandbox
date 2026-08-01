"""Strict read-only loading for the pinned nanochat CORE evaluation bundle."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final
import zipfile

from omegaconf import OmegaConf

from scratch_llm._validation import (
    require_non_empty_string,
    require_non_negative_integer,
)
from scratch_llm.core_evaluation import CoreEvaluationError, CoreTaskType
from scratch_llm.identity import file_identity


CORE_BUNDLE_URL: Final = (
    "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
)
CORE_BUNDLE_SHA256: Final = (
    "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
)
CORE_CONFIG_MEMBER: Final = "eval_bundle/core.yaml"
CORE_METADATA_MEMBER: Final = "eval_bundle/eval_meta_data.csv"
CORE_TASK_LABELS: Final = (
    "hellaswag_zeroshot",
    "jeopardy",
    "bigbench_qa_wikidata",
    "arc_easy",
    "arc_challenge",
    "copa",
    "commonsense_qa",
    "piqa",
    "openbook_qa",
    "lambada_openai",
    "hellaswag",
    "winograd",
    "winogrande",
    "bigbench_dyck_languages",
    "agi_eval_lsat_ar",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
)
CORE_REFERENCE_FILES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai-community-gpt2": "eval_bundle/openai-community-gpt2.csv",
        "openai-community-gpt2-medium": (
            "eval_bundle/openai-community-gpt2-medium.csv"
        ),
        "openai-community-gpt2-large": ("eval_bundle/openai-community-gpt2-large.csv"),
    }
)

_SUPPORTED_TASK_TYPES = frozenset({"multiple_choice", "schema", "language_modeling"})
_CONFIG_TASK_REQUIRED_KEYS = frozenset(
    {"label", "dataset_uri", "num_fewshot", "icl_task_type"}
)
_CONFIG_TASK_OPTIONAL_KEYS = frozenset({"continuation_delimiter", "has_categories"})
_HEX_DIGITS = frozenset("0123456789abcdef")


class CoreBundleError(CoreEvaluationError):
    """A local archive does not satisfy the pinned CORE bundle contract."""


@dataclass(frozen=True)
class CoreBundleSpec:
    """Expected immutable archive identity, task order, and references."""

    archive_sha256: str
    task_labels: tuple[str, ...]
    reference_files: Mapping[str, str]

    def __post_init__(self) -> None:
        digest = require_non_empty_string(
            self.archive_sha256,
            name="archive_sha256",
        ).lower()
        if len(digest) != 64 or any(
            character not in _HEX_DIGITS for character in digest
        ):
            raise ValueError("archive_sha256 must contain 64 lowercase hex digits")
        if not isinstance(self.task_labels, tuple) or not self.task_labels:
            raise ValueError("task_labels must be a non-empty tuple")
        for index, label in enumerate(self.task_labels):
            require_non_empty_string(label, name=f"task_labels[{index}]")
        if len(set(self.task_labels)) != len(self.task_labels):
            raise ValueError("task_labels must be unique")
        if not isinstance(self.reference_files, Mapping) or not self.reference_files:
            raise ValueError("reference_files must be a non-empty mapping")
        references: dict[str, str] = {}
        for model_id, member in self.reference_files.items():
            require_non_empty_string(model_id, name="reference model id")
            _require_safe_archive_member(member, label=f"reference {model_id!r}")
            references[model_id] = member
        object.__setattr__(self, "archive_sha256", digest)
        object.__setattr__(self, "reference_files", MappingProxyType(references))


@dataclass(frozen=True)
class CoreTask:
    """One task definition after config and baseline validation."""

    label: str
    task_type: CoreTaskType
    dataset_member: str
    num_fewshot: int
    continuation_delimiter: str
    random_baseline_percent: float


@dataclass(frozen=True)
class CoreReferenceResult:
    """One pinned reference-model result table from the bundle."""

    model_id: str
    task_accuracies: Mapping[str, float]
    task_centered_scores: Mapping[str, float]
    core_metric: float


@dataclass(frozen=True)
class CoreBundle:
    """Validated archive metadata; task rows remain lazily loaded."""

    path: Path
    identity: str
    config_identity: str
    metadata_identity: str
    tasks: tuple[CoreTask, ...]
    reference_results: tuple[CoreReferenceResult, ...]
    spec: CoreBundleSpec


def load_core_bundle(
    path: str | Path,
    *,
    spec: CoreBundleSpec | None = None,
) -> CoreBundle:
    """Validate one local archive without downloading or extracting it."""

    if spec is None:
        spec = NANOCHAT_CORE_V1_BUNDLE_SPEC
    if not isinstance(spec, CoreBundleSpec):
        raise TypeError("spec must be a CoreBundleSpec")
    resolved = Path(path)
    actual_identity = file_identity(resolved)
    expected_identity = f"sha256:{spec.archive_sha256}"
    if actual_identity != expected_identity:
        raise CoreBundleError(
            "CORE bundle SHA-256 does not match the pinned protocol: "
            f"expected {expected_identity}, got {actual_identity}"
        )
    try:
        with zipfile.ZipFile(resolved) as archive:
            _reject_duplicate_members(archive)
            config_bytes = _read_member(archive, CORE_CONFIG_MEMBER)
            metadata_bytes = _read_member(archive, CORE_METADATA_MEMBER)
            baselines = _load_random_baselines(metadata_bytes)
            tasks = _load_task_config(config_bytes, baselines=baselines, spec=spec)
            references = tuple(
                _load_reference_result(
                    archive,
                    model_id=model_id,
                    member=member,
                    task_labels=spec.task_labels,
                )
                for model_id, member in spec.reference_files.items()
            )
    except (OSError, zipfile.BadZipFile, UnicodeError) as error:
        raise CoreBundleError(
            f"could not read CORE bundle {resolved}: {error}"
        ) from error
    return CoreBundle(
        path=resolved,
        identity=actual_identity,
        config_identity=_bytes_identity(config_bytes),
        metadata_identity=_bytes_identity(metadata_bytes),
        tasks=tasks,
        reference_results=references,
        spec=spec,
    )


def _load_task_config(
    data: bytes,
    *,
    baselines: Mapping[str, float],
    spec: CoreBundleSpec,
) -> tuple[CoreTask, ...]:
    try:
        text = data.decode("utf-8")
        raw = OmegaConf.to_container(OmegaConf.create(text), resolve=True)
    except (UnicodeError, ValueError, TypeError) as error:
        raise CoreBundleError(f"CORE config is invalid YAML: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"icl_tasks"}:
        raise CoreBundleError("CORE config must contain exactly the icl_tasks field")
    raw_tasks = raw["icl_tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise CoreBundleError("CORE config icl_tasks must be a non-empty list")
    tasks = tuple(
        _parse_task_config(value, index=index, baselines=baselines)
        for index, value in enumerate(raw_tasks)
    )
    labels = tuple(task.label for task in tasks)
    if labels != spec.task_labels:
        raise CoreBundleError(
            "CORE task labels or order do not match the pinned protocol; "
            f"expected={list(spec.task_labels)}, got={list(labels)}"
        )
    if not set(spec.task_labels) <= set(baselines):
        missing = sorted(set(spec.task_labels) - set(baselines))
        raise CoreBundleError(
            f"CORE random baselines are missing configured tasks; missing={missing}"
        )
    return tasks


def _parse_task_config(
    value: object,
    *,
    index: int,
    baselines: Mapping[str, float],
) -> CoreTask:
    label_prefix = f"icl_tasks[{index}]"
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CoreBundleError(f"{label_prefix} must be an object with string keys")
    keys = set(value)
    if not _CONFIG_TASK_REQUIRED_KEYS <= keys or not keys <= (
        _CONFIG_TASK_REQUIRED_KEYS | _CONFIG_TASK_OPTIONAL_KEYS
    ):
        missing = sorted(_CONFIG_TASK_REQUIRED_KEYS - keys)
        unexpected = sorted(
            keys - _CONFIG_TASK_REQUIRED_KEYS - _CONFIG_TASK_OPTIONAL_KEYS
        )
        raise CoreBundleError(
            f"{label_prefix} fields are invalid; missing={missing}, "
            f"unexpected={unexpected}"
        )
    label = _task_string(value["label"], label=f"{label_prefix}.label")
    task_type = value["icl_task_type"]
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise CoreBundleError(
            f"{label_prefix}.icl_task_type is unsupported: {task_type!r}"
        )
    dataset_uri = _task_string(
        value["dataset_uri"],
        label=f"{label_prefix}.dataset_uri",
    )
    _require_safe_relative_path(dataset_uri, label=f"{label_prefix}.dataset_uri")
    dataset_member = f"eval_bundle/eval_data/{dataset_uri}"
    _require_safe_archive_member(dataset_member, label=f"{label_prefix}.dataset_uri")
    fewshot_values = value["num_fewshot"]
    if not isinstance(fewshot_values, list) or len(fewshot_values) != 1:
        raise CoreBundleError(f"{label_prefix}.num_fewshot must contain one integer")
    try:
        num_fewshot = require_non_negative_integer(
            fewshot_values[0],
            name=f"{label_prefix}.num_fewshot[0]",
        )
    except (TypeError, ValueError) as error:
        raise CoreBundleError(str(error)) from error
    delimiter = value.get("continuation_delimiter", " ")
    if not isinstance(delimiter, str):
        raise CoreBundleError(f"{label_prefix}.continuation_delimiter must be a string")
    try:
        baseline = baselines[label]
    except KeyError as error:
        raise CoreBundleError(f"CORE baseline is missing for task {label!r}") from error
    return CoreTask(
        label=label,
        task_type=task_type,  # type: ignore[arg-type]
        dataset_member=dataset_member,
        num_fewshot=num_fewshot,
        continuation_delimiter=delimiter,
        random_baseline_percent=baseline,
    )


def _load_random_baselines(data: bytes) -> Mapping[str, float]:
    rows = _csv_rows(data, label="CORE metadata")
    if not rows or rows[0] != ["Eval Task", "Random baseline"]:
        if not rows or not {"Eval Task", "Random baseline"} <= set(rows[0]):
            raise CoreBundleError(
                "CORE metadata must include Eval Task and Random baseline columns"
            )
        task_index = rows[0].index("Eval Task")
        baseline_index = rows[0].index("Random baseline")
    else:
        task_index, baseline_index = 0, 1
    baselines: dict[str, float] = {}
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) <= max(task_index, baseline_index):
            raise CoreBundleError(f"CORE metadata row {row_number} is incomplete")
        label = _task_string(row[task_index], label=f"metadata row {row_number} task")
        if label in baselines:
            raise CoreBundleError(f"CORE metadata duplicates task {label!r}")
        baseline = _finite_float(
            row[baseline_index],
            label=f"metadata baseline for {label!r}",
        )
        if not 0 <= baseline < 100:
            raise CoreBundleError(
                f"metadata baseline for {label!r} must be in [0, 100)"
            )
        baselines[label] = baseline
    return MappingProxyType(baselines)


def _load_reference_result(
    archive: zipfile.ZipFile,
    *,
    model_id: str,
    member: str,
    task_labels: tuple[str, ...],
) -> CoreReferenceResult:
    rows = _csv_rows(_read_member(archive, member), label=f"reference {model_id!r}")
    if not rows or rows[0] != ["Task", "Accuracy", "Centered"]:
        raise CoreBundleError(
            f"reference {model_id!r} must have Task, Accuracy, Centered columns"
        )
    expected_labels = (*task_labels, "CORE")
    labels = tuple(row[0] if row else "" for row in rows[1:])
    if labels != expected_labels:
        raise CoreBundleError(
            f"reference {model_id!r} task order does not match the CORE config"
        )
    accuracies: dict[str, float] = {}
    centered: dict[str, float] = {}
    for label, row in zip(task_labels, rows[1:-1], strict=True):
        if len(row) != 3:
            raise CoreBundleError(f"reference {model_id!r} row {label!r} is invalid")
        accuracies[label] = _finite_float(
            row[1], label=f"reference {model_id!r} accuracy {label!r}"
        )
        centered[label] = _finite_float(
            row[2], label=f"reference {model_id!r} centered {label!r}"
        )
    core_row = rows[-1]
    if len(core_row) != 3 or core_row[1] != "":
        raise CoreBundleError(f"reference {model_id!r} CORE row is invalid")
    return CoreReferenceResult(
        model_id=model_id,
        task_accuracies=MappingProxyType(accuracies),
        task_centered_scores=MappingProxyType(centered),
        core_metric=_finite_float(
            core_row[2], label=f"reference {model_id!r} CORE metric"
        ),
    )


def _task_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise CoreBundleError(f"{label} must be a non-empty string")
    return value


def _csv_rows(data: bytes, *, label: str) -> list[list[str]]:
    try:
        text = data.decode("utf-8")
        return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text))]
    except (UnicodeError, csv.Error) as error:
        raise CoreBundleError(f"{label} is invalid CSV: {error}") from error


def _finite_float(value: object, *, label: str) -> float:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise CoreBundleError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CoreBundleError(f"{label} must be a finite number") from error
    if not math.isfinite(numeric):
        raise CoreBundleError(f"{label} must be a finite number")
    return numeric


def _read_member(archive: zipfile.ZipFile, member: str) -> bytes:
    try:
        return archive.read(member)
    except KeyError as error:
        raise CoreBundleError(f"CORE bundle is missing {member!r}") from error


def _reject_duplicate_members(archive: zipfile.ZipFile) -> None:
    names = [entry.filename for entry in archive.infolist()]
    if len(names) != len(set(names)):
        raise CoreBundleError("CORE bundle contains duplicate archive members")


def _require_safe_relative_path(value: str, *, label: str) -> None:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise CoreBundleError(f"{label} must be a safe relative path")


def _require_safe_archive_member(value: object, *, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    _require_safe_relative_path(value, label=label)


def _bytes_identity(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


NANOCHAT_CORE_V1_BUNDLE_SPEC: Final = CoreBundleSpec(
    archive_sha256=CORE_BUNDLE_SHA256,
    task_labels=CORE_TASK_LABELS,
    reference_files=CORE_REFERENCE_FILES,
)


__all__ = [
    "CORE_BUNDLE_SHA256",
    "CORE_BUNDLE_URL",
    "CORE_CONFIG_MEMBER",
    "CORE_METADATA_MEMBER",
    "CORE_REFERENCE_FILES",
    "CORE_TASK_LABELS",
    "CoreBundle",
    "CoreBundleError",
    "CoreBundleSpec",
    "CoreReferenceResult",
    "CoreTask",
    "NANOCHAT_CORE_V1_BUNDLE_SPEC",
    "load_core_bundle",
]
