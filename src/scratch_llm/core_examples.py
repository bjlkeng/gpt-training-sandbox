"""Typed JSONL example loading for validated CORE bundle tasks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
from typing import TypeAlias
import zipfile

from scratch_llm._validation import JsonValueValidator, require_integer
from scratch_llm.core_bundle import CoreBundle, CoreBundleError, CoreTask


@dataclass(frozen=True)
class MultipleChoiceExample:
    query: str
    choices: tuple[str, ...]
    gold: int


@dataclass(frozen=True)
class SchemaExample:
    context_options: tuple[str, ...]
    continuation: str
    gold: int


@dataclass(frozen=True)
class LanguageModelingExample:
    context: str
    continuation: str


CoreExample: TypeAlias = MultipleChoiceExample | SchemaExample | LanguageModelingExample


@dataclass(frozen=True)
class CoreTaskExamples:
    """Validated examples plus the identity of their exact JSONL bytes."""

    examples: tuple[CoreExample, ...]
    identity: str

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[CoreExample]:
        return iter(self.examples)

    def __getitem__(self, index: int) -> CoreExample:
        return self.examples[index]


def load_core_task_examples(
    bundle: CoreBundle,
    task: CoreTask,
) -> CoreTaskExamples:
    """Load and validate one task JSONL member from a validated archive."""

    if not isinstance(bundle, CoreBundle):
        raise TypeError("bundle must be a CoreBundle")
    if task not in bundle.tasks:
        raise CoreBundleError("task does not belong to the supplied CORE bundle")
    try:
        with zipfile.ZipFile(bundle.path) as archive:
            data = archive.read(task.dataset_member)
    except KeyError as error:
        raise CoreBundleError(
            f"CORE bundle is missing task member {task.dataset_member!r}"
        ) from error
    except (OSError, zipfile.BadZipFile) as error:
        raise CoreBundleError(
            f"could not read CORE task {task.label!r}: {error}"
        ) from error
    examples = _parse_task_jsonl(data, task=task)
    return CoreTaskExamples(examples=examples, identity=_bytes_identity(data))


def _parse_task_jsonl(data: bytes, *, task: CoreTask) -> tuple[CoreExample, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise CoreBundleError(f"task {task.label!r} is not UTF-8: {error}") from error
    validator = JsonValueValidator(CoreBundleError)
    examples: list[CoreExample] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise CoreBundleError(
                f"task {task.label!r} line {line_number} must not be blank"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=validator.duplicate_object_hook(
                    label=f"task {task.label!r} line {line_number}"
                ),
            )
        except json.JSONDecodeError as error:
            raise CoreBundleError(
                f"task {task.label!r} line {line_number} is invalid JSON: {error}"
            ) from error
        examples.append(_parse_example(value, task=task, line_number=line_number))
    if not examples:
        raise CoreBundleError(f"task {task.label!r} must contain at least one example")
    return tuple(examples)


def _parse_example(value: object, *, task: CoreTask, line_number: int) -> CoreExample:
    label = f"task {task.label!r} line {line_number}"
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CoreBundleError(f"{label} must be an object with string keys")
    if task.task_type == "multiple_choice":
        choices = _string_tuple(value.get("choices"), label=f"{label}.choices")
        if len(choices) < 2:
            raise CoreBundleError(f"{label} must contain at least two choices")
        return MultipleChoiceExample(
            query=_required_text(value.get("query"), label=f"{label}.query"),
            choices=choices,
            gold=_gold_index(value.get("gold"), size=len(choices), label=label),
        )
    if task.task_type == "schema":
        options = _string_tuple(
            value.get("context_options"),
            label=f"{label}.context_options",
        )
        if len(options) < 2:
            raise CoreBundleError(f"{label} must contain at least two context options")
        return SchemaExample(
            context_options=options,
            continuation=_required_text(
                value.get("continuation"), label=f"{label}.continuation"
            ),
            gold=_gold_index(value.get("gold"), size=len(options), label=label),
        )
    return LanguageModelingExample(
        context=_required_text(value.get("context"), label=f"{label}.context"),
        continuation=_required_text(
            value.get("continuation"), label=f"{label}.continuation"
        ),
    )


def _gold_index(value: object, *, size: int, label: str) -> int:
    try:
        gold = require_integer(value, name=f"{label}.gold")
    except TypeError as error:
        raise CoreBundleError(str(error)) from error
    if not 0 <= gold < size:
        raise CoreBundleError(f"{label}.gold must index one of {size} choices")
    return gold


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CoreBundleError(f"{label} must be a list")
    return tuple(
        _required_text(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise CoreBundleError(f"{label} must be a non-empty string")
    return value


def _bytes_identity(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


__all__ = [
    "CoreExample",
    "CoreTaskExamples",
    "LanguageModelingExample",
    "MultipleChoiceExample",
    "SchemaExample",
    "load_core_task_examples",
]
