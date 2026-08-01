"""Bounded, deterministic tokenizer evaluation and report contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Final, Literal

from scratch_llm._validation import (
    require_non_negative_integer,
    require_positive_integer,
)
from scratch_llm.data.climbmix import CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX
from scratch_llm.data.loaders import (
    list_parquet_files,
    parquets_iter_batched,
    select_parquet_files,
)
from scratch_llm.tokenization.tokenizer import Tokenizer
from scratch_llm.utils import atomic_write


CorpusKind = Literal["builtin", "climbmix"]
ComparisonStatus = Literal["measured", "skipped", "unavailable"]

TOKENIZER_EVALUATION_FORMAT: Final = "scratch_llm_tokenizer_evaluation"
TOKENIZER_EVALUATION_FORMAT_VERSION: Final = 1
_COMPARISON_NAMES: Final = ("gpt2", "cl100k_base")

_BUILTIN_CORPORA: Final = (
    (
        "news",
        "A coastal town opened a solar-powered library on Tuesday. "
        "Residents can borrow books, tools, and seeds.",
    ),
    (
        "korean",
        "작은 도서관이 오늘 문을 열었습니다.\n"
        "이곳에서는 책과 생활 도구를 함께 빌릴 수 있습니다.",
    ),
    (
        "code",
        "def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:\n"
        "    return {(left, right): 1 for left, right in zip(ids, ids[1:])}",
    ),
    (
        "math",
        "For n ≥ 1, the triangular number is T_n = n(n + 1) / 2. "
        "Therefore 1 + 2 + ··· + n = T_n.",
    ),
    (
        "science",
        "Photosynthesis stores light energy in chemical bonds. "
        "Chlorophyll absorbs photons, and carbon dioxide helps form sugars.",
    ),
)


@dataclass(frozen=True)
class EvaluationCorpus:
    """One immutable text category plus its exact source and bounds."""

    name: str
    kind: CorpusKind
    identifier: str
    documents: tuple[str, ...]
    data_dir: str | None = None
    split: str | None = None
    text_column: str | None = None
    num_train_shards: int | None = None
    selected_shards: tuple[str, ...] = ()
    max_documents: int | None = None
    max_characters: int | None = None
    document_char_cap: int | None = None

    @property
    def text(self) -> str:
        """Return the exact category payload evaluated by tokenizers."""

        return "\n".join(self.documents)

    def source_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible source metadata."""

        source: dict[str, Any] = {
            "character_count": sum(len(document) for document in self.documents),
            "document_count": len(self.documents),
            "identifier": self.identifier,
            "kind": self.kind,
            "utf8_bytes": sum(
                len(document.encode("utf-8")) for document in self.documents
            ),
        }
        if self.kind == "builtin":
            source["limits"] = {
                "document_char_cap": None,
                "max_characters": None,
                "max_documents": None,
            }
            return source
        source.update(
            {
                "data_dir": self.data_dir,
                "limits": {
                    "document_char_cap": self.document_char_cap,
                    "max_characters": self.max_characters,
                    "max_documents": self.max_documents,
                },
                "selected_shard_count": len(self.selected_shards),
                "selected_shards": list(self.selected_shards),
                "selection": {
                    "num_train_shards": self.num_train_shards,
                    "split": self.split,
                    "text_column": self.text_column,
                },
            }
        )
        return source


@dataclass(frozen=True)
class TokenizerComparisonResult:
    """One optional baseline token-count comparison."""

    name: str
    status: ComparisonStatus
    vocab_size: int | None
    tokens: int | None
    relative_token_count_difference: float | None
    detail: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible comparison fields."""

        return {
            "detail": self.detail,
            "relative_token_count_difference": (self.relative_token_count_difference),
            "status": self.status,
            "tokens": self.tokens,
            "vocab_size": self.vocab_size,
        }


@dataclass(frozen=True)
class TokenizerCategoryResult:
    """Compression and round-trip results for one evaluation category."""

    name: str
    source: EvaluationCorpus
    utf8_bytes: int
    tokens: int
    bytes_per_token: float | None
    round_trip: bool
    comparisons: tuple[TokenizerComparisonResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the category's deterministic JSON-compatible fields."""

        return {
            "bytes": self.utf8_bytes,
            "bytes_per_token": self.bytes_per_token,
            "comparisons": {
                comparison.name: comparison.to_dict()
                for comparison in sorted(
                    self.comparisons,
                    key=lambda comparison: comparison.name,
                )
            },
            "name": self.name,
            "round_trip": self.round_trip,
            "source": self.source.source_dict(),
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """One timed encode or decode measurement."""

    seconds: float
    timed_token_count: int

    @property
    def tokens_per_second(self) -> float:
        """Return token IDs processed per measured second."""

        return self.timed_token_count / self.seconds

    def to_dict(self) -> dict[str, int | float]:
        """Return deterministic JSON-compatible timing fields."""

        return {
            "seconds": self.seconds,
            "timed_token_count": self.timed_token_count,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class TokenizerCategoryBenchmark:
    """Encode and decode measurements for one category."""

    name: str
    encode: BenchmarkMeasurement
    decode: BenchmarkMeasurement

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible category timings."""

        return {
            "decode": self.decode.to_dict(),
            "encode": self.encode.to_dict(),
            "name": self.name,
        }


@dataclass(frozen=True)
class TokenizerBenchmarkResult:
    """Warmup protocol plus per-category and aggregate measurements."""

    warmup_iterations: int
    timed_iterations: int
    categories: tuple[TokenizerCategoryBenchmark, ...]

    @property
    def encode(self) -> BenchmarkMeasurement:
        """Return the total timed encode work across all categories."""

        return _aggregate_measurements(
            tuple(category.encode for category in self.categories)
        )

    @property
    def decode(self) -> BenchmarkMeasurement:
        """Return the total timed decode work across all categories."""

        return _aggregate_measurements(
            tuple(category.decode for category in self.categories)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the benchmark's deterministic JSON-compatible fields."""

        return {
            "aggregate": {
                "decode": self.decode.to_dict(),
                "encode": self.encode.to_dict(),
            },
            "categories": [category.to_dict() for category in self.categories],
            "protocol": {
                "clock": "monotonic",
                "denominator": "token IDs processed during timed calls",
                "timed_iterations": self.timed_iterations,
                "warmup_iterations": self.warmup_iterations,
            },
        }


@dataclass(frozen=True)
class TokenizerEvaluationResult:
    """Immutable result shared by reports, the CLI, and later tracking."""

    tokenizer_identity: str
    vocab_size: int
    categories: tuple[TokenizerCategoryResult, ...]
    benchmark: TokenizerBenchmarkResult

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible result contract."""

        total_bytes = sum(category.utf8_bytes for category in self.categories)
        total_tokens = sum(category.tokens for category in self.categories)
        benchmark_by_name = {
            category.name: category for category in self.benchmark.categories
        }
        categories: list[dict[str, Any]] = []
        for category in self.categories:
            category_payload = category.to_dict()
            category_benchmark = benchmark_by_name[category.name]
            category_payload.update(
                {
                    "decode_tokens_per_second": (
                        category_benchmark.decode.tokens_per_second
                    ),
                    "encode_tokens_per_second": (
                        category_benchmark.encode.tokens_per_second
                    ),
                }
            )
            categories.append(category_payload)
        aggregate_comparisons = _aggregate_comparison_results(
            self.categories,
            ours_tokens=total_tokens,
        )
        return {
            "aggregate": {
                "bytes": total_bytes,
                "bytes_per_token": _bytes_per_token(total_bytes, total_tokens),
                "comparisons": {
                    comparison.name: comparison.to_dict()
                    for comparison in sorted(
                        aggregate_comparisons,
                        key=lambda comparison: comparison.name,
                    )
                },
                "decode_tokens_per_second": self.benchmark.decode.tokens_per_second,
                "encode_tokens_per_second": self.benchmark.encode.tokens_per_second,
                "round_trip": all(category.round_trip for category in self.categories),
                "tokens": total_tokens,
            },
            "benchmark": self.benchmark.to_dict(),
            "categories": categories,
            "format": TOKENIZER_EVALUATION_FORMAT,
            "format_version": TOKENIZER_EVALUATION_FORMAT_VERSION,
            "tokenizer": {
                "identity": self.tokenizer_identity,
                "vocab_size": self.vocab_size,
            },
        }


def collect_evaluation_corpora(
    data_dir: str | Path,
    *,
    num_train_shards: int = 1,
    max_documents: int = 32,
    max_characters: int = 100_000,
    document_char_cap: int = 10_000,
    validation_shard_index: int = CLIMBMIX_FINAL_VALIDATION_SHARD_INDEX,
    batch_size: int = 1024,
    text_column: str = "text",
) -> tuple[EvaluationCorpus, ...]:
    """Collect the five fixed fixtures and two bounded ClimbMix categories."""

    num_train_shards = require_positive_integer(
        num_train_shards,
        name="num_train_shards",
    )
    max_documents = require_positive_integer(max_documents, name="max_documents")
    max_characters = require_positive_integer(max_characters, name="max_characters")
    document_char_cap = require_positive_integer(
        document_char_cap,
        name="document_char_cap",
    )
    batch_size = require_positive_integer(batch_size, name="batch_size")
    directory = Path(data_dir)
    discovered = list_parquet_files(directory)

    corpora = [
        EvaluationCorpus(
            name=name,
            kind="builtin",
            identifier=f"scratch-llm:{name}:v1",
            documents=(text,),
        )
        for name, text in _BUILTIN_CORPORA
    ]
    for name, split in (
        ("climbmix-train", "train"),
        ("climbmix-validation", "val"),
    ):
        selected_files = select_parquet_files(
            discovered,
            split,
            num_train_shards=num_train_shards if split == "train" else None,
            validation_shard_index=validation_shard_index,
        )
        documents = _collect_bounded_documents(
            split,
            data_dir=directory,
            num_train_shards=num_train_shards if split == "train" else None,
            validation_shard_index=validation_shard_index,
            batch_size=batch_size,
            text_column=text_column,
            max_documents=max_documents,
            max_characters=max_characters,
            document_char_cap=document_char_cap,
        )
        if not documents:
            raise ValueError(
                f"{name} did not yield any documents under the configured limits"
            )
        corpora.append(
            EvaluationCorpus(
                name=name,
                kind="climbmix",
                identifier=name,
                documents=documents,
                data_dir=str(directory),
                split=split,
                text_column=text_column,
                num_train_shards=num_train_shards if split == "train" else None,
                selected_shards=tuple(path.name for path in selected_files),
                max_documents=max_documents,
                max_characters=max_characters,
                document_char_cap=document_char_cap,
            )
        )
    return tuple(corpora)


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    corpora: Iterable[EvaluationCorpus],
    *,
    compare: bool = False,
    benchmark_warmup_iterations: int = 1,
    benchmark_iterations: int = 3,
    clock: Callable[[], float] = time.monotonic,
) -> TokenizerEvaluationResult:
    """Evaluate compression, optional baselines, round-trip, and throughput."""

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"tokenizer must implement Tokenizer, got {type(tokenizer).__name__}"
        )
    if not isinstance(compare, bool):
        raise TypeError(f"compare must be a boolean, got {type(compare).__name__}")
    if not callable(clock):
        raise TypeError(f"clock must be callable, got {type(clock).__name__}")
    warmup_iterations = require_non_negative_integer(
        benchmark_warmup_iterations,
        name="benchmark_warmup_iterations",
    )
    timed_iterations = require_positive_integer(
        benchmark_iterations,
        name="benchmark_iterations",
    )
    normalized_corpora = _normalize_corpora(corpora)
    comparison_encodings: Mapping[str, Any] | None = None
    comparison_unavailable_detail: str | None = None
    if compare:
        try:
            comparison_encodings = _load_tiktoken_encodings()
        except ModuleNotFoundError:
            comparison_unavailable_detail = (
                "optional dependency 'tiktoken' is unavailable; install the "
                "tokenizer-comparison extra"
            )
        except Exception as error:
            comparison_unavailable_detail = (
                "optional tiktoken comparisons could not be initialized: "
                f"{type(error).__name__}: {error}"
            )

    encoded_inputs: list[tuple[int, ...]] = []
    category_results: list[TokenizerCategoryResult] = []
    for corpus in normalized_corpora:
        encoded = tuple(tokenizer.encode(corpus.text))
        encoded_inputs.append(encoded)
        utf8_bytes = len(corpus.text.encode("utf-8"))
        category_results.append(
            TokenizerCategoryResult(
                name=corpus.name,
                source=corpus,
                utf8_bytes=utf8_bytes,
                tokens=len(encoded),
                bytes_per_token=_bytes_per_token(utf8_bytes, len(encoded)),
                round_trip=tokenizer.decode(encoded) == corpus.text,
                comparisons=_category_comparisons(
                    corpus.text,
                    ours_tokens=len(encoded),
                    compare=compare,
                    encodings=comparison_encodings,
                    unavailable_detail=comparison_unavailable_detail,
                ),
            )
        )

    benchmark = _benchmark_tokenizer(
        tokenizer,
        tuple(corpus.name for corpus in normalized_corpora),
        tuple(corpus.text for corpus in normalized_corpora),
        tuple(encoded_inputs),
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        clock=clock,
    )
    return TokenizerEvaluationResult(
        tokenizer_identity=tokenizer.get_identity(),
        vocab_size=tokenizer.get_vocab_size(),
        categories=tuple(category_results),
        benchmark=benchmark,
    )


def write_tokenizer_evaluation_reports(
    result: TokenizerEvaluationResult,
    metrics_dir: str | Path,
) -> tuple[Path, Path]:
    """Atomically write deterministic JSON and Markdown from one result."""

    if not isinstance(result, TokenizerEvaluationResult):
        raise TypeError(
            f"result must be TokenizerEvaluationResult, got {type(result).__name__}"
        )
    payload = result.to_dict()
    serialized_json = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    serialized_markdown = _render_tokenizer_evaluation_markdown(result)
    directory = Path(metrics_dir)
    json_path = directory / "tokenizer_eval.json"
    markdown_path = directory / "tokenizer_eval.md"
    atomic_write(json_path, f"{serialized_json}\n")
    atomic_write(markdown_path, serialized_markdown)
    return json_path, markdown_path


def _render_tokenizer_evaluation_markdown(
    result: TokenizerEvaluationResult,
) -> str:
    payload = result.to_dict()
    lines = [
        "# Tokenizer evaluation",
        "",
        f"- Tokenizer identity: `{result.tokenizer_identity}`",
        f"- Vocabulary size: {result.vocab_size}",
        "",
        "## Compression and round-trip",
        "",
        (
            "| Category | Bytes | Tokens | Bytes/token | Round trip | "
            "Encode tok/s | Decode tok/s | GPT-2 token Δ | cl100k token Δ |"
        ),
        "| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for category in payload["categories"]:
        comparisons = category["comparisons"]
        lines.append(
            f"| {category['name']} | {category['bytes']} | "
            f"{category['tokens']} | "
            f"{_format_optional_float(category['bytes_per_token'])} | "
            f"{'pass' if category['round_trip'] else 'fail'} | "
            f"{category['encode_tokens_per_second']:.3f} | "
            f"{category['decode_tokens_per_second']:.3f} | "
            f"{_format_comparison(comparisons['gpt2'])} | "
            f"{_format_comparison(comparisons['cl100k_base'])} |"
        )
    aggregate = payload["aggregate"]
    lines.extend(
        [
            (
                f"| **Aggregate** | {aggregate['bytes']} | "
                f"{aggregate['tokens']} | "
                f"{_format_optional_float(aggregate['bytes_per_token'])} | "
                f"{'pass' if aggregate['round_trip'] else 'fail'} | "
                f"{aggregate['encode_tokens_per_second']:.3f} | "
                f"{aggregate['decode_tokens_per_second']:.3f} | — | — |"
            ),
            "",
            (
                "Relative token-count difference uses "
                "`(baseline_tokens - project_tokens) / baseline_tokens`; "
                "positive means the project tokenizer uses fewer tokens."
            ),
            "",
            "## Sources and limits",
            "",
            "| Category | Source | Documents | Selected shards | Limits |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for category in payload["categories"]:
        source = category["source"]
        limits = source["limits"]
        selected_shards = source.get("selected_shards", [])
        limit_text = ", ".join(
            f"{name}={value if value is not None else 'none'}"
            for name, value in sorted(limits.items())
        )
        lines.append(
            f"| {category['name']} | {source['identifier']} "
            f"({source['kind']}) | {source['document_count']} | "
            f"{', '.join(selected_shards) if selected_shards else '—'} | "
            f"{limit_text} |"
        )
    benchmark = payload["benchmark"]
    protocol = benchmark["protocol"]
    lines.extend(
        [
            "",
            "## Throughput benchmark",
            "",
            f"- Warmup iterations: {protocol['warmup_iterations']}",
            f"- Timed iterations: {protocol['timed_iterations']}",
            f"- Clock: {protocol['clock']}",
            f"- Denominator: {protocol['denominator']}",
            (
                f"- Encode: "
                f"{benchmark['aggregate']['encode']['tokens_per_second']:.3f} "
                "tokens/second "
                f"({benchmark['aggregate']['encode']['timed_token_count']} "
                "token IDs in "
                f"{benchmark['aggregate']['encode']['seconds']:.6f} seconds)"
            ),
            (
                f"- Decode: "
                f"{benchmark['aggregate']['decode']['tokens_per_second']:.3f} "
                "tokens/second "
                f"({benchmark['aggregate']['decode']['timed_token_count']} "
                "token IDs in "
                f"{benchmark['aggregate']['decode']['seconds']:.6f} seconds)"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_comparison(comparison: Mapping[str, Any]) -> str:
    if comparison["status"] != "measured":
        return str(comparison["status"])
    relative_difference = comparison["relative_token_count_difference"]
    if relative_difference is None:
        return "n/a"
    return f"{relative_difference:+.1%}"


def _category_comparisons(
    text: str,
    *,
    ours_tokens: int,
    compare: bool,
    encodings: Mapping[str, Any] | None,
    unavailable_detail: str | None,
) -> tuple[TokenizerComparisonResult, ...]:
    if not compare:
        return tuple(
            TokenizerComparisonResult(
                name=name,
                status="skipped",
                vocab_size=None,
                tokens=None,
                relative_token_count_difference=None,
                detail="comparison not requested",
            )
            for name in _COMPARISON_NAMES
        )
    if unavailable_detail is not None:
        return tuple(
            TokenizerComparisonResult(
                name=name,
                status="unavailable",
                vocab_size=None,
                tokens=None,
                relative_token_count_difference=None,
                detail=unavailable_detail,
            )
            for name in _COMPARISON_NAMES
        )
    if encodings is None:
        raise RuntimeError("comparison encodings were not initialized")

    results: list[TokenizerComparisonResult] = []
    for name in _COMPARISON_NAMES:
        encoding = encodings.get(name)
        if encoding is None:
            results.append(
                TokenizerComparisonResult(
                    name=name,
                    status="unavailable",
                    vocab_size=None,
                    tokens=None,
                    relative_token_count_difference=None,
                    detail=f"tiktoken encoding {name!r} is unavailable",
                )
            )
            continue
        try:
            vocab_size = encoding.n_vocab
            baseline_tokens = len(encoding.encode(text))
            if (
                not isinstance(vocab_size, int)
                or isinstance(vocab_size, bool)
                or vocab_size <= 0
            ):
                raise ValueError("n_vocab must be a positive integer")
        except Exception as error:
            results.append(
                TokenizerComparisonResult(
                    name=name,
                    status="unavailable",
                    vocab_size=None,
                    tokens=None,
                    relative_token_count_difference=None,
                    detail=(
                        f"tiktoken encoding {name!r} failed: "
                        f"{type(error).__name__}: {error}"
                    ),
                )
            )
            continue
        relative_difference = (
            (baseline_tokens - ours_tokens) / baseline_tokens
            if baseline_tokens
            else None
        )
        detail = (
            "(baseline_tokens - project_tokens) / baseline_tokens; "
            "positive means the project tokenizer uses fewer tokens"
        )
        if baseline_tokens == 0:
            detail += "; undefined for an empty baseline token sequence"
        results.append(
            TokenizerComparisonResult(
                name=name,
                status="measured",
                vocab_size=vocab_size,
                tokens=baseline_tokens,
                relative_token_count_difference=relative_difference,
                detail=detail,
            )
        )
    return tuple(results)


def _load_tiktoken_encodings() -> Mapping[str, Any]:
    """Lazily import and construct the two opt-in comparison encodings."""

    import tiktoken  # type: ignore[import-not-found,import-untyped]

    return {name: tiktoken.get_encoding(name) for name in _COMPARISON_NAMES}


def _aggregate_comparison_results(
    categories: Sequence[TokenizerCategoryResult],
    *,
    ours_tokens: int,
) -> tuple[TokenizerComparisonResult, ...]:
    aggregate: list[TokenizerComparisonResult] = []
    for name in _COMPARISON_NAMES:
        comparisons = tuple(
            next(
                comparison
                for comparison in category.comparisons
                if comparison.name == name
            )
            for category in categories
        )
        if all(comparison.status == "measured" for comparison in comparisons):
            baseline_tokens = sum(comparison.tokens or 0 for comparison in comparisons)
            vocab_sizes = {comparison.vocab_size for comparison in comparisons}
            if len(vocab_sizes) != 1:
                raise RuntimeError(
                    f"{name} comparison vocabulary changed across categories"
                )
            relative_difference = (
                (baseline_tokens - ours_tokens) / baseline_tokens
                if baseline_tokens
                else None
            )
            detail = comparisons[0].detail
            if baseline_tokens == 0 and detail is not None:
                detail = (
                    detail
                    if "undefined" in detail
                    else detail + "; undefined for an empty baseline token sequence"
                )
            aggregate.append(
                TokenizerComparisonResult(
                    name=name,
                    status="measured",
                    vocab_size=comparisons[0].vocab_size,
                    tokens=baseline_tokens,
                    relative_token_count_difference=relative_difference,
                    detail=detail,
                )
            )
            continue

        statuses = {comparison.status for comparison in comparisons}
        details = {comparison.detail for comparison in comparisons}
        status: ComparisonStatus = (
            comparisons[0].status if len(statuses) == 1 else "unavailable"
        )
        detail = (
            comparisons[0].detail
            if len(details) == 1
            else "comparison was not available for every category"
        )
        aggregate.append(
            TokenizerComparisonResult(
                name=name,
                status=status,
                vocab_size=None,
                tokens=None,
                relative_token_count_difference=None,
                detail=detail,
            )
        )
    return tuple(aggregate)


def _aggregate_measurements(
    measurements: Sequence[BenchmarkMeasurement],
) -> BenchmarkMeasurement:
    if not measurements:
        raise ValueError("benchmark must contain at least one category")
    return BenchmarkMeasurement(
        seconds=sum(measurement.seconds for measurement in measurements),
        timed_token_count=sum(
            measurement.timed_token_count for measurement in measurements
        ),
    )


def _benchmark_tokenizer(
    tokenizer: Tokenizer,
    names: Sequence[str],
    texts: Sequence[str],
    encoded_inputs: Sequence[Sequence[int]],
    *,
    warmup_iterations: int,
    timed_iterations: int,
    clock: Callable[[], float],
) -> TokenizerBenchmarkResult:
    category_benchmarks: list[TokenizerCategoryBenchmark] = []
    for name, text, token_ids in zip(
        names,
        texts,
        encoded_inputs,
        strict=True,
    ):
        for _ in range(warmup_iterations):
            tokenizer.encode(text)
        encode_started = clock()
        encode_token_count = 0
        for _ in range(timed_iterations):
            encode_token_count += len(tokenizer.encode(text))
        encode_seconds = _elapsed_seconds(
            encode_started,
            clock(),
            operation=f"encode category {name!r}",
        )

        for _ in range(warmup_iterations):
            tokenizer.decode(token_ids)
        decode_started = clock()
        decode_token_count = 0
        for _ in range(timed_iterations):
            tokenizer.decode(token_ids)
            decode_token_count += len(token_ids)
        decode_seconds = _elapsed_seconds(
            decode_started,
            clock(),
            operation=f"decode category {name!r}",
        )
        category_benchmarks.append(
            TokenizerCategoryBenchmark(
                name=name,
                encode=BenchmarkMeasurement(
                    seconds=encode_seconds,
                    timed_token_count=encode_token_count,
                ),
                decode=BenchmarkMeasurement(
                    seconds=decode_seconds,
                    timed_token_count=decode_token_count,
                ),
            )
        )
    return TokenizerBenchmarkResult(
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        categories=tuple(category_benchmarks),
    )


def _elapsed_seconds(start: float, end: float, *, operation: str) -> float:
    elapsed = end - start
    if elapsed <= 0:
        raise ValueError(
            f"monotonic clock must advance during {operation} benchmark; "
            f"measured {elapsed:g} seconds"
        )
    return elapsed


def _normalize_corpora(
    corpora: Iterable[EvaluationCorpus],
) -> tuple[EvaluationCorpus, ...]:
    if isinstance(corpora, (str, bytes)):
        raise TypeError("corpora must be an iterable of EvaluationCorpus values")
    try:
        normalized = tuple(corpora)
    except TypeError as error:
        raise TypeError(
            "corpora must be an iterable of EvaluationCorpus values"
        ) from error
    if not normalized:
        raise ValueError("corpora must contain at least one evaluation category")
    for index, corpus in enumerate(normalized):
        if not isinstance(corpus, EvaluationCorpus):
            raise TypeError(
                f"corpus at position {index} must be EvaluationCorpus, "
                f"got {type(corpus).__name__}"
            )
    names = tuple(corpus.name for corpus in normalized)
    if len(set(names)) != len(names):
        raise ValueError("evaluation category names must be unique")
    return normalized


def _bytes_per_token(utf8_bytes: int, tokens: int) -> float | None:
    return utf8_bytes / tokens if tokens else None


def _collect_bounded_documents(
    split: str,
    *,
    data_dir: Path,
    num_train_shards: int | None,
    validation_shard_index: int,
    batch_size: int,
    text_column: str,
    max_documents: int,
    max_characters: int,
    document_char_cap: int,
) -> tuple[str, ...]:
    documents: list[str] = []
    characters = 0
    for batch in parquets_iter_batched(
        split,
        data_dir=data_dir,
        num_train_shards=num_train_shards,
        validation_shard_index=validation_shard_index,
        batch_size=batch_size,
        text_column=text_column,
    ):
        for text in batch:
            if len(documents) >= max_documents or characters >= max_characters:
                return tuple(documents)
            bounded = text[:document_char_cap]
            bounded = bounded[: max_characters - characters]
            documents.append(bounded)
            characters += len(bounded)
    return tuple(documents)


__all__ = [
    "TOKENIZER_EVALUATION_FORMAT",
    "TOKENIZER_EVALUATION_FORMAT_VERSION",
    "BenchmarkMeasurement",
    "EvaluationCorpus",
    "TokenizerBenchmarkResult",
    "TokenizerCategoryBenchmark",
    "TokenizerCategoryResult",
    "TokenizerComparisonResult",
    "TokenizerEvaluationResult",
    "collect_evaluation_corpora",
    "evaluate_tokenizer",
    "write_tokenizer_evaluation_reports",
]
