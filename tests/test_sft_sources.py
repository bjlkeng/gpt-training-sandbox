"""SFT dataset adapter and deterministic source-view contracts."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from scratch_llm.chat.conversation import (
    AssistantMessage,
    Conversation,
    PythonOutputPart,
    PythonPart,
    SystemMessage,
    TextPart,
    UserMessage,
    parse_conversation,
)
from scratch_llm.data.sft_sources import (
    NANOCHAT_SFT_REFERENCE_COMMIT,
    SFTConversationDataset,
    SFTDatasetError,
    SFTDatasetRowError,
    get_sft_dataset_spec,
    normalize_gsm8k_row,
    normalize_mmlu_row,
    normalize_smoltalk_row,
    parse_gsm8k_answer_parts,
)
from scratch_llm.data.hub import publish_local_parquet_cache
from scratch_llm.chat.rendering import render_conversation
from scratch_llm.tokenization.tokenizer import ByteTokenizer


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)


def _smoltalk_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ]
        }
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("dataset", "split", "repository", "subset"),
    [
        ("smoltalk", "train", "HuggingFaceTB/smol-smoltalk", "default"),
        ("smoltalk", "test", "HuggingFaceTB/smol-smoltalk", "default"),
        ("mmlu", "auxiliary_train", "cais/mmlu", "all"),
        ("mmlu", "test", "cais/mmlu", "all"),
        ("gsm8k", "train", "openai/gsm8k", "main"),
        ("gsm8k", "test", "openai/gsm8k", "main"),
    ],
)
def test_dataset_specs_pin_repository_subset_and_split(
    dataset: str,
    split: str,
    repository: str,
    subset: str,
) -> None:
    spec = get_sft_dataset_spec(dataset, split)

    assert spec.dataset == dataset
    assert spec.repository == repository
    assert spec.subset == subset
    assert spec.split == split
    assert spec.reference_commit == NANOCHAT_SFT_REFERENCE_COMMIT
    assert spec.source_identity.startswith("sha256:")
    assert spec.cache_key.endswith(spec.source_identity.removeprefix("sha256:")[:12])


@pytest.mark.parametrize(
    ("dataset", "split"),
    [("unknown", "train"), ("smoltalk", "validation"), ("mmlu", "train")],
)
def test_dataset_specs_reject_unknown_contracts(dataset: str, split: str) -> None:
    with pytest.raises(SFTDatasetError, match="supported"):
        get_sft_dataset_spec(dataset, split)


def test_smoltalk_preserves_exact_messages_and_optional_system() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Unicode?"},
            {"role": "assistant", "content": "Café ☕"},
        ],
        "ignored_source_column": "unchanged",
    }

    conversation = normalize_smoltalk_row(row, context="smoltalk train row 7")

    assert conversation == Conversation(
        messages=(
            SystemMessage(content="Be concise."),
            UserMessage(content="Hello"),
            AssistantMessage(content="Hi"),
            UserMessage(content="Unicode?"),
            AssistantMessage(content="Café ☕"),
        )
    )
    assert row["messages"][1] == {"role": "user", "content": "Hello"}  # type: ignore[index]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "missing assistant"}],
        [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ],
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
        ],
    ],
)
def test_smoltalk_reports_source_context_for_invalid_rows(
    messages: list[dict[str, object]],
) -> None:
    with pytest.raises(
        SFTDatasetRowError,
        match=r"HuggingFaceTB/smol-smoltalk/default/train row 3",
    ):
        normalize_smoltalk_row(
            {"messages": messages},
            context="HuggingFaceTB/smol-smoltalk/default/train row 3",
        )


def test_mmlu_uses_exact_small_model_choice_binding_format() -> None:
    conversation = normalize_mmlu_row(
        {
            "question": "What color is a clear daytime sky?",
            "choices": ["Blue", "Green", "Red", "Orange"],
            "answer": 0,
            "subject": "common_sense",
        },
        context="cais/mmlu/all/auxiliary_train row 2",
    )

    assert conversation.messages == (
        UserMessage(
            content=(
                "Multiple Choice question: What color is a clear daytime sky?\n"
                "- Blue=A\n"
                "- Green=B\n"
                "- Red=C\n"
                "- Orange=D\n"
                "\nRespond only with the letter of the correct answer."
            )
        ),
        AssistantMessage(content="A"),
    )
    assert "= A" not in conversation.messages[0].content


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (
            {"question": "Q", "choices": ["a", "b", "c"], "answer": 0},
            "exactly four choices",
        ),
        (
            {
                "question": "Q",
                "choices": ["a", "b", "c", "d"],
                "answer": 4,
            },
            "answer must be an integer in",
        ),
        (
            {
                "question": "Q",
                "choices": ["a", 2, "c", "d"],
                "answer": 0,
            },
            r"choices\[1\] must be a string",
        ),
    ],
)
def test_mmlu_rejects_invalid_rows_with_context(
    row: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(
        SFTDatasetRowError,
        match=rf"mmlu test row 9: .*{match}",
    ):
        normalize_mmlu_row(row, context="mmlu test row 9")


def test_gsm8k_preserves_text_and_splits_repeated_calculator_calls() -> None:
    answer = "Start <<12/60=0.2>> then <<-1,200+200=-1,000>>.\n#### -1,000"

    parts = parse_gsm8k_answer_parts(answer, context="gsm8k train row 4")
    conversation = normalize_gsm8k_row(
        {"question": "Compute.", "answer": answer},
        context="gsm8k train row 4",
    )

    assert parts == (
        TextPart(text="Start "),
        PythonPart(text="12/60"),
        PythonOutputPart(text="0.2"),
        TextPart(text=" then "),
        PythonPart(text="-1,200+200"),
        PythonOutputPart(text="-1,000"),
        TextPart(text=".\n#### -1,000"),
    )
    assert conversation == Conversation(
        messages=(
            UserMessage(content="Compute."),
            AssistantMessage(content=parts),
        )
    )


def test_gsm8k_keeps_empty_text_fragments_at_tool_boundaries() -> None:
    assert parse_gsm8k_answer_parts(
        "<<1+1=2>><<2+2=4>>",
        context="gsm8k row 0",
    ) == (
        TextPart(text=""),
        PythonPart(text="1+1"),
        PythonOutputPart(text="2"),
        TextPart(text=""),
        PythonPart(text="2+2"),
        PythonOutputPart(text="4"),
        TextPart(text=""),
    )


@pytest.mark.parametrize(
    "answer",
    [
        "broken <<1+1=2",
        "broken 1+1=2>>",
        "missing equals <<1+1>>",
        "missing expression <<=2>>",
        "missing result <<1+1=>>",
        "nested <<1+<<1=2>>",
    ],
)
def test_gsm8k_malformed_markers_fail_explicitly(answer: str) -> None:
    with pytest.raises(
        SFTDatasetRowError,
        match=r"gsm8k test row 6: malformed calculator marker",
    ):
        parse_gsm8k_answer_parts(answer, context="gsm8k test row 6")


def test_gsm8k_adapter_composes_with_renderer_tool_mask() -> None:
    conversation = normalize_gsm8k_row(
        {"question": "Two plus three?", "answer": "Use <<2+3=5>>. #### 5"},
        context="gsm8k train row 1",
    )
    parse_conversation(conversation)

    rendered = render_conversation(conversation, ByteTokenizer())
    decoded = ByteTokenizer().decode(rendered.token_ids)
    python_start = rendered.token_ids.index(
        ByteTokenizer().encode_special("<|python_start|>")
    )
    output_start = rendered.token_ids.index(
        ByteTokenizer().encode_special("<|output_start|>")
    )

    assert "<|python_start|>2+3<|python_end|>" in decoded
    assert "<|output_start|>5<|output_end|>" in decoded
    assert all(rendered.loss_mask[python_start : python_start + 5])
    assert not any(rendered.loss_mask[output_start : output_start + 3])


def test_parquet_source_has_stable_identity_count_seeded_order_and_slices(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "smoltalk.parquet"
    _write_rows(parquet, _smoltalk_rows(12))
    spec = get_sft_dataset_spec("smoltalk", "train")
    cache = publish_local_parquet_cache(spec, tmp_path / "cache", (parquet,))
    source = SFTConversationDataset(cache, shuffle_buffer_size=4)
    reloaded = SFTConversationDataset(
        publish_local_parquet_cache(spec, tmp_path / "cache", (parquet,)),
        shuffle_buffer_size=4,
    )

    first = tuple(source.iter_examples(seed=17))
    again = tuple(reloaded.iter_examples(seed=17))
    other = tuple(source.iter_examples(seed=18))
    sliced = tuple(source.iter_examples(seed=17, start=1, stop=8, step=3))

    assert len(source) == 12
    assert source.source_identity == reloaded.source_identity
    assert [example.identity for example in first] == [
        example.identity for example in again
    ]
    assert [example.identity for example in first] != [
        example.identity for example in other
    ]
    assert [example.identity for example in sliced] == [
        first[index].identity for index in (1, 4, 7)
    ]
    assert all(example.source_identity == source.source_identity for example in first)
    assert sorted(example.source_row for example in first) == list(range(12))


def test_parquet_source_stops_without_materializing_all_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet = tmp_path / "smoltalk.parquet"
    _write_rows(parquet, _smoltalk_rows(20))
    spec = get_sft_dataset_spec("smoltalk", "train")
    cache = publish_local_parquet_cache(spec, tmp_path / "cache", (parquet,))
    source = SFTConversationDataset(cache, shuffle_buffer_size=2)
    seen: list[int] = []
    original = source._iter_indexed_rows

    def observed_rows():
        for row_index, row in original():
            seen.append(row_index)
            if len(seen) > 5:
                raise AssertionError("source eagerly consumed the complete parquet")
            yield row_index, row

    monkeypatch.setattr(source, "_iter_indexed_rows", observed_rows)

    examples = tuple(source.iter_examples(seed=5, stop=2))

    assert len(examples) == 2
    assert len(seen) <= 4


@pytest.mark.parametrize(
    ("start", "stop", "step", "match"),
    [
        (-1, None, 1, "start"),
        (0, -1, 1, "stop"),
        (2, 1, 1, "stop"),
        (0, None, 0, "step"),
    ],
)
def test_parquet_source_validates_bounded_views(
    tmp_path: Path,
    start: int,
    stop: int | None,
    step: int,
    match: str,
) -> None:
    parquet = tmp_path / "smoltalk.parquet"
    _write_rows(parquet, _smoltalk_rows(2))
    spec = get_sft_dataset_spec("smoltalk", "train")
    cache = publish_local_parquet_cache(spec, tmp_path / "cache", (parquet,))
    source = SFTConversationDataset(cache)

    with pytest.raises((TypeError, ValueError), match=match):
        tuple(source.iter_examples(start=start, stop=stop, step=step))
