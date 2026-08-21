"""Terminal chat adapter and subprocess acceptance tests."""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import scripts.chat as chat_script
from scripts._checkpoint_fixtures import create_tiny_sft_checkpoint
from scratch_llm.chat import TokenEvent, read_conversations
from scratch_llm.config import GenerationConfig
from scratch_llm.tokenization.tokenizer import ByteTokenizer
from tests.test_chat_engine import (
    _PrecisionTransitionModel,
    _assistant_end,
    _assistant_start,
    _engine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _events(text: str) -> Iterator[TokenEvent]:
    yield TokenEvent(
        type="start",
        token_ids=(),
        text_delta="",
        prompt_token_count=5,
        generated_token_count=0,
        sampled_token_count=0,
        elapsed_seconds=0,
        completion_reason=None,
        stop_token_id=None,
    )
    yield TokenEvent(
        type="token",
        token_ids=(65,),
        text_delta=text,
        prompt_token_count=5,
        generated_token_count=1,
        sampled_token_count=1,
        elapsed_seconds=1,
        completion_reason=None,
        stop_token_id=None,
    )
    yield TokenEvent(
        type="complete",
        token_ids=(),
        text_delta="",
        prompt_token_count=5,
        generated_token_count=1,
        sampled_token_count=1,
        elapsed_seconds=1,
        completion_reason="max_new_tokens",
        stop_token_id=None,
    )


class _FakeEngine:
    def __init__(self, responses: tuple[str, ...] = ("answer",)) -> None:
        self.default_generation_config = GenerationConfig(
            temperature=0.8,
            top_k=50,
            max_new_tokens=256,
            seed=None,
        )
        self.responses = iter(responses)
        self.appended: list[str] = []
        self.settings: list[GenerationConfig] = []
        self.resets = 0
        self.saved: list[Path] = []

    def append_user_message(self, text: str) -> None:
        self.appended.append(text)

    def generate_stream(self, settings: GenerationConfig) -> Iterator[TokenEvent]:
        self.settings.append(settings)
        return _events(next(self.responses))

    def reset(self) -> None:
        self.resets += 1

    def save_transcript(self, path: Path) -> Path:
        self.saved.append(path)
        return path


def test_main_resolves_checkpoint_defaults_and_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    monkeypatch.setattr(chat_script, "ChatEngine", lambda *_args, **_kwargs: engine)
    transcript = tmp_path / "chat.jsonl"

    exit_code = chat_script.main(
        [
            "--checkpoint",
            "unused.pt",
            "--device",
            "cpu",
            "--prompt",
            "hello",
            "--temperature",
            "0",
            "--top-k",
            "3",
            "--max-new-tokens",
            "4",
            "--seed",
            "5",
            "--transcript",
            str(transcript),
        ]
    )

    assert exit_code == 0
    assert engine.appended == ["hello"]
    assert engine.settings == [
        GenerationConfig(
            temperature=0,
            top_k=3,
            max_new_tokens=4,
            seed=5,
        )
    ]
    assert engine.saved == [transcript]
    assert capsys.readouterr().out == "answer\n"


def test_interactive_commands_preserve_turns_reset_and_exit_without_generation(
    tmp_path: Path,
) -> None:
    engine = _FakeEngine(("first answer", "second answer"))
    output = StringIO()
    transcript = tmp_path / "chat.jsonl"

    chat_script.run_terminal_chat(
        engine,  # type: ignore[arg-type]
        GenerationConfig(temperature=0),
        prompt=None,
        transcript_path=transcript,
        input_stream=StringIO("first\n/reset\nsecond\n/quit\n"),
        output_stream=output,
    )

    assert engine.appended == ["first", "second"]
    assert engine.resets == 1
    assert len(engine.settings) == 2
    assert engine.saved == [transcript, transcript]
    assert output.getvalue() == (
        "user> assistant> first answer\n"
        "user> Conversation reset.\n"
        "user> assistant> second answer\n"
        "user> "
    )


@pytest.mark.parametrize("command", ["/exit", "/quit"])
def test_exit_commands_and_eof_never_generate(command: str) -> None:
    engine = _FakeEngine()
    output = StringIO()

    chat_script.run_terminal_chat(
        engine,  # type: ignore[arg-type]
        GenerationConfig(),
        prompt=None,
        input_stream=StringIO(f"{command}\n"),
        output_stream=output,
    )
    chat_script.run_terminal_chat(
        engine,  # type: ignore[arg-type]
        GenerationConfig(),
        prompt=None,
        input_stream=StringIO(""),
        output_stream=output,
    )

    assert engine.appended == []
    assert engine.settings == []
    assert engine.saved == []


def test_keyboard_interrupt_and_generation_failure_close_cleanly_without_save(
    tmp_path: Path,
) -> None:
    class _InterruptingInput:
        def readline(self) -> str:
            raise KeyboardInterrupt

    engine = _FakeEngine()
    output = StringIO()
    chat_script.run_terminal_chat(
        engine,  # type: ignore[arg-type]
        GenerationConfig(),
        prompt=None,
        input_stream=_InterruptingInput(),  # type: ignore[arg-type]
        output_stream=output,
    )
    assert engine.appended == []

    closed = False

    class _FailingEngine(_FakeEngine):
        def generate_stream(
            self,
            settings: GenerationConfig,
        ) -> Iterator[TokenEvent]:
            def fail() -> Iterator[TokenEvent]:
                nonlocal closed
                try:
                    yield next(_events("partial"))
                    raise RuntimeError("generation failed")
                finally:
                    closed = True

            return fail()

    failing = _FailingEngine()
    with pytest.raises(RuntimeError, match="generation failed"):
        chat_script.run_terminal_chat(
            failing,  # type: ignore[arg-type]
            GenerationConfig(),
            prompt="hello",
            transcript_path=tmp_path / "must-not-exist.jsonl",
            output_stream=StringIO(),
        )

    assert closed is True
    assert failing.saved == []
    assert not (tmp_path / "must-not-exist.jsonl").exists()


def test_invalid_generation_override_is_actionable_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = _FakeEngine()
    monkeypatch.setattr(chat_script, "ChatEngine", lambda *_args, **_kwargs: engine)

    with pytest.raises(SystemExit) as raised:
        chat_script.main(
            [
                "--checkpoint",
                "unused.pt",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "0",
            ]
        )

    assert raised.value.code == 2
    error_output = capsys.readouterr().err
    assert "generation.max_new_tokens" in error_output
    assert "Traceback" not in error_output


def test_one_shot_subprocess_uses_real_engine_and_writes_parseable_transcript(
    tmp_path: Path,
) -> None:
    checkpoint = create_tiny_sft_checkpoint(tmp_path / "sft.pt")
    transcript = tmp_path / "chat.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.chat",
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "Hi",
            "--transcript",
            str(transcript),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "\x00\x00\n"
    assert "Traceback" not in result.stderr
    conversations = read_conversations(transcript)
    assert len(conversations) == 1
    assert [
        (message.role, message.content) for message in conversations[0].messages
    ] == [
        ("user", "Hi"),
        ("assistant", "\x00\x00"),
    ]


def test_terminal_main_uses_shared_checkpoint_precision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokenizer = ByteTokenizer()
    model = _PrecisionTransitionModel(
        {
            _assistant_start(tokenizer): ord("A"),
            ord("A"): _assistant_end(tokenizer),
        },
        expected_autocast_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        chat_script,
        "ChatEngine",
        lambda *_args, **_kwargs: _engine(model, dtype="bfloat16")[0],
    )

    assert (
        chat_script.main(
            [
                "--checkpoint",
                "unused.pt",
                "--prompt",
                "hello",
                "--temperature",
                "0",
                "--max-new-tokens",
                "2",
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == "A\n"
    assert model.precision_observations == [
        (True, torch.bfloat16),
        (True, torch.bfloat16),
    ]


def test_tracking_config_logs_only_opted_in_prompt_and_warns_locally(
    tmp_path: Path,
) -> None:
    checkpoint = create_tiny_sft_checkpoint(tmp_path / "sft.pt")
    run_name = "tracked-terminal-chat"
    output_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.chat",
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "PROMPT_COMMAND_SECRET",
            "--config",
            str(PROJECT_ROOT / "configs" / "smoke.yaml"),
            "--override",
            f"run.output_dir={output_dir}",
            "--override",
            f"run.name={run_name}",
            "--override",
            "tracking.wandb.log_prompts=true",
            "--override",
            "tracking.wandb.log_responses=false",
            "--no-wandb",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Privacy warning" in result.stderr
    metrics_path = output_dir / run_name / "metrics" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    chat_record = next(
        record["metrics"]
        for record in records
        if record.get("record_type") == "metrics"
    )
    assert chat_record["chat/prompt"] == "PROMPT_COMMAND_SECRET"
    assert "chat/response" not in chat_record
    assert chat_record["chat/transport"] == "cli"
    assert chat_record["chat/generated_tokens"] == 2
    summary = (output_dir / run_name / "metrics" / "summary.json").read_text()
    assert "PROMPT_COMMAND_SECRET" in summary
    assert "chat/response" not in summary
