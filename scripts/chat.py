"""Chat with a supervised-finetuned checkpoint in the terminal."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
import sys
from typing import TextIO

from scratch_llm.chat import ChatEngine, ChatEngineError, TokenEvent
from scratch_llm.config import GenerationConfig
from scripts._common import (
    add_generation_arguments,
    checkpoint_parser,
    resolve_generation_arguments,
)


COMMAND = "chat"


def build_parser() -> argparse.ArgumentParser:
    """Return the terminal chat command parser."""

    parser = checkpoint_parser(COMMAND, "Chat with a model in the terminal.")
    parser.add_argument(
        "-p",
        "--prompt",
        help="Run one prompt non-interactively instead of opening a chat loop.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Atomically save completed conversation history as one JSONL record.",
    )
    add_generation_arguments(parser)
    return parser


def _close_iterator(events: Iterator[TokenEvent]) -> None:
    close = getattr(events, "close", None)
    if callable(close):
        close()


def _generate_turn(
    engine: ChatEngine,
    settings: GenerationConfig,
    user_text: str,
    *,
    transcript_path: Path | None,
    output_stream: TextIO,
    interactive: bool,
) -> None:
    engine.append_user_message(user_text)
    if interactive:
        output_stream.write("assistant> ")
        output_stream.flush()
    events = engine.generate_stream(settings)
    completed = False
    try:
        for event in events:
            if event.text_delta:
                output_stream.write(event.text_delta)
                output_stream.flush()
            completed = completed or event.type == "complete"
    finally:
        _close_iterator(events)
    if not completed:
        raise ChatEngineError("chat generation ended without a completion event")
    output_stream.write("\n")
    output_stream.flush()
    if transcript_path is not None:
        engine.save_transcript(transcript_path)


def run_terminal_chat(
    engine: ChatEngine,
    settings: GenerationConfig,
    *,
    prompt: str | None,
    transcript_path: Path | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Drive one-shot or interactive terminal I/O through ``ChatEngine`` only."""

    active_input = sys.stdin if input_stream is None else input_stream
    active_output = sys.stdout if output_stream is None else output_stream
    try:
        if prompt is not None:
            _generate_turn(
                engine,
                settings,
                prompt,
                transcript_path=transcript_path,
                output_stream=active_output,
                interactive=False,
            )
            return

        while True:
            active_output.write("user> ")
            active_output.flush()
            line = active_input.readline()
            if line == "":
                return
            user_text = line.rstrip("\r\n")
            command = user_text.strip().lower()
            if command in {"/exit", "/quit"}:
                return
            if command == "/reset":
                engine.reset()
                active_output.write("Conversation reset.\n")
                active_output.flush()
                continue
            _generate_turn(
                engine,
                settings,
                user_text,
                transcript_path=transcript_path,
                output_stream=active_output,
                interactive=True,
            )
    except KeyboardInterrupt:
        active_output.write("\n")
        active_output.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the terminal chat command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        engine = ChatEngine(arguments.checkpoint, device=arguments.device)
        settings = resolve_generation_arguments(
            engine.default_generation_config,
            arguments,
        )
        run_terminal_chat(
            engine,
            settings,
            prompt=arguments.prompt,
            transcript_path=arguments.transcript,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
