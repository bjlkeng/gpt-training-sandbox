"""Chat with a supervised-finetuned checkpoint in the terminal."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import TextIO

from scratch_llm.chat import (
    ChatEngine,
    ChatEngineError,
    ChatEventTracker,
    ChatTrackingSession,
    TokenEvent,
    close_token_stream,
)
from scratch_llm.config import GenerationConfig
from scripts._common import (
    add_generation_arguments,
    add_optional_chat_tracking_arguments,
    checkpoint_parser,
    optional_chat_tracking,
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
    add_optional_chat_tracking_arguments(parser)
    return parser


def _generate_turn(
    engine: ChatEngine,
    settings: GenerationConfig,
    user_text: str,
    *,
    transcript_path: Path | None,
    output_stream: TextIO,
    interactive: bool,
    tracking_session: ChatTrackingSession | None,
) -> None:
    engine.append_user_message(user_text)
    if interactive:
        output_stream.write("assistant> ")
        output_stream.flush()
    events = engine.generate_stream(settings)
    completion: TokenEvent | None = None
    try:
        for event in events:
            if event.text_delta:
                output_stream.write(event.text_delta)
                output_stream.flush()
            if event.type == "complete":
                completion = event
    finally:
        close_token_stream(events)
    if completion is None:
        raise ChatEngineError("chat generation ended without a completion event")
    output_stream.write("\n")
    output_stream.flush()
    if transcript_path is not None:
        engine.save_transcript(transcript_path)
    if tracking_session is not None:
        tracking_session.record_completed_turn(
            completion,
            prompt_factory=lambda: engine.get_last_completed_message("user"),
            response_factory=lambda: engine.get_last_completed_message("assistant"),
        )


def run_terminal_chat(
    engine: ChatEngine,
    settings: GenerationConfig,
    *,
    prompt: str | None,
    transcript_path: Path | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    chat_tracking: ChatEventTracker | None = None,
) -> None:
    """Drive one-shot or interactive terminal I/O through ``ChatEngine`` only."""

    active_input = sys.stdin if input_stream is None else input_stream
    active_output = sys.stdout if output_stream is None else output_stream
    tracking_session = (
        None if chat_tracking is None else chat_tracking.start_session("cli")
    )
    try:
        if prompt is not None:
            _generate_turn(
                engine,
                settings,
                prompt,
                transcript_path=transcript_path,
                output_stream=active_output,
                interactive=False,
                tracking_session=tracking_session,
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
                if tracking_session is not None:
                    tracking_session.reset()
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
                tracking_session=tracking_session,
            )
    except KeyboardInterrupt:
        active_output.write("\n")
        active_output.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the terminal chat command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        with optional_chat_tracking(
            parser,
            arguments,
            command=COMMAND,
        ) as chat_tracking:
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
                chat_tracking=chat_tracking,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
