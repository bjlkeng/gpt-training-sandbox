"""Architecture guardrails shared by terminal and web chat adapters."""

from __future__ import annotations

import ast
import inspect

import scripts.chat as terminal_adapter
import scratch_llm.chat as shared_chat
import scratch_llm.web.service as web_adapter


def _called_attributes(module: object) -> list[str]:
    tree = ast.parse(inspect.getsource(module))
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


def test_terminal_and_web_adapters_share_engine_and_token_event_contracts() -> None:
    assert terminal_adapter.ChatEngine is shared_chat.ChatEngine
    assert web_adapter.ChatEngine is shared_chat.ChatEngine
    assert web_adapter.TokenEvent is shared_chat.TokenEvent


def test_adapters_delegate_generation_without_a_second_sampling_loop() -> None:
    for adapter in (terminal_adapter, web_adapter):
        calls = _called_attributes(adapter)
        assert "generate_stream" in calls
        assert "forward" not in calls
        assert "multinomial" not in calls
        assert "stream_generate_sequence" not in calls
