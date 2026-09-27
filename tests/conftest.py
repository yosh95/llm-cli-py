"""Shared fixtures for the test suite.

A fake client points at a non-routable URL, so a test that forgets to patch the
transport fails loudly instead of hitting a real provider.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_cli_py.models import LlmResponse, ToolCall
from llm_cli_py.providers.llm_api import LlmApiClient
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.registry import ToolFunc, ToolRegistry
from llm_cli_py.tools.types import ExecResult

TEST_URL = "https://api.example.invalid/v1"


def make_client(
    model: str = "m",
    api_key: str = "k",
    *,
    system_prompt: str = "",
) -> LlmApiClient:
    """Build a client against a fake URL (nothing is sent until ``send``)."""
    return LlmApiClient(model=model, api_url=TEST_URL, api_key=api_key, system_prompt=system_prompt)


def completion(text: str | None = None, tool_calls: list[dict] | None = None) -> dict:
    """An OpenAI-style chat-completion payload (what ``post_json`` returns)."""
    message: dict[str, object] = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def tool_call(call_id: str, name: str, arguments: str) -> dict:
    """One entry of a response's ``tool_calls`` (``arguments`` is JSON text)."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def register(registry: ToolRegistry, name: str, func: ToolFunc | None = None) -> ToolRegistry:
    """Register ``func`` (default: a tool that succeeds) as tool ``name``."""
    if func is None:

        def func(**_kwargs: object) -> ExecResult:
            return ExecResult(stdout="ok")

    registry.register(name, f"{name} tool", {"type": "object", "properties": {}}, func)
    return registry


def tool_then_text(tool: ToolCall, text: str = "Done") -> list[LlmResponse]:
    """Scripted turns: the model asks for ``tool``, then answers with ``text``."""
    return [LlmResponse(text=None, tool_calls=[tool]), LlmResponse(text=text)]


@pytest.fixture
def session() -> ActiveSession:
    """A session whose client only answers when a test patches ``send``."""
    return ActiveSession(make_client("gpt-4o"), SessionContext(tool_registry=ToolRegistry()))


@pytest.fixture
def turn(session: ActiveSession, capsys: pytest.CaptureFixture[str]):
    """Run one ``process_and_print`` against scripted responses; return stdout."""

    def _run(responses: list[LlmResponse], text: str = "hi") -> str:
        with patch.object(session.client, "send", side_effect=responses):
            session.process_and_print(text)
        return capsys.readouterr().out

    return _run
