"""Shared helpers and fixtures for the test suite.

Only helpers that are genuinely reused live here: SSE stream building, a client
factory, a session fixture and a ``turn`` helper that runs one scripted
``process_and_print`` and returns the terminal output.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_cli_py.models import DataSource, LlmResponse, ToolCall
from llm_cli_py.providers.llm_api import LlmApiClient
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult

TEST_URL = "https://api.example.com/v1"


def sse_chunk(delta: dict[str, object], finish_reason: str | None = None) -> str:
    """Build a single SSE data line from a delta dict."""
    return json.dumps({"choices": [{"delta": delta, "finish_reason": finish_reason}]})


def stream_response(chunks: list[str], status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response that yields SSE data lines."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.iter_lines.return_value = [c.encode("utf-8") for c in chunks]
    return resp


def tool_call_chunk(arguments: str, *, name: str = "python", index: int = 0, call_id: str = "call_1") -> str:
    """SSE line carrying one tool-call delta (``arguments`` as raw JSON text)."""
    return sse_chunk(
        {"tool_calls": [{"index": index, "id": call_id, "function": {"name": name, "arguments": arguments}}]}
    )


def text_stream(*texts: str) -> list[str]:
    """SSE lines for a plain text answer followed by a stop + [DONE]."""
    return [sse_chunk({"content": t}) for t in texts] + [sse_chunk({}, finish_reason="stop"), "data: [DONE]"]


def make_client(model: str = "m", api_key: str = "k") -> LlmApiClient:
    return LlmApiClient(model=model, api_url=TEST_URL, api_key=api_key)


def exec_tool(**kwargs: object) -> ExecResult:  # noqa: ARG001
    """A tool that always succeeds (used where the tool body is irrelevant)."""
    return ExecResult(stdout="ok")


def register(registry: ToolRegistry, name: str, func=exec_tool) -> ToolRegistry:
    """Register ``func`` as tool ``name`` and return the registry."""
    registry.register(name, f"{name} tool", {"type": "object", "properties": {}}, func)
    return registry


def tool_then_text(tool: ToolCall, text: str = "Done") -> list[LlmResponse]:
    """Scripted turns: the model asks for ``tool``, then answers with ``text``."""
    return [LlmResponse(text=None, tool_calls=[tool]), LlmResponse(text=text)]


@pytest.fixture
def session() -> ActiveSession:
    """An ActiveSession whose client never reaches the network unless patched."""
    return ActiveSession(make_client("gpt-4o"), SessionContext(tool_registry=ToolRegistry()))


@pytest.fixture
def turn(session: ActiveSession, capsys: pytest.CaptureFixture[str]):
    """Run one ``process_and_print`` against scripted responses; return stdout."""

    def _run(responses: list[LlmResponse], text: str = "hi") -> str:
        with patch.object(session.client, "send", side_effect=responses):
            session.process_and_print([DataSource(text=text)])
        return capsys.readouterr().out

    return _run
