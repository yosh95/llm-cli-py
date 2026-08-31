"""Tests for the OpenRouter web search server tool integration.

Covers:
- Tool registration (server_tool flag, verbatim schema)
- Request building (server tools emitted verbatim, user tools wrapped)
- Session handling (server tools are skipped client-side, loop continues)
- main.initialize_tools (registered only for OpenRouter API URLs)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_cli_py.main import initialize_tools
from llm_cli_py.models import DataSource, LlmResponse, ToolCall, ToolSchema
from llm_cli_py.providers.llm_api import LlmApiClient
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.web_search import (
    OPENROUTER_WEB_SEARCH_DESCRIPTION,
    OPENROUTER_WEB_SEARCH_SCHEMA,
    OPENROUTER_WEB_SEARCH_TOOL_NAME,
)


class TestWebSearchToolDefinition:
    """The tool definition constants."""

    def test_tool_name(self) -> None:
        assert OPENROUTER_WEB_SEARCH_TOOL_NAME == "openrouter:web_search"

    def test_schema_has_query_required(self) -> None:
        assert OPENROUTER_WEB_SEARCH_SCHEMA["type"] == "object"
        assert "query" in OPENROUTER_WEB_SEARCH_SCHEMA["required"]  # type: ignore[operator]
        props = OPENROUTER_WEB_SEARCH_SCHEMA["properties"]
        assert isinstance(props, dict)
        assert props["query"]["type"] == "string"

    def test_description_nonempty(self) -> None:
        assert OPENROUTER_WEB_SEARCH_DESCRIPTION


class TestInitializeTools:
    """main.initialize_tools registers the server tool only for OpenRouter."""

    def test_openrouter_url_registers_server_tool(self) -> None:
        registry = initialize_tools(api_url="https://openrouter.ai/api/v1")
        assert "execute_python" in registry
        assert OPENROUTER_WEB_SEARCH_TOOL_NAME in registry
        tool = registry.get(OPENROUTER_WEB_SEARCH_TOOL_NAME)
        assert tool is not None
        assert tool.server_tool is True
        assert tool.func is None

    def test_local_url_does_not_register_server_tool(self) -> None:
        registry = initialize_tools(api_url="http://localhost:11434/v1")
        assert "execute_python" in registry
        assert OPENROUTER_WEB_SEARCH_TOOL_NAME not in registry

    def test_none_url_does_not_register_server_tool(self) -> None:
        registry = initialize_tools(api_url=None)
        assert OPENROUTER_WEB_SEARCH_TOOL_NAME not in registry

    def test_schema_is_verbatim_server_tool(self) -> None:
        registry = initialize_tools(api_url="https://openrouter.ai/api/v1")
        tool = registry.get(OPENROUTER_WEB_SEARCH_TOOL_NAME)
        assert tool is not None
        schema = tool.schema
        assert schema.server_tool is True
        assert schema.name == "openrouter:web_search"


class TestRequestBuilding:
    """LlmApiClient._build_request emits server tools verbatim."""

    def _client(self) -> LlmApiClient:
        return LlmApiClient(
            model="openai/gpt-5.2",
            api_url="https://openrouter.ai/api/v1",
            api_key="key",
        )

    def test_server_tool_emitted_verbatim(self) -> None:
        client = self._client()
        server_schema = ToolSchema(
            name="openrouter:web_search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            server_tool=True,
        )
        body = client._build_request([{"role": "user", "content": "hi"}], [server_schema])
        tools = body["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "openrouter:web_search"
        assert "function" not in tools[0]
        assert tools[0]["parameters"]["properties"]["query"]["type"] == "string"

    def test_mixed_server_and_user_tools(self) -> None:
        client = self._client()
        server_schema = ToolSchema(
            name="openrouter:web_search",
            description="Search",
            parameters={"type": "object", "properties": {}},
            server_tool=True,
        )
        user_schema = ToolSchema(
            name="execute_python",
            description="Run Python",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )
        body = client._build_request([{"role": "user", "content": "hi"}], [server_schema, user_schema])
        tools = body["tools"]
        by_type = {t["type"]: t for t in tools}
        assert "openrouter:web_search" in by_type
        assert "function" in by_type
        assert by_type["function"]["function"]["name"] == "execute_python"


class TestSessionServerToolHandling:
    """ActiveSession skips server tools client-side and continues the loop."""

    @pytest.fixture
    def session(self) -> ActiveSession:
        client = LlmApiClient(
            model="openai/gpt-5.2",
            api_url="https://openrouter.ai/api/v1",
            api_key="key",
        )
        ctx = SessionContext(tool_registry=initialize_tools(api_url="https://openrouter.ai/api/v1"))
        return ActiveSession(client, ctx)

    def test_server_tool_call_is_skipped_and_loop_continues(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A server tool call is not executed locally; the next turn answers."""
        tool_call = ToolCall(
            id="call_server_1",
            name=OPENROUTER_WEB_SEARCH_TOOL_NAME,
            arguments={"query": "latest AI news"},
        )
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="server_tool_calls"),
            LlmResponse(text="Here are the latest AI news..."),
        ]
        with patch.object(session.client, "send", side_effect=responses) as mock_send:
            session.process_and_print([DataSource(text="What is new in AI?")])

        captured = capsys.readouterr()
        assert "Here are the latest AI news" in captured.out
        # The server tool was NOT executed locally (no local func).
        assert "Executing tool: openrouter:web_search" not in captured.out
        # The agent loop made two requests: search turn + final answer turn.
        assert mock_send.call_count == 2

    def test_user_tool_still_executed_locally(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """User-defined tools still run locally alongside server tools."""

        def dummy(code: str = "") -> object:  # noqa: ARG001
            from llm_cli_py.tools.types import ExecResult

            return ExecResult(stdout="42")

        session.ctx.tool_registry.register(
            "calculate",
            "Calculate",
            {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            dummy,
        )
        tool_call = ToolCall(id="call_1", name="calculate", arguments={"code": "40+2"})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="The answer is 42"),
        ]
        with patch.object(session.client, "send", side_effect=responses):
            session.process_and_print([DataSource(text="What is 40+2?")])

        captured = capsys.readouterr()
        assert "Executing tool: calculate" in captured.out
        assert "The answer is 42" in captured.out
