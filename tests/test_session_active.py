"""Tests for ActiveSession - the core session processing logic.

Covers process_and_print, tool call handling, verifier integration,
error handling.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from llm_cli_py.consts import APPROVAL_MODE_VERIFIER
from llm_cli_py.models import DataSource, LlmResponse, Message, Role, ToolCall
from llm_cli_py.providers.llm_api import LlmApiClient
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult, SearchResult, SearchResultItem, ToolError
from llm_cli_py.verifier import Verifier


@pytest.fixture
def mock_client() -> LlmApiClient:
    client = LlmApiClient(
        model="gpt-4o",
        api_url="https://api.example.com/v1",
        api_key="key",
    )
    return client


@pytest.fixture
def session(mock_client: LlmApiClient) -> ActiveSession:
    verifier = Verifier()
    verifier.set_enabled(False)
    ctx = SessionContext(tool_registry=ToolRegistry(), verifier=verifier)
    return ActiveSession(mock_client, ctx)


class TestActiveSessionInit:
    """Test ActiveSession initialization."""

    def test_initialization(self, session: ActiveSession) -> None:
        assert session.trace_id is not None
        assert len(session.trace_id) > 0
        assert session.client is not None
        assert session.ctx is not None


class TestProcessAndPrint:
    """Test the main processing loop."""

    def test_no_model_configured(self, session: ActiveSession, capsys: pytest.CaptureFixture[str]) -> None:
        session.client.state.model = ""
        # HTTP request would be attempted; mock the client to keep it fast.
        with patch.object(
            session.client,
            "send",
            return_value=LlmResponse(text="Hello from LLM"),
        ):
            session.process_and_print([DataSource(text="Hello")])
        captured = capsys.readouterr()
        assert "No model specified locally" in captured.out

    def test_verifier_not_configured(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session.ctx.verifier = Verifier()  # not configured
        session.process_and_print([DataSource(text="Hello")])
        captured = capsys.readouterr()
        assert "not configured" in captured.out

    def test_llm_request_failure(self, session: ActiveSession, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(session.client, "send", side_effect=Exception("API error")):
            session.process_and_print([DataSource(text="Hello")])
        captured = capsys.readouterr()
        assert "API error" in captured.out

    def test_simple_text_response(self, session: ActiveSession, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(
            session.client,
            "send",
            return_value=LlmResponse(text="Hello back!"),
        ):
            session.process_and_print([DataSource(text="Hi")])
        captured = capsys.readouterr()
        assert "Hello back!" in captured.out

    def test_tool_call_then_text_response(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Simulate a tool call followed by a text response."""

        # Register a simple tool
        def dummy_tool(code: str = "") -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="42")

        session.ctx.tool_registry.register(
            "calculate",
            "Calculate something",
            {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            dummy_tool,
        )

        # First call returns a tool call, second returns text
        tool_call = ToolCall(id="call_1", name="calculate", arguments={"code": "40+2"})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="The answer is 42"),
        ]

        with patch.object(session.client, "send", side_effect=responses):
            session.process_and_print([DataSource(text="What is 40+2?")])

        captured = capsys.readouterr()
        assert "The answer is 42" in captured.out
        assert "Tool: calculate" in captured.out

    def test_web_search_result_shown_truncated(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Web search results are shown in the terminal, but truncated to a
        single top result."""

        def search_tool(query: str = "") -> SearchResult:  # noqa: ARG001
            return SearchResult(
                query=query,
                results=[
                    SearchResultItem(title="Top hit", url="https://a.com", snippet="Snippet A"),
                    SearchResultItem(title="Second hit", url="https://b.com", snippet="Snippet B"),
                ],
                result_count=2,
            )

        session.ctx.tool_registry.register(
            "web_search",
            "Search the web",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            search_tool,
        )

        tool_call = ToolCall(id="call_1", name="web_search", arguments={"query": "python"})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="Done"),
        ]

        with patch.object(session.client, "send", side_effect=responses):
            session.process_and_print([DataSource(text="Search python")])

        captured = capsys.readouterr()
        # Tool call (arguments) is shown.
        assert "Tool: web_search" in captured.out
        assert "query: python" in captured.out
        # Truncated result is shown: top result only.
        assert "Tool Result:" in captured.out
        assert "Top hit" in captured.out
        assert "Second hit" not in captured.out

    def test_tool_not_found(self, session: ActiveSession, capsys: pytest.CaptureFixture[str]) -> None:
        tool_call = ToolCall(id="call_1", name="nonexistent_tool", arguments={})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="Done", tool_calls=[]),
        ]
        with patch.object(
            session.client,
            "send",
            side_effect=responses,
        ):
            session.process_and_print([DataSource(text="Do something")])

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_broken_tool_call_exits_loop_with_error(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A truncated (unparseable) tool call is NOT executed: error shown and loop exits."""
        broken = ToolCall(id="call_1", name="execute_python", arguments={"raw": '{"code": "pri'})
        # Both turns feed a broken tool call, so no real network request is made.
        # (Leaving the second turn un-mocked would trigger a live HTTP request
        # against api.example.com and block for seconds.)
        with patch.object(
            session.client,
            "send",
            side_effect=[
                LlmResponse(text=None, tool_calls=[broken], finish_reason="tool_calls"),
                LlmResponse(text=None, tool_calls=[broken], finish_reason="tool_calls"),
            ],
        ) as mock_send:
            session.process_and_print([DataSource(text="Run it")])

            captured = capsys.readouterr()
            assert "truncated" in captured.out
            assert "NOT executed" in captured.out

            # We exit the loop without a re-request and without running any tool.
            assert mock_send.call_count == 1

            # Pre-populate history with the same broken assistant tool call (as the
            # real client's _record_assistant would) and confirm it is dropped so
            # the next turn reuses a clean assistant message.
            session.client.state.conversation.append(
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "execute_python", "arguments": '{"raw": "x"}'},
                        }
                    ],
                )
            )
            session.process_and_print([DataSource(text="Run it again")])
            assert session.client.state.conversation[-1].tool_calls is None

    def test_tool_execution_error(self, session: ActiveSession, capsys: pytest.CaptureFixture[str]) -> None:
        def failing_tool(**kwargs: object) -> ExecResult:  # noqa: ARG001
            msg = "Division by zero"
            raise RuntimeError(msg)

        session.ctx.tool_registry.register(
            "failing_tool",
            "Fails",
            {"type": "object", "properties": {}, "required": []},
            failing_tool,
        )

        tool_call = ToolCall(id="call_1", name="failing_tool", arguments={})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="Done", tool_calls=[]),
        ]
        with patch.object(
            session.client,
            "send",
            side_effect=responses,
        ):
            session.process_and_print([DataSource(text="Run it")])

        captured = capsys.readouterr()
        assert "failed" in captured.out

    def test_tool_returns_tool_error(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def error_tool(**kwargs: object) -> ToolError:  # noqa: ARG001
            return ToolError(error="API limit exceeded")

        session.ctx.tool_registry.register(
            "error_tool",
            "Returns error",
            {"type": "object", "properties": {}, "required": []},
            error_tool,
        )

        tool_call = ToolCall(id="call_1", name="error_tool", arguments={})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="Done", tool_calls=[]),
        ]
        with patch.object(
            session.client,
            "send",
            side_effect=responses,
        ):
            session.process_and_print([DataSource(text="Run it")])

        captured = capsys.readouterr()
        assert "API limit exceeded" in captured.out


class TestVerifierIntegration:
    """Test verifier integration in tool call handling."""

    def test_verifier_approves_tool(self, session: ActiveSession, capsys: pytest.CaptureFixture[str]) -> None:
        verifier = MagicMock(spec=Verifier)
        verifier.enabled = True
        verifier.is_configured = True
        verifier.verify.return_value = (True, "Safe operation")
        session.ctx.verifier = verifier

        def safe_tool(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="done")

        session.ctx.tool_registry.register(
            "safe_tool",
            "Safe",
            {"type": "object", "properties": {}, "required": []},
            safe_tool,
        )

        tool_call = ToolCall(id="call_1", name="safe_tool", arguments={})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="All good!", tool_calls=[]),
        ]
        with patch.object(
            session.client,
            "send",
            side_effect=responses,
        ):
            session.process_and_print([DataSource(text="Run safe tool")])

        captured = capsys.readouterr()
        assert "Verifier approved 'safe_tool'." in captured.out
        # Since the verifier approved the call, the tool must be executed.
        assert "Executing tool: safe_tool" in captured.out

    def test_verifier_context_has_only_purpose_and_tool_call(self, session: ActiveSession) -> None:
        """The verifier receives only the tool's purpose + execution content,
        not the full past history."""
        verifier = MagicMock(spec=Verifier)
        verifier.enabled = True
        verifier.is_configured = True
        verifier.verify.return_value = (True, "Safe operation")
        session.ctx.verifier = verifier
        session.ctx.approval_mode = APPROVAL_MODE_VERIFIER

        # Pre-populate older history that the verifier must ignore.
        session.client.state.conversation.append(
            Message(role=Role.USER, content="Earlier unrelated question")
        )
        session.client.state.conversation.append(
            Message(role=Role.ASSISTANT, content="Earlier answer about something")
        )
        # The current assistant message proposing the tool call: content is the
        # tool's purpose, tool_calls holds the execution content.
        session.client.state.conversation.append(
            Message(
                role=Role.ASSISTANT,
                content="I will look up the weather.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "safe_tool",
                            "arguments": '{"query": "current weather"}',
                        },
                    }
                ],
            )
        )

        def safe_tool(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="done")

        session.ctx.tool_registry.register(
            "safe_tool",
            "Safe",
            {"type": "object", "properties": {}, "required": []},
            safe_tool,
        )

        tool_call = ToolCall(
            id="call_1",
            name="safe_tool",
            arguments={"query": "current weather"},
        )
        session._handle_tool_calls([tool_call])

        # Verifier got exactly the tool purpose + the tool call content.
        assert verifier.verify.called
        ctx = verifier.verify.call_args.args[1]
        assert len(ctx) == 2
        # 1st: the assistant's purpose explanation.
        assert ctx[0] == {"role": "assistant", "content": "I will look up the weather."}
        # 2nd: the tool execution content (name + arguments).
        assert "safe_tool" in ctx[1]["content"]
        assert "current weather" in ctx[1]["content"]
        # Older history is NOT included.
        assert not any("Earlier question" in m["content"] for m in ctx)
        assert not any("Earlier answer" in m["content"] for m in ctx)

    def test_verifier_rejects_tool_user_overrides(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        verifier = MagicMock(spec=Verifier)
        verifier.enabled = True
        verifier.is_configured = True
        verifier.verify.return_value = (False, "Potentially dangerous")
        session.ctx.verifier = verifier

        tool_call = ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="Done", tool_calls=[]),
        ]
        with (
            patch.object(
                session.client,
                "send",
                side_effect=responses,
            ),
            patch("llm_cli_py.session.session.prompt", return_value="y"),
        ):
            session.process_and_print([DataSource(text="Run code")])

        captured = capsys.readouterr()
        assert "Executing tool" in captured.out
        assert "User override" in captured.out

    def test_verifier_rejects_user_skips(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        verifier = MagicMock(spec=Verifier)
        verifier.enabled = True
        verifier.is_configured = True
        verifier.verify.return_value = (False, "Writes to disk")
        session.ctx.verifier = verifier

        tool_call = ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="Done", tool_calls=[]),
        ]
        with (
            patch.object(
                session.client,
                "send",
                side_effect=responses,
            ),
            patch("llm_cli_py.session.session.prompt", return_value="n"),
        ):
            session.process_and_print([DataSource(text="Run code")])

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()

    def test_verifier_streams_response(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        verifier = MagicMock(spec=Verifier)
        verifier.enabled = True
        verifier.is_configured = True
        verifier.model = "verifier-model"

        def fake_verify(
            _tc: object,
            _ctx: object,
            on_content: Callable[[str], None] | None = None,
        ) -> tuple[bool, str]:
            if on_content:
                on_content('{"approved": true, "reason": "safe"}')
            return (True, "safe")

        verifier.verify.side_effect = fake_verify
        session.ctx.verifier = verifier

        def safe_tool(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="done")

        session.ctx.tool_registry.register(
            "safe_tool",
            "Safe",
            {"type": "object", "properties": {}, "required": []},
            safe_tool,
        )

        tool_call = ToolCall(id="call_1", name="safe_tool", arguments={})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="All good!", tool_calls=[]),
        ]
        with patch.object(
            session.client,
            "send",
            side_effect=responses,
        ):
            session.process_and_print([DataSource(text="Run safe tool")])

        captured = capsys.readouterr()
        assert "Verifier:" in captured.out
        # The content streaming callback was wired through to verify().
        kwargs = verifier.verify.call_args.kwargs
        assert kwargs["on_content"] is not None

    def test_verifier_rejected_with_feedback(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        verifier = MagicMock(spec=Verifier)
        verifier.enabled = True
        verifier.is_configured = True
        verifier.verify.return_value = (False, "Suspicious")
        session.ctx.verifier = verifier

        tool_call = ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})
        responses = [
            LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LlmResponse(text="Done", tool_calls=[]),
        ]
        with (
            patch.object(
                session.client,
                "send",
                side_effect=responses,
            ),
            patch("llm_cli_py.session.session.prompt", return_value="This is actually safe because..."),
        ):
            session.process_and_print([DataSource(text="Run code")])

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
        # Check that user feedback was added to conversation
        assert any("User feedback" in m.content for m in session.client.state.conversation)
