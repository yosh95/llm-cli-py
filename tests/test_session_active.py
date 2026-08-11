"""Tests for ActiveSession - the core session processing logic.

Covers process_and_print, tool call handling, verifier integration,
error handling, and MAX_TOOL_ITERATIONS enforcement.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from llm_cli_py.consts import MAX_TOOL_ITERATIONS
from llm_cli_py.models import DataSource, LlmResponse, ToolCall
from llm_cli_py.providers.llm_api import LlmApiClient
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult, ToolError
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

    def test_response_with_reasoning_not_displayed(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Reasoning is parsed/kept in the model for multi-turn history, but it
        # is intentionally NOT printed to the terminal. Only the final answer
        # should appear.
        with patch.object(
            session.client,
            "send",
            return_value=LlmResponse(text="Final answer", reasoning="I think..."),
        ):
            session.process_and_print([DataSource(text="Hi")])
        captured = capsys.readouterr()
        assert "Final answer" in captured.out
        assert "I think..." not in captured.out
        assert "Reasoning" not in captured.out

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
            patch.object(session.ctx.backend, "prompt", return_value="y"),
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
            patch.object(session.ctx.backend, "prompt", return_value="n"),
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
            on_reasoning: Callable[[str], None] | None = None,
            on_content: Callable[[str], None] | None = None,
        ) -> tuple[bool, str]:
            if on_reasoning:
                on_reasoning("Thinking about it...")
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
        # Reasoning is no longer streamed/displayed by the session.
        assert "Verifier reasoning:" not in captured.out
        assert "Thinking about it..." not in captured.out
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
            patch.object(session.ctx.backend, "prompt", return_value="This is actually safe because..."),
        ):
            session.process_and_print([DataSource(text="Run code")])

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
        # Check that user feedback was added to conversation
        assert any("User feedback" in m.content for m in session.client.state.conversation)


class TestTokenUsage:
    """Test token_usage property."""

    def test_token_usage_default(self, session: ActiveSession) -> None:
        prompt, completion, total = session.token_usage
        assert prompt == 0
        assert completion == 0
        assert total == 0

    def test_token_usage_after_parse(self, session: ActiveSession) -> None:
        session.client.state.token_usage.prompt_tokens = 100
        session.client.state.token_usage.completion_tokens = 50
        session.client.state.token_usage.total_tokens = 150
        prompt, completion, total = session.token_usage
        assert prompt == 100
        assert completion == 50
        assert total == 150


class TestMaxToolIterations:
    """Test MAX_TOOL_ITERATIONS enforcement."""

    def test_max_iterations_stops_loop(
        self, session: ActiveSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The tool-call loop should stop after max_tool_iterations iterations."""
        # Every response returns a tool call that requires another iteration
        tool_call = ToolCall(id="call_1", name="looper", arguments={})

        def looper_tool(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="loop")

        session.ctx.tool_registry.register(
            "looper",
            "Loop tool",
            {"type": "object", "properties": {}, "required": []},
            looper_tool,
        )

        max_iter = session.ctx.max_tool_iterations
        # Generate enough responses to exceed max_tool_iterations
        responses = [LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls")] * (
            max_iter + 5
        )

        with patch.object(session.client, "send", side_effect=responses) as mock_send:
            session.process_and_print([DataSource(text="Start loop")])
            send_call_count = mock_send.call_count

        captured = capsys.readouterr()
        assert "maximum tool-call iterations" in captured.out.lower()
        # Verify send was called at most max_iter + 1 times
        # (the +1 accounts for the initial request before tool calls start)
        assert send_call_count <= max_iter + 1

    def test_max_iterations_default_value(self) -> None:
        assert MAX_TOOL_ITERATIONS == 500

    def test_max_iterations_custom_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Custom max_tool_iterations should be respected."""
        from llm_cli_py.providers.llm_api import LlmApiClient
        from llm_cli_py.session.session import ActiveSession, SessionContext
        from llm_cli_py.tools.registry import ToolRegistry
        from llm_cli_py.tools.types import ExecResult
        from llm_cli_py.verifier import Verifier

        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        verifier = Verifier()
        verifier.set_enabled(False)
        ctx = SessionContext(
            tool_registry=ToolRegistry(),
            verifier=verifier,
            max_tool_iterations=3,
        )
        custom_session = ActiveSession(client, ctx)

        tool_call = ToolCall(id="call_1", name="looper", arguments={})

        def looper_tool(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="loop")

        custom_session.ctx.tool_registry.register(
            "looper",
            "Loop tool",
            {"type": "object", "properties": {}, "required": []},
            looper_tool,
        )

        responses = [LlmResponse(text=None, tool_calls=[tool_call], finish_reason="tool_calls")] * 10

        with patch.object(custom_session.client, "send", side_effect=responses) as mock_send:
            custom_session.process_and_print([DataSource(text="Start loop")])
            send_call_count = mock_send.call_count

        captured = capsys.readouterr()
        assert "maximum tool-call iterations" in captured.out.lower()
        # With max_tool_iterations=3, send should be called at most 4 times
        # (initial request + 3 tool iterations)
        assert send_call_count <= 4
