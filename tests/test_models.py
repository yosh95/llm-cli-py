"""Tests for data models."""

from llm_cli_py.models import (
    ClientState,
    DataSource,
    LlmResponse,
    Message,
    Role,
    ToolCall,
    ToolSchema,
)


class TestRole:
    """Test Role enum."""

    def test_role_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_is_string(self) -> None:
        assert isinstance(Role.SYSTEM, str)
        assert Role.SYSTEM.value == "system"


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_message_with_tool_info(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content='{"result": "ok"}',
            tool_call_id="call_123",
        )
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_123"


class TestDataSource:
    """Test DataSource dataclass."""

    def test_text_source(self) -> None:
        ds = DataSource(text="hello")
        assert ds.text == "hello"
        assert ds.source_type == "text"

    def test_file_source(self) -> None:
        ds = DataSource(text="file content", source_type="file")
        assert ds.text == "file content"
        assert ds.source_type == "file"


class TestLlmResponse:
    """Test LlmResponse dataclass."""

    def test_empty_response(self) -> None:
        resp = LlmResponse()
        assert resp.text is None
        assert resp.tool_calls == []
        assert resp.finish_reason is None

    def test_text_response(self) -> None:
        resp = LlmResponse(text="Hello!")
        assert resp.text == "Hello!"
        assert resp.tool_calls == []

    def test_tool_call_response(self) -> None:
        tc = ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})
        resp = LlmResponse(
            text=None,
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        assert resp.text is None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "python"
        assert resp.finish_reason == "tool_calls"


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_tool_call_with_explanation(self) -> None:
        tc = ToolCall(
            id="call_1",
            name="execute_python",
            arguments={"code": "print('hi')", "explanation": "Run some Python code"},
        )
        assert tc.id == "call_1"
        assert tc.name == "execute_python"
        assert tc.arguments["explanation"] == "Run some Python code"


class TestMessageTimestamp:
    """Messages carry an optional local timestamp (log/dump only)."""

    def test_timestamp_defaults_to_none(self) -> None:
        assert Message(role=Role.USER, content="Hello").timestamp is None

    def test_timestamp_can_be_set(self) -> None:
        msg = Message(role=Role.USER, content="Hello", timestamp="2026-09-16T12:34:56+09:00")
        assert msg.timestamp == "2026-09-16T12:34:56+09:00"

    def test_timestamps_do_not_affect_equality(self) -> None:
        """Positional/keyword construction stays backwards compatible."""
        assert Message(Role.USER, "Hi") == Message(role=Role.USER, content="Hi")


class TestClientStateNotifyChanged:
    """ClientState notifies a listener so the chat log can be written per message."""

    def test_notify_calls_listener(self) -> None:
        calls: list[int] = []
        state = ClientState(on_change=lambda: calls.append(1))
        state.notify_changed()
        assert calls == [1]

    def test_notify_without_listener_is_noop(self) -> None:
        ClientState().notify_changed()  # must not raise

    def test_listener_errors_are_swallowed(self) -> None:
        def boom() -> None:
            msg = "log write failed"
            raise RuntimeError(msg)

        state = ClientState(on_change=boom)
        state.notify_changed()  # logging must never break a request


class TestClientState:
    """Test ClientState dataclass."""

    def test_default_state(self) -> None:
        state = ClientState()
        assert state.model == ""
        assert state.conversation == []

    def test_custom_state(self) -> None:
        state = ClientState(
            model="gpt-4o",
        )
        assert state.model == "gpt-4o"


class TestToolSchema:
    """Test ToolSchema dataclass."""

    def test_schema_creation(self) -> None:
        schema = ToolSchema(
            name="python",
            description="Execute Python code",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )
        assert schema.name == "python"
        assert "code" in str(schema.parameters)
