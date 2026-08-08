"""Tests for data models."""

from llm_cli_py.models import (
    ClientState,
    DataSource,
    LlmResponse,
    Message,
    Role,
    TokenUsage,
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
        assert msg.tool_name is None
        assert msg.name is None

    def test_message_with_tool_info(self) -> None:
        msg = Message(
            role=Role.TOOL,
            content='{"result": "ok"}',
            tool_call_id="call_123",
            tool_name="python",
        )
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.tool_name == "python"


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
            name="web_search",
            arguments={"query": "python", "explanation": "Need to search for Python info"},
        )
        assert tc.id == "call_1"
        assert tc.name == "web_search"
        assert tc.arguments["explanation"] == "Need to search for Python info"


class TestTokenUsage:
    """Test TokenUsage dataclass."""

    def test_default_usage(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


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
