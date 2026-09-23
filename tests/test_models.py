"""Tests for data models."""

from llm_cli_py.models import (
    ClientState,
    DataSource,
    LlmResponse,
    Message,
    Role,
    ToolCall,
    ToolCallPayload,
)


def test_role_values() -> None:
    assert [r.value for r in Role] == ["system", "user", "assistant", "tool"]


def test_message_defaults_and_timestamp() -> None:
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.tool_call_id is None
    assert msg.tool_calls is None
    assert msg.timestamp is None
    assert Message(Role.USER, "Hi", timestamp="2026-09-16T12:34:56+09:00").timestamp


def test_tool_call_payload_has_the_openai_shape() -> None:
    payload = ToolCallPayload(
        id="call_1",
        type="function",
        function={"name": "python", "arguments": '{"code": "print(1)"}'},
    )
    assert payload["function"]["arguments"] == '{"code": "print(1)"}'


def test_data_source_defaults_to_text() -> None:
    assert DataSource(text="hello").source_type == "text"
    assert DataSource(text="body", source_type="url").source_type == "url"


def test_llm_response_defaults() -> None:
    resp = LlmResponse()
    assert resp.text is None
    assert resp.tool_calls == []


def test_tool_call_carries_parse_error_separately_from_arguments() -> None:
    """Truncated arguments are reported via ``parse_error``, not inside ``arguments``."""
    good = ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})
    assert good.parse_error is None
    broken = ToolCall(id="call_1", name="python", arguments={}, parse_error='{"')
    assert broken.parse_error == '{"' and broken.arguments == {}


def test_client_state_notifies_listener_but_swallows_errors() -> None:
    calls: list[str] = []
    state = ClientState(on_change=lambda: calls.append("x"))
    state.notify_changed()
    assert calls == ["x"]

    def boom() -> None:
        msg = "log write failed"
        raise RuntimeError(msg)

    ClientState(on_change=boom).notify_changed()  # must not raise
