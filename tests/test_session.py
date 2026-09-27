"""Tests for ActiveSession: the request/tool loop and its display."""

from __future__ import annotations

import json
from unittest.mock import patch

from llm_cli_py.models import LlmResponse, Role, ToolCall
from llm_cli_py.tools.types import ExecResult, ToolError
from tests.conftest import register, tool_then_text


def test_a_plain_answer_is_printed(turn) -> None:
    out = turn([LlmResponse(text="Hello back!")], text="Hi")
    assert "gpt-4o is thinking..." in out
    assert "Hello back!" in out


def test_a_failing_request_is_reported_and_the_loop_stops(session, capsys) -> None:
    with patch.object(session.client, "send", side_effect=Exception("API error")) as send:
        session.process_and_print("Hello")

    assert "API error" in capsys.readouterr().out
    assert send.call_count == 1


def test_a_tool_is_executed_automatically_and_its_result_is_sent_back(session) -> None:
    register(session.ctx.tool_registry, "calc", lambda **_: ExecResult(stdout="42"))
    tool = ToolCall(id="c1", name="calc", arguments={"code": "40+2"})

    with patch.object(session.client, "send", side_effect=tool_then_text(tool, "The answer is 42")) as send:
        session.process_and_print("compute")

    # One request for the prompt, one for the tool result.
    assert [c.args[0] for c in send.call_args_list] == ["compute", ""]
    tool_msgs = [m for m in session.client.state.conversation if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert json.loads(tool_msgs[0].content)["stdout"] == "42"


def test_tool_arguments_and_result_are_displayed(session, turn) -> None:
    register(session.ctx.tool_registry, "calc")
    out = turn(tool_then_text(ToolCall(id="c1", name="calc", arguments={"code": "print(1)"})))

    assert "Executing tool: calc" in out
    assert "  [code] print(1)" in out
    assert "Tool Result" in out and "Exit code: 0" in out


def test_unknown_tool_is_reported_to_the_model(session, turn) -> None:
    out = turn(tool_then_text(ToolCall(id="c1", name="ghost", arguments={})))
    assert "not found" in out
    tool_msgs = [m for m in session.client.state.conversation if m.role == Role.TOOL]
    assert tool_msgs and "not found" in tool_msgs[0].content


def test_tool_failure_is_reported_to_the_model(session, turn) -> None:
    register(session.ctx.tool_registry, "flaky", lambda **_: ToolError(error="API limit exceeded"))
    out = turn(tool_then_text(ToolCall(id="c1", name="flaky", arguments={})))

    assert "API limit exceeded" in out
    assert any(m.role == Role.TOOL for m in session.client.state.conversation)


def test_a_broken_tool_call_is_not_executed(session) -> None:
    """A tool call with unparseable arguments ends the turn without running it."""
    broken = ToolCall(id="c1", name="execute_python", arguments={}, parse_error='{"code": "pri')
    with patch.object(session.client, "send", side_effect=[LlmResponse(tool_calls=[broken])]) as send:
        session.process_and_print("Run it")

    assert send.call_count == 1
    assert not any(m.role == Role.TOOL for m in session.client.state.conversation)


def test_broken_tool_calls_are_dropped_from_the_history(session) -> None:
    """The assistant turn must not keep tool_calls that have no tool result."""
    from llm_cli_py.models import Message

    session.client.state.conversation.append(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{"}}],
        )
    )
    session._drop_broken_tool_calls_from_history([ToolCall(id="c1", name="x", arguments={}, parse_error="{")])
    assert session.client.state.conversation[-1].tool_calls is None
