"""Tests for ActiveSession: tool execution, streaming display and error paths."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from llm_cli_py.models import DataSource, LlmResponse, Message, Role, ToolCall
from llm_cli_py.session.session import ActiveSession
from llm_cli_py.tools.types import ExecResult, ToolError
from tests.conftest import register, tool_then_text


def test_request_failure_is_reported(session: ActiveSession, capsys) -> None:
    with patch.object(session.client, "send", side_effect=Exception("API error")):
        session.process_and_print([DataSource(text="Hello")])
    assert "API error" in capsys.readouterr().out


def test_no_model_is_reported_before_the_first_turn(session: ActiveSession, turn) -> None:
    session.client.state.model = ""
    out = turn([LlmResponse(text="Hello from LLM")], text="Hello")
    assert "No model specified locally" in out


def test_thinking_line_names_the_model(turn) -> None:
    out = turn([LlmResponse(text="Hi")])
    assert "gpt-4o is thinking..." in out


def test_plain_answer_is_printed_when_no_delta_streamed(turn) -> None:
    """Providers that send no deltas still get their answer displayed once."""
    out = turn([LlmResponse(text="Hello back!")], text="Hi")
    assert "Hello back!" in out


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        (ToolCall(id="c1", name="calc", arguments={"code": "40+2"}), "Executing tool: calc"),
        (ToolCall(id="c1", name="missing_tool", arguments={}), "not found"),
    ],
)
def test_tool_dispatch(session: ActiveSession, turn, tool: ToolCall, expected: str) -> None:
    """A registered tool runs; an unknown tool name is reported as an error."""
    if tool.name == "calc":
        register(session.ctx.tool_registry, "calc")
    out = turn(tool_then_text(tool, "The answer is 42"))
    assert expected in out
    assert "The answer is 42" in out


def test_tool_arguments_and_result_are_displayed(session: ActiveSession, turn) -> None:
    register(session.ctx.tool_registry, "calc")
    out = turn(tool_then_text(ToolCall(id="c1", name="calc", arguments={"code": "print(1)"})))
    assert "Executing tool: calc" in out
    assert "  [code] print(1)" in out  # rendered under its Args label
    assert "Exit code: 0" in out  # the ExecResult is rendered
    assert "Tool Result" in out


def test_unknown_tool_is_recorded_as_a_tool_message(session: ActiveSession, turn) -> None:
    turn(tool_then_text(ToolCall(id="c1", name="ghost", arguments={})))
    tool_msgs = [m for m in session.client.state.conversation if m.role == Role.TOOL]
    assert tool_msgs and "not found" in tool_msgs[0].content


@pytest.mark.parametrize(
    ("tool_func", "expected"),
    [
        (lambda **_: (_ for _ in ()).throw(RuntimeError("Division by zero")), "failed"),
        (lambda **_: ToolError(error="API limit exceeded"), "API limit exceeded"),
    ],
)
def test_tool_failures_are_surfaced_and_recorded(
    session: ActiveSession, turn, tool_func, expected: str
) -> None:
    register(session.ctx.tool_registry, "flaky", tool_func)
    out = turn(tool_then_text(ToolCall(id="c1", name="flaky", arguments={})))
    assert expected in out
    # The failure is recorded as a tool message so the model can react to it.
    assert any(m.role == Role.TOOL for m in session.client.state.conversation)


def test_tool_returning_a_bare_string_is_reported_as_an_invalid_result(session: ActiveSession, turn) -> None:
    """A tool must return ExecResult/ToolError; anything else is a clear error."""
    register(session.ctx.tool_registry, "bad", lambda **_: "error happened")
    out = turn(tool_then_text(ToolCall(id="c1", name="bad", arguments={})))
    assert "invalid result type" in out
    tool_msgs = [m for m in session.client.state.conversation if m.role == Role.TOOL]
    assert json.loads(tool_msgs[0].content)["error"].startswith("Tool returned an invalid result type")


def test_tool_result_is_recorded_in_history(session: ActiveSession, turn) -> None:
    register(session.ctx.tool_registry, "calc", lambda **_: ExecResult(stdout="42"))
    turn(tool_then_text(ToolCall(id="c1", name="calc", arguments={})))
    tool_msgs = [m for m in session.client.state.conversation if m.role == Role.TOOL]
    assert len(tool_msgs) == 1
    assert json.loads(tool_msgs[0].content)["stdout"] == "42"
    assert tool_msgs[0].tool_call_id == "c1"


def test_broken_tool_call_is_not_executed(session: ActiveSession) -> None:
    """A truncated tool call exits the loop without a re-request or execution."""
    broken = ToolCall(id="c1", name="execute_python", arguments={}, parse_error='{"code": "pri')
    with patch.object(
        session.client, "send", side_effect=[LlmResponse(text=None, tool_calls=[broken])]
    ) as mock_send:
        session.process_and_print([DataSource(text="Run it")])
    assert mock_send.call_count == 1


def test_broken_tool_calls_are_dropped_from_history(session: ActiveSession) -> None:
    session.client.state.conversation.append(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{"}}],
        )
    )
    session._drop_broken_tool_calls_from_history([ToolCall(id="c1", name="x", arguments={}, parse_error="{")])
    assert session.client.state.conversation[-1].tool_calls is None


def test_dropping_broken_tool_calls_refreshes_the_persisted_log(session: ActiveSession) -> None:
    """The repair must be flushed so the log matches the in-memory history."""
    flushes: list[int] = []
    session.client.state.on_change = lambda: flushes.append(len(session.client.state.conversation))
    session.client.state.conversation.append(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{"}}],
        )
    )
    session._drop_broken_tool_calls_from_history([ToolCall(id="c1", name="x", arguments={}, parse_error="{")])
    assert flushes == [1]


def test_tool_arguments_formatting() -> None:
    assert ActiveSession._format_tool_arguments({}) is None
    assert ActiveSession._format_tool_arguments({"a": 1, "b": "x"}) == ["  [a] 1", "  [b] x"]


def test_multiline_code_argument_keeps_its_indentation() -> None:
    """Multi-line code is shown verbatim on its own lines, not inlined."""
    lines = ActiveSession._format_tool_arguments({"code": "def run(cmd):\n    print(cmd)\nrun(1)"})
    assert lines == ["  [code]", "    def run(cmd):", "        print(cmd)", "    run(1)"]


def test_trailing_newline_does_not_add_an_empty_display_line() -> None:
    """Code written with a trailing newline ends on its last real line."""
    assert ActiveSession._format_tool_arguments({"code": "print(1)\n"}) == ["  [code] print(1)"]
    lines = ActiveSession._format_tool_arguments({"code": "def run():\n    print(1)\n"})
    assert lines == ["  [code]", "    def run():", "        print(1)"]


def test_multiline_tool_arguments_are_printed_verbatim(session: ActiveSession, turn) -> None:
    register(session.ctx.tool_registry, "run")
    code = "import subprocess\n\ndef run(cmd):\n    subprocess.run(cmd, shell=True)"
    out = turn(tool_then_text(ToolCall(id="c1", name="run", arguments={"code": code})))
    assert "    subprocess.run(cmd, shell=True)" in out
    assert "code=import subprocess" not in out  # not flattened onto one line
    assert out.index("Args:") < out.index("[code]")  # header opens the argument block


def test_auto_execution_has_no_approval_prompt(session: ActiveSession, turn) -> None:
    register(session.ctx.tool_registry, "safe")
    out = turn(tool_then_text(ToolCall(id="c1", name="safe", arguments={}), "All good!"))
    assert "Executing tool: safe" in out
    assert "Approve" not in out
