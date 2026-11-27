"""Tests for ActiveSession: the request/tool loop and its display."""

from __future__ import annotations

import json
import re
import time
from unittest.mock import patch

import pytest

from llm_cli_py.models import LlmResponse, Message, Role, ToolCall
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult, ToolError
from tests.conftest import completion, make_client, register, tool_call, tool_then_text


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


def test_the_call_s_explanation_is_displayed_after_the_code(session, turn) -> None:
    """The model's reason for the call reaches the transcript, under the code."""
    register(session.ctx.tool_registry, "calc")
    call = ToolCall(
        id="c1",
        name="calc",
        arguments={"explanation": "Add the two numbers.", "code": "40+2"},
    )

    out = turn(tool_then_text(call))

    lines = out.splitlines()
    assert lines.index("  [code] 40+2") < lines.index("  [explanation] Add the two numbers.")


def test_a_slow_tool_reports_how_long_it_ran(session, turn) -> None:
    """The tool's duration is on screen, so a slow tool is not mistaken for a hang."""

    def slow(**_kwargs: object) -> ExecResult:
        time.sleep(0.05)
        return ExecResult(stdout="ok")

    register(session.ctx.tool_registry, "slow", slow)
    out = turn(tool_then_text(ToolCall(id="c1", name="slow", arguments={})))

    assert re.search(r"\[Tool\] slow running\.\.\. \d+\.\ds", out)


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


def test_a_broken_tool_call_rolls_the_whole_turn_back(session) -> None:
    """Neither the broken call nor a healthy sibling may survive in the history.

    An assistant ``tool_calls`` entry with no matching tool result is what some
    OpenAI-compatible APIs reject with HTTP 400 on the next request, so the whole
    turn is discarded -- while the user's own prompt stays, so the request is
    still in the conversation and can be asked again.
    """
    register(session.ctx.tool_registry, "calc")
    response = completion(
        tool_calls=[
            tool_call("c1", "calc", "{}"),
            tool_call("c2", "calc", '{"code": "pri'),  # truncated JSON
        ]
    )

    with patch("llm_cli_py.providers.llm_api.post_json", return_value=response) as post:
        session.process_and_print("Run both")

    assert post.call_count == 1  # the agent loop stopped, it did not retry
    assert [(m.role.value, m.content) for m in session.client.state.conversation] == [
        ("user", "Run both"),
    ]


def test_an_interrupted_tool_call_does_not_leave_a_dangling_request(session) -> None:
    """Ctrl+C during a tool run must not leave assistant tool_calls without results."""

    def interrupted(**_kwargs: object) -> ExecResult:
        raise KeyboardInterrupt

    register(session.ctx.tool_registry, "slow", interrupted)
    response = completion(tool_calls=[tool_call("c1", "slow", "{}")])

    with (
        patch("llm_cli_py.providers.llm_api.post_json", return_value=response),
        pytest.raises(KeyboardInterrupt),
    ):
        session.process_and_print("Run it")

    assert [(m.role.value, m.content) for m in session.client.state.conversation] == [("user", "Run it")]


def test_an_interrupted_tool_run_says_so(session, capsys) -> None:
    """The status line stops at the interrupt, so the transcript says what became of the run."""

    def interrupted(**_kwargs: object) -> ExecResult:
        raise KeyboardInterrupt

    register(session.ctx.tool_registry, "slow", interrupted)
    response = completion(tool_calls=[tool_call("c1", "slow", "{}")])

    with (
        patch("llm_cli_py.providers.llm_api.post_json", return_value=response),
        pytest.raises(KeyboardInterrupt),
    ):
        session.process_and_print("Run it")

    assert "Interrupted" in capsys.readouterr().out


def test_rolling_back_without_an_assistant_reply_does_nothing(session) -> None:
    """``rollback_last_turn`` is a no-op when there is nothing to undo."""
    session.client.state.conversation.append(Message(role=Role.USER, content="hello"))

    session.client.rollback_last_turn()

    assert [(m.role.value, m.content) for m in session.client.state.conversation] == [("user", "hello")]


def test_the_status_line_says_what_a_call_is_waiting_for() -> None:
    """A wait names its condition, so a long call does not read as a hung one.

    ``[Tool] execute_python running...`` on a call that is waiting for a build is
    the same line a stuck process produces; naming the condition is what tells
    the two apart on the one line the terminal shows while it waits.
    """
    from llm_cli_py.session.session import _running_label

    assert _running_label("execute_python", {}) == "[Tool] execute_python running..."
    assert (
        _running_label("execute_python", {"wait": {"until": "exit"}})
        == "[Tool] execute_python waiting for the job to exit..."
    )
    assert (
        _running_label("execute_python", {"wait": {"until": "file", "path": "/tmp/out.bin"}})
        == "[Tool] execute_python waiting for /tmp/out.bin..."
    )
    assert (
        _running_label("execute_python", {"wait": {"job": "build"}})
        == "[Tool] execute_python waiting for job build..."
    )
    assert (
        _running_label("execute_python", {"wait": {"until": "port", "port": 8080}})
        == "[Tool] execute_python waiting for port 8080..."
    )
    # Anything at all can arrive from the model: a label is not worth failing a call over.
    assert _running_label("execute_python", {"wait": 42}) == "[Tool] execute_python running..."


def test_a_waiting_tool_run_does_not_read_as_a_plain_run(capsys) -> None:
    """The label reaches the screen, not just the helper."""
    session = ActiveSession(make_client("gpt-4o"), SessionContext(tool_registry=ToolRegistry()))
    register(session.ctx.tool_registry, "waits", lambda **_: ExecResult(stdout="k"))
    tool = ToolCall(id="c1", name="waits", arguments={"wait": {"until": "exit"}})

    with patch.object(session.client, "send", side_effect=tool_then_text(tool, "ok")):
        session.process_and_print("go")

    out = capsys.readouterr().out
    assert "waiting for the job to exit" in out
