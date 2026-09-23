"""Tests for the interactive loop: slash commands, dump and the chat log."""

from __future__ import annotations

import json
import tomllib
from unittest.mock import patch

import pytest

from llm_cli_py.models import Message, Role
from llm_cli_py.session import interactive as interactive_mod
from llm_cli_py.session.interactive import ChatLogWriter
from llm_cli_py.session.session import ActiveSession


@pytest.fixture
def session() -> ActiveSession:
    from llm_cli_py.session.session import SessionContext
    from llm_cli_py.tools.registry import ToolRegistry
    from tests.conftest import make_client, register

    ctx = SessionContext(tool_registry=register(ToolRegistry(), "execute_python"))
    return ActiveSession(make_client("gpt-4o"), ctx)


# ── Slash commands ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/help", "/quit"),
        ("/h", "/quit"),
        ("/notacommand", "Unknown command"),
    ],
)
def test_command_output(session: ActiveSession, capsys, line: str, expected: str) -> None:
    interactive_mod._handle_slash_command(session, line)
    assert expected in capsys.readouterr().out


def test_quit_returns_exit(session: ActiveSession) -> None:
    assert interactive_mod._handle_slash_command(session, "/quit") == "exit"
    assert interactive_mod._handle_slash_command(session, "/exit") == "exit"


def test_info_shows_api_url_model_and_tools(session: ActiveSession, capsys) -> None:
    interactive_mod._handle_slash_command(session, "/info")
    out = capsys.readouterr().out
    assert "API URL" in out and "https://api.example.com/v1" in out
    assert "gpt-4o" in out
    assert "execute_python" in out


def test_dump_emits_parseable_toml_with_timestamps(session: ActiveSession, capsys) -> None:
    session.client.state.conversation = [
        Message(role=Role.USER, content="Hi there", timestamp="2026-09-16T12:00:00+09:00"),
        Message(role=Role.TOOL, content="42", tool_call_id="call_1"),
    ]
    interactive_mod._cmd_dump(session, "")
    parsed = tomllib.loads(capsys.readouterr().out)
    assert [m["role"] for m in parsed["message"]] == ["user", "tool"]
    assert parsed["message"][0]["timestamp"] == "2026-09-16T12:00:00+09:00"
    assert parsed["message"][1]["tool_call_id"] == "call_1"


def test_dump_keeps_assistant_tool_calls_inside_the_message(session: ActiveSession, capsys) -> None:
    """Regression: TOML nested tables detached tool_calls from their message."""
    session.client.state.conversation = [
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            timestamp="2026-09-16T12:00:00+09:00",
        )
    ]
    interactive_mod._cmd_dump(session, "")
    parsed = tomllib.loads(capsys.readouterr().out)

    assert "tool_calls" not in parsed  # no detached top-level table
    calls = json.loads(parsed["message"][0]["tool_calls"])
    assert calls[0]["id"] == "c1"
    assert calls[0]["function"]["name"] == "x"


# ── User input plumbing ───────────────────────────────────────────


def test_user_input_is_forwarded_as_text_source(session: ActiveSession) -> None:
    with patch.object(session, "process_and_print") as mock:
        interactive_mod._handle_user_input(session, "hello")
    args = mock.call_args.args[0]
    assert [(a.text, a.source_type) for a in args] == [("hello", "text")]


def test_user_input_error_is_reported(session: ActiveSession, capsys) -> None:
    with patch.object(session, "process_and_print", side_effect=Exception("boom")):
        interactive_mod._handle_user_input(session, "hello")
    assert "Failed to process input: boom" in capsys.readouterr().out


# ── Session loop ──────────────────────────────────────────────────


def test_eof_exits_silently(session: ActiveSession) -> None:
    with patch.object(interactive_mod, "prompt", side_effect=EOFError()):
        interactive_mod.run_interactive(session)


def test_keyboard_interrupt_then_eof(session: ActiveSession, capsys) -> None:
    with patch.object(interactive_mod, "prompt", side_effect=[KeyboardInterrupt(), EOFError()]):
        interactive_mod.run_interactive(session)
    assert "Use /quit to exit" in capsys.readouterr().out


def test_unexpected_loop_error_keeps_the_session_alive(session: ActiveSession, capsys) -> None:
    with patch.object(interactive_mod, "prompt", side_effect=[RuntimeError("boom"), EOFError()]):
        interactive_mod.run_interactive(session)
    out = capsys.readouterr().out
    assert "Unexpected error: boom" in out
    assert "The session continues" in out


# ── Chat log ──────────────────────────────────────────────────────


def _run_with_prompts(session: ActiveSession, prompts) -> None:
    with patch.object(interactive_mod, "prompt", side_effect=prompts):
        interactive_mod.run_interactive(session)


def test_log_is_flushed_while_the_session_is_still_running(
    session: ActiveSession, tmp_path, monkeypatch
) -> None:
    log_file = tmp_path / "chat.log"
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
    seen: list[str] = []

    def fake_prompt(_text: str) -> str:
        session.client.state.conversation.append(
            Message(role=Role.USER, content="Hi there", timestamp="2026-09-16T12:00:00+09:00")
        )
        session.client.state.notify_changed()
        seen.append(log_file.read_text(encoding="utf-8"))
        raise EOFError

    _run_with_prompts(session, fake_prompt)
    assert "Hi there" in seen[0]  # written mid-session, not only at exit


def test_log_mirrors_the_conversation_at_exit(session: ActiveSession, tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "chat.log"
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
    session.client.state.conversation = [Message(role=Role.USER, content="Bye")]

    _run_with_prompts(session, EOFError())
    assert log_file.read_text(encoding="utf-8") == interactive_mod.render_conversation(
        session.client.state.conversation
    )
    assert not (tmp_path / "chat.log.tmp").exists()  # atomic replace cleans up


def test_append_mode_keeps_previous_sessions(session: ActiveSession, tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "chat.log"
    log_file.write_text('[[message]]\nrole = "user"\ncontent = "previous"\n', encoding="utf-8")
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_APPEND", "1")
    session.client.state.conversation = [Message(role=Role.USER, content="Bye")]

    _run_with_prompts(session, EOFError())
    parsed = tomllib.loads(log_file.read_text(encoding="utf-8"))
    assert [m["content"] for m in parsed["message"]] == ["previous", "Bye"]


def test_log_repairs_a_dropped_broken_tool_call(session: ActiveSession, tmp_path, monkeypatch) -> None:
    """After a broken tool call is dropped, the log must not keep the stale turn."""
    log_file = tmp_path / "chat.log"
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
    session.client.state.conversation = [
        Message(
            role=Role.ASSISTANT,
            content="partial",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{"}}],
        )
    ]

    def fake_prompt(_text: str) -> str:
        from llm_cli_py.models import ToolCall

        session.client.state.notify_changed()
        session._drop_broken_tool_calls_from_history(
            [ToolCall(id="c1", name="x", arguments={}, parse_error="{")]
        )
        raise EOFError

    _run_with_prompts(session, fake_prompt)

    log = log_file.read_text(encoding="utf-8")
    assert "tool_calls" not in log
    assert "partial" in log


def test_no_log_file_is_written_when_unset(session: ActiveSession, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLI_CHAT_LOG_FILE", raising=False)
    session.client.state.conversation = [Message(role=Role.USER, content="Hi")]
    _run_with_prompts(session, EOFError())
    assert list(tmp_path.iterdir()) == []


def test_unwritable_log_warns_but_does_not_break_the_session(
    session: ActiveSession, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(tmp_path))  # a directory
    _run_with_prompts(session, EOFError())
    assert "Failed to write chat log" in capsys.readouterr().out


def test_change_listener_is_detached_after_the_loop(session: ActiveSession, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(tmp_path / "chat.log"))
    _run_with_prompts(session, EOFError())
    assert session.client.state.on_change is None


# ── Append mode rewrite (rollback) ────────────────────────────────


def _tool_call_msg(content: str = "partial") -> Message:
    return Message(
        role=Role.ASSISTANT,
        content=content,
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{"}}],
    )


def test_append_mode_rolls_back_an_in_place_repair(tmp_path) -> None:
    """A repaired (shrunk) message is rewritten, not left stale in the file."""
    path = tmp_path / "chat.log"
    path.write_text('[[message]]\nrole = "user"\ncontent = "previous"\n', encoding="utf-8")
    writer = ChatLogWriter(str(path), append=True)

    message = _tool_call_msg()
    writer.write([message])
    assert "tool_calls" in path.read_text(encoding="utf-8")

    message.tool_calls = None  # simulate _drop_broken_tool_calls_from_history
    writer.write([message])

    text = path.read_text(encoding="utf-8")
    assert "tool_calls" not in text
    assert "partial" in text
    assert "previous" in text  # the earlier session is untouched
    parsed = tomllib.loads(text)
    assert [m["content"] for m in parsed["message"]] == ["previous", "partial"]


def test_append_mode_rolls_back_a_shrinking_history(tmp_path) -> None:
    path = tmp_path / "chat.log"
    writer = ChatLogWriter(str(path), append=True)
    messages = [Message(role=Role.USER, content=f"m{i}") for i in range(3)]

    writer.write(messages)
    writer.write(messages[:1])

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert [m["content"] for m in parsed["message"]] == ["m0"]


def test_append_mode_does_not_duplicate_on_an_unchanged_rewrite(tmp_path) -> None:
    path = tmp_path / "chat.log"
    writer = ChatLogWriter(str(path), append=True)
    messages = [Message(role=Role.USER, content="m0")]

    writer.write(messages)
    before = path.read_text(encoding="utf-8")
    writer.write(messages)
    assert path.read_text(encoding="utf-8") == before


def test_append_mode_appends_only_the_new_tail(tmp_path) -> None:
    path = tmp_path / "chat.log"
    writer = ChatLogWriter(str(path), append=True)
    writer.write([Message(role=Role.USER, content="m0")])
    writer.write([Message(role=Role.USER, content="m0"), Message(role=Role.USER, content="m1")])

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert [m["content"] for m in parsed["message"]] == ["m0", "m1"]


def test_append_mode_log_repairs_a_dropped_broken_tool_call(session, tmp_path, monkeypatch) -> None:
    """The same repair guarantee holds in append mode, end to end."""
    log_file = tmp_path / "chat.log"
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
    monkeypatch.setenv("LLM_CLI_CHAT_LOG_APPEND", "1")
    session.client.state.conversation = [_tool_call_msg()]

    def fake_prompt(_text: str) -> str:
        from llm_cli_py.models import ToolCall

        session.client.state.notify_changed()
        session._drop_broken_tool_calls_from_history(
            [ToolCall(id="c1", name="x", arguments={}, parse_error="{")]
        )
        raise EOFError

    _run_with_prompts(session, fake_prompt)

    log = log_file.read_text(encoding="utf-8")
    assert "tool_calls" not in log
    assert "partial" in log
