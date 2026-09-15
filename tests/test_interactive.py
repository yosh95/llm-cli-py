"""Tests for interactive slash command handling and session helpers."""

from __future__ import annotations

import pytest

from llm_cli_py.models import Message, Role
from llm_cli_py.providers.llm_api import LlmApiClient
from llm_cli_py.session.interactive import _handle_slash_command
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.registry import ToolRegistry


def _make_session() -> ActiveSession:
    client = LlmApiClient(
        model="gpt-4o",
        api_url="https://api.example.com/v1",
        api_key="key",
    )
    ctx = SessionContext(tool_registry=ToolRegistry())
    return ActiveSession(client, ctx)


class TestSlashCommands:
    """Test slash command dispatch."""

    def test_help_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()
        result = _handle_slash_command(session, "/help")
        captured = capsys.readouterr()
        assert result == ""
        assert "/quit" in captured.out

    def test_quit_command(self) -> None:
        session = _make_session()
        result = _handle_slash_command(session, "/quit")
        assert result == "exit"

    def test_info_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()
        _handle_slash_command(session, "/info")
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out
        assert "https://api.example.com/v1" in captured.out

    def test_info_command_shows_api_url(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify /info displays the configured API URL."""
        session = _make_session()
        _handle_slash_command(session, "/info")
        captured = capsys.readouterr()
        assert "API URL" in captured.out
        assert "https://api.example.com/v1" in captured.out

    def test_unknown_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()
        _handle_slash_command(session, "/notacommand")
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out


class TestDumpCommand:
    """The /dump slash command emits the conversation as TOML."""

    def test_dump_includes_timestamps_and_parses(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The dumped TOML carries timestamps/tool_call_id and round-trips."""
        import tomllib

        session = _make_session()
        session.client.state.conversation = [
            Message(role=Role.USER, content="Hi there", timestamp="2026-09-16T12:00:00+09:00"),
            Message(
                role=Role.TOOL, content="42", tool_call_id="call_1", timestamp="2026-09-16T12:00:05+09:00"
            ),
        ]
        from llm_cli_py.session.interactive import _cmd_dump

        _cmd_dump(session, "")
        captured = capsys.readouterr()
        parsed = tomllib.loads(captured.out)
        assert [m["role"] for m in parsed["message"]] == ["user", "tool"]
        assert parsed["message"][0]["timestamp"] == "2026-09-16T12:00:00+09:00"
        assert parsed["message"][1]["tool_call_id"] == "call_1"

    def test_dump_emits_toml(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = _make_session()
        session.client.state.conversation = [
            Message(role=Role.USER, content="Hi there"),
            Message(role=Role.ASSISTANT, content="Hello"),
        ]
        from llm_cli_py.session.interactive import _cmd_dump

        result = _cmd_dump(session, "")
        assert result is None
        captured = capsys.readouterr()
        assert 'role = "user"' in captured.out
        assert 'role = "assistant"' in captured.out
        assert "Hi there" in captured.out


class TestInteractiveUserInput:
    """The interactive input handler forwards text as a DataSource."""

    def test_handle_user_input_builds_text_source(self) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import _handle_user_input

        session = _make_session()
        with patch.object(session, "process_and_print") as mock:
            _handle_user_input(session, "hello")
            mock.assert_called_once()
            args = mock.call_args.args[0]
            assert len(args) == 1
            assert args[0].text == "hello"
            assert args[0].source_type == "text"

    def test_handle_user_input_surfaces_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import _handle_user_input

        session = _make_session()
        with patch.object(session, "process_and_print", side_effect=Exception("boom")):
            _handle_user_input(session, "hello")
        captured = capsys.readouterr()
        assert "Failed to process input: boom" in captured.out


class TestRunInteractive:
    """The interactive loop terminates cleanly on EOF / Ctrl-C."""

    def test_eof_exits_loop(self) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        session = _make_session()
        with patch("llm_cli_py.session.interactive.prompt", side_effect=EOFError()):
            run_interactive(session)  # must not raise

    def test_keyboard_interrupt_then_eof(self, capsys: pytest.CaptureFixture[str]) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        session = _make_session()
        with patch(
            "llm_cli_py.session.interactive.prompt",
            side_effect=[KeyboardInterrupt(), EOFError()],
        ):
            run_interactive(session)
        captured = capsys.readouterr()
        assert "Use /quit to exit" in captured.out


class TestChatLogIncrementalWrite:
    """The chat log is flushed as messages are added, not only at exit."""

    def test_log_contains_message_while_session_is_still_running(self, tmp_path, monkeypatch) -> None:
        """A user message is already on disk before the loop even ends."""
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        log_file = tmp_path / "chat.log"
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
        monkeypatch.delenv("LLM_CLI_CHAT_LOG_APPEND", raising=False)

        session = _make_session()
        observed: list[str] = []

        def fake_prompt(_text: str) -> str:
            # Mimic what the client does for a user turn, then read the log from
            # inside the still-running session.
            session.client.state.conversation.append(
                Message(role=Role.USER, content="Hi there", timestamp="2026-09-16T12:00:00+09:00")
            )
            session.client.state.notify_changed()
            observed.append(log_file.read_text(encoding="utf-8"))
            raise EOFError

        with patch("llm_cli_py.session.interactive.prompt", side_effect=fake_prompt):
            run_interactive(session)

        assert observed, "the loop must have run at least one prompt"
        assert "Hi there" in observed[0]  # written mid-session, not only at exit
        assert "2026-09-16T12:00:00+09:00" in observed[0]
        assert "Hi there" in log_file.read_text(encoding="utf-8")

    def test_empty_session_still_creates_the_log_file(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        log_file = tmp_path / "chat.log"
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
        # No system prompt: the conversation really is empty.
        monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)

        session = _make_session()
        with patch("llm_cli_py.session.interactive.prompt", side_effect=EOFError()):
            run_interactive(session)

        assert log_file.exists()
        assert log_file.read_text(encoding="utf-8") == ""

    def test_snapshot_write_leaves_no_temp_file(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        log_file = tmp_path / "chat.log"
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))

        session = _make_session()
        session.client.state.conversation = [Message(role=Role.USER, content="Hi")]
        with patch("llm_cli_py.session.interactive.prompt", side_effect=EOFError()):
            run_interactive(session)

        assert not (tmp_path / "chat.log.tmp").exists()

    def test_append_mode_keeps_previous_sessions(self, tmp_path, monkeypatch) -> None:
        """LLM_CLI_CHAT_LOG_APPEND=1 only appends new messages to the file."""
        import tomllib
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        log_file = tmp_path / "chat.log"
        log_file.write_text('[[message]]\nrole = "user"\ncontent = "previous"\n', encoding="utf-8")
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_APPEND", "1")

        session = _make_session()
        session.client.state.conversation = [Message(role=Role.USER, content="Bye")]
        with patch("llm_cli_py.session.interactive.prompt", side_effect=EOFError()):
            run_interactive(session)

        text = log_file.read_text(encoding="utf-8")
        assert text.count("[[message]]") == 2
        parsed = tomllib.loads(text)
        assert [m["content"] for m in parsed["message"]] == ["previous", "Bye"]

    def test_unwritable_log_does_not_break_the_session(self, tmp_path, monkeypatch, capsys) -> None:
        """A failing log write warns once and the chat keeps working."""
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        # A directory can never be opened as a file.
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(tmp_path))

        session = _make_session()
        with patch("llm_cli_py.session.interactive.prompt", side_effect=EOFError()):
            run_interactive(session)  # must not raise

        captured = capsys.readouterr()
        assert "Failed to write chat log" in captured.out


class TestChatLogSave:
    """The interactive loop saves the conversation before exiting when a log file
    is configured, and skips saving when it is not."""

    def test_saves_on_eof_when_env_set(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import _dump_toml, run_interactive

        log_file = tmp_path / "chat.log"
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))

        session = _make_session()
        session.client.state.conversation = [
            Message(role=Role.USER, content="Hi there"),
            Message(role=Role.ASSISTANT, content="Hello"),
        ]

        with patch("llm_cli_py.session.interactive.prompt", side_effect=EOFError()):
            run_interactive(session)

        assert log_file.exists()
        assert log_file.read_text(encoding="utf-8") == _dump_toml(session)

    def test_saves_on_quit_when_env_set(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        log_file = tmp_path / "chat.log"
        monkeypatch.setenv("LLM_CLI_CHAT_LOG_FILE", str(log_file))

        session = _make_session()
        session.client.state.conversation = [Message(role=Role.USER, content="Bye")]

        with patch("llm_cli_py.session.interactive.prompt", side_effect=["/quit"]):
            run_interactive(session)

        assert log_file.exists()
        assert 'content = "Bye"' in log_file.read_text(encoding="utf-8")

    def test_does_not_save_when_env_unset(self, tmp_path, monkeypatch) -> None:
        from unittest.mock import patch

        from llm_cli_py.session.interactive import run_interactive

        monkeypatch.delenv("LLM_CLI_CHAT_LOG_FILE", raising=False)

        session = _make_session()
        session.client.state.conversation = [Message(role=Role.USER, content="Hi")]

        candidate = tmp_path / "should_not_exist.log"

        with patch("llm_cli_py.session.interactive.prompt", side_effect=EOFError()):
            run_interactive(session)

        assert not candidate.exists()
        assert list(tmp_path.iterdir()) == []
