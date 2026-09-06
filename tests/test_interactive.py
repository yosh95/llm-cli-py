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
