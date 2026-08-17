"""Tests for interactive slash command handling and session helpers."""

from __future__ import annotations

import pytest

from llm_cli_py.models import Message, Role
from llm_cli_py.providers.llm_api import LlmApiClient
from llm_cli_py.session.interactive import _handle_slash_command
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.verifier import Verifier


def _make_session() -> ActiveSession:
    client = LlmApiClient(
        model="gpt-4o",
        api_url="https://api.example.com/v1",
        api_key="key",
    )
    ctx = SessionContext(tool_registry=ToolRegistry(), verifier=Verifier())
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

    def test_clear_command(self) -> None:
        session = _make_session()
        session.client.state.conversation = [Message(role=Role.USER, content="Hi")]
        _handle_slash_command(session, "/clear")
        assert session.client.state.conversation == []

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

    def test_verifier_toggle(self) -> None:
        session = _make_session()
        assert session.ctx.verifier
        assert session.ctx.verifier.enabled is True
        _handle_slash_command(session, "/v off")
        assert session.ctx.verifier.enabled is False
        _handle_slash_command(session, "/verifier on")
        assert session.ctx.verifier.enabled is True
