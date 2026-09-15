"""Tests for the base LLM client interface.

Covers display name and abstract method enforcement.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from llm_cli_py.base import LlmClient
from llm_cli_py.models import ClientState, DataSource, LlmResponse, ToolSchema


class _ConcreteClient(LlmClient):
    """Concrete implementation for testing the abstract base class."""

    def send(
        self,
        _data: list[DataSource],
        _tool_schemas: list[ToolSchema],
        on_text: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        # Parameters exist to satisfy the abstract base signature.
        _ = on_text
        return LlmResponse(text="mocked")


class TestLlmClientBase:
    """Test the abstract base class functionality."""

    def test_initialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
        client = _ConcreteClient(model="gpt-4o")
        assert client.state.model == "gpt-4o"
        assert client.state.conversation == []

    def test_state_property(self) -> None:
        client = _ConcreteClient(model="claude-3")
        assert isinstance(client.state, ClientState)
        assert client.state.model == "claude-3"

    def test_system_message_is_timestamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The seeded system message carries a local timestamp for the log."""
        monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "You are a test assistant.")
        client = _ConcreteClient(model="gpt-4o")
        system_msg = client.state.conversation[0]
        assert system_msg.role.value == "system"
        assert system_msg.timestamp is not None
        datetime.fromisoformat(system_msg.timestamp)

    def test_send_is_abstract(self) -> None:
        """Verify that LlmClient.send is abstract and must be overridden."""
        with pytest.raises(TypeError):
            LlmClient(model="test")
