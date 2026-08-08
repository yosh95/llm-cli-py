"""Tests for the base LLM client interface.

Covers display name and abstract method enforcement.
"""

from __future__ import annotations

import pytest

from llm_cli_py.base import LlmClient
from llm_cli_py.models import ClientState, DataSource, LlmResponse, ToolSchema


class _ConcreteClient(LlmClient):
    """Concrete implementation for testing the abstract base class."""

    def send(
        self,
        _data: list[DataSource],
        _tool_schemas: list[ToolSchema],
    ) -> LlmResponse:
        return LlmResponse(text="mocked")


class TestLlmClientBase:
    """Test the abstract base class functionality."""

    def test_initialization(self) -> None:
        client = _ConcreteClient(model="gpt-4o")
        assert client.state.model == "gpt-4o"
        assert client.state.conversation == []

    def test_state_property(self) -> None:
        client = _ConcreteClient(model="claude-3")
        assert isinstance(client.state, ClientState)
        assert client.state.model == "claude-3"

    def test_get_display_name(self) -> None:
        client = _ConcreteClient(model="gpt-4o")
        assert client.get_display_name() == "gpt-4o"

    def test_get_display_name_empty(self) -> None:
        client = _ConcreteClient(model="")
        assert client.get_display_name() == ""

    def test_send_is_abstract(self) -> None:
        """Verify that LlmClient.send is abstract and must be overridden."""
        with pytest.raises(TypeError):
            LlmClient(model="test")  # type: ignore[abstract]
