"""Tests for the base LLM client interface.

Covers display name and abstract method enforcement.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from llm_cli_py.base import LlmClient
from llm_cli_py.models import ClientState, DataSource, LlmResponse, ToolSchema


class _ConcreteClient(LlmClient):
    """Concrete implementation for testing the abstract base class."""

    def send(
        self,
        _data: list[DataSource],
        _tool_schemas: list[ToolSchema],
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        # Parameters exist to satisfy the abstract base signature.
        _ = (stream, on_text)
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

    def test_send_is_abstract(self) -> None:
        """Verify that LlmClient.send is abstract and must be overridden."""
        with pytest.raises(TypeError):
            LlmClient(model="test")
