"""Tests for the base LLM client interface."""

from collections.abc import Callable

import pytest

from llm_cli_py.base import LlmClient
from llm_cli_py.models import ClientState, DataSource, LlmResponse, ToolSchema


class _ConcreteClient(LlmClient):
    def send(
        self,
        _data: list[DataSource],
        _tool_schemas: list[ToolSchema],
        on_text: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        _ = on_text
        return LlmResponse(text="mocked")


def test_send_is_abstract() -> None:
    with pytest.raises(TypeError):
        LlmClient(model="test")  # type: ignore[abstract]


def test_no_system_message_without_a_prompt() -> None:
    assert _ConcreteClient(model="gpt-4o").state.conversation == []


def test_system_prompt_is_seeded_with_a_timestamp() -> None:
    client = _ConcreteClient(model="gpt-4o", system_prompt="You are a test assistant.")
    assert isinstance(client.state, ClientState)
    assert client.state.system_prompt == "You are a test assistant."
    system_msg = client.state.conversation[0]
    assert system_msg.role.value == "system"
    assert system_msg.timestamp is not None


def test_system_prompt_is_passed_in_not_read_from_the_environment(monkeypatch) -> None:
    """Configuration is injected, so a client is unaffected by the process env."""
    monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "from the environment")
    assert _ConcreteClient(model="gpt-4o").state.conversation == []
