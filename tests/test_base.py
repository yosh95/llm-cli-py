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


def test_system_prompt_is_snapshotted_with_a_timestamp(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
    assert _ConcreteClient(model="gpt-4o").state.conversation == []

    monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "You are a test assistant.")
    client = _ConcreteClient(model="gpt-4o")
    assert isinstance(client.state, ClientState)
    system_msg = client.state.conversation[0]
    assert system_msg.role.value == "system"
    assert system_msg.timestamp is not None
