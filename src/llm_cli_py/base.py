"""Base LLM client interface."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable

from .models import ClientState, DataSource, LlmResponse, Message, Role, ToolSchema


class LlmClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, model: str) -> None:
        """Initialize state, reading the system prompt once at startup.

        The ``LLM_CLI_SYSTEM_PROMPT`` environment variable is snapshotted here and
        seeded as the first message of the conversation. It is intentionally
        NOT re-read per request: mid-session changes would make later turns
        inconsistent with earlier context. When unset or empty, no system
        message is seeded (no default/date prompt is injected).
        """
        system_prompt = os.environ.get("LLM_CLI_SYSTEM_PROMPT", "")
        self._state = ClientState(
            model=model,
            system_prompt=system_prompt,
            conversation=[Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else [],
        )

    @property
    def state(self) -> ClientState:
        return self._state

    @abstractmethod
    def send(
        self,
        data: list[DataSource],
        tool_schemas: list[ToolSchema],
        on_text: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        """Send a chat completion request (streaming).

        Args:
            data: User input sources for this turn.
            tool_schemas: Tool schemas to advertise.
            on_text: Optional callback invoked with each text delta.
        """
        ...

    @property
    def api_url(self) -> str:
        """Return the API base URL this client connects to."""
        return ""
