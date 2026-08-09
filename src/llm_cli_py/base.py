"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from .models import ClientState, DataSource, LlmResponse, ToolSchema


class LlmClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, model: str) -> None:
        self._state = ClientState(model=model)

    @property
    def state(self) -> ClientState:
        return self._state

    @abstractmethod
    def send(
        self,
        data: list[DataSource],
        tool_schemas: list[ToolSchema],
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        """Send a chat completion request.

        Args:
            data: User input sources for this turn.
            tool_schemas: Tool schemas to advertise.
            stream: When True, stream tokens via ``on_text`` / ``on_reasoning``.
            on_text: Optional callback invoked with each text delta.
            on_reasoning: Optional callback invoked with each reasoning delta.
        """
        ...

    @property
    def api_url(self) -> str:
        """Return the API base URL this client connects to."""
        return ""
