"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
    ) -> LlmResponse:
        """Send a chat completion request."""
        ...

    @property
    def api_url(self) -> str:
        """Return the API base URL this client connects to."""
        return ""
