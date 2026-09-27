"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ClientState, LlmResponse, Message, Role, ToolSchema


class LlmClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, model: str, *, system_prompt: str = "") -> None:
        """Initialize state, seeding ``system_prompt`` as the first message.

        The prompt is supplied by the caller (the CLI composition root reads
        ``LLM_CLI_SYSTEM_PROMPT`` once at startup) rather than read from the
        environment here: configuration is passed in explicitly, so tests and
        other embedders can construct a client without touching ``os.environ``.
        It is intentionally NOT re-read per request: mid-session changes would
        make later turns inconsistent with earlier context. When empty, no
        system message is seeded (no default/date prompt is injected).
        """
        self._state = ClientState(
            model=model,
            system_prompt=system_prompt,
            conversation=([Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []),
        )

    @property
    def state(self) -> ClientState:
        return self._state

    @abstractmethod
    def send(
        self,
        prompt: str,
        tool_schemas: list[ToolSchema],
    ) -> LlmResponse:
        """Send one chat completion request and return the whole response.

        Args:
            prompt: The user's prompt text for this turn, sent verbatim.
            tool_schemas: Tool schemas to advertise.
        """
        ...

    @property
    def api_url(self) -> str:
        """Return the API base URL this client connects to."""
        raise NotImplementedError
