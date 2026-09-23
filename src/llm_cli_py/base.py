"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from .models import ClientState, DataSource, LlmResponse, Message, Role, ToolSchema
from .utils.timeutil import now_iso


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
        # Timestamps are local metadata (chat log / dump only); they are never
        # sent to the API -- see LlmApiClient._build_messages.
        self._state = ClientState(
            model=model,
            system_prompt=system_prompt,
            conversation=(
                [Message(role=Role.SYSTEM, content=system_prompt, timestamp=now_iso())]
                if system_prompt
                else []
            ),
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
        raise NotImplementedError
