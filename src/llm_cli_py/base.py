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
        system message is seeded (no default/date prompt is injected), and the
        seeded message is the only place the prompt is kept.
        """
        self._state = ClientState(
            model=model,
            conversation=([Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []),
        )

    @property
    def state(self) -> ClientState:
        return self._state

    def rollback_last_turn(self) -> None:
        """Discard the assistant reply recorded for the current turn.

        Used when a turn cannot be finished -- a tool call whose arguments did
        not parse, or a tool run the user interrupted -- because what was
        recorded is then an assistant message whose ``tool_calls`` have no
        results, the one shape some OpenAI-compatible APIs reject with HTTP 400
        on the next request. The user's own prompt is kept: the question stays in
        the conversation and can simply be asked again. A no-op when no assistant
        message has been recorded (nothing to undo).
        """
        conversation = self._state.conversation
        for index in range(len(conversation) - 1, -1, -1):
            if conversation[index].role is Role.ASSISTANT:
                del conversation[index:]
                return

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
