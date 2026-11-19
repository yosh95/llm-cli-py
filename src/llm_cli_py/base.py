"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol

from .models import ClientState, LlmResponse, Message, Role, ToolSchema
from .utils.clock import Timer


class ConversationObserver(Protocol):
    """A hook that watches the conversation change.

    Implemented by the optional transcript (``session.transcript``): the client
    tells the observer about each recorded message, and the observer decides what
    -- if anything -- to do with it. The client itself never learns whether
    anything is listening, so a run without a log pays nothing for the hook.
    """

    def message_recorded(self, state: ClientState, message: Message) -> None:
        """Called once ``message`` has been appended to ``state``."""
        ...

    def turn_discarded(self, state: ClientState) -> None:
        """Called once a turn has been rolled back out of ``state``."""
        ...


class LlmClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(
        self,
        model: str,
        *,
        instructions: str = "",
        timer_factory: Callable[[], Timer] = Timer,
    ) -> None:
        """Initialize state, seeding the project instructions if there are any.

        ``instructions`` (the composition root reads ``LLM_CLI.md`` once at
        startup and passes what it found) becomes the conversation's system
        message: the first message, ahead of every user turn, sent with each
        request. It is supplied by the caller rather than read from the
        filesystem here, so a client can be built without touching the disk, and
        it is not re-read per request -- mid-session changes would make later
        turns inconsistent with earlier ones. Empty means no system message at
        all, and the conversation is then exactly the turns that were exchanged.

        ``timer_factory`` builds the timer each request is measured with, so how
        long the model took is recorded next to what it said. Tests pass a
        scripted timer instead of waiting for a real one.
        """
        self._state = ClientState(
            model=model,
            conversation=[Message(role=Role.SYSTEM, content=instructions)] if instructions else [],
        )
        # Nothing is watching unless a log file was configured; see observe.
        self._observer: ConversationObserver | None = None
        self._timer_factory = timer_factory

    @property
    def state(self) -> ClientState:
        return self._state

    def observe(self, observer: ConversationObserver) -> None:
        """Have ``observer`` be told about every change to the conversation.

        Set after construction (rather than passed in) so the log stays optional
        and the client is complete without one.
        """
        self._observer = observer

    def remember(self, message: Message) -> None:
        """Append ``message`` to the conversation and show it to the observer.

        The one place a message is recorded, so the log cannot miss a turn: user
        turns, assistant replies (tool calls included) and tool results all go
        through here.
        """
        self._state.conversation.append(message)
        if self._observer is not None:
            self._observer.message_recorded(self._state, message)

    def rollback_last_turn(self) -> None:
        """Discard the assistant reply recorded for the current turn.

        Used when a turn cannot be finished -- a tool call whose arguments did
        not parse, or a tool run the user interrupted -- because what was
        recorded is then an assistant message whose ``tool_calls`` have no
        results, the one shape some OpenAI-compatible APIs reject with HTTP 400
        on the next request. The user's own prompt is kept: the question stays in
        the conversation and can simply be asked again. A no-op when no assistant
        message has been recorded (nothing to undo).

        The system message -- the project instructions -- is the one message
        that is never discarded, whatever else goes: it sits at the head of the
        conversation, so a turn is undone down to it and no further.
        """
        conversation = self._state.conversation
        for index in range(len(conversation) - 1, -1, -1):
            if conversation[index].role is Role.ASSISTANT:
                del conversation[index:]
                if self._observer is not None:
                    self._observer.turn_discarded(self._state)
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
