"""Data models for llm-cli-py."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    """Message role enum matching OpenAI format."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in the conversation."""

    role: Role
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    """Tool calls data (for assistant messages)."""
    timestamp: str | None = None
    """Local creation time as an ISO 8601 string (e.g. ``2026-09-16T12:34:56+09:00``).

    Local metadata only: it is written to ``/dump`` and the chat log, but is
    deliberately NOT part of the API request (see
    :meth:`LlmApiClient._build_messages`).
    """


@dataclass
class DataSource:
    """Represents a data input (text, file content, URL result)."""

    text: str
    source_type: str = "text"  # text, file, url


@dataclass
class LlmResponse:
    """Response from an LLM API call."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass
class ToolSchema:
    """Schema definition for a tool."""

    name: str
    description: str
    parameters: dict[str, object]


@dataclass
class ClientState:
    """State of an LLM client session."""

    model: str = ""
    system_prompt: str = ""
    """System prompt read once at client initialization (startup snapshot)."""
    conversation: list[Message] = field(default_factory=list)
    on_change: Callable[[], None] | None = None
    """Optional listener called by :meth:`notify_changed` after the conversation grows.

    Used to persist the chat log incrementally: the CLI registers a writer here,
    so every new message (user turn, assistant answer, tool result) reaches disk
    immediately instead of only when the session ends.
    """

    def notify_changed(self) -> None:
        """Notify the change listener (if any) that the conversation has grown.

        Called right after appending a message, before any network request, so
        the log on disk is up to date even if the process dies mid-request.
        The listener is best-effort by design: exceptions are swallowed here so
        that a failing log write can never break the chat request.
        """
        if self.on_change is None:
            return
        # Logging must never break a request, so failures are swallowed.
        with contextlib.suppress(Exception):
            self.on_change()
