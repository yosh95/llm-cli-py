"""Data models for llm-cli-py."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypedDict


class Role(StrEnum):
    """Message role enum matching OpenAI format."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallFunction(TypedDict):
    """The ``function`` object of an OpenAI-style tool call."""

    name: str
    arguments: str
    """JSON-encoded argument string (the API requires a string, not an object)."""


class ToolCallPayload(TypedDict):
    """One entry of an assistant message's ``tool_calls`` array.

    Used for the OpenAI wire format and when replaying a recorded assistant
    turn. Typing it (rather than ``dict[str, object]``) keeps attribute-style
    access checked and documents that ``arguments`` is always a JSON string.
    """

    id: str
    type: str
    function: ToolCallFunction


@dataclass
class Message:
    """A single message in the conversation."""

    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCallPayload] | None = None
    """Tool calls data (for assistant messages)."""


@dataclass
class LlmResponse:
    """Response from an LLM API call."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, object]
    parse_error: str | None = None
    """Raw (unparseable) arguments string when the tool call's JSON was broken."""


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
