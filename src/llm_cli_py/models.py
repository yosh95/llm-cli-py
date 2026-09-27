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
    """The ``function`` object of an OpenAI-style tool call.

    Attributes:
        name: Name of the tool to call.
        arguments: JSON-encoded argument string (the API requires a string, not
            an object).
    """

    name: str
    arguments: str


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
    """A single message in the conversation.

    Attributes:
        role: Who the message is from.
        content: The message text (may be empty).
        tool_call_id: The tool call this message answers (``Role.TOOL`` only).
        tool_calls: Tool calls the assistant asked for (``Role.ASSISTANT`` only).
    """

    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCallPayload] | None = None


@dataclass
class LlmResponse:
    """Response from an LLM API call."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolCall:
    """A tool call requested by the LLM.

    Attributes:
        id: Identifier the result has to be sent back with.
        name: Name of the tool to call.
        arguments: Parsed arguments (empty when they could not be parsed).
        parse_error: The raw arguments string when it was not valid JSON; such a
            call must not be executed.
    """

    id: str
    name: str
    arguments: dict[str, object]
    parse_error: str | None = None


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
    conversation: list[Message] = field(default_factory=list)
