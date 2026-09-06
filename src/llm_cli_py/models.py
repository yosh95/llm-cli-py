"""Data models for llm-cli-py."""

from __future__ import annotations

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
