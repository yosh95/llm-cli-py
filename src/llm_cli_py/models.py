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
    reasoning: str | None = None
    """Reasoning / thinking trace attached to an assistant message."""


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
    reasoning: str | None = None
    """Model's internal reasoning / thinking trace (e.g. OpenRouter
    ``reasoning`` field). Distinct from the final ``text`` answer."""


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
class TokenUsage:
    """Token usage tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ClientState:
    """State of an LLM client session."""

    model: str = ""
    conversation: list[Message] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
