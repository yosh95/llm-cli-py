"""Typed result classes for tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Self


def _indent_lines(text: str, prefix: str) -> list[str]:
    """Split ``text`` into stripped lines, each indented with ``prefix``."""
    return [f"{prefix}{line}" for line in text.rstrip().splitlines()]


@dataclass
class ExecResult:
    """Result of Python code execution in a subprocess.

    Fields:
        stdout: Standard output from the executed code.
        stderr: Standard error from the executed code.
        exit_code: Exit code (0 = success, non-zero = error, -1 = exception).
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        """Rebuild an ``ExecResult`` from its serialised form (``to_dict``)."""
        raw_exit = data.get("exit_code", 0)
        exit_code = raw_exit if isinstance(raw_exit, int) else 0
        return cls(
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            exit_code=exit_code,
        )

    def to_lines(self) -> list[str]:
        """Render this result as display lines for the terminal."""
        lines = [f"Exit code: {self.exit_code}"]
        if self.stdout.strip():
            lines.append("[stdout]")
            lines.extend(_indent_lines(self.stdout, "  "))
        if self.stderr.strip():
            lines.append("[stderr]")
            lines.extend(_indent_lines(self.stderr, "  "))
        return lines


@dataclass
class ToolError:
    """Represents a tool execution error.

    When a tool encounters an error (missing API key, network failure,
    etc.), it returns a ToolError.
    """

    error: str

    def to_dict(self) -> dict[str, str]:
        return {"error": self.error}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        """Rebuild a ``ToolError`` from its serialised form (``to_dict``)."""
        return cls(error=str(data.get("error", "")))

    def to_lines(self) -> list[str]:
        """Render this error as display lines for the terminal."""
        return [f"Error: {self.error}"]


ToolResult = ExecResult | ToolError
"""Union type for all possible tool execution results."""


def parse_tool_result(content_str: str) -> ToolResult | None:
    """Parse a serialised tool result.

    Returns the concrete :class:`ExecResult` / :class:`ToolError` when
    ``content_str`` is the JSON produced by ``to_dict``, and ``None`` when it is
    not that JSON shape (plain text, a non-object, or an unrelated JSON value).
    Callers decide how to display such raw content.
    """
    try:
        data = json.loads(content_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "stdout" in data or "stderr" in data:
        return ExecResult.from_dict(data)
    if "error" in data:
        return ToolError.from_dict(data)
    return None


def normalise_tool_result(result: object) -> ToolResult:
    """Coerce whatever a tool returned into an :class:`ExecResult`/``ToolError``.

    Tools are declared as ``Callable[..., ToolResult]``; a tool returning some
    other type is a programming error. Rather than silently reinterpreting the
    value as a successful result (or letting ``.get`` crash on a string), it is
    turned into an explicit :class:`ToolError` so the model sees a clear,
    structured failure.
    """
    if isinstance(result, (ExecResult, ToolError)):
        return result
    return ToolError(error=f"Tool returned an invalid result type: {type(result).__name__}")


def render_tool_result(content_str: str) -> list[str]:
    """Render serialised tool output as terminal display lines.

    Structured results reuse the dataclasses' own ``to_lines``. Anything that is
    not a recognisable tool result (an empty body, plain text, an unexpected
    JSON value) is displayed verbatim instead of being mistaken for an error.
    """
    if not content_str:
        return ["(empty result)"]

    parsed = parse_tool_result(content_str)
    if parsed is not None:
        return parsed.to_lines()

    return content_str.splitlines() or [content_str]
