"""Typed result classes for tool execution.

A tool returns one of :class:`ExecResult` / :class:`ToolError`, and those two
classes own both of the things the CLI does with such a value: ``as_tool_content``
produces the JSON string recorded for the model, and ``as_display_lines``
produces the human-readable transcript lines. There is no separate parse step:
nothing re-reads the JSON the CLI itself produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


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
        """Return this result as a JSON-serialisable payload."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
        }

    def as_display_lines(self) -> list[str]:
        """Return this result as display lines for the terminal."""
        lines = [f"Exit code: {self.exit_code}"]
        if self.stdout.strip():
            lines.append("[stdout]")
            lines.extend(_indent_lines(self.stdout, "  "))
        if self.stderr.strip():
            lines.append("[stderr]")
            lines.extend(_indent_lines(self.stderr, "  "))
        return lines

    def as_tool_content(self) -> str:
        """Return the JSON string recorded as this turn's tool message."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ToolError:
    """Represents a tool execution error.

    When a tool encounters an error (missing API key, network failure,
    etc.), it returns a ToolError.
    """

    error: str

    def to_dict(self) -> dict[str, str]:
        """Return this error as a JSON-serialisable payload."""
        return {"error": self.error}

    def as_display_lines(self) -> list[str]:
        """Return this error as display lines for the terminal."""
        return [f"Error: {self.error}"]

    def as_tool_content(self) -> str:
        """Return the JSON string recorded as this turn's tool message."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


ToolResult = ExecResult | ToolError
"""Union type for all possible tool execution results."""


def normalise_tool_result(result: object) -> ToolResult:
    """Coerce whatever a tool returned into an :class:`ExecResult`/``ToolError``.

    Tools are declared as ``Callable[..., ToolResult]``; a tool returning some
    other type is a programming error. Rather than silently reinterpreting the
    value as a successful result (or letting a later attribute access crash on
    a string), it is turned into an explicit :class:`ToolError` so the model
    sees a clear, structured failure.
    """
    if isinstance(result, (ExecResult, ToolError)):
        return result
    return ToolError(error=f"Tool returned an invalid result type: {type(result).__name__}")
