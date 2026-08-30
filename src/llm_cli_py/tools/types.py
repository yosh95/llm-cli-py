"""Typed result classes for tool execution."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class ToolError:
    """Represents a tool execution error.

    When a tool encounters an error (missing API key, network failure,
    etc.), it returns a ToolError.
    """

    error: str

    def to_dict(self) -> dict[str, str]:
        return {"error": self.error}


ToolResult = ExecResult | ToolError
"""Union type for all possible tool execution results."""
