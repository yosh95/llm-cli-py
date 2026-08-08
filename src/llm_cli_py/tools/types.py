"""Typed result classes for tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResultItem:
    """A single search result from Brave Search."""

    title: str = ""
    url: str = ""
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }


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
class SearchResult:
    """Result of a web search via Brave Search.

    Fields:
        query: The original search query.
        results: List of search result items.
        result_count: Number of results returned.
    """

    query: str = ""
    results: list[SearchResultItem] = field(default_factory=list)
    result_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "result_count": self.result_count,
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


ToolResult = ExecResult | SearchResult | ToolError
"""Union type for all possible tool execution results."""
