"""Display utilities - terminal output."""

from __future__ import annotations

import json
import re
import shutil

from ..tools.types import ExecResult, SearchResult, ToolError, ToolResult


def _term_width() -> int:
    """Get the current terminal width."""
    return shutil.get_terminal_size().columns


def _sanitize_terminal_output(text: str) -> str:
    """Remove ANSI escape sequences and control characters that break terminal display.

    Strips:
      - ANSI escape sequences (colours, cursor movement, screen clears, etc.)
      - Carriage returns (\r) — these cause the cursor to jump to column 0,
        which misaligns subsequent output.
      - Other control characters in the range 0x00-0x1f (except \n, \t, and
        the already-handled \r) that can corrupt terminal state.
    """
    # ANSI escape sequences: ESC [ <params> <letter>
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    # Carriage returns
    text = text.replace("\r", "")
    # Remaining control characters (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def print_rule() -> None:
    """Print a horizontal rule matching the terminal width."""
    width = _term_width()
    print("\u2500" * width)


def print_block(content: str, title: str | None = None) -> None:
    """Print content with optional title."""
    if title:
        print(f"{title}:")
    text = content.strip()
    print(text)


def print_assistant(text: str) -> None:
    """Display the assistant's final answer with a clear, borderless label."""
    print_rule()
    print_block(text, title="\U0001f600 Assistant")


def stream_start(title: str) -> None:
    """Emit the opening label for a streaming answer block."""
    print_rule()
    print(title)
    print("", end="", flush=True)


def stream_text(delta: str) -> None:
    """Print an incremental text delta without a trailing newline (live)."""
    print(delta, end="", flush=True)


def stream_end() -> None:
    """Terminate a streaming line with a newline."""
    print("", flush=True)


def print_tool_call(name: str, args: dict[str, object]) -> None:
    """Display a tool call with its arguments as plain text."""
    print_rule()
    print(f"\U0001f527 Tool: {name}")
    print("  Arguments:")
    for k, v in args.items():
        if name == "execute_python" and k == "code" and isinstance(v, str):
            print(f"    {k}:")
            for line in v.strip().splitlines():
                print(f"      {_sanitize_terminal_output(line)}")
        else:
            # Sanitize non-code argument values too: URLs, queries, etc. may
            # carry control characters that corrupt the terminal state.
            if isinstance(v, str):
                v = _sanitize_terminal_output(v)
            print(f"    {k}: {v}")


def format_tool_result(result: ToolResult) -> str:
    """Format a tool execution result into a human-readable string."""
    if isinstance(result, ExecResult):
        lines: list[str] = []
        if result.stdout:
            lines.append("- stdout:")
            lines.append(_sanitize_terminal_output(result.stdout.rstrip("\n")))
        else:
            lines.append("- stdout: (no output)")
        if result.stderr:
            lines.append("")
            lines.append("- stderr:")
            lines.append(_sanitize_terminal_output(result.stderr.rstrip("\n")))
        else:
            lines.append("")
            lines.append("- stderr: (no output)")
        lines.append("")
        lines.append(f"- exit_code: {result.exit_code}")
        return "\n".join(lines)

    if isinstance(result, SearchResult):
        # Web search results can be very long, so only the top result is shown
        # on the human-visible terminal (the full results are still passed to
        # the LLM via conversation history). The snippet is also truncated to
        # a fixed number of characters to avoid cluttering the display.
        display_top = 1
        snippet_max_chars = 300

        lines = []
        lines.append(f"- Query: {result.query}")
        lines.append("")
        shown = min(display_top, result.result_count)
        lines.append(f"- Results ({result.result_count}, showing top {shown}):")
        lines.append("")
        for i, item in enumerate(result.results[:display_top], 1):
            lines.append(f"  - {i}. {item.title}")
            lines.append(f"    URL: {item.url}")
            if item.snippet:
                snippet = _sanitize_terminal_output(item.snippet)
                if len(snippet) > snippet_max_chars:
                    snippet = snippet[:snippet_max_chars] + "..."
                lines.append(f"    Snippet: {snippet}")
            lines.append("")
        return "\n".join(lines).rstrip("\n")

    if isinstance(result, ToolError):
        return f"- **Error:** {_sanitize_terminal_output(result.error)}"

    return json.dumps(result, indent=2, ensure_ascii=False)


def print_tool_result(result: str) -> None:
    """Display a tool execution result as plain text."""
    if result.strip():
        print_rule()
        print("\U0001f4e6 Tool Result:")
        print(_sanitize_terminal_output(result.strip()))


def print_info(label: str, value: str) -> None:
    """Print an info key-value pair."""
    print(f"  {label}: {value}")


def report_info(message: str) -> None:
    """Report an informational message."""
    print(f"INFO: {message}")


def report_success(message: str) -> None:
    """Report a success message."""
    print(f"\U0001f44c {message}")


def report_error(message: str) -> None:
    """Report an error message."""
    print(f"\u26d4 ERROR: {message}")


def report_warning(message: str) -> None:
    """Report a warning message."""
    print(f"\U0001f91a {message}")
