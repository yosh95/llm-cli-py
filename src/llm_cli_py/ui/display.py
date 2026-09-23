"""Display utilities - terminal output.

Every user-visible line goes through this module, so output style (rules,
labels, indentation, prefixes) is defined in one place rather than sprinkled
as bare ``print`` calls across the session loop.
"""

from __future__ import annotations

import shutil


def _term_width() -> int:
    """Get the current terminal width."""
    return shutil.get_terminal_size().columns


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
    print_block(text, title="Assistant")


def print_thinking(model: str) -> None:
    """Announce that a request is in flight for ``model``."""
    print(f"{model} is thinking...")


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


def print_tool_call(name: str, arg_lines: list[str]) -> None:
    """Print a tool invocation and its arguments.

    The rule and the ``Executing tool: ...`` header open the block. When the
    call carries arguments, an ``Args:`` line follows them in as a header and
    each pre-formatted argument line is printed unchanged (the caller labels
    and indents them), so multi-line values such as code keep their own
    internal indentation. A call without arguments prints no ``Args:`` block.
    """
    print_rule()
    print(f"[Tool] Executing tool: {name}...")
    if not arg_lines:
        return
    print("Args:")
    for line in arg_lines:
        print(line)


def print_tool_result(lines: list[str]) -> None:
    """Print the result of a tool execution.

    A horizontal rule is drawn first so the result block is clearly
    separated from the tool invocation (``Executing tool: ...``) above
    it. Each line in the list is printed with an indent for visual
    separation from the surrounding output.
    """
    print_rule()
    print("  Tool Result:")
    for line in lines:
        print(f"    {line}")


def print_info(label: str, value: str) -> None:
    """Print an info key-value pair."""
    print(f"  {label}: {value}")


def report_info(message: str) -> None:
    """Report an informational message."""
    print(f"INFO: {message}")


def report_error(message: str) -> None:
    """Report an error message."""
    print(f"[ERROR] {message}")


def report_warning(message: str) -> None:
    """Report a warning message."""
    print(f"[WARN] {message}")
