"""Display utilities - terminal output."""

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
