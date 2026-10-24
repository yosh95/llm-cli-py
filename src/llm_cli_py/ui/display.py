"""Display utilities - terminal output.

Every user-visible line goes through this module, so output style (rules,
labels, indentation, prefixes, timestamps) is defined in one place rather than
sprinkled as bare ``print`` calls across the session loop.

Two clocks are shown, and they answer different questions. Each line is stamped
with the wall-clock time (``[09:32:10]``), so a transcript read afterwards -- or
watched through a ``tail -f`` -- lines up with everything else that happened at
that moment. The rule also carries how long the session has been running
(``[+00:03:21]``) at its right end, so the gap between two rules is visible
without doing arithmetic on the stamps.

Neither of those answers the question a long silence raises: *is this still
working, or is it stuck?* For that, an operation that blocks -- a request to the
model, a tool run -- gets a status line (:func:`working`) that redraws itself
once a second with the time elapsed so far. Under a terminal it redraws in
place, so the line just grows: ``gpt-4o is thinking... 1.0s``, ``... 2.0s``,
``... done`` -- a counter that keeps moving is the whole proof that the process
is alive.
"""

from __future__ import annotations

import shutil
import sys
import threading
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

from ..utils.clock import Clock, Timer

RULE_CHAR = "\u2500"
"""The character a rule is drawn with: a light horizontal box-drawing line."""

STATUS_INTERVAL = 1.0
"""How often the status line of a running operation is redrawn, in seconds."""

_clock = Clock()
"""The clock stamps and durations are read from -- one per run.

Created at import so a module used on its own (tests, an embedder) works without
setup; :func:`configure_clock` replaces it at the start of a real run so the
session's uptime counts from the session, not from the import.
"""


def configure_clock(clock: Clock) -> None:
    """Read stamps and durations from ``clock`` from now on."""
    global _clock
    _clock = clock


def _term_width() -> int:
    """Get the current terminal width."""
    return shutil.get_terminal_size().columns


# ── Stamps ─────────────────────────────────────────────────────────


def stamp() -> str:
    """Return the current wall-clock time as the ``[HH:MM:SS]`` prefix of a line."""
    return f"[{_clock.now():%H:%M:%S}]"


def format_duration(seconds: float) -> str:
    """Render a duration compactly: ``3.2s``, ``1m 12s``, ``1h 07m``.

    Tenths of a second matter for a tool run that takes a moment, and stop
    mattering once the number is large enough to read at a glance, so the
    smaller units are dropped as the duration grows -- which is also what keeps
    a status line from changing width as it ticks.
    """
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _format_uptime(seconds: float) -> str:
    """Render the session's uptime as ``+HH:MM:SS`` for the right end of a rule."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"+{hours:02d}:{minutes:02d}:{secs:02d}"


def _display_width(text: str) -> int:
    """Width of ``text`` in terminal columns (wide characters such as Japanese count twice)."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


# ── Rules and blocks ───────────────────────────────────────────────


def print_rule() -> None:
    """Print a horizontal rule matching the terminal width.

    The wall-clock time opens the line and the session's uptime closes it, so
    the rule -- which is printed at every turn and every tool call -- doubles as
    the timeline: ``[09:32:10] ──────────────── [+00:03:21]``.

    On a narrow terminal the uptime is dropped rather than the rule: the rule
    still separates the blocks, and a stamp that wraps onto a second line would
    be worse than no uptime at all.
    """
    width = _term_width()
    left = stamp()
    if width < 40:
        print(left + " " + RULE_CHAR * max(0, width - _display_width(left) - 1))
        return

    right = f"[{_format_uptime(_clock.uptime())}]"
    fill = width - _display_width(left) - _display_width(right) - 2
    print(left + " " + RULE_CHAR * fill + " " + right)


def print_assistant(text: str) -> None:
    """Display the assistant's final answer with a clear, borderless label."""
    print_rule()
    print("Assistant:")
    print(text.strip())


def print_tool_call(name: str, arg_lines: list[str]) -> None:
    """Print a tool invocation and its arguments.

    The rule and the ``Executing tool: ...`` header open the block. When the
    call carries arguments, an ``Args:`` line follows them in as a header and
    each pre-formatted argument line is printed unchanged (the caller labels
    and indents them), so multi-line values such as code keep their own
    internal indentation. A call without arguments prints no ``Args:`` block.
    """
    print_rule()
    print(f"{stamp()} [Tool] Executing tool: {name}...")
    if not arg_lines:
        return
    print(f"{stamp()} Args:")
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
    print(f"{stamp()}   Tool Result:")
    for line in lines:
        print(f"    {line}")


# ── A live status line for a blocking operation ────────────────────


def _paint(stream: TextIO, text: str, painted: int) -> int:
    """Redraw ``text`` in place over the last line and return its width.

    ``\\r`` returns to the start of the line and the padding overwrites whatever
    the previous, possibly longer, text left behind. No ANSI escape is used
    (``\\x1b[K`` would be shorter but prints as garbage on consoles that do not
    interpret it), and one that wraps is never produced: labels are short.
    """
    width = _display_width(text)
    stream.write("\r" + text + " " * max(0, painted - width))
    stream.flush()
    return width


def _redraws_in_place(stream: TextIO) -> bool:
    """Whether ``stream`` is a terminal, and so can be redrawn rather than appended to.

    Output redirected to a file is not: writing carriage returns into it would
    leave one unreadable line per second. There the operation is announced once,
    when it is over (see :func:`working`).
    """
    return hasattr(stream, "isatty") and stream.isatty()


@contextmanager
def working(label: str) -> Iterator[Timer]:
    """Announce a blocking operation and show how long it has been running.

    Inside a terminal the line redraws itself once a second, replacing the
    previous reading, and is left in place when the block ends::

        [09:32:10] gpt-4o is thinking... 1.0s
        [09:32:13] gpt-4o is thinking... 3.2s

    A counter that keeps moving is what distinguishes a slow provider from a
    hung one; the final reading stays on screen as the operation's duration.
    Anywhere else (output redirected to a file, a test) nothing is written until
    the block ends, and then the same final line is printed once -- a stream
    nobody is watching has no use for intermediate frames.

    The duration is measured on the monotonic clock (see
    :class:`~llm_cli_py.utils.clock.Timer`), so a system clock adjustment during
    a long request cannot make it negative. The block's exception, if any,
    propagates unchanged; the line is closed either way.

    Yields:
        The timer, so a caller that wants the number can read it afterwards.
    """
    timer = _clock.timer()
    stream = sys.stdout
    painted = 0

    if not _redraws_in_place(stream):
        try:
            yield timer
        finally:
            print(f"{stamp()} {label} {format_duration(timer.elapsed())}")
        return

    stop = threading.Event()
    last = ""  # the line as last drawn: a redraw that would read the same is skipped

    def draw() -> None:
        """Repaint the line with the current reading, if it has changed."""
        nonlocal painted, last
        text = f"{stamp()} {label} {format_duration(timer.elapsed())}"
        if text != last:
            painted = _paint(stream, text, painted)
            last = text

    def tick() -> None:
        """Redraw the line once a second until the block finishes."""
        while not stop.wait(STATUS_INTERVAL):
            draw()

    draw()
    thread = threading.Thread(target=tick, name="llm-cli-status", daemon=True)
    thread.start()
    try:
        yield timer
    finally:
        # Whatever ended the block -- the answer, an error, Ctrl+C -- the ticker
        # must stop before anything else writes to the same line.
        stop.set()
        try:
            thread.join()
        finally:
            # The line is closed even when the interrupt that ended the block
            # arrives while the ticker is being waited for: an unterminated line
            # would leave the next output on top of the reading left there.
            draw()
            stream.write("\n")
            stream.flush()


# ── Reports ────────────────────────────────────────────────────────


def report_info(message: str) -> None:
    """Report an informational message."""
    print(f"{stamp()} INFO: {message}")


def report_error(message: str) -> None:
    """Report an error message."""
    print(f"{stamp()} [ERROR] {message}")
