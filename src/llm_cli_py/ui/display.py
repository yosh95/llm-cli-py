"""Display utilities - terminal output, rendered with ``rich``.

Every user-visible line goes through this module, so output style (rules,
labels, indentation, prefixes, timestamps, colours) is defined in one place
rather than sprinkled as bare ``print`` calls across the session loop.

Two rules decide what a line looks like:

* **A terminal gets formatting; anything else gets plain text.** Output
  redirected to a file, piped to ``grep``, or captured by a test is not watched
  by a person, and escape sequences (or a Markdown renderer's ``•`` bullets)
  would only make it harder to read. ``Console`` already drops colour when its
  file is not a terminal, and this module goes one step further: it also skips
  the re-rendering of answers as Markdown and the syntax highlighting of code.
  So ``llm-cli-py ... > answer.md`` still writes the answer as the model wrote
  it, and ``NO_COLOR=1`` (which ``rich`` honours) turns colours off in a
  terminal.
* **The clock is part of the transcript.** Every line is stamped with the
  wall-clock time (``[09:32:10]``), so a transcript read afterwards -- or
  watched through ``tail -f`` -- lines up with everything else that happened at
  that moment. The rule also carries how long the session has been running
  (``[+00:03:21]``) at its right end, so the gap between two rules is visible
  without doing arithmetic on the stamps -- and a session that has been open all
  afternoon is not mistaken for an unbroken run of work.

Neither of those answers the question a long silence raises: *is this still
working, or is it stuck?* For that, an operation that blocks -- a request to the
model, a tool run -- gets a status line (:func:`working`) with a spinner and the
time elapsed so far. A counter that keeps moving is the whole proof that the
process is alive.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from rich.cells import cell_len
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ..utils.clock import Clock, Timer

if TYPE_CHECKING:  # imported lazily at use: they are only needed for a terminal
    from rich.console import ConsoleOptions, RenderableType, RenderResult
    from rich.markdown import Markdown, MarkdownElement

    from ..tools.types import ToolResult

RULE_CHAR = "\u2500"
"""The character a rule is drawn with: a light horizontal box-drawing line."""

LIVE_REFRESH_PER_SECOND = 10
"""How often the status line of a running operation is redrawn, in seconds."""

CODE_INDENT = "    "
"""Indentation of a multi-line argument's value, and of code shown under a label."""

_clock = Clock()
"""The clock stamps and durations are read from -- one per run.

Created at import so a module used on its own (tests, an embedder) works without
setup; :func:`configure_clock` replaces it at the start of a real run so the
session's uptime counts from the session, not from the import.
"""

_console = Console(soft_wrap=True, markup=False, highlight=False, emoji=False)
"""The console every line is written through.

``markup=False`` is what keeps a model's or a tool's own text literal: an answer
containing ``[INFO]`` or ``[1, 2]`` must not be read as rich markup.
``soft_wrap=True`` is what keeps a line from being re-wrapped at the console
width -- the text arrives already formatted, and a redirected transcript should
keep the shape it was given.
"""


def configure_clock(clock: Clock) -> None:
    """Read stamps and durations from ``clock`` from now on."""
    global _clock
    _clock = clock


def configure_console(console: Console) -> None:
    """Write through ``console`` from now on (tests pass one pointed at a buffer)."""
    global _console
    _console = console


def _term_width() -> int:
    """Get the width the console renders at (the real terminal's, or ``COLUMNS``)."""
    return _console.width


def _is_terminal() -> bool:
    """Whether the console writes to a terminal, and so may be formatted."""
    return _console.is_terminal


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


def _stamped(rest: Text | RenderableType, *, label: str = "", label_style: str = "") -> Text:
    """Prefix ``rest`` with the current stamp, and optionally a styled label."""
    line = Text()
    line.append(stamp(), style="dim")
    line.append(" ")
    if label:
        line.append(label, style=label_style)
        line.append(" ")
    if isinstance(rest, Text):
        line.append_text(rest)
    else:
        line.append(str(rest))
    return line


def _print(line: Text | RenderableType) -> None:
    """Write one line through the console."""
    _console.print(line)


# ── Rules and blocks ───────────────────────────────────────────────


def print_rule() -> None:
    """Print a horizontal rule matching the terminal width.

    The wall-clock time opens the line and the session's uptime closes it, so
    the rule -- which is printed at every turn and every tool call -- doubles as
    the timeline: ``[09:32:10] ──────────────── [+00:03:21]``.

    On a narrow terminal the uptime is dropped rather than the rule: the rule
    still separates the blocks, and a stamp that wrapped onto a second line would
    be worse than no uptime at all.
    """
    width = _term_width()
    left = stamp()
    line = Text(left, style="dim")
    if width < 40:
        line.append(" " + RULE_CHAR * max(0, width - cell_len(left) - 1), style="dim")
        _print(line)
        return

    right = f"[{_format_uptime(_clock.uptime())}]"
    fill = width - cell_len(left) - cell_len(right) - 2
    line.append(" " + RULE_CHAR * fill + " ", style="dim")
    line.append(right, style="dim")
    _print(line)


def _wrapping_markdown(markup: str) -> Markdown:
    """Build a Markdown renderable whose tables wrap their cells instead of cutting them.

    ``rich``'s Markdown gives every table cell ``overflow="ellipsis"``, which is
    the wrong failure mode for a transcript: a chatty model writes tables far
    wider than any terminal, and the part that is cut is usually the part that
    says what happened. Cells are set to ``"fold"`` -- wrap onto another line
    within the column -- so the whole table is always readable.

    Japanese makes this worse than it is in English, which is why it cannot be
    left to chance: a wide character takes two columns, and a table of them is
    over the width limit far sooner than the character count suggests.
    ``fold`` breaks at the column edge, mid-word when it has to, which for
    Japanese (no spaces to break at) is the only option anyway.
    """
    from rich.markdown import Markdown, TableElement
    from rich.table import Table

    class WrappingTableElement(TableElement):
        """A table whose columns wrap their content."""

        def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
            for renderable in super().__rich_console__(console, options):
                if isinstance(renderable, Table):
                    for column in renderable.columns:
                        column.overflow = "fold"
                        column.no_wrap = False
                yield renderable

    class WrappingMarkdown(Markdown):
        """Markdown with the wrapping table in place of the truncating one."""

        elements: ClassVar[dict[str, type[MarkdownElement]]] = {
            **Markdown.elements,
            "table_open": WrappingTableElement,
        }

    return WrappingMarkdown(markup)


def print_assistant(text: str) -> None:
    """Display the assistant's final answer with a clear, borderless label.

    On a terminal the answer is rendered as Markdown, which is how models write:
    headings, lists, emphasis and fenced code come out as they were meant to be
    read. Everywhere else the text is written exactly as it arrived, because a
    file is read as its own source rather than as a rendering of one -- and
    Markdown's bullets and code-block padding would be noise in it.
    """
    print_rule()
    body = text.strip()
    if _is_terminal():
        _console.print(Text("Assistant:", style="bold"))
        # ``soft_wrap`` is off here, though the console itself has it on: with it,
        # ``print`` tells the renderable not to wrap, and the Markdown elements
        # that lay their own text out -- list items, block quotes -- measure their
        # text, hand it to ``render_lines`` and crop each line at the width. The
        # end of every long item is then dropped rather than moved to the next
        # line, which is exactly what a Japanese answer shows first: a wide
        # character takes two columns, so an item is over the width at half the
        # character count a reader would guess. Wrapping normally also keeps
        # Markdown's own alignment (table columns, hanging indents) intact, and
        # ``crop=False`` is the backstop for a line that still cannot be folded.
        _console.print(_wrapping_markdown(body), soft_wrap=False, crop=False)
        return

    _print(Text("Assistant:", style="bold"))
    _print(Text(body))


def print_tool_call(name: str, arguments: dict[str, object]) -> None:
    """Print a tool invocation and its arguments.

    The rule and the ``Executing tool: ...`` header open the block. A call with
    arguments adds an ``Args:`` header, then one ``[name] value`` line per
    argument at a glance; a value that spans several lines (``code`` being the
    obvious case) gets a ``[name]`` label of its own and is reproduced
    underneath, so the code keeps the indentation it was written with instead of
    being mangled into one long line. On a terminal such a value is also
    syntax-highlighted -- when an agent shows you the Python it is about to run,
    the code should look like code.
    """
    print_rule()
    header = _stamped(Text(), label="[Tool]", label_style="bold cyan")
    header.append(f"Executing tool: {name}...", style="bold")
    _print(header)

    if not arguments:
        return

    _print(_stamped(Text("Args:", style="dim")))
    for arg_name, value in arguments.items():
        text = value if isinstance(value, str) else str(value)
        # Code is almost always written with a trailing newline; dropping it
        # first keeps the block from ending on an empty indented line and lets
        # one-line code stay on a single ``[name] value`` line.
        value_lines = text.rstrip("\n").split("\n")
        if len(value_lines) == 1:
            line = Text()
            line.append("  ")
            line.append(f"[{arg_name}]", style="cyan")
            line.append(f" {value_lines[0]}")
            _print(line)
            continue

        _print(Text(f"  [{arg_name}]", style="cyan"))
        _print_code(text.rstrip("\n"))


def _print_code(code: str) -> None:
    """Print a block of code, indented, highlighted on a terminal."""
    if not _is_terminal():
        for line in code.split("\n"):
            _print(Text(CODE_INDENT + line))
        return

    from rich.syntax import Syntax

    syntax = Syntax(
        code,
        "python",
        # No background: the code sits in the transcript rather than in a panel,
        # so it should use the terminal's own colours for everything it does not
        # highlight.
        background_color="default",
        word_wrap=False,
        padding=0,
        indent_guides=False,
    )
    _console.print(Padding(syntax, (0, 0, 0, len(CODE_INDENT))))


def print_tool_result(result: ToolResult) -> None:
    """Print the result of a tool execution.

    A horizontal rule is drawn first so the result block is clearly separated
    from the tool invocation above it, then the result's own lines: the exit code
    in green or red, ``[stdout]``/``[stderr]`` as labels, and the output itself
    indented under each. A failed result is red from its first line, so the
    outcome is visible without reading the block.
    """
    print_rule()
    ok = result.ok
    label = _stamped(Text("Tool Result:", style="dim bold green" if ok else "dim bold red"))
    _print(label)

    lines = result.as_display_lines()
    for line in lines:
        style = _result_line_style(line, ok)
        _print(Text(CODE_INDENT + line, style=style) if style else Text(CODE_INDENT + line))


def _result_line_style(line: str, ok: bool) -> str:
    """Pick a style for one result line, or ``""`` to leave it plain.

    The labels are this module's own vocabulary (``as_display_lines`` writes
    them for it), so recognising them here keeps the styling out of the tool
    types, which stay display-agnostic.
    """
    if line.startswith("Exit code:"):
        return "green" if ok else "bold red"
    if line == "[stderr]" or line.startswith("Error:"):
        return "bold red"
    if line == "[stdout]":
        return "bold"
    return ""


# ── A live status line for a blocking operation ────────────────────


def _status_text(label: str, seconds: float) -> Text:
    """The text of a status line: ``[09:32:13] gpt-4o is thinking... 2.5s``."""
    body = Text()
    body.append(stamp(), style="dim")
    body.append(" ")
    body.append(label, style="yellow")
    body.append(" ")
    body.append(format_duration(seconds), style="bold green")
    return body


def _working_line(label: str, seconds: float, *, spinner: bool) -> Table:
    """Assemble the status line, with or without the spinner that turns it."""
    body = _status_text(label, seconds)

    grid = Table.grid(padding=0)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    # One column for the spinner and one for the text, with a blank of the same
    # width standing in for the spinner once the run is over -- so the finished
    # line begins where the spinning ones did, rather than jumping to the left.
    grid.add_row(
        Spinner("dots", style="yellow") if spinner else Text(" "),
        Text(" ") + body,
    )
    return grid


@contextmanager
def working(label: str) -> Iterator[Timer]:
    """Announce a blocking operation and show how long it has been running.

    Inside a terminal the line carries a spinner and redraws ten times a second,
    so it reads as alive rather than as a number that happens to change::

        ⠹ [09:32:10] gpt-4o is thinking... 1.0s
        ⠼ [09:32:13] gpt-4o is thinking... 3.2s

    A counter that keeps moving (and a spinner that keeps turning) is what
    distinguishes a slow provider from a hung one; the final reading stays on
    screen, without the spinner, as the operation's duration.

    Anywhere else -- output redirected to a file, captured by a test -- nothing is
    written until the block ends, and then the same final line is printed once: a
    stream nobody is watching has no use for intermediate frames, and carriage
    returns in a file are unreadable.

    The duration is measured on the monotonic clock (see
    :class:`~llm_cli_py.utils.clock.Timer`), so a system clock adjustment during
    a long request cannot make it negative. The block's exception, if any,
    propagates unchanged; the line is closed either way.

    Yields:
        The timer, so a caller that wants the number can read it afterwards.
    """
    timer = _clock.timer()

    if not _is_terminal():
        try:
            yield timer
        finally:
            # No spinner, and no grid making room for one: a file wants the line
            # itself and nothing else.
            _print(_status_text(label, timer.elapsed()))
        return

    finished = [False]

    def render() -> Table:
        """The line as it should look right now -- read at every refresh.

        Live keeps calling this until it stops, including once more while it
        stops, so the finished block must say so here: otherwise the last frame
        drawn would have the spinner back on a run that is over.
        """
        return _working_line(label, timer.elapsed(), spinner=not finished[0])

    with Live(
        render(),
        console=_console,
        get_renderable=render,
        refresh_per_second=LIVE_REFRESH_PER_SECOND,
        # The finished line must stay on screen: it is the operation's duration.
        transient=False,
        # A status line is one line; letting it be cropped would be the one thing
        # worse than not showing it.
        vertical_overflow="visible",
    ):
        try:
            yield timer
        finally:
            # Whatever ended the block -- the answer, an error, Ctrl+C -- the last
            # frame must be the finished line, without the spinner, before the
            # display is released back to the transcript.
            # Only the flag: leaving the block redraws once more, and that redraw is
            # the finished line. Updating here as well would draw it twice.
            finished[0] = True


# ── Reports ────────────────────────────────────────────────────────


def report_info(message: str) -> None:
    """Report an informational message."""
    line = _stamped(Text(message), label="INFO:", label_style="bold cyan")
    _print(line)


def report_error(message: str) -> None:
    """Report an error message."""
    line = _stamped(Text(message), label="[ERROR]", label_style="bold red")
    _print(line)
