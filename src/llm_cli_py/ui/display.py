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
* **The clock is part of the transcript.** A line that records something that
  happened -- an answer, a report, a tool's result -- is stamped with the
  wall-clock time (``[09:32:10]``), so a transcript read afterwards -- or watched
  through ``tail -f`` -- lines up with everything else that happened at that
  moment. The rule also carries how long the session has been running
  (``[+00:03:21]``) at its right end, so the gap between two rules is visible
  without doing arithmetic on the stamps -- and a session that has been open all
  afternoon is not mistaken for an unbroken run of work.

Neither of those answers the question a long silence raises: *is this still
working, or is it stuck?* For that, an operation that blocks -- a request to the
model, a tool run -- gets a status line (:func:`working`) with the time elapsed
so far, redrawn in place while it runs. A counter that keeps moving is the whole
proof that the process is alive.

Progress is the exception: the status line of a running operation
(:func:`working`) and the block that announces a tool call
(:func:`print_tool_call`) carry no stamp. They describe the present rather than
an event -- the status line already says how long it has lasted, and the call has
a rule stamped just above it -- so a stamp there would be a number rewritten ten
times a second, or repeated on every line of a block whose time is already on
screen.

One thing here is deliberately not styled at all: :func:`print_verbatim` writes
text that is meant to be copied out of the terminal -- ``/raw``'s Markdown -- so
anything this module would add to it (a stamp, a label, a re-rendering of its
Markdown) would end up in the paste.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from rich.cells import cell_len
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
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

MIN_FOLD_WIDTH = 16
"""The fewest columns there must be left to fold a tool's output into.

Below it there is no row to fold onto: the labels (``Exit code: 0`` is twelve
columns, ``[stdout]`` ten) would be broken up themselves, and the output would
arrive as fragments of words. A terminal that narrow is left to wrap the line
itself, which loses nothing and keeps the labels whole.
"""

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


def _stamped(
    rest: Text | RenderableType,
    *,
    label: str = "",
    label_style: str = "",
    with_stamp: bool = True,
) -> Text:
    """Prefix ``rest`` with the current stamp, and optionally a styled label.

    ``with_stamp=False`` drops the stamp from a line that is about what is
    happening now rather than about when it happened: a progress line, or the
    header of a tool call, whose time is on the rule printed above it. The label
    and the text are unchanged -- only the leading ``[HH:MM:SS]`` goes -- so such
    a line starts at the left edge.
    """
    line = Text()
    if with_stamp:
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


EXPLANATION_ARGUMENT = "explanation"
"""The argument that says why a call is being made, and is printed last.

``execute_python`` asks for it (the schema has it in ``required``), and the whole
point of it is the reader: the sentence belongs in the transcript directly after
the code it is about, where the question "what is this for?" is asked. So the
argument is held back from the loop below and printed once the code is on screen,
whatever order the model happened to write the keys in.
"""


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

    One argument is deliberately last: ``explanation``, the reason the model
    gives for the call. It is printed after every other argument -- after the
    code, for the tool that has one -- because it answers a question a reader has
    about what is above it, not about what is below it (see
    :data:`EXPLANATION_ARGUMENT`).

    The block is not stamped. It describes a call that has not run yet, and the
    rule that opened it carries the time, so the only clock the call itself needs
    is the duration its status line shows (:func:`working`).
    """
    print_rule()
    header = _stamped(Text(), label="[Tool]", label_style="bold cyan", with_stamp=False)
    header.append(f"Executing tool: {name}...", style="bold")
    _print(header)

    if not arguments:
        return

    _print(_stamped(Text("Args:", style="dim"), with_stamp=False))
    for arg_name, value in arguments.items():
        if arg_name == EXPLANATION_ARGUMENT:
            continue  # printed below, after the code it explains
        _print_argument(arg_name, value)

    # An empty (or blank) explanation is not printed: a bare ``[explanation]``
    # label with nothing under it would only be a gap in the block. A call that
    # carries none at all ends here, exactly as it did before -- a tool that does
    # not ask for one, or a call the tool will refuse for the missing sentence
    # (the refusal is the result printed below, which says what to send instead).
    explanation = arguments.get(EXPLANATION_ARGUMENT)
    if explanation is not None and str(explanation).strip():
        _print_argument(EXPLANATION_ARGUMENT, explanation, code=False)


def _print_argument(arg_name: str, value: object, *, code: bool = True) -> None:
    """Print one argument: ``[name] value`` on a line, or labelled with the value under it.

    A value that spans several lines gets a label of its own and is reproduced
    underneath (see :func:`print_tool_call`): code is written with its own
    indentation, and folding it onto one line would lose it. ``code=False`` is
    for a value that is prose (the call's ``explanation``): a value of several
    lines is written out as it was typed, indented like code but *not* coloured
    as if it were Python -- a sentence highlighted as code reads as a mistake.
    """
    text = value if isinstance(value, str) else str(value)
    # Code is almost always written with a trailing newline; dropping it
    # first keeps the block from ending on an empty indented line and lets
    # one-line code stay on a single ``[name] value`` line.
    value_lines = text.rstrip("\n").split("\n")
    if len(value_lines) == 1:
        label_text = f"[{arg_name}] "
        line = Text("  ")
        line.append(f"[{arg_name}]", style="cyan")
        line.append(f" {value_lines[0]}")
        # A folded value continues under the value, not under the label:
        # what is being read is the argument, and the label is its heading.
        _print_folded(line, continued="  " + " " * cell_len(label_text))
        return

    _print(Text(f"  [{arg_name}]", style="cyan"))
    if code:
        _print_code(text.rstrip("\n"))
        return
    for value_line in value_lines:
        _print_folded(Text(CODE_INDENT + value_line))


def _print_code(code: str) -> None:
    """Print a block of code, indented, highlighted on a terminal.

    A line longer than the terminal is wrapped onto the next one rather than cut
    at the right edge. ``Syntax`` crops -- it does not wrap -- unless asked, so
    the last thing on a long line is lost: the URL a script fetches, the
    ``timeout=`` it passes, the closing bracket that shows where an expression
    ended. Those are the details a reader of a transcript is scanning the code
    for, and an agent's one-line ``sh("curl ...")`` is exactly the line that
    runs off the edge.
    """
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
        word_wrap=True,
        padding=0,
        indent_guides=False,
    )
    # ``soft_wrap`` is off here, as in :func:`print_assistant`, and for the same
    # reason: with it the renderable is told never to wrap or crop, and a line
    # that does not fit is cut instead of continued. ``crop=False`` is the
    # backstop for a line that still cannot be folded.
    _console.print(Padding(syntax, (0, 0, 0, len(CODE_INDENT))), soft_wrap=False, crop=False)


def print_tool_result(result: ToolResult) -> None:
    """Print the result of a tool execution.

    A horizontal rule is drawn first so the result block is clearly separated
    from the tool invocation above it, then the result's own lines: the exit code
    in green or red, ``[stdout]``/``[stderr]`` as labels, and the output itself
    indented under each. A failed result is red from its first line, so the
    outcome is visible without reading the block.

    A line longer than the terminal is folded onto the next row rather than run
    off the right edge (see :func:`_print_folded`): a tool's output is the thing
    the reader came for, and half of it is not a transcript.
    """
    print_rule()
    ok = result.ok
    label = _stamped(Text("Tool Result:", style="dim bold green" if ok else "dim bold red"))
    _print(label)

    lines = result.as_display_lines()
    for line in lines:
        style = _result_line_style(line, ok)
        _print_folded(Text(CODE_INDENT + line, style=style) if style else Text(CODE_INDENT + line))


def _print_folded(line: Text, *, continued: str = "") -> None:
    """Print one line, folded at the console width on a terminal.

    A tool's output line can be arbitrarily long -- a serialised payload, a
    traceback, one JSON object -- and a terminal left to fold it does so at
    column zero, mid-word and mid-wide-character, with the indentation that says
    which label (``[stdout]``, ``[stderr]``) it belongs under gone on the
    continuation rows. Folding here keeps every row inside the width and every
    row under its label, which is the same reasoning that wraps the answer's
    Markdown and the code of a call. A one-line argument of a call is folded the
    same way, under its ``[name]``.

    The break is at the last space that fits where there is one, and between
    characters where there is not: output is mostly not prose (a hash, a base64
    blob, a path), and a run with nowhere to break has to continue on the next
    row rather than run past the edge. See :func:`_fold_offsets`.

    A stream nobody is watching keeps the line as it arrived: a file has no width
    to fold to, an output redirected to it has to stay greppable, and folded text
    would no longer be the text the tool wrote.
    """
    if not _is_terminal():
        _print(line)
        return

    # Tabs are expanded first: a tab is one character but however many columns
    # it takes to reach the next tab stop, and the console would expand it after
    # the fold had already counted it as nothing. Tab-separated output is common
    # enough (a table, a tab in a log line) that it must not be the way a row
    # overruns the width.
    line = line.copy()
    line.expand_tabs()

    # The line arrives already indented under its label, and the fold happens in
    # the width that is left. The continuation rows go under ``continued`` --
    # the indentation of the line itself by default, or the label's width when
    # the caller wants them to line up under the value rather than under the
    # label.
    plain = line.plain
    indent = plain[: len(plain) - len(plain.lstrip(" "))]
    body = line[len(indent) :]
    first_width = _term_width() - cell_len(indent)
    continued = continued or indent
    width = _term_width() - cell_len(continued)
    if min(width, first_width) < MIN_FOLD_WIDTH:
        _print(line)
        return

    offsets = _fold_offsets(body.plain, width, first_width=first_width)
    if not offsets:
        # It fits: written as it arrived, no row rebuilt.
        _print(line)
        return
    for number, piece in enumerate(body.divide(offsets)):
        row = Text(indent if number == 0 else continued)
        row.append_text(piece)
        _print(row)


def _fold_offsets(plain: str, width: int, *, first_width: int | None = None) -> list[int]:
    """Return the character indices at which ``plain`` is broken to fit ``width``.

    A row ends after the last space that fits, so prose is broken between words.
    A word too long to fit on a row of its own -- a hash, a URL, a base64 blob, a
    path, a whole expression -- is filled to the edge of the row it is on and
    continued on the next, because the part past the edge is the part a reader
    needs. The space a break lands on ends the row it is on rather than being
    swallowed, so the text keeps every character it arrived with, and a wide
    character that does not fit moves whole to the next row instead of being
    split across two.

    ``first_width`` is the room the first row has when it differs from the rest:
    a one-line argument starts after its ``[name]`` and continues under it, in
    the narrower width that the label leaves.
    """
    offsets: list[int] = []
    start = 0
    length = len(plain)
    while start < length:
        row_width = first_width if first_width is not None and not offsets else width
        used = 0
        index = start
        last_space = -1
        while index < length:
            cells = cell_len(plain[index])
            if used + cells > row_width:
                break
            used += cells
            if plain[index] == " ":
                last_space = index
            index += 1
        if index >= length:
            break  # what is left fits, so there is nothing more to break
        if index == start:
            index += 1  # a character wider than a whole row still has to pass
        elif last_space >= start and _word_ends_within(plain, last_space + 1, width):
            index = last_space + 1  # break between words, keeping the space
        offsets.append(index)
        start = index
    return offsets


def _word_ends_within(plain: str, start: int, width: int) -> bool:
    """Whether the word at ``start`` ends -- at a space or the end -- within ``width``.

    A word that does not is longer than a row of its own, and breaking before it
    would leave the rest of the row empty and push the long word onto a row where
    it still does not fit.
    """
    used = 0
    for char in plain[start:]:
        if char == " ":
            return True
        used += cell_len(char)
        if used > width:
            return False
    return True


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


def _working_line(label: str, seconds: float) -> Text:
    """The status line of a running operation: ``gpt-4o is thinking... 2.5s``.

    The line starts at the left edge, without a stamp: what it has to say is how
    long the operation has been running, and the rule printed before it already
    says when it started. A stamp here would be the one number on a line that is
    redrawn ten times a second that never needed to move -- and it would leave the
    label that much less room before the line has to wrap.
    """
    body = Text()
    body.append(label, style="yellow")
    body.append(" ")
    body.append(format_duration(seconds), style="bold green")
    return body


@contextmanager
def working(label: str) -> Iterator[Timer]:
    """Announce a blocking operation and show how long it has been running.

    Inside a terminal the line redraws ten times a second, so it reads as alive
    rather than as a number that happens to change::

        gpt-4o is thinking... 0.0s
        gpt-4o is thinking... 3.2s

    A counter that keeps moving is what distinguishes a slow provider from a hung
    one, and the final reading stays on screen as the operation's duration. The
    line is not stamped: it is progress, not an event, and the rule printed
    before the operation carries the time it started at.

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
            # A file gets the line itself and nothing else.
            _print(_working_line(label, timer.elapsed()))
        return

    def render() -> Text:
        """The line as it should look right now -- read at every refresh."""
        return _working_line(label, timer.elapsed())

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
        # Leaving the block draws the last frame -- the finished reading -- and
        # ends the line, so there is nothing to restore here. Whatever ended the
        # block -- the answer, an error, Ctrl+C -- the line is closed either way.
        yield timer


# ── Verbatim text ──────────────────────────────────────────────────


def print_verbatim(text: str) -> None:
    """Write ``text`` to stdout exactly as it is, and let the terminal wrap it.

    Text that is going to be copied elsewhere -- the conversation, as ``/raw``
    shows it -- must not be re-rendered: no Markdown (bullets, emphasis, syntax
    colour), no stamps, no indentation added, no newlines of our own. Copying the
    screen is then copying the text.

    So the console here is built for this one write rather than being the
    transcript's: it writes plain text, with nothing to interpret and nothing to
    highlight. Read at the time of the call, it also follows ``sys.stdout``
    wherever that points -- the terminal, a redirected file -- which is where the
    rest of the transcript goes too.

    ``soft_wrap=True`` at the print call is what leaves the wrapping to the
    terminal: a line too long for the terminal is folded on screen while the line
    itself stays one line, so a paragraph is copied with its Markdown intact
    instead of arriving broken where ``rich`` would have wrapped it. And nothing
    is added to the text either -- its own trailing newline ends the block, and
    one is supplied only when the caller's text has none, so that whatever is
    printed next starts on a line of its own.
    """
    console = Console(
        file=sys.stdout,
        markup=False,
        highlight=False,
        emoji=False,
        # The opposite of the transcript console's setting: text written through
        # this one is a candidate for wrapping at all (see the docstring).
        soft_wrap=False,
    )
    console.print(text, end="" if text.endswith("\n") else "\n", soft_wrap=True)
    sys.stdout.flush()


# ── Reports ────────────────────────────────────────────────────────


def report_info(message: str) -> None:
    """Report an informational message."""
    line = _stamped(Text(message), label="INFO:", label_style="bold cyan")
    _print(line)


def report_error(message: str) -> None:
    """Report an error message."""
    line = _stamped(Text(message), label="[ERROR]", label_style="bold red")
    _print(line)
