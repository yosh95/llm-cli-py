"""Tests for the display: line stamps, the rule's uptime, the status line, styling.

What a terminal gets and what a file gets are deliberately different -- Markdown
and syntax highlighting are for a person watching, plain text is for a reader --
so each test says which one it exercises. Every test runs against a ``rich``
console over an in-memory buffer, which is also what keeps them independent of
pytest's capturing and of the terminal they happen to run in.
"""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timedelta

import pytest
from rich.console import Console

from llm_cli_py.tools.types import ExecResult, ToolError
from llm_cli_py.ui import display
from llm_cli_py.utils.clock import Clock

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class Clocks:
    """A ``Clock`` whose two clocks a test drives by hand.

    One run needs both: a wall time to stamp lines with, and a monotonic reading
    to measure durations on. They are movable independently here, so a test can
    spend a minute of uptime without any wall time at all passing -- and the
    scripted monotonic is in place *before* the ``Clock`` is built, so its start
    is the scripted one rather than the machine's.
    """

    def __init__(self) -> None:
        self.wall = datetime(2026, 10, 24, 9, 32, 10)
        self.monotonic = 1000.0

    def now(self) -> datetime:
        return self.wall

    def tick(self) -> float:
        return self.monotonic

    def advance(self, seconds: float) -> None:
        """Let ``seconds`` pass on both clocks."""
        self.monotonic += seconds
        self.wall += timedelta(seconds=seconds)

    def clock(self) -> Clock:
        return Clock(now=self.now, monotonic=self.tick)


class Capture:
    """The installed console, over a buffer, with two ways to read it."""

    def __init__(self, *, terminal: bool, width: int = 100) -> None:
        self.stream = io.StringIO()
        display.configure_console(
            Console(
                file=self.stream,
                width=width,
                force_terminal=terminal,
                soft_wrap=True,
                markup=False,
                highlight=False,
                emoji=False,
            )
        )

    @property
    def raw(self) -> str:
        """Exactly what was written, escape sequences included."""
        return self.stream.getvalue()

    @property
    def text(self) -> str:
        """What a person would see: the colour codes removed."""
        return ANSI.sub("", self.raw)


def capture(*, terminal: bool, width: int = 100) -> Capture:
    """Install a console over a buffer of the given kind and width."""
    return Capture(terminal=terminal, width=width)


@pytest.fixture(autouse=True)
def _restore_console(monkeypatch):
    """Put the module's console back after each test.

    ``capture()`` replaces a module-level object, which would otherwise follow a
    test into the next one -- and a later test's output would vanish into this
    one's buffer.
    """
    monkeypatch.setattr(display, "_console", display._console)


@pytest.fixture
def clocks(monkeypatch) -> Clocks:
    """Install a hand-driven clock: wall time 09:32:10, uptime 0.0 seconds."""
    clocks = Clocks()
    monkeypatch.setattr(display, "_clock", clocks.clock())
    return clocks


@pytest.fixture
def plain() -> Capture:
    """Output as a redirected file gets it: no colour, no re-rendering."""
    return capture(terminal=False)


@pytest.fixture
def terminal() -> Capture:
    """Output as a terminal gets it: colour and rendering on."""
    return capture(terminal=True)


# ── Values ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0.0s"),
        (3.24, "3.2s"),
        (59.9, "59.9s"),
        (60.0, "1m 00s"),
        (72.0, "1m 12s"),
        (3725.0, "1h 02m"),
        (-1.0, "0.0s"),
    ],
)
def test_a_duration_is_rendered_compactly(seconds: float, expected: str) -> None:
    assert display.format_duration(seconds) == expected


@pytest.mark.usefixtures("clocks")
def test_a_stamp_is_wall_clock_hours_minutes_seconds() -> None:
    assert display.stamp() == "[09:32:10]"


def test_the_wall_clock_is_read_per_line_not_once(clocks) -> None:
    """Two lines a second apart carry two stamps -- that is what a stamp is for."""
    first = display.stamp()
    clocks.advance(1)
    assert (first, display.stamp()) == ("[09:32:10]", "[09:32:11]")


# ── The rule ──────────────────────────────────────────────────────


def test_the_rule_carries_the_stamp_and_the_session_uptime(clocks) -> None:
    """The rule doubles as the timeline: ``[HH:MM:SS] ──── [+HH:MM:SS]``."""
    out = capture(terminal=False, width=40)
    clocks.advance(3725)

    display.print_rule()

    line = out.text.rstrip("\n")
    # The wall clock moved with the monotonic one, so the stamp shows the later
    # time while the uptime shows how long it has been since the session started.
    assert line.startswith("[10:34:15] \u2500")
    assert line.endswith(" [+01:02:05]")
    assert len(line) == 40


@pytest.mark.usefixtures("clocks")
def test_the_rule_fills_the_terminal_exactly() -> None:
    """No wrapping: the rule's width is the terminal's, wide characters included."""
    out = capture(terminal=True, width=60)

    display.print_rule()

    line = out.text.rstrip("\n")
    assert len(line) == 60
    assert re.fullmatch(r"\[\d{2}:\d{2}:\d{2}\] \u2500+ \[\+\d{2}:\d{2}:\d{2}\]", line)


@pytest.mark.usefixtures("clocks")
def test_a_narrow_terminal_drops_the_uptime_rather_than_wrapping() -> None:
    """A rule that wrapped onto a second line would be worse than one without the uptime."""
    out = capture(terminal=False, width=20)

    display.print_rule()

    line = out.text.rstrip("\n")
    assert "+00:00:00" not in line
    assert line.startswith("[09:32:10] ") and set(line[11:]) == {"\u2500"}
    assert len(line) == 20


# ── Reports ───────────────────────────────────────────────────────


@pytest.mark.usefixtures("clocks")
def test_report_lines_are_stamped(plain) -> None:
    display.report_info("hello")
    display.report_error("boom")

    assert plain.text == "[09:32:10] INFO: hello\n[09:32:10] [ERROR] boom\n"


@pytest.mark.usefixtures("clocks")
def test_a_report_is_coloured_on_a_terminal_and_plain_in_a_file(terminal) -> None:
    display.report_error("boom")

    assert "\x1b[" in terminal.raw  # the label is coloured
    assert terminal.text == "[09:32:10] [ERROR] boom\n"


# ── The assistant's answer ────────────────────────────────────────


MARKDOWN_ANSWER = "# Title\n\n- one\n- two\n\n**bold**\n"


def test_an_answer_is_rendered_as_markdown_on_a_terminal(terminal) -> None:
    """Models write Markdown; a terminal shows it as such."""
    display.print_assistant(MARKDOWN_ANSWER)

    text = terminal.text
    assert "Title" in text
    assert "\u2022 one" in text  # Markdown's bullet, not the source's "- one"
    assert "**bold**" not in text  # emphasis, not its source


def test_an_answer_is_written_verbatim_to_a_file(plain) -> None:
    """A redirected transcript is its own source: no bullets, no escapes."""
    display.print_assistant(MARKDOWN_ANSWER)

    assert "- one" in plain.text
    assert "**bold**" in plain.text
    assert "\u2022" not in plain.text
    assert "\x1b[" not in plain.raw


def test_an_answer_with_markup_characters_is_not_read_as_markup(plain) -> None:
    """A model's or a tool's own words are literal: ``[INFO]`` is not a style tag."""
    display.print_assistant("see [INFO] and [/dim]")

    assert "see [INFO] and [/dim]" in plain.text


# ── Tool calls and results ────────────────────────────────────────


@pytest.mark.usefixtures("clocks")
def test_a_tool_block_is_stamped(plain) -> None:
    display.print_tool_call("calc", {"code": "print(1)"})
    display.print_tool_result(ExecResult(stdout="1\n"))

    assert "[09:32:10] [Tool] Executing tool: calc..." in plain.text
    assert "[09:32:10] Args:" in plain.text
    assert "[09:32:10] Tool Result:" in plain.text


def test_a_one_line_argument_stays_on_one_line(plain) -> None:
    display.print_tool_call("calc", {"code": "1+1", "note": 7})  # type: ignore[dict-item]

    lines = plain.text.splitlines()
    assert "  [code] 1+1" in lines
    assert "  [note] 7" in lines


def test_a_multi_line_argument_keeps_its_own_indentation(plain) -> None:
    """The code is reproduced as written, indented under a label of its own."""
    code = "def f():\n    if True:\n        return 1"

    display.print_tool_call("execute_python", {"code": code})

    assert plain.text.splitlines()[3:] == [
        "  [code]",
        "    def f():",
        "        if True:",
        "            return 1",
    ]


def test_a_call_without_arguments_prints_no_args_block(plain) -> None:
    display.print_tool_call("calc", {})

    assert "Args:" not in plain.text


def test_code_is_highlighted_on_a_terminal_but_not_in_a_file(plain, terminal) -> None:
    """An agent showing the Python it is about to run should show it as code."""
    code = "def f():\n    return 1"

    display.print_tool_call("execute_python", {"code": code})

    assert "\x1b[" in terminal.raw
    assert "def f():" in terminal.text  # the code itself is unchanged
    assert "\x1b[" not in plain.raw


@pytest.mark.usefixtures("clocks")
def test_a_successful_result_reports_its_exit_code_and_output(plain) -> None:
    display.print_tool_result(ExecResult(stdout="done\n"))

    assert plain.text.splitlines()[2:] == [
        "    Exit code: 0",
        "    [stdout]",
        "      done",
    ]


def test_a_failed_result_is_red_on_a_terminal(terminal) -> None:
    """The outcome is visible from the first line of the block, without reading it."""
    display.print_tool_result(ExecResult(stderr="boom\n", exit_code=1))

    assert "\x1b[1;31m" in terminal.raw or "\x1b[31m" in terminal.raw  # red
    assert "Exit code: 1" in terminal.text


def test_a_tool_error_result_is_red_too(terminal) -> None:
    display.print_tool_result(ToolError(error="API limit exceeded"))

    assert "31m" in terminal.raw  # red, whatever the extra attributes
    assert "Error: API limit exceeded" in terminal.text


# ── The status line ───────────────────────────────────────────────


@pytest.mark.usefixtures("clocks")
def test_working_prints_one_finished_line_when_not_a_terminal(plain) -> None:
    """A redirected stream gets no frames, only the duration once the block is over."""
    with display.working("gpt-4o is thinking..."):
        pass

    assert plain.text.splitlines() == ["[09:32:10] gpt-4o is thinking... 0.0s"]


@pytest.mark.usefixtures("clocks")
def test_working_reports_the_block_s_duration(clocks, plain) -> None:
    """The number is measured on the monotonic clock, and is readable afterwards."""
    with display.working("calc running...") as timer:
        clocks.advance(2.5)

    assert display.format_duration(timer.elapsed()) == "2.5s"
    assert plain.text.splitlines()[-1].endswith("calc running... 2.5s")


def test_working_redraws_in_place_on_a_terminal(clocks, monkeypatch, terminal) -> None:
    """On a terminal the line is redrawn, and only the finished one is terminated."""
    monkeypatch.setattr(display, "LIVE_REFRESH_PER_SECOND", 50)

    with display.working("gpt-4o is thinking..."):
        clocks.advance(2.0)
        time.sleep(0.1)

    # The frames overwrite each other; what remains visible is the last one, and it
    # is the finished reading -- no spinner, since the run is over.
    last = terminal.text.rstrip().splitlines()[-1]
    assert last.endswith("[09:32:12] gpt-4o is thinking... 2.0s")
    assert "\u2807" not in last and "\u2839" not in last  # no spinner on the finished line
    assert "\r\x1b[2K" in terminal.raw  # each frame is erased before the next
    assert "\x1b[?25l" in terminal.raw  # the cursor was hidden while it ran
    assert "\x1b[?25h" in terminal.raw  # and shown again afterwards
    # One line on the terminal: each frame starts where the previous one began, which
    # is exactly what the "\r" + erase pairs above are for.
    assert "\n" not in terminal.text.rstrip("\n")


@pytest.mark.usefixtures("terminal")
def test_the_status_line_keeps_updating_while_the_block_runs(clocks, monkeypatch) -> None:
    """A counter that keeps moving is what tells a slow operation from a hung one."""
    monkeypatch.setattr(display, "LIVE_REFRESH_PER_SECOND", 50)
    seen: list[float] = []
    real = display._working_line

    def record(label: str, seconds: float, *, spinner: bool) -> object:
        seen.append(seconds)
        return real(label, seconds, spinner=spinner)

    monkeypatch.setattr(display, "_working_line", record)

    with display.working("gpt-4o is thinking..."):
        for _ in range(5):
            clocks.advance(1.0)
            time.sleep(0.05)  # let a refresh happen at this reading

    assert max(seen) >= 4.0
    assert len(set(seen)) > 2  # the counter moved while the block ran


@pytest.mark.usefixtures("clocks")
def test_an_exception_leaves_the_status_line_closed(terminal) -> None:
    """Whatever ended the block, the next output starts on a line of its own."""
    with pytest.raises(RuntimeError), display.working("calc running..."):
        msg = "boom"
        raise RuntimeError(msg)

    assert terminal.text.splitlines()[-1].endswith("[09:32:10] calc running... 0.0s")
    assert terminal.text.endswith("\n")


def test_an_interrupted_block_still_shows_how_long_it_ran(clocks, monkeypatch, terminal) -> None:
    """Ctrl+C ends the block too; the duration it reached is worth seeing."""

    def interrupted(**_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(display, "LIVE_REFRESH_PER_SECOND", 50)
    with pytest.raises(KeyboardInterrupt), display.working("calc running..."):
        clocks.advance(3.0)
        interrupted()

    assert terminal.text.splitlines()[-1].endswith("calc running... 3.0s")


# ── Wiring ────────────────────────────────────────────────────────


def test_configure_clock_replaces_the_clock_read_at_import(monkeypatch) -> None:
    """The session's uptime counts from the run's start, not from the import."""
    clocks = Clocks()
    monkeypatch.setattr(display, "_clock", Clock(now=lambda: datetime(2000, 1, 1)))

    display.configure_clock(clocks.clock())

    assert display.stamp() == "[09:32:10]"


def test_the_session_uses_the_one_clock_it_was_given(clocks) -> None:
    """Stamps and durations come from the same clock, so they cannot disagree."""
    timer = display._clock.timer()
    clocks.advance(1.5)

    assert display.format_duration(timer.elapsed()) == "1.5s"
    assert display.stamp() == "[09:32:11]"
