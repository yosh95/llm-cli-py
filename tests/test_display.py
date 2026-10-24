"""Tests for the display's clock: line stamps, the rule's uptime, the live status line."""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timedelta
from typing import TextIO

import pytest

from llm_cli_py.ui import display
from llm_cli_py.utils.clock import Clock


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


@pytest.fixture
def clocks(monkeypatch) -> Clocks:
    """Install a hand-driven clock: wall time 09:32:10, uptime 0.0 seconds."""
    clocks = Clocks()
    monkeypatch.setattr(display, "_clock", clocks.clock())
    return clocks


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


def test_the_rule_carries_the_stamp_and_the_session_uptime(clocks, monkeypatch, capsys) -> None:
    """The rule doubles as the timeline: ``[HH:MM:SS] ──── [+HH:MM:SS]``."""
    monkeypatch.setattr(display, "_term_width", lambda: 40)
    clocks.advance(3725)

    display.print_rule()

    line = capsys.readouterr().out.rstrip("\n")
    # The wall clock moved with the monotonic one, so the stamp shows the later
    # time while the uptime shows how long it has been since the session started.
    assert line.startswith("[10:34:15] \u2500")
    assert line.endswith(" [+01:02:05]")
    assert display._display_width(line) == 40


@pytest.mark.usefixtures("clocks")
def test_the_rule_fills_the_terminal_exactly(monkeypatch, capsys) -> None:
    """No wrapping: the rule's width is the terminal's, wide characters included."""
    monkeypatch.setattr(display, "_term_width", lambda: 60)

    display.print_rule()

    line = capsys.readouterr().out.rstrip("\n")
    assert display._display_width(line) == 60
    assert re.fullmatch(r"\[\d{2}:\d{2}:\d{2}\] \u2500+ \[\+\d{2}:\d{2}:\d{2}\]", line)


@pytest.mark.usefixtures("clocks")
def test_a_narrow_terminal_drops_the_uptime_rather_than_wrapping(monkeypatch, capsys) -> None:
    """A rule that wrapped onto a second line would be worse than one without the uptime."""
    monkeypatch.setattr(display, "_term_width", lambda: 20)

    display.print_rule()

    line = capsys.readouterr().out.rstrip("\n")
    assert "+00:00:00" not in line
    assert line.startswith("[09:32:10] ") and set(line[11:]) == {"\u2500"}
    assert display._display_width(line) == 20


@pytest.mark.usefixtures("clocks")
def test_report_lines_are_stamped(capsys) -> None:
    display.report_info("hello")
    display.report_error("boom")

    assert capsys.readouterr().out == "[09:32:10] INFO: hello\n[09:32:10] [ERROR] boom\n"


@pytest.mark.usefixtures("clocks")
def test_a_tool_block_is_stamped(capsys) -> None:
    display.print_tool_call("calc", ["  [code] 1+1"])
    display.print_tool_result(["Exit code: 0"])

    out = capsys.readouterr().out
    assert "[09:32:10] [Tool] Executing tool: calc..." in out
    assert "[09:32:10] Args:" in out
    assert "[09:32:10]   Tool Result:" in out


class FakeTty(io.StringIO):
    """A stream that claims to be a terminal, so the status line redraws in place."""

    def isatty(self) -> bool:
        return True


@pytest.mark.usefixtures("clocks")
def test_working_prints_one_finished_line_when_not_a_terminal(capsys) -> None:
    """A redirected stream gets no frames, only the duration once the block is over."""
    with display.working("gpt-4o is thinking..."):
        pass

    assert capsys.readouterr().out == "[09:32:10] gpt-4o is thinking... 0.0s\n"


def test_working_reports_the_block_s_duration(clocks, capsys) -> None:
    """The number is measured on the monotonic clock, and is readable afterwards."""
    with display.working("calc running...") as timer:
        clocks.advance(2.5)

    assert display.format_duration(timer.elapsed()) == "2.5s"
    assert capsys.readouterr().out.endswith("calc running... 2.5s\n")


def test_working_redraws_in_place_on_a_terminal(clocks, monkeypatch) -> None:
    """On a terminal the line is repainted with ``\\r`` and ends with one newline."""
    monkeypatch.setattr(display, "STATUS_INTERVAL", 0.01)
    stream = FakeTty()

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(display.sys, "stdout", stream)
        with display.working("gpt-4o is thinking..."):
            clocks.advance(2.0)

    written = stream.getvalue()
    assert written.startswith("\r[09:32:10] gpt-4o is thinking... 0.0s")
    assert written.rstrip("\n").endswith("gpt-4o is thinking... 2.0s")
    # Every frame is the same line redrawn, not a new line: the terminal keeps the
    # last one, and only the final reading is terminated.
    assert written.endswith("\n")
    assert written.count("\n") == 1


def test_the_status_line_keeps_ticking_while_the_block_runs(clocks, monkeypatch) -> None:
    """A counter that keeps moving is what tells a slow operation from a hung one."""
    monkeypatch.setattr(display, "STATUS_INTERVAL", 0.01)
    stream = FakeTty()
    frames: list[str] = []
    original_paint = display._paint

    def record(stream_: TextIO, text: str, painted: int) -> int:
        frames.append(text)
        return original_paint(stream_, text, painted)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(display.sys, "stdout", stream)
        patched.setattr(display, "_paint", record)
        with display.working("gpt-4o is thinking..."):
            for _ in range(5):
                clocks.advance(1.0)
                time.sleep(0.02)  # let the ticker draw this reading

    durations = [frame.rsplit(" ", 1)[-1] for frame in frames]
    assert durations[-1] == "5.0s"
    assert len(set(durations)) > 2  # the counter moved while the block ran


@pytest.mark.usefixtures("clocks")
def test_an_exception_leaves_the_status_line_closed() -> None:
    """Whatever ended the block, the next output starts on a line of its own."""
    stream = FakeTty()

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(display.sys, "stdout", stream)
        with pytest.raises(RuntimeError), display.working("calc running..."):
            msg = "boom"
            raise RuntimeError(msg)

    assert stream.getvalue().endswith("\n")


def test_configure_clock_replaces_the_clock_read_at_import(monkeypatch) -> None:
    """The session's uptime counts from the run's start, not from the import."""
    clocks = Clocks()
    monkeypatch.setattr(display, "_clock", Clock(now=lambda: datetime(2000, 1, 1)))

    display.configure_clock(clocks.clock())

    assert display.stamp() == "[09:32:10]"


@pytest.mark.usefixtures("clocks")
def test_the_ticker_stops_before_the_line_is_closed(monkeypatch) -> None:
    """Once the block is over, no frame may land after the newline that closes the line."""
    monkeypatch.setattr(display, "STATUS_INTERVAL", 0.01)
    stream = FakeTty()
    in_flight: list[int] = []
    frames: list[int] = []

    real_paint = display._paint

    def slow_paint(stream_: TextIO, text: str, painted: int) -> int:
        in_flight.append(1)
        try:
            return real_paint(stream_, text, painted)
        finally:
            in_flight.pop()
            frames.append(len(frames))

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(display.sys, "stdout", stream)
        patched.setattr(display, "_paint", slow_paint)
        with display.working("calc running..."):
            time.sleep(0.05)

    # Nothing was drawn after the line was closed: the trailing newline is last.
    assert stream.getvalue().endswith("\n")
    assert not in_flight
