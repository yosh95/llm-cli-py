"""The clock the CLI reads: wall time for stamps, a monotonic timer for durations.

Two times are needed here and they are not interchangeable. Stamps come from the
wall clock, so a line can be lined up with a log, another terminal, or the clock
on the wall. Durations come from a monotonic clock, which never jumps backwards
when the system time is adjusted (an NTP step, a DST change) -- and a duration is
exactly what answers "is this still working?".

Both readings are injected, so a test can drive time instead of sleeping on it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime


class Timer:
    """Measures how long one operation takes, on the monotonic clock.

    ``elapsed`` may be read as often as wanted while the operation is still
    running -- that is what the live status line redraws once a second -- and it
    keeps working afterwards, so the finished line and the log can both report
    the same number.
    """

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._started = monotonic()

    def elapsed(self) -> float:
        """Seconds since this timer was created, as of right now."""
        return self._monotonic() - self._started

    def elapsed_ms(self) -> float:
        """The same duration in milliseconds, rounded to what a log is read for.

        Whole milliseconds are finer than any network or subprocess timing is
        meaningful to, and ``0.0`` is worth seeing as such rather than as a float
        with fifteen digits.
        """
        return round(self.elapsed() * 1000, 1)


class Clock:
    """The wall clock, plus how long this session has been running.

    One ``Clock`` is created per run (``ui.display`` owns it; see
    :func:`llm_cli_py.ui.display.stamp`), so its start is the session's start and
    ``uptime`` counts from there.
    """

    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._now = now or datetime.now
        self._monotonic = monotonic or time.monotonic
        self._started = self._monotonic()

    def now(self) -> datetime:
        """The current local time."""
        return self._now()

    def uptime(self) -> float:
        """Seconds since this clock was created."""
        return self._monotonic() - self._started

    def timer(self) -> Timer:
        """Start timing one operation, on this clock's monotonic reading."""
        return Timer(self._monotonic)
