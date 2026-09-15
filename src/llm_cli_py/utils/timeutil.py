"""Time helpers for local, human-readable timestamps."""

from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    """Return the current local time as an ISO 8601 string with UTC offset.

    Example: ``2026-09-16T12:34:56+09:00``. Local (not UTC) so that log entries
    line up with the wall clock the user sees; the offset keeps them unambiguous.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")
