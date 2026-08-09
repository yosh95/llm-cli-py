"""Streaming turn state for the interactive session."""

from __future__ import annotations


class StreamState:
    """Tracks whether the reasoning / answer display blocks have been opened.

    The streaming callbacks in ``process_and_print`` need a stable, mutable
    holder that is *not* a loop variable, so that closing over it is safe
    (avoids late-binding of the iteration variable in a closure).
    """

    def __init__(self) -> None:
        self.reasoning_open: bool = False
        self.answer_open: bool = False
