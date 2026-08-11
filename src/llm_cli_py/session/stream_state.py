"""Streaming turn state for the interactive session."""

from __future__ import annotations


class StreamState:
    """Tracks whether the answer display block has been opened.

    The streaming callback in ``process_and_print`` needs a stable, mutable
    holder that is *not* a loop variable, so that closing over it is safe
    (avoids late-binding of the iteration variable in a closure).
    """

    def __init__(self) -> None:
        self.answer_open: bool = False
