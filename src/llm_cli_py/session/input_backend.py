"""Input backend abstraction for the interactive session.

``prompt_toolkit`` manipulates the terminal directly (raw / non-canonical
mode, alternate screen buffer, its own event loop and signal handling).
When an external automation harness (e.g. HTB) drives the same tty, these
can conflict and leave the terminal in a broken state (no echo, unresponsive
keyboard).

To avoid that, the prompt call is abstracted behind :class:`InputBackend` so
callers can choose between:

- :class:`PromptToolkitBackend` -- full-featured editor (history, completion,
  multiline). Best for interactive humans on a real tty.
- :class:`PlainInputBackend` -- plain ``input()`` that only reads a single
  line and never touches terminal modes or runs an event loop. Safe for
  automation and non-tty stdin.

Use :func:`create_backend` to pick one.
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings


class InputBackend:
    """Base interface for prompting the user."""

    def prompt(self, text: str) -> str:
        """Prompt the user and return their input as a single string."""
        raise NotImplementedError


class PlainInputBackend(InputBackend):
    """Backend that reads a single line with plain ``input()``.

    ``input()`` only reads one line and does not touch the terminal in raw
    mode, alternate screen, or run any event loop, so it is safe for
    automation harnesses and non-tty stdin.
    """

    def prompt(self, text: str) -> str:
        return input(text)


class PromptToolkitBackend(InputBackend):
    """Backend backed by a full-featured ``prompt_toolkit`` prompt session."""

    def __init__(
        self,
        history_path: Path,
        completer: Completer | None = None,
        bindings: KeyBindings | None = None,
    ) -> None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if not history_path.exists():
            history_path.touch()

        history = FileHistory(str(history_path))
        self._session: PromptSession[str] = PromptSession(
            history=history,
            completer=completer,
            key_bindings=bindings,
            multiline=True,
            enable_open_in_editor=True,
            vi_mode=False,
            complete_while_typing=False,
        )

    def prompt(self, text: str) -> str:
        return self._session.prompt(text)


def create_backend(
    history_path: Path,
    plain: bool = False,
    completer: Completer | None = None,
    bindings: KeyBindings | None = None,
) -> InputBackend:
    """Return an input backend.

    When ``plain`` is True, a :class:`PlainInputBackend` is returned that does
    not touch the terminal, which is safe for automation and non-tty stdin.
    Otherwise a :class:`PromptToolkitBackend` with the given completion and key
    bindings is returned.
    """
    if plain:
        return PlainInputBackend()
    return PromptToolkitBackend(
        history_path,
        completer=completer,
        bindings=bindings,
    )
