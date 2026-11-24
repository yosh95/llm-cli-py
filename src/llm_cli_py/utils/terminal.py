"""Keeping the terminal's settings the way the user had them.

``prompt_toolkit`` puts the terminal into raw mode for the duration of a prompt --
no echo, no line editing, and no ``ISIG`` -- and restores what it found when the
prompt ends (see ``prompt_toolkit.input.vt100.raw_mode``). That is fine as long
as nothing else touches the settings in between: what the prompt puts back is
the state it captured when it started.

Tool code is exactly such a thing. ``execute_python`` runs it in a child
process, and a child that inherits this process's stdin inherits the terminal
with it, so it can put the terminal into raw mode and leave it there. The next
prompt then captures *that* as the state to return to, and from then on neither
Ctrl+C (without ``ISIG`` the key is an ordinary byte, not a signal) nor Enter
(without ``ICANON``) does what it should -- in this process, and in the shell it
is left behind when the CLI exits. Ctrl+Z is no help: it is a byte too.

``execute_python`` therefore keeps the child away from the terminal entirely
(see the ``stdin`` it passes), and this module is the second half of the
guarantee: settings can be captured before work that might disturb them, and put
back afterwards, so what the user is left with is what they started with.

There is nothing to do on Windows, which has no ``termios`` and does not
configure a console this way: every function here is then a no-op.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from typing import Any

try:  # Unix only: Windows has no terminal settings of this kind
    import termios
except ImportError:  # pragma: no cover - exercised on Windows
    termios = None  # type: ignore[assignment]


TerminalState = Any
"""What :func:`save` returned: the terminal's settings, or ``None`` for no terminal.

Opaque on purpose -- :func:`restore` is the only thing that reads it -- and
``None`` wherever there was nothing to save (Windows, or input redirected to a
file, where ``termios`` does not apply).
"""


def _terminal_fd() -> int | None:
    """Return the file descriptor of the terminal input, or ``None``.

    ``sys.stdin`` is what the CLI reads from, so its descriptor is what the
    settings belong to -- and it is only a terminal when ``isatty`` says so: a
    redirected or piped stdin has no settings to keep, and asking ``termios``
    about it would only raise.
    """
    if termios is None:
        return None
    try:
        if not sys.stdin.isatty():
            return None
        return sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):
        # A detached or closed stdin has no descriptor to ask about.
        return None


def save() -> TerminalState:
    """Capture the terminal's settings, or ``None`` when there is no terminal."""
    fd = _terminal_fd()
    if fd is None:
        return None
    try:
        return termios.tcgetattr(fd)
    except (termios.error, OSError):
        return None


def restore(state: TerminalState) -> None:
    """Put ``state`` (from :func:`save`) back, if there is anything to put back.

    Failures are ignored: the terminal may have been closed (the shell may even
    have taken it back), and a settings problem must not become the error the
    user sees instead of the one that actually happened.
    """
    if not state:
        return
    fd = _terminal_fd()
    if fd is None:
        return
    with contextlib.suppress(termios.error, OSError, ValueError):
        termios.tcsetattr(fd, termios.TCSANOW, state)


@contextlib.contextmanager
def preserved() -> Iterator[None]:
    """Put the terminal's settings back when the block ends, however it ends.

    The settings in force when the block is entered are the ones restored, so a
    raw-mode terminal left behind by something inside the block does not survive
    it -- not on success, not on an exception, and not on Ctrl+C.
    """
    state = save()
    try:
        yield
    finally:
        restore(state)
