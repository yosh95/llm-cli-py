"""Small filesystem helpers shared by the log and the prompt history."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

UNUSABLE_PATH = (OSError, ValueError, RuntimeError)
"""What a bad path can raise: the filesystem, a bad value, a ``~`` with no home.

``expanduser`` signals "I cannot tell what ``~`` means" with ``RuntimeError``,
not ``OSError``, so the three are handled together wherever a path from the
environment is turned into a file.
"""


def clean_path(value: str | Path) -> Path:
    """Return ``value`` as an absolute path, as the user meant it.

    Paths reach the CLI through environment variables, so they carry whatever
    the shell put there. Three things are undone here, once, for both the log and
    the history:

    * surrounding whitespace -- ``  ~/log.json``;
    * one pair of surrounding quotes -- ``set LLM_CLI_LOG_FILE="C:\\My Logs\\log.json"``
      in ``cmd.exe`` leaves the quotes *in* the value, unlike a POSIX shell or
      PowerShell, which is how a Windows path with a space gets here;
    * a leading ``~``, expanded against the home directory (``USERPROFILE`` on
      Windows).

    The result is absolutised, so a run that changes directory cannot move where
    a file is written.
    """
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
        text = text.strip()

    if not text:
        msg = "no path given"
        raise ValueError(msg)

    # expanduser raises RuntimeError when it cannot tell what ``~`` means (no
    # home directory, or a ``~name`` that does not exist), so callers treat that
    # along with ValueError and OSError as "this path cannot be used".
    return Path(text).expanduser().absolute()


def write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, never leaving a half-written file behind.

    The text goes to a temporary file in the same directory, which is then
    renamed onto ``path``: a rename within one filesystem is atomic, so a reader
    sees either the previous contents or the new ones. That matters for a log
    that is rewritten as the conversation grows -- an interrupted run must still
    leave a file that parses.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        temp_path.replace(path)
    except BaseException:
        # Includes KeyboardInterrupt: whatever happened, the temporary file must
        # not be left lying next to the log.
        temp_path.unlink(missing_ok=True)
        raise
