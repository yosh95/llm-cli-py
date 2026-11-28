"""Small filesystem helpers shared by the log and the prompt history."""

from __future__ import annotations

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

    * surrounding whitespace -- ``  ~/state/llm-cli/runs``;
    * one pair of surrounding quotes -- ``set LLM_CLI_LOG_DIR="C:\\My Logs\\runs"``
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
