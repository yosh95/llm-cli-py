"""The project instructions file: ``LLM_CLI.md``, read once at startup.

``LLM_CLI.md`` is this CLI's ``CLAUDE.md``: a file that sits in a project's
directory and says how the agent should work there -- which commands to run,
what to leave alone, how to write the answer. Finding one is not a command-line
flag but a default, so nothing has to be typed for it to apply.

What is loaded becomes the conversation's **system message**: the first message
of the run, kept ahead of every user turn (`-s` and the prompt alike) and sent
with each request. That is what a system message is for -- standing instructions
rather than a request -- and it is why the file is not merged into the first
prompt: the model is being told how to behave, not asked to do something. Being
part of the conversation, it shows up in ``/raw`` and in the log like everything
else the model was sent.

The file is read once, at startup, and exactly one file is used: the nearest
``LLM_CLI.md``, looked for in the current directory and then in each of its
parents, so running the CLI in a subdirectory of a project still finds that
project's file. A file that cannot be read is reported and the run continues
without it -- instructions are an aid, not a prerequisite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import ui

FILENAME = "LLM_CLI.md"
"""The name of the file that is looked for: ``LLM_CLI.md``."""

MAX_CHARS = 64_000
"""How much of the file is used at most, in characters.

The text is sent with *every* request of the run, so a file that has grown (a
pasted changelog, a data dump) would quietly eat the model's context. Past this
size the file is cut at a line boundary and the truncation is reported: a file
that says why it is long is more use than one that is silently ignored.
"""


@dataclass(frozen=True)
class ProjectFile:
    """The project instructions that were loaded.

    Attributes:
        path: The absolute path the text came from, so the startup report can
            name the file the instructions are actually coming from.
        text: The contents, as the system message will carry them.
    """

    path: Path
    text: str


def find(start: Path) -> Path | None:
    """Return the nearest ``FILENAME``, from ``start`` upwards, or ``None``.

    ``start`` itself comes first, then each parent directory up to the
    filesystem root, so the file that is found is the one belonging to the
    project you are in -- not one belonging to a directory above it unless
    nothing nearer has one. A directory that happens to be called ``LLM_CLI.md``
    is not a file and is skipped.
    """
    for directory in (start, *start.parents):
        candidate = directory / FILENAME
        if candidate.is_file():
            return candidate
    return None


def _capped(text: str, path: Path) -> str:
    """Cut ``text`` to ``MAX_CHARS`` at a line boundary, reporting it if cut."""
    if len(text) <= MAX_CHARS:
        return text
    cut = text.rfind("\n", 0, MAX_CHARS)
    kept = text[: cut if cut > 0 else MAX_CHARS]
    ui.display.report_error(
        f"{path} is longer than {MAX_CHARS} characters; only its first {len(kept)} were loaded."
    )
    return kept


def load(start: Path | None = None) -> ProjectFile | None:
    """Return the project instructions to start the conversation with.

    ``start`` defaults to the current directory (tests and embedders pass their
    own). ``None`` means there is nothing to load: no file was found, or the one
    that was found holds nothing but whitespace -- an empty file states nothing
    and so adds no message to the conversation.

    A file that cannot be read (a permissions problem, a binary file that is not
    UTF-8) is reported here and the run continues without it: losing the
    instructions must not lose the session.
    """
    path = find((start if start is not None else Path.cwd()).absolute())
    if path is None:
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        ui.display.report_error(f"Could not read the project instructions {path}: {e}")
        ui.display.report_info("Continuing without them; the session is unaffected.")
        return None

    text = _capped(text, path)
    if not text.strip():
        return None
    # The surrounding blank lines are the file's, not the message's: what goes on
    # the wire starts at the first line that says something and ends after the
    # last one.
    return ProjectFile(path=path, text=text.strip() + "\n")
