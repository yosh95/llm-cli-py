"""Optional on-disk transcript of the conversation, including tool calls.

With ``LLM_CLI_LOG_FILE`` unset -- the default -- nothing here is used and
nothing is written. When it is set, the file holds the conversation as it stood
after every recorded message, so a run that ends in an exception still leaves
everything that had been recorded up to that point.

The conversation is the one source of what is written: the log is
:class:`ClientState.conversation` rendered as JSON, not a second record kept in
parallel, so it cannot drift from what the model was sent.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import ui
from ..models import ClientState, Message
from ..utils.fileio import UNUSABLE_PATH, clean_path, write_text_atomically

LOG_FORMAT_VERSION = 1
"""Version of the log layout (bumped when the shape below changes)."""


def message_record(message: Message) -> dict[str, object]:
    """Return one conversation message as a JSON-serialisable record.

    The fields are the ones a message carries, so a tool turn keeps the
    ``tool_call_id`` it answers and an assistant turn keeps its ``tool_calls``:
    everything needed to read the conversation back, tool calls included. Fields
    that do not apply are omitted rather than written as ``null``.
    """
    record: dict[str, object] = {"role": message.role.value, "content": message.content}
    if message.tool_call_id is not None:
        record["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        record["tool_calls"] = message.tool_calls
    return record


def render(state: ClientState) -> str:
    """Render the conversation as pretty-printed JSON, tool calls included.

    Pretty-printing (rather than one line per turn) is for the human who reads
    the log afterwards: a tool call's arguments and result are what a failure is
    usually diagnosed from, and they are only legible indented.
    """
    document: dict[str, object] = {
        "version": LOG_FORMAT_VERSION,
        "model": state.model,
        "conversation": [message_record(message) for message in state.conversation],
    }
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


class ConversationLog:
    r"""Rewrite the conversation into ``path`` whenever a message is recorded.

    Rewriting, rather than appending, is what makes the file survive a run that
    dies mid-turn: writing the whole conversation and then replacing the old file
    with it means the log on disk is valid JSON at every instant -- either the
    previous version or the current one, never a half-written turn. A
    conversation is small (the CLI records no streaming output), so rewriting it
    is not worth a more intricate format.

    The path is read the way the user wrote it (see :func:`clean_path`): quotes
    that a Windows ``set`` left in the value, surrounding blanks and a leading
    ``~`` are all handled, and ``C:\My Logs\log.json`` is fine. It is resolved to
    an absolute path at construction, so a run that changes directory (which
    ``execute_python`` code is free to do) cannot send later writes somewhere
    else. A path that is not a regular file (``/dev/null``, a FIFO) is written to
    directly instead: replacing those is not what they are for, and ``/dev/null``
    is a reasonable way to switch the log back off without unsetting the
    variable.

    A write that fails -- a full disk, or on Windows a file another program is
    holding open -- is reported and turns logging off for the rest of the run,
    rather than being raised into the turn that happened to trigger it. What was
    written last stays on disk and stays readable.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = clean_path(path)
        self.disabled = False

    def save(self, state: ClientState) -> None:
        """Write the conversation as it currently stands.

        Never raises: losing the log is not worth losing the session, so a
        failure is reported once and logging stops. The previous contents of the
        file are left untouched by a failed write (see
        :func:`~llm_cli_py.utils.fileio.write_text_atomically`).
        """
        if self.disabled:
            return

        try:
            text = render(state)
            if self.path.exists() and not self.path.is_file():
                with self.path.open("w", encoding="utf-8") as handle:
                    handle.write(text)
            else:
                write_text_atomically(self.path, text)
        except UNUSABLE_PATH as e:
            self.disabled = True
            ui.display.report_error(f"Could not write the conversation log to {self.path}: {e}")
            ui.display.report_info("Continuing without a log; the session is unaffected.")

    def message_recorded(self, state: ClientState, _message: Message) -> None:
        """Write the conversation once a message has been recorded.

        Implements :class:`~llm_cli_py.base.ConversationObserver`, which is how
        the client reaches the log. The recorded message itself is not looked at:
        the conversation it now belongs to is written whole.
        """
        self.save(state)

    def turn_discarded(self, state: ClientState) -> None:
        """Write the conversation as it stands after a discarded turn.

        The log then says what really happened: a reply the model could not be
        held to is gone from the record, and so it is gone from the file.
        """
        self.save(state)
