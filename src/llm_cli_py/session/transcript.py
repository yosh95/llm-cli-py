"""The conversation on disk: one append-only file per run, in one directory.

With ``LLM_CLI_LOG_DIR`` unset -- the default -- nothing here is used and
nothing is written. When it is set, that directory holds one ``.jsonl`` file per
run, named after the moment the run started, and each line is one record:
a header, the run's session record, then one record per message, appended as
the conversation grows. A run that ends in an exception still leaves every
record written up to that point, and every earlier run's file is untouched --
a new run creates a new file rather than rewriting the old ones.

Nothing is read at startup: a run adds a file, so the cost of logging does not
depend on how much has been logged before. Nothing is read back into the
conversation either; the directory is where a past run is looked up afterwards,
not where the next one picks up its context.

JSON Lines (one object per line) rather than one JSON document, because the
record only ever grows at its end. Appending a line is one ``write`` of one
``bytes`` to a file opened ``O_APPEND``: no parse, no re-serialise, no temporary
file, and -- the reason the format is worth its cost -- a reader of a file
being appended to sees whole lines, and a killed run leaves at most one
truncated last line rather than an unparseable document. A turn that is
discarded is rolled back by truncating the file back to the line before it, so
what is on disk is still exactly the conversation the model will be sent.

The same conversation has a second rendering, :func:`render_markdown`, which is
what ``/raw`` shows in the terminal: the same messages, laid out as Markdown
rather than as records, for copying elsewhere.

What is written is whatever the conversation holds, the system message included:
when ``LLM_CLI.md`` supplied project instructions, the run's first record is that
``system`` message, and ``/raw`` shows it as the first section like any other
message the model was sent.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from .. import ui
from ..models import ClientState, Message, Role
from ..utils.fileio import UNUSABLE_PATH, clean_path

LOG_FORMAT_VERSION = 4
"""Version of the records below (bumped when their shape changes).

4 is one JSON Lines file per run: a ``log`` header, the run's ``session``, then
one ``message`` line per message. There is no reading of older layouts -- the
earlier one kept every run in a single JSON document, which this version cannot
add to without rewriting it, and rewriting is what the layout exists to avoid.
Such a file named by ``LLM_CLI_LOG_DIR`` is reported and left as it stands.

3 moved each run into an entry of a single document's ``sessions`` array, and 2
added ``duration_ms`` to messages that were produced by a measured operation (a
request to the model, a tool run).
"""


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
    if message.duration_ms is not None:
        record["duration_ms"] = message.duration_ms
    return record


class LogUnwritableError(Exception):
    """The directory named by ``LLM_CLI_LOG_DIR`` is not one a log can be added to.

    Raised while the run's file is being opened. ``ConversationLog`` turns this
    into a disabled log rather than an exception: an unusable directory costs the
    log, not the session.
    """


def session_record(state: ClientState, started_at: datetime) -> dict[str, object]:
    """Return the record naming the run a file holds: what it started at, and what it used.

    Written once, ahead of the first message, so a reader knows which run a file
    holds without reading any of its messages -- the same reason the file is
    named after the moment it started rather than after a counter. It comes after
    the header line, which says how to read the records rather than which run
    wrote them.
    """
    return {
        "type": "session",
        "started_at": started_at.isoformat(timespec="seconds"),
        "model": state.model,
    }


_HEADER: dict[str, object] = {"type": "log", "version": LOG_FORMAT_VERSION}
"""The first line of every file: the layout of the records after it."""


class ConversationLog:
    r"""Append this run's conversation to ``directory``, as one JSON Lines file.

    One run is one file. Starting the CLI again creates another file beside the
    ones already there rather than rewriting them: earlier runs cannot be lost to
    a later one, two runs at once write two files instead of overwriting each
    other, and nothing logged before has to be read, parsed or written back --
    which is what made the previous single-document log cost more the longer it
    had been used (every message rewrote every session ever logged).

    Each message becomes one line, appended when the client records it, so the
    file is a journal rather than a snapshot: a run that is killed keeps every
    line up to the moment it died, and a half-written last line is the most an
    append can damage. A turn that is discarded -- a tool call whose arguments did
    not parse, a tool run interrupted with Ctrl+C -- is rolled back by truncating
    that line off again, so what is on disk is still exactly the conversation the
    model will be sent next.

    The directory is read the way the user wrote it (see :func:`clean_path`):
    quotes a Windows ``set`` left in the value, surrounding blanks and a leading
    ``~`` are all handled, and ``C:\\My Logs\\runs`` is fine. It is resolved to an
    absolute path at construction, so a run that changes directory (which
    ``execute_python`` code is free to do) cannot send later writes somewhere
    else.

    A directory that cannot be used -- a path that is a file, a directory that
    cannot be created or written into -- is reported and turns logging off for
    the rest of the run, as does a write that fails (a full disk). What was
    written before that stays on disk and stays readable; the session itself is
    unaffected.
    """

    def __init__(self, directory: str | Path, *, started_at: datetime | None = None) -> None:
        self.directory = clean_path(directory)
        # The stamp the file is named after: it tells this run from the ones
        # already in the directory, and is the one the session record carries.
        self.started_at = started_at if started_at is not None else datetime.now()
        self.path: Path | None = None
        self.disabled = False
        # The file this run appends to, open for the run. Bytes, not text: what is
        # appended is one encoded line, and its length is what the offsets below
        # are counted in.
        self._handle: BinaryIO | None = None
        # The byte offset at which each message line starts, in the order the
        # conversation holds them. Rolling a turn back is a truncation to one of
        # these, which is why they are kept.
        self._offsets: list[int] = []
        self._opened = False
        self._session_written = False

    # ── Opening, appending, rolling back ───────────────────────────

    def _file_for(self) -> Path:
        """Return the name this run's file gets, without taking a name in use.

        The name is the local start time to the second. Two runs started in the
        same second would name the same file, so a numbered suffix is added until
        a name is free -- and the file is created exclusively, so that check
        cannot be raced by another run doing the same.
        """
        stamp = self.started_at.strftime("%Y%m%dT%H%M%S")
        for attempt in range(1000):
            name = f"{stamp}.jsonl" if attempt == 0 else f"{stamp}-{attempt}.jsonl"
            path = self.directory / name
            try:
                handle = path.open("xb")
            except FileExistsError:
                continue
            except UNUSABLE_PATH as e:
                msg = f"cannot be written ({e})"
                raise LogUnwritableError(msg) from e
            self._handle = handle
            return path
        msg = "no free file name for this run"
        raise LogUnwritableError(msg)

    def prepare(self) -> None:
        """Make sure this run can be logged, and open its file.

        Called once at startup, so a directory that cannot be created or written
        into says so now rather than at the end of the first turn. The run's file
        is created here, holding only the header line; the session record is added
        with the first message, when the conversation it belongs to exists.

        Never raises: a log that cannot be used is reported and switched off.
        """
        if self._opened or self.disabled:
            return
        self._opened = True
        if self.directory.exists() and not self.directory.is_dir():
            self._disable("is not a directory")
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self._file_for()
        except UNUSABLE_PATH as e:
            msg = f"cannot be used ({e})"
            self._disable(msg)
            return
        except LogUnwritableError as e:
            self._disable(str(e))
            return
        self._append(_HEADER)

    def _append(self, record: dict[str, object]) -> bool:
        """Append ``record`` as one line; return whether it was written.

        A failure -- a full disk, or on Windows a file another program is holding
        open -- is reported once and stops logging for the rest of the run, rather
        than being raised into the turn that happened to trigger it.
        """
        if self.disabled or self._handle is None:
            return False
        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            self._handle.write(line.encode("utf-8"))
            # One flush per record, not one per turn: a run killed with SIGKILL
            # still leaves the records the kernel has, and a reader tailing the
            # file sees whole lines as they are recorded.
            self._handle.flush()
        except UNUSABLE_PATH as e:
            self._disable(f"cannot be written ({e})")
            return False
        return True

    def _disable(self, reason: str) -> None:
        """Stop logging for the run, saying why -- once."""
        self.disabled = True
        self._close_handle()
        ui.display.report_error(f"Could not use the conversation log {self.directory}: {reason}")
        ui.display.report_info("Continuing without a log; the session is unaffected.")

    def _close_handle(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(*UNUSABLE_PATH):
                handle.close()

    def close(self) -> None:
        """Close this run's file. Called when the session ends; never raises."""
        self._close_handle()

    # ── What the CLI calls ─────────────────────────────────────────

    def _write_session(self, state: ClientState) -> bool:
        """Write the session record, once, before the first message.

        The model is only known with the conversation, so the record is written
        when there is one to write it for -- at the first message, not at startup.
        """
        if self._session_written:
            return True
        if not self._append(session_record(state, self.started_at)):
            return False
        self._session_written = True
        return True

    def message_recorded(self, state: ClientState, message: Message) -> None:
        """Append whatever ``state`` now holds that the file does not yet.

        Implements :class:`~llm_cli_py.base.ConversationObserver`, which is how
        the client reaches the log. The conversation is walked by index, so a
        message that never went through ``remember`` -- the ``system`` message
        the client is constructed with, when ``LLM_CLI.md`` supplied one -- is
        written too, in its place in the conversation, rather than being the one
        thing a run's file is missing. In the ordinary case this appends the one
        message just recorded: one line, one write, and the cost of logging does
        not grow with the length of the run.
        """
        if self.disabled or not self._write_session(state):
            return
        if self._handle is None:  # the session line could not be written
            return
        while len(self._offsets) < len(state.conversation):
            offset = self._handle.tell()
            message = state.conversation[len(self._offsets)]
            if not self._append({"type": "message", **message_record(message)}):
                return
            self._offsets.append(offset)

    def turn_discarded(self, state: ClientState) -> None:
        """Cut the discarded turn's lines off the file again.

        The conversation has already been rolled back when the client calls this,
        so its length is the number of messages that remain: the file is truncated
        back to where the first of the discarded ones started. The log then says
        what really happened -- a reply the model could not be held to is gone
        from the record, and so it is gone from the file.
        """
        keep = len(state.conversation)
        if self.disabled or self._handle is None or keep >= len(self._offsets):
            return
        target = self._offsets[keep]
        try:
            self._handle.flush()
            self._handle.truncate(target)
            # The offset has to be moved too: appending from where the handle
            # stands would leave the bytes past ``target`` as a hole of NULs.
            self._handle.seek(target)
        except UNUSABLE_PATH as e:
            self._disable(f"cannot be rewritten ({e})")
            return
        del self._offsets[keep:]


# ── The same conversation as Markdown ──────────────────────────────

_FENCE_CHAR = "`"


def _longest_run(text: str, char: str) -> int:
    """Return the length of the longest unbroken run of ``char`` in ``text``."""
    longest = current = 0
    for ch in text:
        current = current + 1 if ch == char else 0
        longest = max(longest, current)
    return longest


def _fenced(code: str, language: str = "") -> str:
    """Fence ``code`` with more backticks than it contains.

    A fence the length of the text's own longest run of backticks would be closed
    by it: a tool result that itself prints a fenced block would end the block
    early and spill the rest of the conversation into the document as prose.
    """
    fence = _FENCE_CHAR * max(3, _longest_run(code, _FENCE_CHAR) + 1)
    return f"{fence}{language}\n{code}\n{fence}"


def _json_block(text: str) -> str:
    """Fence a JSON payload, indented, or fence the text as it is if it is not JSON.

    Tool arguments and results are JSON strings on the wire, and indented they
    are something a reader can check -- a call that was one long line becomes a
    few short ones. A payload that does not parse (arguments the model sent
    truncated) or that is not JSON at all (the text of a tool that could not be
    found) is kept exactly as it arrived, and its block is left unlabelled
    rather than claiming to be JSON it is not.
    """
    try:
        return _fenced(json.dumps(json.loads(text), ensure_ascii=False, indent=2), "json")
    except json.JSONDecodeError:
        return _fenced(text)


def _message_heading(message: Message) -> str:
    """The section heading of one message: ``## assistant``, ``## tool (c1)``.

    A tool message names the call it answers, which is how a result is matched to
    the call above it once the two are read as text rather than as a structure.
    """
    if message.role is Role.TOOL and message.tool_call_id:
        return f"## {message.role.value} ({message.tool_call_id})"
    return f"## {message.role.value}"


def _message_body(message: Message) -> str | None:
    """The message's own text, or ``None`` when it has none.

    Text is reproduced as it stands -- a model's answer is already Markdown, and
    re-wrapping or escaping it would be editing what the model said. A tool
    message is different: its content is the JSON string the model was sent, so
    it goes into a code block, indented.
    """
    body = message.content.rstrip("\n")
    if not body.strip():
        return None
    if message.role is Role.TOOL:
        return _json_block(body)
    return body


def _tool_call_blocks(message: Message) -> list[str]:
    """One block per tool call the assistant asked for, with its call id."""
    blocks: list[str] = []
    for call in message.tool_calls or []:
        function = call["function"]
        label = f"**Tool call:** `{function['name']}`"
        if call["id"]:
            label += f" (`{call['id']}`)"
        arguments = function["arguments"]
        if arguments.strip():
            label += "\n\n" + _json_block(arguments)
        blocks.append(label)
    return blocks


def _message_markdown(message: Message) -> str:
    """Render one conversation message as a Markdown section."""
    parts = [_message_heading(message)]
    body = _message_body(message)
    if body is not None:
        parts.append(body)
    parts.extend(_tool_call_blocks(message))
    return "\n\n".join(parts)


def render_markdown(state: ClientState) -> str:
    """Render the conversation as plain Markdown, for copying out of the terminal.

    :func:`render` is the machine-readable record of the whole log; this is the
    human one, of one run's conversation. Every
    message becomes a ``## role`` section holding its text verbatim (tool calls
    and tool results as JSON code blocks), so what is pasted elsewhere is the
    conversation the model was sent, laid out -- not a re-rendering of it: no
    colour, no wrapping, no bullets that were not written by the model.

    The document opens with a heading naming the model, so a transcript pasted
    into a file or another chat still says which model produced it.
    """
    title = f"# Conversation ({state.model})" if state.model else "# Conversation"
    sections = [title, *(_message_markdown(message) for message in state.conversation)]
    return "\n\n".join(sections) + "\n"
