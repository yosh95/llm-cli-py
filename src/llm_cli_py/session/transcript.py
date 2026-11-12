"""Optional on-disk transcript of the conversation, including tool calls.

With ``LLM_CLI_LOG_FILE`` unset -- the default -- nothing here is used and
nothing is written. When it is set, the file holds the conversation as it stood
after every recorded message, so a run that ends in an exception still leaves
everything that had been recorded up to that point.

The conversation is the one source of what is written: the log is
:class:`ClientState.conversation` rendered as JSON, not a second record kept in
parallel, so it cannot drift from what the model was sent.

The same conversation has a second rendering, :func:`render_markdown`, which is
what ``/raw`` shows in the terminal: the same messages, laid out as Markdown
rather than as a document, for copying elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from .. import ui
from ..models import ClientState, Message, Role
from ..utils.fileio import UNUSABLE_PATH, clean_path, write_text_atomically

LOG_FORMAT_VERSION = 3
"""Version of the log layout (bumped when the shape below changes).

3 made the document cumulative: ``sessions`` holds one entry per run, so
starting the CLI again adds a session instead of replacing the file.

2 added ``duration_ms`` to messages that were produced by a measured operation
(a request to the model, a tool run). Readers of version 1 keep working: the
field is additive and absent when nothing was measured.

1 and 2 were a single session at the top level (``model`` and ``conversation``
beside ``version``). Those files are still read, and are written back as the
first entry of ``sessions``, with ``started_at`` null -- that run's start was
never recorded.
"""


class LogUnreadableError(Exception):
    """The file named by ``LLM_CLI_LOG_FILE`` is not a log that can be added to.

    Raised while reading it. ``ConversationLog`` turns this into a disabled log
    rather than an exception: the one thing that must not happen to a file
    holding earlier runs is a blind overwrite.
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


def session_record(state: ClientState, started_at: datetime | None = None) -> dict[str, object]:
    """Return one run of the CLI as the entry it becomes in the log.

    A session is the conversation that run recorded, its model, and when it
    started -- enough for a reader to tell two runs apart in a file that holds
    many. Pretty-printing (rather than one line per turn) is for the human who
    reads the log afterwards: a tool call's arguments and result are what a
    failure is usually diagnosed from, and they are only legible indented.
    """
    records = [message_record(message) for message in state.conversation]
    session: dict[str, object] = {
        "started_at": started_at.isoformat(timespec="seconds") if started_at is not None else None,
        "model": state.model,
        "conversation": records,
    }
    totals = _totals(records)
    if totals:
        session["durations_ms"] = totals
    return session


def render(sessions: Iterable[dict[str, object]]) -> str:
    """Render the sessions the log holds as one pretty-printed JSON document.

    ``sessions`` is every entry read from the file followed by this run's own;
    the document is written whole, so what is on disk is always a complete log.
    """
    document: dict[str, object] = {
        "version": LOG_FORMAT_VERSION,
        "sessions": list(sessions),
    }
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


_DURATION_LABELS = {
    Role.ASSISTANT: "llm",
    Role.TOOL: "tool",
}
"""Which bucket a measured message's time belongs to: the model's or a tool's."""


def _totals(records: list[dict[str, object]]) -> dict[str, float]:
    """Sum the measured durations per kind, for a log that can be read at a glance.

    The per-message numbers say where one slow turn went; the totals say it
    without adding up eighteen of them by hand. Measured messages only, so a log
    without any is written exactly as it was before.
    """
    totals: dict[str, float] = {}
    for record in records:
        kind = _DURATION_LABELS.get(Role(str(record["role"])))
        duration = record.get("duration_ms")
        if kind is None or not isinstance(duration, (int, float)):
            continue
        totals[kind] = round(totals.get(kind, 0.0) + float(duration), 1)
    return totals


def _check_session(entry: object, index: int) -> dict[str, object]:
    """Return ``entry`` if it is a session record, or say why it cannot be read.

    A file that has been edited by hand, truncated by a full disk, or written by
    something else must be reported rather than quietly replaced, so every entry
    is checked before anything is written back beside it.
    """
    if not isinstance(entry, dict):
        msg = f"sessions[{index}] is not an object"
        raise LogUnreadableError(msg)
    if not isinstance(entry.get("conversation"), list):
        msg = f"sessions[{index}] has no conversation list"
        raise LogUnreadableError(msg)
    return entry


def _sessions_from_document(document: object) -> list[dict[str, object]]:
    """Return the sessions an older log holds, in the layout this version writes.

    ``sessions`` is the current shape. A version 1 or 2 file holds one session at
    the top level instead; it is carried over as the first entry, so upgrading
    does not lose the run that was already there.
    """
    if not isinstance(document, dict):
        msg = "the log is not a JSON object"
        raise LogUnreadableError(msg)

    if "sessions" in document:
        sessions = document["sessions"]
        if not isinstance(sessions, list):
            msg = "'sessions' is not a list"
            raise LogUnreadableError(msg)
        return [_check_session(entry, index) for index, entry in enumerate(sessions)]

    if "conversation" in document:
        session = _check_session(
            {
                "started_at": None,
                "model": document.get("model", ""),
                "conversation": document["conversation"],
                **({"durations_ms": document["durations_ms"]} if "durations_ms" in document else {}),
            },
            0,
        )
        return [session]

    msg = "the log is neither a 'sessions' document nor a single session"
    raise LogUnreadableError(msg)


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


class ConversationLog:
    r"""Keep every run in ``path``, rewriting it whenever a message is recorded.

    The file accumulates: each run is an entry in the document's ``sessions``
    array, so starting the CLI again appends a session instead of replacing the
    file. Earlier runs are read once, at startup, and written back unchanged
    beside this run's own -- the log is where a past session is looked up
    afterwards, not where the next one picks up its context (nothing read here
    reaches the conversation; see :class:`~llm_cli_py.models.ClientState`).

    Rewriting, rather than appending, is what makes the file survive a run that
    dies mid-turn: writing the whole document and then replacing the old file
    with it means the log on disk is valid JSON at every instant -- either the
    previous version or the current one, never a half-written turn. Appending
    would also mean a killed run leaves a truncated last session behind. A log
    is small (the CLI records no streaming output), so rewriting it is not worth
    a more intricate format.

    A file that is not a log -- edited by hand, truncated by something else,
    written by another program -- is reported and left exactly as it is: a log
    that holds earlier runs is never overwritten blind. Logging then stops for
    the run, and the session carries on without it.

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

    def __init__(self, path: str | Path, *, started_at: datetime | None = None) -> None:
        self.path = clean_path(path)
        # The stamp that tells this run apart from the ones already in the file.
        self.started_at = started_at if started_at is not None else datetime.now()
        self.disabled = False
        # The sessions already on disk, read once (None until then).
        self._earlier: list[dict[str, object]] | None = None

    # ── Reading and writing ────────────────────────────────────────

    def _read_sessions(self) -> list[dict[str, object]]:
        """Return the sessions ``path`` already holds, or ``[]`` if there is no log.

        Raises:
            LogUnreadableError: If the file exists and is not a log this version can
                be added to. Its contents are not touched in that case.
        """
        # A device or a FIFO is not a log, and reading one could block, so only a
        # regular file is read -- but writing to one still happens (see _write),
        # which is what keeps /dev/null a way to switch the log off.
        if not self.path.is_file():
            if self.path.is_dir():
                msg = "is a directory"
                raise LogUnreadableError(msg)
            return []

        try:
            text = self.path.read_text(encoding="utf-8")
        except (OSError, ValueError) as e:
            msg = f"cannot be read ({e})"
            raise LogUnreadableError(msg) from e

        if not text.strip():
            return []

        try:
            document = json.loads(text)
        except json.JSONDecodeError as e:
            msg = f"is not valid JSON ({e})"
            raise LogUnreadableError(msg) from e

        return _sessions_from_document(document)

    def _write(self, sessions: list[dict[str, object]]) -> None:
        """Write ``sessions`` to ``path`` whole, atomically where it is a file."""
        text = render(sessions)
        if self.path.exists() and not self.path.is_file():
            with self.path.open("w", encoding="utf-8") as handle:
                handle.write(text)
        else:
            write_text_atomically(self.path, text)

    def _earlier_sessions(self) -> list[dict[str, object]] | None:
        """The sessions already on disk, read once; ``None`` if the log is unusable.

        A failure is reported once, here, and turns logging off for the run --
        the caller only has to check for ``None``.
        """
        if self.disabled:
            return None
        if self._earlier is None:
            try:
                self._earlier = self._read_sessions()
            except LogUnreadableError as e:
                self.disabled = True
                ui.display.report_error(f"Could not use the conversation log {self.path}: {e}")
                ui.display.report_info("Continuing without a log; the file is left as it is.")
                return None
        return self._earlier

    def _save_sessions(self, sessions: list[dict[str, object]]) -> None:
        """Write ``sessions``, or report why not and stop logging."""
        try:
            self._write(sessions)
        except UNUSABLE_PATH as e:
            self.disabled = True
            ui.display.report_error(f"Could not write the conversation log to {self.path}: {e}")
            ui.display.report_info("Continuing without a log; the session is unaffected.")

    # ── What the CLI calls ─────────────────────────────────────────

    def prepare(self) -> None:
        """Read the log that is already there, and make sure it can be written.

        Called once at startup: a path that cannot be read, or a file that is not
        a log, says so now rather than at the end of the first turn. A path with
        no file yet is created (and its directory with it) as a log with no
        sessions -- this run's first message adds one -- so a path that cannot be
        written is reported now too.

        Never raises: a log that cannot be used is reported and switched off.
        """
        if self._earlier_sessions() is None:
            return
        if not self.path.exists():
            self._save_sessions([])

    def save(self, state: ClientState) -> None:
        """Write this run's session as the conversation currently stands.

        The sessions already in the file are written back beside it, unchanged,
        so a previous run survives the restart. Never raises: losing the log is
        not worth losing the session, so a failure is reported once and logging
        stops. The previous contents of the file are left untouched by a failed
        write (see :func:`~llm_cli_py.utils.fileio.write_text_atomically`).
        """
        previous = self._earlier_sessions()
        if previous is None:
            return
        self._save_sessions([*previous, session_record(state, self.started_at)])

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
