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
from pathlib import Path

from .. import ui
from ..models import ClientState, Message, Role
from ..utils.fileio import UNUSABLE_PATH, clean_path, write_text_atomically

LOG_FORMAT_VERSION = 2
"""Version of the log layout (bumped when the shape below changes).

2 added ``duration_ms`` to messages that were produced by a measured operation
(a request to the model, a tool run). Readers of version 1 keep working: the
field is additive and absent when nothing was measured.
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


def render(state: ClientState) -> str:
    """Render the conversation as pretty-printed JSON, tool calls included.

    Pretty-printing (rather than one line per turn) is for the human who reads
    the log afterwards: a tool call's arguments and result are what a failure is
    usually diagnosed from, and they are only legible indented.
    """
    records = [message_record(message) for message in state.conversation]
    document: dict[str, object] = {
        "version": LOG_FORMAT_VERSION,
        "model": state.model,
        "conversation": records,
    }
    totals = _totals(records)
    if totals:
        document["durations_ms"] = totals
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

    :func:`render` is the machine-readable record; this is the human one. Every
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
