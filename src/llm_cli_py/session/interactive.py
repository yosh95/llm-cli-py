"""Interactive chat session using the shared prompt_toolkit session."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import tomli_w

from .. import ui
from ..consts import ENV_CHAT_LOG_APPEND, ENV_CHAT_LOG_FILE, TRUTHY_VALUES
from ..models import DataSource, Message
from .prompt import prompt
from .session import ActiveSession
from .slash import find_spec, help_rows

PROMPT_TEXT = "> "


# ── TOML rendering ────────────────────────────────────────────────


def _message_to_dict(message: Message) -> dict[str, object]:
    """Render one message as a TOML-serialisable dict (timestamps included)."""
    entry: dict[str, object] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.timestamp:
        entry["timestamp"] = message.timestamp
    if message.tool_call_id is not None:
        entry["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        # TOML cannot keep a nested array-of-tables attached to its own
        # ``[[message]]`` entry: tomli_w emits it as a sibling top-level
        # ``[[tool_calls]]``, so when the entries are concatenated (append mode)
        # the calls end up detached from the message or dropped entirely. Store
        # them as a JSON string instead -- it stays inside this message's table
        # and mirrors the OpenAI wire format, where tool-call arguments are
        # already JSON text.
        entry["tool_calls"] = json.dumps(message.tool_calls, ensure_ascii=False)
    return entry


def render_conversation(messages: list[Message]) -> str:
    """Render messages as TOML ``[[message]]`` tables.

    One table per message keeps the file appendable (see ``ChatLogWriter``) and
    stays readable for long, multi-line content -- tool results and the system
    prompt would otherwise collapse into one enormous line. Assistant tool calls
    are stored as a JSON string (see ``_message_to_dict``), which is the only
    way to keep them inside their own message table.
    """
    return "".join(
        "[[message]]\n" + tomli_w.dumps(_message_to_dict(m), multiline_strings=True) for m in messages
    )


# ── Chat log ──────────────────────────────────────────────────────


class ChatLogWriter:
    """Incrementally persist the conversation to the chat log file.

    Called after every message appended to the history (user turn, assistant
    answer, tool result) instead of only when the session ends, so the log is
    already on disk when the session dies abruptly (crash, SIGKILL, power loss).
    Failures are reported once and never interrupt the chat.

    Two modes:

    - **snapshot** (default): the file always mirrors the current session. It is
      rewritten atomically (temp file + ``replace``) so a reader never sees a
      half-written log.
    - **append** (``LLM_CLI_CHAT_LOG_APPEND=1``): only newly added messages are
      appended as ``[[message]]`` tables. Earlier sessions stay in the same file
      and the file remains valid TOML overall (handy with a per-day filename).

    Append mode still guarantees the "log == ``/dump``" invariant. The byte
    offset at which each message was written is recorded, so when a message that
    was already persisted is *rewritten in place* (an assistant turn whose
    broken tool calls are dropped) or the history shrinks, the file is rolled
    back to that point and the affected tail is written again -- without
    touching the earlier sessions that precede it.
    """

    def __init__(self, path: str, *, append: bool = False) -> None:
        self.path = Path(path)
        self.append = append
        self._written = 0
        """Number of messages already on disk (drives append mode)."""
        self._offsets: list[int] = []
        """Byte offset where each written message starts (``len == _written + 1``)."""
        self._last_error: str | None = None

    def write(self, messages: list[Message]) -> None:
        """Write the conversation (or just its new/rewritten tail) to disk."""
        try:
            if self.append:
                self._append(messages)
            else:
                self._write_snapshot(messages)
        except OSError as e:
            self._report_once(f"Failed to write chat log to '{self.path}': {e}")
        self._written = len(messages)

    def _write_snapshot(self, messages: list[Message]) -> None:
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                fh.write(render_conversation(messages))
                fh.flush()
                os.fsync(fh.fileno())
            tmp_path.replace(self.path)
        except OSError:
            with contextlib.suppress(OSError):  # best-effort temp-file cleanup
                tmp_path.unlink()
            raise

    def _resume_index(self, count: int) -> int:
        """First message index that must be (re)written for a history of ``count``.

        Normally only the messages added since the last write; when the history
        did not grow, the last message may have been repaired in place, so it is
        rewritten too.
        """
        if count > self._written:
            return self._written
        if count == self._written and count > 0:
            return count - 1
        return count

    def _truncate_to(self, offset: int) -> None:
        """Drop everything after ``offset`` bytes (best effort)."""
        try:
            if self.path.stat().st_size > offset:
                with self.path.open("r+b") as fh:
                    fh.truncate(offset)
        except OSError:
            return

    def _append(self, messages: list[Message]) -> None:
        if not self._offsets:
            # First write of this process: remember where our own content starts,
            # so a later rollback can never eat an earlier session's log.
            size = 0
            with contextlib.suppress(OSError):
                size = self.path.stat().st_size
            self._offsets = [size]

        start = max(0, min(self._resume_index(len(messages)), len(self._offsets) - 1))
        self._truncate_to(self._offsets[start])
        self.path.touch(exist_ok=True)
        offset = self.path.stat().st_size
        offsets = [*self._offsets[:start], offset]

        pieces = [render_conversation([m]) for m in messages[start:]]
        with self.path.open("ab") as fh:
            for piece in pieces:
                data = piece.encode("utf-8")
                fh.write(data)
                offset += len(data)
                offsets.append(offset)
            fh.flush()
            os.fsync(fh.fileno())
        self._offsets = offsets

    def _report_once(self, message: str) -> None:
        if message == self._last_error:
            return
        self._last_error = message
        ui.display.report_warning(message)


def _open_chat_log_writer() -> ChatLogWriter | None:
    """Build the writer from the environment, or return ``None`` when unset.

    If ``LLM_CLI_CHAT_LOG_FILE`` is unset (or empty), nothing is saved.
    """
    log_path = os.environ.get(ENV_CHAT_LOG_FILE, "").strip()
    if not log_path:
        return None
    append = os.environ.get(ENV_CHAT_LOG_APPEND, "").strip().lower() in TRUTHY_VALUES
    return ChatLogWriter(log_path, append=append)


@contextlib.contextmanager
def _chat_log_attached(session: ActiveSession) -> Iterator[None]:
    """Attach the incremental chat-log writer for the duration of the block.

    The writer is local to the call (no module-level global to leak between
    sessions/tests) and is registered as the state's change listener. The
    history collected before the loop started (system prompt, initial ``-s``
    sources) is flushed immediately, and once more on the way out -- even if the
    body raises.
    """
    writer = _open_chat_log_writer()
    if writer is None:
        yield
        return

    def save() -> None:
        writer.write(session.client.state.conversation)

    session.client.state.on_change = save
    save()
    try:
        yield
    finally:
        save()
        session.client.state.on_change = None


# ── Command handlers ──────────────────────────────────────────────


def _cmd_help(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    for aliases, description in help_rows():
        ui.display.print_info(aliases, description)
    return None


def _cmd_quit(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    return "exit"


def _cmd_info(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    state = session.client.state
    api_url = session.client.api_url
    ui.display.print_info("API URL", api_url if api_url else "not configured")
    display_model = state.model if state.model else "not specified"
    ui.display.print_info("Model", display_model)
    tools = session.ctx.tool_registry.get_tool_names()
    ui.display.print_info("Available Tools", ", ".join(tools) if tools else "None")
    ui.display.print_info("Messages", str(len(state.conversation)))
    return None


def _cmd_dump(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    print(render_conversation(session.client.state.conversation))
    return None


_HANDLERS: dict[str, Callable[[ActiveSession, str], str | None]] = {
    "help": _cmd_help,
    "quit": _cmd_quit,
    "info": _cmd_info,
    "dump": _cmd_dump,
}


def _handle_slash_command(session: ActiveSession, input_str: str) -> str:
    """Handle a slash command. Returns 'exit' if session should terminate."""
    parts = input_str[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    spec = find_spec(cmd)
    if spec is None:
        ui.display.report_error(f"Unknown command: /{cmd}. Type /help for commands.")
        return ""

    result = _HANDLERS[spec.names[0]](session, args)
    return result if result is not None else ""


def _handle_user_input(session: ActiveSession, text: str) -> None:
    """Process user text input through the LLM."""
    sources = [DataSource(text=text, source_type="text")]
    try:
        session.process_and_print(sources)
    except Exception as e:
        ui.display.report_error(f"Failed to process input: {e}")


# ── Session loop ──────────────────────────────────────────────────


def run_interactive(
    session: ActiveSession,
    initial_sources: list[DataSource] | None = None,
) -> None:
    """Run the interactive chat session loop.

    Args:
        session: The active session to drive.
        initial_sources: Optional initial inputs to process before prompting.
    """
    # Attach before the first turn so even the initial ``-s`` processing is
    # logged as it happens; the writer is flushed and detached on the way out
    # (EOF, /quit, Ctrl+C halt, unexpected error).
    with _chat_log_attached(session):
        if initial_sources:
            session.process_and_print(initial_sources)

        while True:
            try:
                ui.display.print_rule()
                user_input = prompt(PROMPT_TEXT)

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    if _handle_slash_command(session, user_input) == "exit":
                        break
                    continue

                _handle_user_input(session, user_input)

            except KeyboardInterrupt:
                ui.display.report_info("Use /quit to exit, or press Ctrl+D.")
                continue
            except EOFError:
                break
            except Exception as e:
                ui.display.report_error(f"Unexpected error: {e}")
                ui.display.report_info("The session continues. You can try again.")
                continue
