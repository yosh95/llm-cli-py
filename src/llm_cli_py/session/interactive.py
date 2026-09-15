"""Interactive chat session using the shared prompt_toolkit session."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable

import tomli_w

from .. import ui
from ..consts import ENV_CHAT_LOG_APPEND, ENV_CHAT_LOG_FILE
from ..models import DataSource, Message
from .prompt import prompt
from .session import ActiveSession

_CHAT_LOG_TRUTHY = ("1", "true", "yes", "on")


def _get_prompt_text() -> str:
    return "> "


# ── Command handlers ──────────────────────────────────────────────


def _cmd_help(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    commands = [
        ("/help, /h", "Show this help message"),
        ("/quit, /q", "Exit the session"),
        ("/info, /i", "Show session info (API URL, model, tools)"),
        ("/dump", "Dump conversation history as TOML to stdout"),
    ]
    for cmd, desc in commands:
        ui.display.print_info(cmd, desc)
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
    return entry


def _render_messages(messages: list[Message]) -> str:
    """Render messages as TOML ``[[message]]`` tables.

    One table per message keeps the file appendable (see ``ChatLogWriter``) and
    stays readable for long, multi-line content -- tool results and the system
    prompt would otherwise collapse into one enormous line.
    """
    return "".join(
        "[[message]]\n" + tomli_w.dumps(_message_to_dict(m), multiline_strings=True) for m in messages
    )


def _dump_toml(session: ActiveSession) -> str:
    """Render the conversation history as TOML (same content as ``/dump``)."""
    return _render_messages(session.client.state.conversation)


class ChatLogWriter:
    """Incrementally persist the conversation to the chat log file.

    Called after every message appended to the history (user turn, assistant
    answer, tool result) instead of only when the session ends, so the log is
    already on disk when the session dies abruptly (crash, SIGKILL, power loss).
    Failures are reported once and never interrupt the chat.

    Two modes:

    - **snapshot** (default): the file always mirrors the current session. It is
      rewritten atomically (temp file + ``os.replace``) so a reader never sees a
      half-written log.
    - **append** (``LLM_CLI_CHAT_LOG_APPEND=1``): only newly added messages are
      appended as ``[[message]]`` tables. Earlier sessions stay in the same file
      and the file remains valid TOML overall (handy with a per-day filename).
    """

    def __init__(self, path: str, *, append: bool = False) -> None:
        self.path = path
        self.append = append
        self._written = 0
        """Number of messages already on disk (drives append mode)."""
        self._last_error: str | None = None

    def write(self, messages: list[Message]) -> None:
        """Write the conversation (or just its new tail) to disk."""
        try:
            if self.append:
                self._append(messages)
            else:
                self._write_snapshot(messages)
        except OSError as e:
            self._report_once(f"Failed to write chat log to '{self.path}': {e}")
        self._written = len(messages)

    def _write_snapshot(self, messages: list[Message]) -> None:
        tmp_path = f"{self.path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(_render_messages(messages))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.path)
        except OSError:
            with contextlib.suppress(OSError):  # best-effort temp-file cleanup
                os.unlink(tmp_path)
            raise

    def _append(self, messages: list[Message]) -> None:
        text = _render_messages(messages[self._written :])
        if not text:
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    def _report_once(self, message: str) -> None:
        if message == self._last_error:
            return
        self._last_error = message
        ui.display.report_warning(message)


_chat_log_writer: ChatLogWriter | None = None


def _attach_chat_log(session: ActiveSession) -> None:
    """Register the incremental chat-log writer on the session's state.

    If ``LLM_CLI_CHAT_LOG_FILE`` is unset (or empty), no writer is attached and
    nothing is saved. The history collected before the loop started (system
    prompt, initial ``-s`` sources) is flushed immediately.
    """
    global _chat_log_writer
    log_path = os.environ.get(ENV_CHAT_LOG_FILE, "").strip()
    if not log_path:
        _chat_log_writer = None
        return
    append = os.environ.get(ENV_CHAT_LOG_APPEND, "").strip().lower() in _CHAT_LOG_TRUTHY
    _chat_log_writer = ChatLogWriter(log_path, append=append)
    session.client.state.on_change = lambda: _save_chat_log(session)
    _save_chat_log(session)


def _detach_chat_log(session: ActiveSession) -> None:
    """Drop the change listener so the writer is not kept alive after the loop."""
    global _chat_log_writer
    session.client.state.on_change = None
    _chat_log_writer = None


def _save_chat_log(session: ActiveSession) -> None:
    """Flush the conversation to the configured chat log file.

    Invoked on every history change (via ``ClientState.on_change``) and once more
    when the session ends. Does nothing when no log file is configured.
    """
    if _chat_log_writer is None:
        return
    _chat_log_writer.write(session.client.state.conversation)


def _cmd_dump(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    print(_dump_toml(session))
    return None


# ── Command dispatch dictionary ────────────────────────────────────

_SLASH_COMMANDS: dict[str, Callable[[ActiveSession, str], str | None]] = {
    "h": _cmd_help,
    "help": _cmd_help,
    "q": _cmd_quit,
    "quit": _cmd_quit,
    "exit": _cmd_quit,
    "i": _cmd_info,
    "info": _cmd_info,
    "dump": _cmd_dump,
}


def run_interactive(
    session: ActiveSession,
    initial_sources: list[DataSource] | None = None,
) -> None:
    """Run the interactive chat session loop.

    Args:
        session: The active session to drive.
        initial_sources: Optional initial inputs to process before prompting.
    """
    print("Type /h for help, /q to quit.")

    try:
        # Attach before the first turn so even the initial ``-s`` processing is
        # logged as it happens.
        _attach_chat_log(session)

        if initial_sources:
            session.process_and_print(initial_sources)

        while True:
            try:
                ui.display.print_rule()
                prompt_text = _get_prompt_text()
                user_input = prompt(prompt_text)

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    handled = _handle_slash_command(session, user_input)
                    if handled == "exit":
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
    finally:
        # Safety net: the log is already written after every message, but flush
        # once more on the way out (EOF, /quit, Ctrl+C halt, unexpected error).
        _save_chat_log(session)
        _detach_chat_log(session)


def _handle_slash_command(session: ActiveSession, input_str: str) -> str:
    """Handle a slash command. Returns 'exit' if session should terminate."""
    parts = input_str[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    handler = _SLASH_COMMANDS.get(cmd)
    if handler is None:
        ui.display.report_error(f"Unknown command: /{cmd}. Type /help for commands.")
        return ""

    result = handler(session, args)
    return result if result is not None else ""


def _handle_user_input(session: ActiveSession, text: str) -> None:
    """Process user text input through the LLM."""
    sources = [DataSource(text=text, source_type="text")]
    try:
        session.process_and_print(sources)
    except Exception as e:
        ui.display.report_error(f"Failed to process input: {e}")
