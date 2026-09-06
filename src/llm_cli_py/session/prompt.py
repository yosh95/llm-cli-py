"""Single shared prompt_toolkit session.

prompt_toolkit manipulates the terminal directly (raw mode, alternate
screen buffer, its own event loop). Initializing it more than once on the
same tty can leave the terminal in a broken state, so this module creates
one ``PromptSession`` lazily and reuses it for every prompt in the CLI --
the main chat loop.

The previous ``InputBackend`` abstraction (which picked between plain
``input()`` and prompt_toolkit) has been removed: prompt_toolkit is used
directly, but the session is still owned in exactly one place (here).
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

from ..consts import ENV_PROMPT_HISTORY_FILE


class SlashCommandCompleter(Completer):
    """Completer for slash commands."""

    def __init__(self) -> None:
        self._commands = {
            "/help": "Show this help message",
            "/quit": "Exit the session",
            "/info": "Show session information",
            "/dump": "Dump conversation history as TOML to stdout",
        }

    def get_completions(
        self,
        document: Document,
        _complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        if " " not in text:
            for cmd, desc in self._commands.items():
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=f"{cmd}  ({desc})",
                    )
            return


def build_key_bindings() -> KeyBindings:
    """Build the key bindings for single-line / multiline editing."""
    kb = KeyBindings()

    @kb.add("enter")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("c-j")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return kb


_session: PromptSession[str] | None = None


def get_prompt_session() -> PromptSession[str]:
    """Return the single shared ``PromptSession``, creating it lazily.

    The session is created exactly once and reused, so prompt_toolkit's
    terminal handling is never initialized more than once.
    """
    global _session
    if _session is None:
        # Persist the input history to the file given by LLM_CLI_PROMPT_HISTORY_FILE
        # so it survives across invocations. If the env var is unset, fall back
        # to an in-memory history that only lives for this run.
        history_path = os.environ.get(ENV_PROMPT_HISTORY_FILE, "").strip()
        history = FileHistory(history_path) if history_path else InMemoryHistory()
        _session = PromptSession(
            history=history,
            completer=SlashCommandCompleter(),
            key_bindings=build_key_bindings(),
            multiline=True,
            enable_open_in_editor=True,
            vi_mode=False,
            complete_while_typing=False,
            # Allow Ctrl+Z to suspend the process to the background (like a
            # plain terminal). Without this, prompt_toolkit binds Ctrl+Z to
            # inserting a literal ^Z character instead.
            enable_suspend=True,
        )
    return _session


def prompt(text: str) -> str:
    """Prompt the user via the single shared prompt_toolkit session."""
    return get_prompt_session().prompt(text)
