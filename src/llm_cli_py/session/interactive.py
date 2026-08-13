"""Interactive chat session using prompt_toolkit."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import tomli_w
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

from .. import ui
from ..consts import APPROVAL_MODE_VERIFIER
from ..models import DataSource
from .input_backend import InputBackend, create_backend
from .session import ActiveSession


class SlashCommandCompleter(Completer):
    """Completer for slash commands."""

    def __init__(self) -> None:
        self._commands = {
            "/help": "Show this help message",
            "/quit": "Exit the session",
            "/clear": "Clear conversation history",
            "/info": "Show session information",
            "/dump": "Dump conversation history as TOML to stdout",
            "/verifier": "Toggle verifier on/off (only affects --approval-mode verifier)",
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


def _get_prompt_text() -> str:
    return "> "


def _build_key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("c-j")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return kb


# ── Command handlers ──────────────────────────────────────────────


def _cmd_help(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    commands = [
        ("/help, /h", "Show this help message"),
        ("/quit, /q", "Exit the session"),
        ("/clear, /c", "Clear conversation history"),
        ("/info, /i", "Show session info (API URL, model, verifier, tools, token usage)"),
        ("/dump", "Dump conversation history as TOML to stdout"),
        ("/verifier [on|off], /v", "Toggle verifier (only when --approval-mode verifier)"),
    ]
    for cmd, desc in commands:
        ui.display.print_info(cmd, desc)
    return None


def _cmd_quit(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    return "exit"


def _cmd_clear(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    session.client.state.conversation.clear()
    ui.display.report_success("Conversation history cleared.")
    return None


def _cmd_info(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    state = session.client.state
    api_url = session.client.api_url
    ui.display.print_info("API URL", api_url if api_url else "not configured")
    display_model = state.model if state.model else "not specified (proxy will inject)"
    ui.display.print_info("Model", display_model)
    ui.display.print_info("Approval mode", session.ctx.approval_mode)
    v = session.ctx.verifier
    if v is None:
        ui.display.print_info("Verifier", "Not configured")
    elif session.ctx.approval_mode == APPROVAL_MODE_VERIFIER:
        ui.display.print_info("Verifier", "Enabled" if v.enabled else "Disabled")
    else:
        ui.display.print_info("Verifier", "Unused (approval mode is not 'verifier')")
    tools = session.ctx.tool_registry.get_tool_names()
    ui.display.print_info("Available Tools", ", ".join(tools) if tools else "None")
    ui.display.print_info("Messages", str(len(state.conversation)))
    prompt_tok, completion_tok, total_tok = session.token_usage
    ui.display.print_info(
        "Tokens (prompt/completion/total)", f"{prompt_tok} / {completion_tok} / {total_tok}"
    )
    return None


def _cmd_dump(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    data: dict[str, object] = {
        "message": [
            {
                "role": m.role.value,
                "content": m.content,
                **({"reasoning": m.reasoning} if m.reasoning else {}),
            }
            for m in session.client.state.conversation
        ]
    }
    print(tomli_w.dumps(data).replace("\\n", "\n"))
    return None


# ── Command dispatch dictionary ────────────────────────────────────


def _cmd_verifier(session: ActiveSession, args: str) -> str | None:
    """Toggle the verifier on/off."""
    v = session.ctx.verifier
    if session.ctx.approval_mode != APPROVAL_MODE_VERIFIER:
        ui.display.report_warning(
            f"Approval mode is '{session.ctx.approval_mode}', not 'verifier'. "
            "Restart with --approval-mode verifier to use the verifier toggle."
        )
        return None

    if v is None:
        ui.display.report_warning("Verifier is not configured.")
        return None

    arg = args.strip().lower()
    if arg in ("on", "true", "1", ""):
        v.set_enabled(True)
        ui.display.report_success("Verifier enabled.")
    elif arg in ("off", "false", "0"):
        v.set_enabled(False)
        ui.display.report_info("Verifier disabled. All tool calls allowed.")
    else:
        ui.display.report_error("Usage: /verifier [on|off]")
    return None


_SLASH_COMMANDS: dict[str, Callable[[ActiveSession, str], str | None]] = {
    "h": _cmd_help,
    "help": _cmd_help,
    "q": _cmd_quit,
    "quit": _cmd_quit,
    "exit": _cmd_quit,
    "c": _cmd_clear,
    "clear": _cmd_clear,
    "i": _cmd_info,
    "info": _cmd_info,
    "dump": _cmd_dump,
    "v": _cmd_verifier,
    "verifier": _cmd_verifier,
}


def run_interactive(
    session: ActiveSession,
    initial_sources: list[DataSource] | None = None,
    plain_input: bool = False,
) -> None:
    """Run the interactive chat session loop.

    Args:
        session: The active session to drive.
        initial_sources: Optional initial inputs to process before prompting.
        plain_input: When True, use a plain ``input()`` backend that never
            touches the terminal. Safe for automation / non-tty stdin.
    """
    completer = SlashCommandCompleter()
    bindings = _build_key_bindings()

    backend: InputBackend = create_backend(
        plain=plain_input,
        completer=completer,
        bindings=bindings,
    )

    print("Type /h for help, /q to quit.")

    if initial_sources:
        session.process_and_print(initial_sources)

    while True:
        try:
            ui.display.print_rule()
            prompt_text = _get_prompt_text()
            user_input = backend.prompt(prompt_text)

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
