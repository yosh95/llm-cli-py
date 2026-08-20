"""Interactive chat session using the shared prompt_toolkit session."""

from __future__ import annotations

from collections.abc import Callable

import tomli_w

from .. import ui
from ..consts import APPROVAL_MODE_VERIFIER
from ..models import DataSource
from .prompt import prompt
from .session import ActiveSession


def _get_prompt_text() -> str:
    return "> "


# ── Command handlers ──────────────────────────────────────────────


def _cmd_help(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    commands = [
        ("/help, /h", "Show this help message"),
        ("/quit, /q", "Exit the session"),
        ("/clear, /c", "Clear conversation history"),
        ("/info, /i", "Show session info (API URL, model, verifier, tools)"),
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
    display_model = state.model if state.model else "not specified"
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
    return None


def _cmd_dump(session: ActiveSession, args: str) -> str | None:  # noqa: ARG001
    data: dict[str, object] = {
        "message": [
            {
                "role": m.role.value,
                "content": m.content,
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
) -> None:
    """Run the interactive chat session loop.

    Args:
        session: The active session to drive.
        initial_sources: Optional initial inputs to process before prompting.
    """
    print("Type /h for help, /q to quit.")

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
