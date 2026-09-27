"""Interactive chat loop over a plain ``input()`` prompt.

The loop is deliberately thin: read a line, hand it to the session, repeat. All
user-visible output goes through ``ui.display`` so the transcript style lives in
one place, and the session itself (``ActiveSession``) owns everything about
talking to the model.

Input is read from the terminal with ``input()``; there is no other way in (no
stdin piping, no history file, no slash commands). EOF (Ctrl+D) ends the
session, and a stray Ctrl+C returns to the prompt.
"""

from __future__ import annotations

from .. import ui
from .session import ActiveSession

PROMPT_TEXT = "> "


def read_prompt() -> str:
    """Read one line of prompt text from the terminal.

    ``EOFError`` is deliberately not caught here: the session loop is the one
    place that decides what end of input means (it ends the session).
    """
    return input(PROMPT_TEXT)


def handle_user_input(session: ActiveSession, text: str) -> None:
    """Process one user message through the LLM."""
    try:
        session.process_and_print(text)
    except Exception as e:
        ui.display.report_error(f"Failed to process input: {e}")


def run_interactive(
    session: ActiveSession,
    initial_prompt: str = "",
) -> None:
    """Run the chat session loop.

    Args:
        session: The active session to drive.
        initial_prompt: Optional prompt (from the command line) processed as the
            first turn, before the terminal prompt is shown.
    """
    if initial_prompt:
        session.process_and_print(initial_prompt)

    while True:
        try:
            ui.display.print_rule()
            user_input = read_prompt()

            if not user_input.strip():
                continue

            handle_user_input(session, user_input)

        except KeyboardInterrupt:
            # A stray Ctrl+C just returns to the prompt; exit with Ctrl+D.
            continue
        except EOFError:
            break
        except Exception as e:
            ui.display.report_error(f"Unexpected error: {e}")
            ui.display.report_info("The session continues. You can try again.")
            continue
