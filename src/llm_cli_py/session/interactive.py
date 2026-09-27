"""Interactive chat loop over a plain ``input()`` prompt.

The loop is deliberately thin: read a line, hand it to the session, repeat. All
user-visible output goes through ``ui.display`` so the transcript style lives in
one place, and the session itself (``ActiveSession``) owns everything about
talking to the model.

Input is read from the terminal with ``input()``; there is no other way in (no
stdin piping, no history file, no slash commands). A stray Ctrl+C at the prompt
returns to it, with the half-typed line abandoned; end of input ends the
session -- Ctrl+D on Linux/macOS and Ctrl+Z then Enter on Windows, where the
terminal reports end-of-file instead. Neither key makes the terminal emit the
newline that Enter would, so ``read_prompt`` prints one on the way out: that
keeps the next output -- a rule, or the shell prompt after the CLI exits -- from
being appended to ``> ^C``.
"""

from __future__ import annotations

from .. import ui
from .session import ActiveSession

PROMPT_TEXT = "> "


def read_prompt() -> str:
    """Read one line of prompt text from the terminal.

    A line is opened here (the ``> `` prompt), so it is closed here too: Ctrl+C
    and Ctrl+D / Ctrl+Z end ``input()`` without the newline Enter would send,
    leaving the cursor after the prompt text. Printing that newline before
    propagating keeps the next thing written -- the rule of the following turn,
    or the shell prompt once the CLI exits -- from starting on the ``> ^C``
    line. The half-typed line itself is abandoned, not resumed.

    The exceptions are deliberately not swallowed: the session loop is the one
    place that decides what an interrupt or end of input means (return to the
    prompt, and end the session, respectively).
    """
    try:
        return input(PROMPT_TEXT)
    except (KeyboardInterrupt, EOFError):
        ui.display.close_prompt_line()
        raise


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
            # A stray Ctrl+C returns to the prompt, which read_prompt already
            # left on a line of its own (Ctrl+C while a request or a tool is
            # running is handled by that operation). End of input ends the
            # session instead: Ctrl+D, or Ctrl+Z then Enter on Windows.
            continue
        except EOFError:
            break
        except Exception as e:
            ui.display.report_error(f"Unexpected error: {e}")
            ui.display.report_info("The session continues. You can try again.")
            continue
