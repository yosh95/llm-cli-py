"""Interactive chat loop using prompt_toolkit.

The loop is deliberately thin: read a line, hand it to the session, repeat. All
user-visible output goes through ``ui.display`` so the transcript style lives in
one place, and the session itself (``ActiveSession``) owns everything about
talking to the model.

Input is read from the terminal using prompt_toolkit; there is no other way in
(no stdin piping, no slash commands), and no history: every line you type at the
``> `` prompt is a new turn, and the arrow keys cannot bring an earlier turn
back. A stray Ctrl+C at the prompt returns to it, with the half-typed line
abandoned; end of input ends the session -- Ctrl+D on Linux/macOS and Ctrl+Z
then Enter on Windows, where the terminal reports end-of-file instead. Closing
the line those keys leave behind is prompt_toolkit's job: when a prompt aborts
it moves the cursor below ``> `` and terminates that line itself, so the next
output -- the rule of the following turn, or the shell prompt once the CLI exits
-- already starts on a line of its own. Nothing is printed here to add to it.
"""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.history import DummyHistory
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output

from .. import ui
from .session import ActiveSession

PROMPT_TEXT = "> "


def read_prompt(prompt_session: PromptSession[str]) -> str:
    """Read one line of prompt text from the terminal using prompt_toolkit.

    The half-typed line is abandoned, not resumed: Ctrl+C and Ctrl+D / Ctrl+Z
    propagate to the caller, which decides what an interrupt (back to the
    prompt) and end of input (end the session) mean.

    Aborting the prompt is also what closes its line -- prompt_toolkit moves the
    cursor past ``> `` and emits the newline Enter would have -- so nothing is
    printed here. A newline of our own would be a second one, leaving a blank
    line between the abandoned prompt and the next output.
    """
    return prompt_session.prompt(PROMPT_TEXT)


def _make_prompt_session() -> PromptSession[str]:
    """Create the session that reads from the terminal.

    ``DummyHistory`` keeps the prompt a plain line reader: the CLI has no
    history feature, so the arrow keys must not replay earlier turns.

    ``output`` and ``input`` are built explicitly rather than left to
    ``PromptSession``'s own defaults, which are resolved from ``sys.stdin`` /
    ``sys.stdout`` at call time and would follow them into a redirected stream.
    """
    return PromptSession(
        history=DummyHistory(),
        output=create_output(),
        input=create_input(),
    )


def handle_user_input(session: ActiveSession, text: str) -> None:
    """Process one user message through the LLM."""
    try:
        session.process_and_print(text)
    except Exception as e:
        ui.display.report_error(f"Failed to process input: {e}")


def run_interactive(
    session: ActiveSession,
    initial_prompt: str = "",
    prompt_session: PromptSession[str] | None = None,
) -> None:
    """Run the chat session loop.

    Args:
        session: The active session to drive.
        initial_prompt: Optional prompt (from the command line) processed as the
            first turn, before the terminal prompt is shown. The session stays
            interactive afterwards -- there is no mode that answers once and
            exits.
        prompt_session: Optional prompt_toolkit PromptSession to read input with.
    """
    if initial_prompt:
        session.process_and_print(initial_prompt)

    if prompt_session is None:
        prompt_session = _make_prompt_session()

    while True:
        try:
            ui.display.print_rule()
            user_input = read_prompt(prompt_session)

            if not user_input.strip():
                continue

            handle_user_input(session, user_input)

        except KeyboardInterrupt:
            # A stray Ctrl+C returns to the prompt: prompt_toolkit closed the
            # abandoned line as it aborted, so nothing has to be written here
            # (Ctrl+C while a request or a tool is running is handled by that
            # operation). End of input ends the session instead: Ctrl+D, or
            # Ctrl+Z then Enter on Windows.
            continue
        except EOFError:
            break
        except Exception as e:
            ui.display.report_error(f"Unexpected error: {e}")
            ui.display.report_info("The session continues. You can try again.")
            continue
