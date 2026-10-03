"""Interactive chat loop using prompt_toolkit.

The loop is deliberately thin: read a line, hand it to the session, repeat. All
user-visible output goes through ``ui.display`` so the transcript style lives in
one place, and the session itself (``ActiveSession``) owns everything about
talking to the model.

Input is read from the terminal using prompt_toolkit; there is no other way in
(no stdin piping, no slash commands), and every line you type at the ``> ``
prompt is a new turn. The prompt is a full prompt_toolkit session rather than a
bare line reader, so the key bindings that come with it are available: the arrow
keys (and Ctrl+R) walk back through earlier turns, Ctrl+X Ctrl+E opens the line
in your editor (``$VISUAL`` / ``$EDITOR``) and sends what you save, and Ctrl+Z
suspends the CLI where the platform has ``SIGTSTP`` to send (see
``SUSPEND_SUPPORTED``). That history is the session's own; what the model sees is
the conversation kept by ``ActiveSession``. It is kept in memory unless
``LLM_CLI_PROMPT_HISTORY_FILE`` names a file, in which case it is read from and
written to that file instead and therefore survives the run. A stray Ctrl+C at
the prompt returns to it, with the half-typed line abandoned; end of input ends the session -- Ctrl+D on
Linux/macOS and Ctrl+Z then Enter on Windows, where the terminal reports
end-of-file instead. Closing the line those keys leave behind is
prompt_toolkit's job: when a prompt aborts it moves the cursor below ``> `` and
terminates that line itself, so the next output -- the rule of the following
turn, or the shell prompt once the CLI exits -- already starts on a line of its
own. Nothing is printed here to add to it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from itertools import islice
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input, create_input
from prompt_toolkit.output import Output, create_output
from prompt_toolkit.utils import suspend_to_background_supported

from .. import ui
from ..consts import ENV_PROMPT_HISTORY_FILE
from ..utils.fileio import clean_path
from .session import ActiveSession

PROMPT_TEXT = "> "

DEFAULT_HISTORY_LIMIT = 1000
"""How many recent turns the prompt keeps for the arrow keys, at most."""

SUSPEND_SUPPORTED = suspend_to_background_supported()
"""Whether Ctrl+Z can suspend the CLI: does this platform have ``SIGTSTP``?

True on Unix, where Ctrl+Z then stops the process the way it stops any job, and
the shell (or whatever else has job control) takes it from there -- the prompt
comes back after ``fg``, with the half-typed line still on it. False on Windows,
which has no ``SIGTSTP``; there Ctrl+Z keeps ``prompt_toolkit``'s default
behaviour and nothing else about the prompt changes.
"""


class LimitedFileHistory(FileHistory):
    """``FileHistory`` that keeps only the newest ``limit`` turns.

    The unbounded ``FileHistory`` grows by one timestamped block per turn
    forever, and every one of those turns is reloaded at startup. Only the
    newest ``limit`` entries are useful at the prompt, so that is how much is
    read back -- ``None`` for no cap; the file itself is left alone.
    """

    def __init__(self, filename: Path, limit: int | None = DEFAULT_HISTORY_LIMIT) -> None:
        super().__init__(str(filename))
        self.limit = limit

    def load_history_strings(self) -> Iterable[str]:
        """Yield the stored entries, newest first, at most ``limit`` of them.

        ``super()`` already returns them newest first, so the cut is a plain
        truncation.
        """
        strings = super().load_history_strings()
        if self.limit is None:
            return strings
        return islice(strings, self.limit)


def _make_history(configured: str) -> tuple[History, Path | None]:
    """Build the prompt history for ``configured`` (the env var's value).

    Unset means memory: ``InMemoryHistory`` gives the arrow keys and Ctrl+R back
    through this run's turns without writing anything anywhere. Set means a file,
    read at startup and appended to as turns are accepted -- the same feature,
    with the history outliving the run.

    The path is read the way the user wrote it (see
    :func:`~llm_cli_py.utils.fileio.clean_path`), so blanks, a leading ``~`` and
    the quotes a Windows ``set`` leaves behind are all handled. It is then made
    absolute and its directory created, so a run that changes directory
    (``execute_python`` code is free to) cannot move where the history goes.
    """
    configured = configured.strip()
    if not configured:
        return InMemoryHistory(), None

    path = clean_path(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    return LimitedFileHistory(path), path


def make_prompt_session(
    history_file: str | Path | None = None,
    input_factory: Callable[[], Input] = create_input,
    output_factory: Callable[[], Output] = create_output,
) -> PromptSession[str]:
    """Create the session that reads from the terminal.

    The history is ``LLM_CLI_PROMPT_HISTORY_FILE`` (or ``history_file``, which
    tests pass): a file when it names one, memory when it does not.
    ``enable_open_in_editor`` turns on Ctrl+X Ctrl+E, which also applies the
    edited text as the turn, as at a readline prompt. ``enable_suspend`` turns on
    Ctrl+Z (where the platform supports it; see ``SUSPEND_SUPPORTED``), so the
    CLI can be put in the background and brought back with ``fg``.

    ``output`` and ``input`` are built explicitly rather than left to
    ``PromptSession``'s own defaults, which are resolved from ``sys.stdin`` /
    ``sys.stdout`` at call time and would follow them into a redirected stream.
    The two factories exist so a test can supply non-tty ones without a
    terminal.
    """
    if history_file is None:
        history_file = os.environ.get(ENV_PROMPT_HISTORY_FILE, "")
    history, _ = _make_history(str(history_file))

    return PromptSession(
        history=history,
        enable_open_in_editor=True,
        # Ctrl+Z suspends on Unix: prompt_toolkit stops the process with
        # SIGTSTP, exactly the key readline binds to suspend. Without SIGTSTP to
        # send (Windows) the condition is false and the key keeps
        # prompt_toolkit's own binding, so the prompt is untouched there.
        enable_suspend=SUSPEND_SUPPORTED,
        output=output_factory(),
        input=input_factory(),
    )


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
        prompt_session = make_prompt_session()

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
