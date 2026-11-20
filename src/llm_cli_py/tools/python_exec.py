"""Python execution tool - runs Python code in a subprocess."""

import ast
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from ..consts import DEFAULT_CHILD_PYTHON
from .types import ExecResult, ToolError

ENV_PYTHON_EXEC = "LLM_CLI_PYTHON_EXEC"
"""Environment variable overriding the interpreter used to run tool code.

The CLI may be installed in an isolated environment (e.g. ``uv tool install``
or ``pipx``) whose interpreter is *not* the one a user's ``python3`` resolves
to. The tool code should run with the same interpreter as the CLI itself, so
``execute_python`` prefers this override first, then ``sys.executable``, and
only falls back to ``python3`` if the CLI process has no usable interpreter
(e.g. under some embedded/frozen builds).
"""

_CHILD_ENCODING = "utf-8"
"""Encoding the child's output is read and decoded with.

The CLI may run where the locale is not UTF-8 (cp932 on Japanese Windows, say),
so the child is told to write UTF-8 (``PYTHONIOENCODING``) and both ends of the
pipe use this. Undecodable bytes are replaced rather than raised: a tool result
that fails to decode is worse than one with a stray character in it.
"""

_CHILD_ERRORS = "replace"
"""How undecodable bytes in the child's output are handled (see above)."""


_SHELL_META = {">", "<", "|", "2>&1", "2>", "1>", ">>", "2>>", ";", "&", "`", "$("}
"""Tokens that mean something to a shell but are inert inside an argv list.

Used only to make the refusal message concrete: the mere presence of one of
these is not by itself harmful (see ``_check_shell_true_with_list``).
"""


def _resolve_child_python() -> str:
    """Return the interpreter used to execute tool code.

    Preference order: ``LLM_CLI_PYTHON_EXEC`` override, the running interpreter
    (``sys.executable``), then ``python3`` on ``PATH``. The CLI runs tool code
    with *its own* interpreter so the tool sees the same installed packages.
    """
    override = os.environ.get(ENV_PYTHON_EXEC, "").strip()
    if override:
        return override
    if sys.executable:
        return sys.executable
    found = shutil.which(DEFAULT_CHILD_PYTHON)
    if found:
        return found
    msg = (
        "No Python interpreter available to run tool code: "
        "sys.executable is empty and 'python3' was not found on PATH."
    )
    raise RuntimeError(msg)


def _shell_dropped_argument(arguments: list[ast.expr]) -> str | None:
    """Name the argument a shell would ignore, for the refusal message, or ``None``.

    ``shell=True`` uses the command string -- the list's first element -- and
    ignores everything after it, so those are the arguments worth quoting back:
    preferring one carrying a shell metacharacter, whose loss is least obvious.
    ``None`` means there is no readable string to point at.
    """
    dropped = arguments[1:] or arguments
    literals = [
        element.value
        for element in dropped
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    return next(
        (text for text in literals if any(meta in text for meta in _SHELL_META)),
        literals[0] if literals else None,
    )


def _check_shell_true_with_list(code: str) -> str | None:
    """Return an explanation if ``code`` calls subprocess with ``shell=True`` and an argv list.

    That combination is almost always a mistake: ``shell=True`` hands ``argv[0]``
    to the shell as a command *string* and ignores the remaining list elements,
    so ``subprocess.run(["cmd", "2>&1"], shell=True)`` runs ``cmd`` alone -- the
    argument the code meant to pass is silently dropped. (The command string may
    also be a command the shell resolves from ``PATH`` rather than the program
    named in the list.) The process does not hang; it simply does not do what
    the call site says, which is silent enough to be worth refusing while the
    agent can still see the reason.

    What this check deliberately does not do: judge individual strings for shell
    metacharacters. ``&``, ``|`` and friends appear in perfectly ordinary
    arguments (URL query strings, log formats), and a list element is not run by
    a shell at all unless the whole command is one string.

    The list itself is not rejected: a code string is the only argument, exactly
    as the subprocess docs recommend.

    Returns:
        A human-readable refusal, or ``None`` when nothing of the sort is found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    class _Checker(ast.NodeVisitor):
        """Report the first refused ``subprocess`` call of the shape this check exists for."""

        def __init__(self) -> None:
            self.error: str | None = None

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr in ("run", "Popen", "call", "check_call", "check_output")
            ):
                has_shell_true = any(
                    kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in node.keywords
                )
                first_arg = node.args[0] if node.args else None
                if has_shell_true and isinstance(first_arg, ast.List):
                    dropped = _shell_dropped_argument(first_arg.elts)
                    shown = repr(dropped) if dropped is not None else "a non-literal argument"
                    lineno = getattr(node, "lineno", "?")
                    self.error = (
                        f"[L{lineno}] subprocess.{func.attr}(..., shell=True) was NOT run\n"
                        f"  Problem: shell=True takes one command *string*, so the "
                        f"remaining list items are ignored\n"
                        f"           ({shown} would be dropped or mishandled)\n"
                        f"  Result: the command runs differently from what the code says "
                        f"(it does not hang)\n"
                        f"  BAD: subprocess.run(['cmd', '--flag'], shell=True)\n"
                        f"  GOOD: subprocess.run(['cmd', '--flag'])\n"
                        f"        subprocess.run('cmd --flag', shell=True)  # "
                        f"single string, if a shell is really wanted\n"
                    )
                    return
            self.generic_visit(node)

    c = _Checker()
    c.visit(tree)
    return c.error


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill the child process and every process it spawned.

    ``proc`` was started with ``start_new_session=True``, so its pid is also the
    id of its own process group: one ``killpg`` takes down the child and all of
    its descendants. The direct child is signalled as well, in case it left the
    group (for example by calling ``setsid()`` itself). Failures are ignored --
    the process may already be gone.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid
    for target, kill in ((pgid, os.killpg), (proc.pid, os.kill)):
        with contextlib.suppress(OSError):
            kill(target, signal.SIGKILL)


def execute_python(code: str, explanation: str = "") -> ExecResult | ToolError:
    """Execute Python code in a subprocess and return the result.

    The child is started in its own session (``start_new_session=True``), so a
    Ctrl+C in the terminal is *not* delivered to the code being run -- it would
    otherwise never stop on its own. On ``KeyboardInterrupt`` the child's whole
    process group is killed first (the code plus everything it spawned) and the
    interrupt is then re-raised, so the prompt comes back with nothing left
    running in the background.

    The run has no time limit: the call returns when the code does, however long
    that takes. Ctrl+C is what ends a run early, and it takes down the whole
    process group before the interrupt goes on to the caller. A tool call
    therefore blocks the turn for as long as the code runs, which is why work
    that is not needed right away is better started in the background.

    Code that calls ``subprocess`` with ``shell=True`` and an argv list is
    refused before anything runs (see ``_check_shell_true_with_list``), because
    the shell would ignore everything after the first element.

    Args:
        code: The Python code to execute.
        explanation: Why this code is being run, written for the user: one
            short sentence. Required, in the schema and here, because the reason
            is part of the call rather than optional commentary: the display
            prints it in the transcript directly after the code, so a reader sees
            why the script ran and not only that it did. It is empty by default
            only so that a call arriving without one is refused with a sentence
            the model can act on, instead of a ``TypeError`` about a signature it
            cannot see.

            What this function does with it is nothing: the display prints it
            from the call's own arguments. It is validated here anyway, so the
            requirement holds whichever layer a call comes through.

    Returns:
        ExecResult on completion, ToolError on failure.

    Raises:
        KeyboardInterrupt: If the user interrupts the execution (Ctrl+C).
    """
    # A call without a reason is refused before anything runs, with the fix
    # spelled out -- the same treatment as a shell=True argv list below. The
    # refusal is a tool result, so the turn survives: the model reads what was
    # missing, sends the call again with it, and nothing has to be typed twice.
    if not explanation.strip():
        return ToolError(
            error=(
                "execute_python was NOT run: 'explanation' is missing or empty.\n"
                "  It is required: one short sentence, written for the user, saying "
                "why this code has to run --\n"
                "  it is shown in the transcript right after the code. Send the call "
                "again with it, e.g.\n"
                '  {"code": "...", "explanation": "Check which Python version is installed."}'
            )
        )

    # Refuse subprocess calls whose argv list would be ignored by the shell,
    # rather than running code that does not do what it looks like it does.
    refusal = _check_shell_true_with_list(code)
    if refusal:
        return ToolError(error=refusal)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(code)

    proc: subprocess.Popen[str] | None = None
    try:
        # Decode the child's output explicitly as UTF-8, and force the child's
        # own stdout/stderr to UTF-8 too. The defaults here are the parent's
        # locale encoding (e.g. cp932 on Japanese Windows), which would either
        # mangle UTF-8 output or raise UnicodeDecodeError in the reader thread.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            [_resolve_child_python(), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=_CHILD_ENCODING,
            errors=_CHILD_ERRORS,
            env=env,
            # Own session = own process group. The terminal's Ctrl+C therefore
            # skips the executed code, which is killed explicitly below instead;
            # Ctrl+Z at the prompt, which stops the CLI, does not reach the code
            # either -- the code keeps running while the CLI is suspended.
            start_new_session=True,
        )
        stdout, stderr = proc.communicate()
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)
    except KeyboardInterrupt:
        # Kill the child and everything it spawned, then let the interrupt
        # continue to the caller (the session loop silently returns to the
        # prompt). Waiting for the child here only waits for a process that has
        # just been SIGKILLed, so it cannot hold the prompt back.
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
            with contextlib.suppress(Exception):
                proc.wait()
        raise
    except Exception as e:
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
        return ExecResult(
            stdout="",
            stderr=str(e),
            exit_code=-1,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# Tool schema definition
PYTHON_TOOL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "The Python code to execute. Use 'print()' for output.",
        },
        # Second, and required: a call has to say why it is being made, and the
        # display prints that sentence after the code it explains.
        "explanation": {
            "type": "string",
            "description": (
                "Why this code has to run, in one short sentence written for the user. "
                "It is shown in the transcript immediately after the code, so write the "
                "reason for the call, not a restatement of what the code does."
            ),
        },
    },
    "required": ["code", "explanation"],
}

PYTHON_TOOL_DESCRIPTION = """\
Execute Python code in a subprocess and return stdout/stderr.

The call blocks until the code exits, with no time limit: it returns when the
run does. Print what you need as you go, and keep a run short enough that the
turn does not have to wait on it longer than necessary. Work that is not needed
right away is better started in the background: start it detached
(subprocess.Popen(..., start_new_session=True) with its output written to a
file) and return immediately, then check the file, the pid or a result in a
later call.

Every call also carries an "explanation": one sentence, written for the user,
saying why this code has to run. It is required, and it is shown in the
transcript right after the code -- so write the reason for the call rather than
a description of the code.
"""
