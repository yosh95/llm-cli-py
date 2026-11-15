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

from ..consts import DEFAULT_CHILD_PYTHON, DEFAULT_TOOL_TIMEOUT
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


def _drain_killed_process(proc: subprocess.Popen[str]) -> tuple[str, str]:
    """Collect what a killed child already wrote, without waiting on a stuck pipe.

    The process group has been killed by the time this runs, so ``communicate``
    returns almost immediately: the pipes close as soon as the last writer is
    gone. A descendant that left the group (it called ``setsid()``) can still
    hold a pipe open, though, and waiting for it would mean waiting for exactly
    the run the timeout exists to stop -- so the pipes are dropped instead and
    the timeout stands without that output.
    """
    try:
        return proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        return "", ""


def _timeout_error(timeout_seconds: float, stdout: str, stderr: str) -> ToolError:
    """Build the ``ToolError`` the model is given when a run is killed for time.

    The message is the result of the call, so it has to say what happened, that
    re-running the same code will hit the same wall, and what to do instead --
    the run that was cut off cannot answer that for itself.
    """
    parts = [
        f"TIMEOUT: the code did not finish within {timeout_seconds:g}s and was killed "
        "(with every process it started).",
        "There is no exit code and no complete output: treat the run as unfinished, not as failed code.",
        "Do not re-run the same code unchanged. Anything that may take longer belongs "
        "in the background: start it detached and return at once, then check it in a "
        "later call. For example:",
        "    subprocess.Popen([sys.executable, 'job.py'], stdout=open('/tmp/job.log', 'w'),"
        " stderr=subprocess.STDOUT, start_new_session=True)",
        "then read /tmp/job.log (or check the pid) in the next tool call.",
    ]
    if stdout.strip() or stderr.strip():
        parts.append("[output written before the kill -- incomplete]")
        for name, text in (("stdout", stdout), ("stderr", stderr)):
            if text.strip():
                parts.append(f"[{name}]")
                parts.extend(f"  {line}" for line in text.rstrip().splitlines())
    return ToolError(error="\n".join(parts))


def execute_python(
    code: str,
    *,
    timeout: float = DEFAULT_TOOL_TIMEOUT,
) -> ExecResult | ToolError:
    """Execute Python code in a subprocess and return the result.

    The child is started in its own session (``start_new_session=True``), so a
    Ctrl+C in the terminal is *not* delivered to the code being run -- it would
    otherwise never stop on its own. On ``KeyboardInterrupt`` the child's whole
    process group is killed first (the code plus everything it spawned) and the
    interrupt is then re-raised, so the prompt comes back with nothing left
    running in the background.

    The run is limited to ``timeout`` seconds (``DEFAULT_TOOL_TIMEOUT``: 3
    minutes). A run that outlives it is killed -- with everything it spawned --
    and the timeout comes back as a :class:`ToolError`, which is what the model
    is told: a tool call blocks the turn, so a runaway script has to end it
    rather than hang it. Anything that legitimately takes longer is meant to be
    started in the background, and the tool description says so.

    Ctrl+C still ends a run early, whatever the remaining time.

    Code that calls ``subprocess`` with ``shell=True`` and an argv list is
    refused before anything runs (see ``_check_shell_true_with_list``), because
    the shell would ignore everything after the first element.

    Args:
        code: The Python code to execute.
        timeout: Seconds the code may run before it is killed.

    Returns:
        ExecResult on completion, ToolError on failure (timeout included).

    Raises:
        KeyboardInterrupt: If the user interrupts the execution (Ctrl+C).
    """
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
            encoding="utf-8",
            errors="replace",
            env=env,
            # Own session = own process group. The terminal's Ctrl+C therefore
            # skips the executed code, which is killed explicitly below instead;
            # Ctrl+Z at the prompt, which stops the CLI, does not reach the code
            # either -- the code keeps running while the CLI is suspended.
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)
    except subprocess.TimeoutExpired:
        # Take the process group down first: the child may be blocked in a way
        # that outlives a signal to the child alone (it is waiting on a
        # grandchild, say), and anything it spawned would otherwise keep running
        # after the model has been told the run ended.
        if proc is not None:
            _kill_process_group(proc)
        stdout, stderr = _drain_killed_process(proc) if proc is not None else ("", "")
        return _timeout_error(timeout, stdout, stderr)
    except KeyboardInterrupt:
        # Kill the child and everything it spawned, reap it, then let the
        # interrupt continue to the caller (the session loop silently returns
        # to the prompt).
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
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
    },
    "required": ["code"],
}

PYTHON_TOOL_DESCRIPTION = f"""\
Execute Python code in a subprocess and return stdout/stderr.

The call blocks until the code exits and is limited to {DEFAULT_TOOL_TIMEOUT} seconds.
Long-running work must not be run in the foreground: start it in the background
as a detached process (subprocess.Popen(..., start_new_session=True) with its
output written to a file) and return immediately, then check the file, the pid
or a result later in another call. A run that reaches the limit is killed,
together with every process it started, and the timeout is returned as this
call's result -- nothing keeps running for you afterwards.
"""
