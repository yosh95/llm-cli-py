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
        def __init__(self) -> None:
            self.error: str | None = None

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr in ("run", "Popen", "call", "check_call", "check_output")
            ):
                has_shell_true = False
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        has_shell_true = True
                        break
                if not has_shell_true:
                    return

                if not node.args:
                    return
                first_arg = node.args[0]
                if not isinstance(first_arg, ast.List):
                    return

                # Quote the first element that a shell would treat specially:
                # these are the arguments the call site most likely meant to
                # pass and the shell will silently ignore.
                suspect = next(
                    (
                        str(elt.value)
                        for elt in first_arg.elts
                        if isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                        and any(meta in elt.value for meta in _SHELL_META)
                    ),
                    first_arg.elts[0] if first_arg.elts else "",
                )
                lineno = getattr(node, "lineno", "?")
                self.error = (
                    f"[L{lineno}] subprocess.{node.func.attr}(..., shell=True) "
                    f"was NOT run\n"
                    f"  Problem: shell=True takes one command *string*, so the "
                    f"remaining list items are ignored\n"
                    f"           ({suspect!r} would be dropped or mishandled)\n"
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


def execute_python(
    code: str,
) -> ExecResult | ToolError:
    """Execute Python code in a subprocess and return the result.

    The child is started in its own session (``start_new_session=True``), so a
    Ctrl+C in the terminal is *not* delivered to the code being run -- it would
    otherwise never stop on its own. On ``KeyboardInterrupt`` the child's whole
    process group is killed first (the code plus everything it spawned) and the
    interrupt is then re-raised, so the prompt comes back with nothing left
    running in the background.

    Runs without a timeout by design; the user interrupts with Ctrl+C.

    Code that calls ``subprocess`` with ``shell=True`` and an argv list is
    refused before anything runs (see ``_check_shell_true_with_list``), because
    the shell would ignore everything after the first element.

    Args:
        code: The Python code to execute.

    Returns:
        ExecResult on completion, ToolError on failure.

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
            # skips the executed code, which is killed explicitly below instead.
            start_new_session=True,
        )
        stdout, stderr = proc.communicate()
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)
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

PYTHON_TOOL_DESCRIPTION = "Execute Python code in a subprocess and return stdout/stderr."
