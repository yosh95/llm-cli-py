"""Python execution tool - runs Python code in a subprocess."""

import ast
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .types import ExecResult, ToolError

_SHELL_META = {">", "<", "|", "2>&1", "2>", "1>", ">>", "2>>", ";", "&", "`", "$("}


def _check_dangerous_subprocess(code: str) -> str | None:
    """Check for dangerous subprocess.run/Popen patterns that cause hangs."""
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

                for elt in first_arg.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        val = elt.value
                        for meta in _SHELL_META:
                            if meta in val:
                                lineno = getattr(elt, "lineno", "?")
                                self.error = (
                                    f"[L{lineno}] Dangerous "
                                    f"subprocess.{node.func.attr}() detected\n"
                                    f"  Problem: shell=True with list argument "
                                    f"containing shell meta-character '{meta}'\n"
                                    f"           ({val!r})\n"
                                    f"  Result: The process will hang indefinitely\n"
                                    f"  BAD: subprocess.run(['cmd', '{meta}'], shell=True)\n"
                                    f"  GOOD: subprocess.run(['cmd'], capture_output=True)\n"
                                )
                                return
            self.generic_visit(node)

    c = _Checker()
    c.visit(tree)
    return c.error


def _kill_process_group(proc: subprocess.Popen) -> None:
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

    Args:
        code: The Python code to execute.

    Returns:
        ExecResult on completion, ToolError on failure.

    Raises:
        KeyboardInterrupt: If the user interrupts the execution (Ctrl+C).
    """
    # Static check: detect dangerous subprocess patterns
    danger = _check_dangerous_subprocess(code)
    if danger:
        return ToolError(error=danger)

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
            [sys.executable, str(tmp_path)],
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
        # interrupt continue to the caller (the session loop prints its usual
        # "Use /quit to exit" notice and returns to the prompt).
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
