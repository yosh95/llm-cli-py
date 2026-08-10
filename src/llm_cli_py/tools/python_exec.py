"""Python execution tool - runs Python code in a subprocess."""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

from ..consts import DEFAULT_PYTHON_EXEC_TIMEOUT
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


def execute_python(
    code: str,
    execution_timeout_seconds: int | None = None,
) -> ExecResult | ToolError:
    """Execute Python code in a subprocess and return the result.

    Args:
        code: The Python code to execute.
        execution_timeout_seconds: TOOL-LEVEL maximum execution time in seconds
            (default: {DEFAULT_PYTHON_EXEC_TIMEOUT}). This is the wall-clock limit on
            the whole execution, applied by the tool itself -- NOT a code-level
            subprocess timeout. To control the runtime use this parameter, do NOT
            rely on subprocess.run(timeout=...) inside the code.

    Returns:
        ExecResult on completion, ToolError on failure.
    """
    # Static check: detect dangerous subprocess patterns
    danger = _check_dangerous_subprocess(code)
    if danger:
        return ToolError(error=danger)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(code)

    try:
        exec_timeout = (
            execution_timeout_seconds
            if execution_timeout_seconds is not None
            else DEFAULT_PYTHON_EXEC_TIMEOUT
        )
        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=exec_timeout,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            stdout=e.stdout if isinstance(e.stdout, str) else "",
            stderr=str(e),
            exit_code=-1,
        )
    except Exception as e:
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
        "execution_timeout_seconds": {
            "type": "integer",
            "description": (
                "TOOL-LEVEL max execution time in seconds (default: "
                f"{DEFAULT_PYTHON_EXEC_TIMEOUT}). Wall-clock limit on the WHOLE "
                "execution, enforced by the tool. Do NOT set subprocess(timeout=...) "
                "inside the code to control runtime."
            ),
        },
    },
    "required": ["code"],
}

PYTHON_TOOL_DESCRIPTION = (
    "Execute Python code in a sandboxed subprocess and return stdout/stderr.\n\n"
    "## Timeout\n"
    f"Execution is limited to {DEFAULT_PYTHON_EXEC_TIMEOUT} seconds at the TOOL level.\n"
    "Long-running operations (e.g. infinite loops, heavy computations) "
    "will be terminated by the tool and return an error.\n"
    "To change this limit, pass the tool-level `execution_timeout_seconds` parameter.\n"
    "This is a TOOL-level wall-clock limit, NOT a Python subprocess timeout. "
    "Setting subprocess.run(timeout=...) inside the code will NOT reliably stop "
    "the whole execution -- the tool-level parameter is the one that enforces the limit.\n\n"
    "## subprocess.run / Popen Warning\n"
    "Do NOT combine shell=True with a list of arguments. "
    "This will cause the process to hang.\n\n"
    "  BAD: subprocess.run(['cmd', '2>&1'], shell=True)   # <- hangs!\n"
    "  GOOD: subprocess.run('cmd 2>&1', shell=True)        # pass a string\n"
    "  GOOD: subprocess.run(['cmd'], capture_output=True)   # shell=False (recommended)\n"
    "  GOOD: subprocess.run(['cmd'], stderr=subprocess.STDOUT, shell=False)\n"
)
