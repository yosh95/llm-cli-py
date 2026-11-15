"""Tests for the Python execution tool and the tool registry."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from llm_cli_py.consts import DEFAULT_TOOL_TIMEOUT
from llm_cli_py.tools.python_exec import (
    PYTHON_TOOL_DESCRIPTION,
    _check_shell_true_with_list,
    execute_python,
)
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult, ToolError


def test_code_runs_and_stdout_is_captured() -> None:
    result = execute_python("print('hello world')")
    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "hello world"


def test_failing_code_reports_stderr_and_a_nonzero_exit_code() -> None:
    result = execute_python("raise ValueError('test error')")
    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert "ValueError" in result.stderr


def test_non_ascii_output_survives_the_subprocess_pipe() -> None:
    result = execute_python('print("\u65e5\u672c\u8a9e")')
    assert isinstance(result, ExecResult)
    assert result.stdout.strip() == "\u65e5\u672c\u8a9e"


def test_shell_true_with_an_argv_list_is_refused() -> None:
    """``shell=True`` drops every list element after the first, so it is refused."""
    result = execute_python('subprocess.run(["cmd", "--flag"], shell=True)')
    assert isinstance(result, ToolError)
    assert "shell=True" in result.error
    assert "--flag" in result.error  # names the argument the shell would drop


def test_shell_true_with_a_command_string_is_allowed() -> None:
    """A single command string is the shape ``shell=True`` actually wants."""
    assert _check_shell_true_with_list("subprocess.run('echo hi', shell=True)") is None


def test_metacharacters_in_an_argv_list_are_not_refused() -> None:
    """Regression: ``&`` in an ordinary URL is not a reason to refuse the call."""
    code = """import subprocess
subprocess.run(["curl", "-s", "https://api.example.com/search?a=1&b=2"])"""
    assert isinstance(execute_python(code), ExecResult)


def test_a_run_that_exceeds_the_timeout_is_killed_and_reported() -> None:
    """The timeout is the call's result, and the call returns at the deadline."""
    started = time.monotonic()
    result = execute_python("import time\ntime.sleep(30)", timeout=0.5)
    elapsed = time.monotonic() - started

    assert isinstance(result, ToolError)
    assert result.error.startswith("TIMEOUT")
    assert "background" in result.error  # tells the model what to do instead
    assert elapsed < 10  # killed at the deadline, not when the code felt like it


def test_a_timed_out_run_leaves_nothing_running() -> None:
    """The whole process group goes, not just the child Python process."""
    result = execute_python(
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)",
        timeout=1.0,
    )
    assert isinstance(result, ToolError)
    pid = int(result.error.split("[stdout]")[1].split()[0])
    # A killed process lingers as a zombie until its (also killed) parent's
    # death is noticed, so "gone" means gone or dead -- not "reaped by now".
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        if _is_zombie(pid):
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"the grandchild {pid} survived the timeout")


def test_the_call_returns_at_the_deadline_even_if_a_pipe_stays_open() -> None:
    """The deadline bounds the whole call, not just the time the code may run.

    Regression: a descendant that left the process group (``setsid``) keeps the
    child's stdout/stderr open after the kill. Waiting for those pipes to close
    gave the run its time back -- five seconds of it -- so the call reported the
    timeout late (a 30-second limit took 35 seconds), and it threw away the
    output it had waited for. The output must survive the shortcut.
    """
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
        " start_new_session=True)\n"
        "print('before the deadline', flush=True)\n"
        "time.sleep(60)"
    )
    started = time.monotonic()
    result = execute_python(code, timeout=1.0)
    elapsed = time.monotonic() - started

    assert isinstance(result, ToolError)
    assert result.error.startswith("TIMEOUT")
    assert elapsed < 2  # the deadline, not the deadline plus a wait on the pipe
    assert "before the deadline" in result.error


def _is_zombie(pid: int) -> bool:
    """Return True if ``pid`` exists but has already exited (a zombie entry)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


def test_output_written_before_the_timeout_is_reported() -> None:
    result = execute_python("print('started', flush=True)\nimport time\ntime.sleep(30)", timeout=0.5)
    assert isinstance(result, ToolError)
    assert "started" in result.error
    assert "incomplete" in result.error


def test_a_run_inside_the_timeout_is_unaffected() -> None:
    result = execute_python("print('done')", timeout=60)
    assert isinstance(result, ExecResult)
    assert result.stdout.strip() == "done"


def test_the_description_mentions_the_timeout_and_backgrounding() -> None:
    assert str(DEFAULT_TOOL_TIMEOUT) in PYTHON_TOOL_DESCRIPTION
    assert "background" in PYTHON_TOOL_DESCRIPTION


def test_tool_result_types_serialise() -> None:
    assert ExecResult(stdout="hello").to_dict() == {"stdout": "hello", "stderr": "", "exit_code": 0}
    assert ToolError(error="boom").to_dict() == {"error": "boom"}


def test_registry_registers_looks_up_and_advertises_a_tool() -> None:
    registry = ToolRegistry()
    registry.register("calc", "Calculate", {"type": "object"}, lambda **_: ExecResult())

    tool = registry.get("calc")
    assert tool is not None and tool.schema.name == "calc"
    assert "calc" in registry and "other" not in registry
    assert registry.get("missing") is None
    assert [s.name for s in registry.get_schemas()] == ["calc"]
    assert tool.schema.description == "Calculate"
