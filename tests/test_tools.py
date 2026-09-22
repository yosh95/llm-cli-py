"""Tests for the Python execution tool, the tool registry and result types."""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from llm_cli_py.tools.python_exec import execute_python
from llm_cli_py.tools.types import ExecResult, ToolError


def _signal_main_thread_when_ready(marker: Path, timeout: float) -> None:
    """Wait for ``marker``, then interrupt the main thread like Ctrl+C does.

    ``signal.pthread_kill`` addresses the main thread specifically: that is where
    the code under test blocks in ``proc.communicate()``, i.e. exactly where a
    terminal Ctrl+C lands. Signalling later would only be noticed once the
    blocking wait returned on its own.
    """
    deadline = time.monotonic() + timeout
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    main_thread = threading.main_thread()
    if main_thread.ident is not None:
        signal.pthread_kill(main_thread.ident, signal.SIGINT)


class TestExecutePython:
    @pytest.mark.parametrize(
        ("code", "expected_stdout"),
        [
            ("print('hello world')", "hello world"),
            ("import math; print(math.sqrt(16))", "4.0"),
            ("", ""),
        ],
    )
    def test_stdout_is_captured(self, code: str, expected_stdout: str) -> None:
        result = execute_python(code)
        assert result == ExecResult(stdout=result.stdout, stderr="", exit_code=0)
        assert result.stdout.strip() == expected_stdout

    @pytest.mark.parametrize(
        ("code", "needle"),
        [
            ("raise ValueError('test error')", "ValueError"),
            ("if True print('bad')", "SyntaxError"),
        ],
    )
    def test_errors_are_reported_in_stderr(self, code: str, needle: str) -> None:
        result = execute_python(code)
        assert result.exit_code == 1
        assert needle in result.stderr

    @pytest.mark.skipif(os.name != "posix", reason="POSIX signal semantics")
    def test_ctrl_c_kills_child_and_its_descendants(self, tmp_path: Path) -> None:
        """Ctrl+C must not leave the executed code (or its children) running.

        The code spawns a process, records its pid, then sleeps. The interrupt
        is delivered while the tool is waiting, and afterwards both the code and
        the process it spawned must be gone.
        """
        marker = tmp_path / "grandchild.pid"
        code = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            f"print(p.pid, flush=True)\n"
            f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
            "time.sleep(60)\n"
        )
        watcher = threading.Thread(target=_signal_main_thread_when_ready, args=(marker, 15.0), daemon=True)
        watcher.start()

        with pytest.raises(KeyboardInterrupt):
            execute_python(code)
        watcher.join(timeout=5)

        assert marker.exists(), "the executed code never started its child"
        grandchild_pid = int(marker.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
        time.sleep(0.3)  # a doomed process would be gone by now
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)

    def test_non_ascii_stdout_roundtrips(self) -> None:
        """Non-ASCII output must survive the subprocess pipe (regression).

        The child's stdout is UTF-8 (pinned via PYTHONIOENCODING) while the
        parent's *default* text decoding is the locale encoding (e.g. cp932 on
        Japanese Windows). Without an explicit ``encoding="utf-8"`` on Popen the
        reader thread raised UnicodeDecodeError and the output was lost.
        """
        result = execute_python('print("\u65e5\u672c\u8a9e")')
        assert isinstance(result, ExecResult)
        assert result.exit_code == 0
        assert result.stdout.strip() == "\u65e5\u672c\u8a9e"

    def test_emoji_stdout_roundtrips(self) -> None:
        """Emoji must not crash the *child* even on a non-UTF-8 console."""
        result = execute_python('print("\U0001f680")')
        assert isinstance(result, ExecResult)
        assert result.exit_code == 0
        assert "\U0001f680" in result.stdout

    def test_dangerous_subprocess_pattern_is_refused(self) -> None:
        result = execute_python('subprocess.run(["cmd", "2>&1"], shell=True)')
        assert isinstance(result, ToolError)
        assert "Dangerous" in result.error
        assert "2>&1" in result.error


class TestCheckDangerousSubprocess:
    """The static check must flag shell=True + list + meta-char, nothing else."""

    @pytest.mark.parametrize(
        "code",
        [
            'subprocess.run(["cmd", "2>&1"], shell=True)',  # run
            'subprocess.Popen(["cmd", "|"], shell=True)',  # Popen
            'subprocess.call(["cmd", ";"], shell=True)',  # call
            'subprocess.check_call(["cmd", "`"], shell=True)',  # check_call
            'subprocess.check_output(["cmd", "$("], shell=True)',  # check_output
        ],
    )
    def test_dangerous_patterns_are_detected(self, code: str) -> None:
        from llm_cli_py.tools.python_exec import _check_dangerous_subprocess

        error = _check_dangerous_subprocess(code)
        assert error is not None
        assert "shell=True" in error

    @pytest.mark.parametrize(
        "code",
        [
            'subprocess.run(["ls"], capture_output=True)',  # no shell
            'subprocess.run(["echo", "hello"], shell=False)',  # shell=False
            'subprocess.run("ls -la", shell=True)',  # string arg, not a list
            "print('hello')",  # no subprocess at all
            "this is not valid python @@@",  # unparseable
        ],
    )
    def test_safe_code_is_not_flagged(self, code: str) -> None:
        from llm_cli_py.tools.python_exec import _check_dangerous_subprocess

        assert _check_dangerous_subprocess(code) is None


class TestToolRegistry:
    def test_register_get_and_schema(self) -> None:
        from llm_cli_py.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register("calc", "Calculate", {"type": "object"}, lambda **_: ExecResult())
        tool = registry.get("calc")
        assert tool is not None and tool.name == "calc"
        assert "calc" in registry
        assert "other" not in registry
        assert registry.get("missing") is None
        assert [s.name for s in registry.get_schemas()] == ["calc"]
        assert tool.schema.description == "Calculate"

    def test_tool_names_are_sorted(self) -> None:
        from llm_cli_py.tools.registry import ToolRegistry

        registry = ToolRegistry()
        for name in ("z_tool", "a_tool"):
            registry.register(name, name, {"type": "object"}, lambda **_: ExecResult())
        assert registry.get_tool_names() == ["a_tool", "z_tool"]


def test_result_types_serialise() -> None:
    assert ExecResult(stdout="hello").to_dict() == {"stdout": "hello", "stderr": "", "exit_code": 0}
    assert ToolError(error="boom").to_dict() == {"error": "boom"}
