"""Tests for the Python execution tool, the tool registry and result types."""

from __future__ import annotations

import pytest

from llm_cli_py.tools.python_exec import execute_python
from llm_cli_py.tools.types import ExecResult, ToolError


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
