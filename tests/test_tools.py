"""Tests for the Python execution tool and the tool registry."""

from __future__ import annotations

from llm_cli_py.tools.python_exec import execute_python
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


def test_dangerous_subprocess_pattern_is_refused() -> None:
    """``shell=True`` with a meta-character in a list hangs forever, so it is refused."""
    result = execute_python('subprocess.run(["cmd", "2>&1"], shell=True)')
    assert isinstance(result, ToolError)
    assert "Dangerous" in result.error


def test_tool_result_types_serialise() -> None:
    assert ExecResult(stdout="hello").to_dict() == {"stdout": "hello", "stderr": "", "exit_code": 0}
    assert ToolError(error="boom").to_dict() == {"error": "boom"}


def test_registry_registers_looks_up_and_advertises_a_tool() -> None:
    registry = ToolRegistry()
    registry.register("calc", "Calculate", {"type": "object"}, lambda **_: ExecResult())

    tool = registry.get("calc")
    assert tool is not None and tool.name == "calc"
    assert "calc" in registry and "other" not in registry
    assert registry.get("missing") is None
    assert [s.name for s in registry.get_schemas()] == ["calc"]
    assert tool.schema.description == "Calculate"
