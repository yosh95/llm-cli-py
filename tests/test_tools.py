"""Tests for tool implementations.

Covers:
- Python execution tool (basic, error, dangerous patterns)
- ToolRegistry (register, get, remove, clear, schemas, containment)
- Result types (ExecResult, ToolError)
"""

import pytest

from llm_cli_py.tools.python_exec import (
    _check_dangerous_subprocess,
    execute_python,
)
from llm_cli_py.tools.registry import Tool, ToolFunc, ToolRegistry
from llm_cli_py.tools.types import ExecResult, ToolError

# ══════════════════════════════════════════════════════════════════════
# Python execution tool tests
# ══════════════════════════════════════════════════════════════════════


class TestPythonExecutionTool:
    """Test Python execution tool."""

    def test_execute_simple_code(self) -> None:
        """Test executing simple Python code."""
        result = execute_python("print('hello world')")
        assert isinstance(result, ExecResult)
        assert result.stdout.strip() == "hello world"
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_execute_with_error(self) -> None:
        """Test executing code that raises an error."""
        result = execute_python("raise ValueError('test error')")
        assert isinstance(result, ExecResult)
        assert "ValueError" in result.stderr
        assert result.exit_code == 1

    def test_execute_complex_code(self) -> None:
        """Test executing more complex Python code."""
        code = "import json\n"
        code += 'data = {"key": "value", "numbers": [1, 2, 3]}\n'
        code += "print(json.dumps(data, indent=2))"
        result = execute_python(code)
        assert isinstance(result, ExecResult)
        assert '"key": "value"' in result.stdout
        assert result.exit_code == 0

    def test_execute_with_import(self) -> None:
        """Test executing code with imports."""
        result = execute_python("import math\nprint(math.sqrt(16))")
        assert isinstance(result, ExecResult)
        assert result.stdout.strip() == "4.0"
        assert result.exit_code == 0

    def test_execute_empty_code(self) -> None:
        """Test executing an empty string."""
        result = execute_python("")
        assert isinstance(result, ExecResult)
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_execute_syntax_error(self) -> None:
        """Test executing code with a syntax error."""
        result = execute_python("if True print('bad')")
        assert isinstance(result, ExecResult)
        assert result.exit_code == 1
        assert "SyntaxError" in result.stderr


class TestCheckDangerousSubprocess:
    """Test the static analyzer for dangerous subprocess patterns."""

    def test_no_danger(self) -> None:
        """Test safe code returns None."""
        code = 'subprocess.run(["ls"], capture_output=True)'
        assert _check_dangerous_subprocess(code) is None

    def test_shell_false_no_danger(self) -> None:
        """Test shell=False with list args is safe."""
        code = 'subprocess.run(["echo", "hello"], shell=False)'
        assert _check_dangerous_subprocess(code) is None

    def test_no_subprocess_import(self) -> None:
        """Test code without subprocess calls is safe."""
        code = "print('hello')\nimport math\nmath.sqrt(4)"
        assert _check_dangerous_subprocess(code) is None

    def test_dangerous_shell_true_with_list_and_meta(self) -> None:
        """Test shell=True + list + meta-char is detected."""
        code = 'subprocess.run(["cmd", "2>&1"], shell=True)'
        error = _check_dangerous_subprocess(code)
        assert error is not None
        assert "Dangerous" in error
        assert "shell=True" in error
        assert "2>&1" in error

    def test_dangerous_popen_shell_true_with_meta(self) -> None:
        """Test Popen with similar dangerous pattern."""
        code = 'subprocess.Popen(["cmd", "|"], shell=True)'
        error = _check_dangerous_subprocess(code)
        assert error is not None
        assert "Popen" in error
        assert "|" in error

    def test_dangerous_call_shell_true_with_meta(self) -> None:
        """Test call with dangerous pattern."""
        code = 'subprocess.call(["cmd", ";"], shell=True)'
        error = _check_dangerous_subprocess(code)
        assert error is not None
        assert "call" in error

    def test_dangerous_check_call_shell_true_with_meta(self) -> None:
        """Test check_call with dangerous pattern."""
        code = 'subprocess.check_call(["cmd", "`"], shell=True)'
        error = _check_dangerous_subprocess(code)
        assert error is not None

    def test_dangerous_check_output_shell_true_with_meta(self) -> None:
        """Test check_output with dangerous pattern."""
        code = 'subprocess.check_output(["cmd", "$("], shell=True)'
        error = _check_dangerous_subprocess(code)
        assert error is not None

    def test_shell_true_string_ok(self) -> None:
        """Test shell=True with a string argument (not list) is OK."""
        code = 'subprocess.run("ls -la", shell=True)'
        assert _check_dangerous_subprocess(code) is None

    def test_syntax_error_code_returns_none(self) -> None:
        """Test that syntactically invalid code returns None (skip check)."""
        code = "this is not valid python @@@"
        assert _check_dangerous_subprocess(code) is None

    def test_dangerous_write_to_stdin_pattern(self) -> None:
        """Test another dangerous pattern: pipe + write."""
        code = 'subprocess.run(["cmd", ">"], shell=True)'
        error = _check_dangerous_subprocess(code)
        assert error is not None
        assert ">" in error


# ══════════════════════════════════════════════════════════════════════
# ToolRegistry tests
# ══════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    """Test ToolRegistry."""

    def make_dummy_func(self) -> ToolFunc:
        def dummy(code: str, explanation: str = "") -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="ok")

        return dummy

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute"},
            },
            "required": ["code"],
        }
        registry.register("test_tool", "A test tool", schema, self.make_dummy_func())

        tool = registry.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert isinstance(tool.func, type(self.make_dummy_func()))

    def test_get_nonexistent(self) -> None:
        """Test retrieving a nonexistent tool returns None."""
        registry = ToolRegistry()
        tool = registry.get("nonexistent")
        assert tool is None

    def test_get_schemas(self) -> None:
        """Test getting tool schemas."""
        registry = ToolRegistry()
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
            },
            "required": ["code"],
        }
        registry.register("test_tool", "A test tool", schema, self.make_dummy_func())

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "test_tool"

    def test_get_schemas_multiple(self) -> None:
        """Test getting schemas from multiple tools."""
        registry = ToolRegistry()

        def dummy1(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="a")

        def dummy2(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="b")

        registry.register(
            "tool_a",
            "Tool A",
            {"type": "object", "properties": {}, "required": []},
            dummy1,
        )
        registry.register(
            "tool_b",
            "Tool B",
            {"type": "object", "properties": {}, "required": []},
            dummy2,
        )

        schemas = registry.get_schemas()
        names = {s.name for s in schemas}
        assert names == {"tool_a", "tool_b"}

    def test_get_tool_names_sorted(self) -> None:
        """Test getting sorted tool names."""
        registry = ToolRegistry()

        def dummy(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="")

        registry.register("z_tool", "Z tool", {"type": "object", "properties": {}}, dummy)
        registry.register("a_tool", "A tool", {"type": "object", "properties": {}}, dummy)

        names = registry.get_tool_names()
        assert names == ["a_tool", "z_tool"]

    def test_contains(self) -> None:
        """Test 'in' operator."""
        registry = ToolRegistry()

        def dummy(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="")

        registry.register("my_tool", "My tool", {"type": "object", "properties": {}}, dummy)

        assert "my_tool" in registry
        assert "other" not in registry

    def test_remove(self) -> None:
        """Test removing a registered tool."""
        registry = ToolRegistry()

        def dummy(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="")

        registry.register("temp_tool", "Temporary", {"type": "object", "properties": {}}, dummy)
        assert "temp_tool" in registry

        registry.remove("temp_tool")
        assert "temp_tool" not in registry
        assert registry.get("temp_tool") is None

    def test_remove_nonexistent_raises_key_error(self) -> None:
        """Test removing a nonexistent tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError):
            registry.remove("not_there")

    def test_clear(self) -> None:
        """Test clearing all tools."""
        registry = ToolRegistry()

        def dummy(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="")

        registry.register("tool_1", "One", {"type": "object", "properties": {}}, dummy)
        registry.register("tool_2", "Two", {"type": "object", "properties": {}}, dummy)
        assert len(registry) == 2

        registry.clear()
        assert len(registry) == 0
        assert registry.get("tool_1") is None
        assert registry.get_tool_names() == []

    def test_len(self) -> None:
        """Test __len__ returns correct count."""
        registry = ToolRegistry()

        def dummy(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="")

        assert len(registry) == 0
        registry.register("a", "A", {"type": "object", "properties": {}}, dummy)
        assert len(registry) == 1
        registry.register("b", "B", {"type": "object", "properties": {}}, dummy)
        assert len(registry) == 2

    def test_explanation_not_in_schema_by_default(self) -> None:
        """Test that explanation is not in the schema unless explicitly added."""
        registry = ToolRegistry()

        def dummy(**kwargs: object) -> ExecResult:  # noqa: ARG001
            return ExecResult(stdout="")

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        registry.register("search", "Search tool", schema, dummy)

        tool = registry.get("search")
        assert tool is not None
        params = tool.parameters
        props = params["properties"]
        assert isinstance(props, dict)
        assert "explanation" not in props

    def test_tool_schema_property(self) -> None:
        """Test the Tool.schema property generates a valid ToolSchema."""
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
            },
            "required": ["code"],
        }
        tool = Tool("my_tool", "Does something", schema, self.make_dummy_func())
        ts = tool.schema
        assert ts.name == "my_tool"
        assert ts.description == "Does something"
        assert ts.parameters["type"] == "object"


# ══════════════════════════════════════════════════════════════════════
# Result type tests
# ══════════════════════════════════════════════════════════════════════


class TestResultTypes:
    """Test the typed result classes."""

    def test_exec_result_defaults(self) -> None:
        """Test ExecResult default values."""
        result = ExecResult()
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_exec_result(self) -> None:
        """Test ExecResult dataclass."""
        result = ExecResult(stdout="hello", stderr="", exit_code=0)
        assert result.stdout == "hello"
        assert result.stderr == ""
        assert result.exit_code == 0

        d = result.to_dict()
        assert d == {"stdout": "hello", "stderr": "", "exit_code": 0}

    def test_exec_result_timeout(self) -> None:
        """Test ExecResult with timeout indicator."""
        result = ExecResult(stdout="", stderr="timed out", exit_code=-1)
        assert result.exit_code == -1

    def test_tool_error(self) -> None:
        """Test ToolError dataclass."""
        err = ToolError(error="Something went wrong")
        assert err.error == "Something went wrong"

        d = err.to_dict()
        assert d == {"error": "Something went wrong"}

    def test_tool_error_empty(self) -> None:
        """Test ToolError with empty string."""
        err = ToolError(error="")
        assert err.error == ""
