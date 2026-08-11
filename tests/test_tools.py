"""Tests for tool implementations.

Covers:
- Python execution tool (basic, error, dangerous patterns)
- Web Search tool (mocked API calls)
- ToolRegistry (register, get, remove, clear, schemas, containment)
- Result types (ExecResult, SearchResult, SearchResultItem, ToolError)
- format_tool_result display formatting
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_cli_py.tools.python_exec import (
    _check_dangerous_subprocess,
    execute_python,
)
from llm_cli_py.tools.registry import Tool, ToolFunc, ToolRegistry
from llm_cli_py.tools.types import ExecResult, SearchResult, SearchResultItem, ToolError
from llm_cli_py.tools.web_search import (
    web_search,
)
from llm_cli_py.ui.display import format_tool_result

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
# Web Search tool tests (mocked)
# ══════════════════════════════════════════════════════════════════════


class TestWebSearchTool:
    """Test Web Search tool with mocked HTTP requests."""

    def test_successful_search(self) -> None:
        """Test a successful search returns SearchResult."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "grounding": {
                "generic": [
                    {
                        "title": "Python Programming",
                        "url": "https://python.org",
                        "snippets": ["Python is a programming language"],
                    },
                    {
                        "title": "Python Docs",
                        "url": "https://docs.python.org",
                        "snippets": ["Official documentation"],
                    },
                ]
            },
            "sources": {
                "https://python.org": {
                    "title": "Python Programming",
                    "hostname": "python.org",
                    "age": None,
                },
                "https://docs.python.org": {
                    "title": "Python Docs",
                    "hostname": "docs.python.org",
                    "age": None,
                },
            },
        }

        with (
            patch.dict(
                "os.environ",
                {"BRAVE_API_KEY": "test-key-123"},
            ),
            patch("llm_cli_py.tools.web_search.requests.get", return_value=mock_response) as mock_get,
        ):
            result = web_search(query="python language")

        assert isinstance(result, SearchResult)
        assert result.query == "python language"
        assert result.result_count == 2
        assert result.results[0].title == "Python Programming"
        assert result.results[0].url == "https://python.org"
        assert result.results[0].snippet == "Python is a programming language"
        assert result.results[1].title == "Python Docs"

        mock_get.assert_called_once_with(
            "https://api.search.brave.com/res/v1/llm/context",
            params={"q": "python language"},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": "test-key-123",
            },
            timeout=30,
        )

    def test_empty_results(self) -> None:
        """Test search with no results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"grounding": {"generic": []}}

        with (
            patch.dict("os.environ", {"BRAVE_API_KEY": "key"}),
            patch("llm_cli_py.tools.web_search.requests.get", return_value=mock_response),
        ):
            result = web_search(query="xyzzy_nonexistent_12345")

        assert isinstance(result, SearchResult)
        assert result.query == "xyzzy_nonexistent_12345"
        assert result.result_count == 0
        assert result.results == []

    def test_search_without_snippets(self) -> None:
        """Test search results that have no snippets."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "grounding": {
                "generic": [
                    {
                        "title": "No Snippet",
                        "url": "https://example.com",
                    },
                ]
            }
        }

        with (
            patch.dict("os.environ", {"BRAVE_API_KEY": "key"}),
            patch("llm_cli_py.tools.web_search.requests.get", return_value=mock_response),
        ):
            result = web_search(query="no snippet")

        assert isinstance(result, SearchResult)
        assert result.results[0].snippet == ""

    def test_search_with_multiple_snippets(self) -> None:
        """Test search results with multiple snippets per source."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "grounding": {
                "generic": [
                    {
                        "title": "Multi Snippet",
                        "url": "https://example.com",
                        "snippets": [
                            "First relevant passage from the page.",
                            "Second relevant passage from the same page.",
                        ],
                    },
                ]
            }
        }

        with (
            patch.dict("os.environ", {"BRAVE_API_KEY": "key"}),
            patch("llm_cli_py.tools.web_search.requests.get", return_value=mock_response),
        ):
            result = web_search(query="multi snippet")

        assert isinstance(result, SearchResult)
        assert (
            result.results[0].snippet
            == "First relevant passage from the page.\nSecond relevant passage from the same page."
        )

    def test_http_error(self) -> None:
        """Test HTTP error returns ToolError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("HTTP 500 Error")

        with (
            patch.dict("os.environ", {"BRAVE_API_KEY": "key"}),
            patch("llm_cli_py.tools.web_search.requests.get", return_value=mock_response),
        ):
            result = web_search(query="test")

        assert isinstance(result, ToolError)
        assert "failed after 3 attempts" in result.error

    def test_rate_limit_retry_eventually_succeeds(self) -> None:
        """Test rate limit (429) retries and eventually succeeds."""
        mock_fail = MagicMock()
        mock_fail.status_code = 429

        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {
            "grounding": {
                "generic": [
                    {
                        "title": "Retry Success",
                        "url": "https://example.com",
                        "snippets": ["Finally worked"],
                    },
                ]
            }
        }

        with (
            patch.dict("os.environ", {"BRAVE_API_KEY": "key"}),
            patch(
                "llm_cli_py.tools.web_search.requests.get",
                side_effect=[mock_fail, mock_success],
            ) as mock_get,
            patch("llm_cli_py.tools.web_search.time.sleep") as mock_sleep,
        ):
            result = web_search(query="retry test")

        assert isinstance(result, SearchResult)
        assert result.results[0].title == "Retry Success"
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(2)

    def test_rate_limit_all_fail(self) -> None:
        """Test rate limit that never recovers."""
        mock_fail = MagicMock()
        mock_fail.status_code = 429

        with (
            patch.dict("os.environ", {"BRAVE_API_KEY": "key"}),
            patch("llm_cli_py.tools.web_search.requests.get", return_value=mock_fail),
            patch("llm_cli_py.tools.web_search.time.sleep"),
        ):
            result = web_search(query="always rate limited")

        assert isinstance(result, ToolError)
        assert "failed after 3 attempts" in result.error
        assert "429" in result.error

    def test_network_error(self) -> None:
        """Test network connectivity error returns ToolError."""
        with (
            patch.dict("os.environ", {"BRAVE_API_KEY": "key"}),
            patch(
                "llm_cli_py.tools.web_search.requests.get",
                side_effect=requests.exceptions.ConnectionError("Connection refused"),
            ),
            patch("llm_cli_py.tools.web_search.time.sleep"),
        ):
            result = web_search(query="network fail")

        assert isinstance(result, ToolError)
        assert "Connection refused" in result.error
        assert "failed after 3 attempts" in result.error

    def test_missing_api_key(self) -> None:
        """Test that missing BRAVE_API_KEY returns ToolError."""
        with patch.dict("os.environ", {}, clear=True):
            result = web_search(query="test")

        assert isinstance(result, ToolError)
        assert "BRAVE_API_KEY" in result.error

    def test_proxy_url_uses_post(self) -> None:
        """Test that when LLM_CLI_PROXY_URL is set, it uses POST."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "grounding": {
                "generic": [
                    {
                        "title": "Proxy Result",
                        "url": "https://example.com",
                        "snippets": ["Via proxy"],
                    },
                ]
            }
        }

        with (
            patch.dict("os.environ", {"LLM_CLI_PROXY_URL": "http://proxy:8080"}),
            patch("llm_cli_py.tools.web_search.requests.post", return_value=mock_response) as mock_post,
        ):
            result = web_search(query="proxy test")

        assert isinstance(result, SearchResult)
        assert result.results[0].title == "Proxy Result"
        mock_post.assert_called_once_with(
            "http://proxy:8080/web_search",
            json={"query": "proxy test"},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )


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

    def test_search_result_item_defaults(self) -> None:
        """Test SearchResultItem default values."""
        item = SearchResultItem()
        assert item.title == ""
        assert item.url == ""
        assert item.snippet == ""

    def test_search_result_item(self) -> None:
        """Test SearchResultItem dataclass."""
        item = SearchResultItem(title="Test", url="https://example.com", snippet="A test snippet")
        assert item.title == "Test"
        assert item.url == "https://example.com"
        assert item.snippet == "A test snippet"

        d = item.to_dict()
        assert d == {"title": "Test", "url": "https://example.com", "snippet": "A test snippet"}

    def test_search_result_defaults(self) -> None:
        """Test SearchResult default values."""
        result = SearchResult()
        assert result.query == ""
        assert result.results == []
        assert result.result_count == 0

    def test_search_result(self) -> None:
        """Test SearchResult dataclass."""
        items = [
            SearchResultItem(title="Result 1", url="https://example.com/1", snippet="First result"),
            SearchResultItem(title="Result 2", url="https://example.com/2", snippet="Second result"),
        ]
        result = SearchResult(query="test query", results=items, result_count=2)
        assert result.query == "test query"
        assert len(result.results) == 2
        assert result.result_count == 2

        d = result.to_dict()
        assert d["query"] == "test query"
        results = d["results"]
        assert isinstance(results, list)
        assert len(results) == 2
        assert d["result_count"] == 2

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


# ══════════════════════════════════════════════════════════════════════
# format_tool_result display tests
# ══════════════════════════════════════════════════════════════════════


class TestFormatToolResult:
    """Test format_tool_result with typed results."""

    def test_format_exec_result_with_output(self) -> None:
        """Test formatting an ExecResult with stdout."""
        result = ExecResult(stdout="Hello World\nLine 2", stderr="", exit_code=0)
        output = format_tool_result(result)
        assert "Hello World" in output
        assert "Line 2" in output
        assert "- exit_code: 0" in output
        assert "- stdout:" in output
        assert "- stderr: (no output)" in output

    def test_format_exec_result_with_stderr(self) -> None:
        """Test formatting an ExecResult with stderr."""
        result = ExecResult(stdout="", stderr="Error: something broke", exit_code=1)
        output = format_tool_result(result)
        assert "- stdout: (no output)" in output
        assert "Error: something broke" in output
        assert "- exit_code: 1" in output

    def test_format_exec_result_both_output(self) -> None:
        """Test formatting an ExecResult with both stdout and stderr."""
        result = ExecResult(stdout="Standard output", stderr="Standard error", exit_code=1)
        output = format_tool_result(result)
        assert "Standard output" in output
        assert "Standard error" in output
        assert "- exit_code: 1" in output

    def test_format_search_result(self) -> None:
        """Test formatting a SearchResult."""
        items = [
            SearchResultItem(title="Python", url="https://python.org", snippet="Python is great"),
        ]
        result = SearchResult(query="python language", results=items, result_count=1)
        output = format_tool_result(result)
        assert "Query: python language" in output
        assert "Python" in output
        assert "https://python.org" in output
        assert "Python is great" in output
        assert "Results (1):" in output

    def test_format_search_result_no_results(self) -> None:
        """Test formatting a SearchResult with no results."""
        result = SearchResult(query="empty search", results=[], result_count=0)
        output = format_tool_result(result)
        assert "Query: empty search" in output
        assert "Results (0):" in output

    def test_format_tool_error(self) -> None:
        """Test formatting a ToolError."""
        result = ToolError(error="API key not found")
        output = format_tool_result(result)
        assert output == "- **Error:** API key not found"

    def test_format_tool_error_empty(self) -> None:
        """Test formatting an empty ToolError."""
        result = ToolError(error="")
        output = format_tool_result(result)
        assert output == "- **Error:** "
