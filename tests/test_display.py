"""Tests for the UI display module.

Covers print functions, formatting and model redaction.
"""

from __future__ import annotations

import pytest

from llm_cli_py.tools.types import ExecResult, SearchResult, SearchResultItem
from llm_cli_py.ui.display import (
    format_tool_result,
    print_assistant,
    print_block,
    print_info,
    print_tool_call,
    print_tool_result,
    report_error,
    report_info,
    report_success,
    report_warning,
)


class TestPrintFunctions:
    """Test display print functions."""

    def test_print_block_empty_content(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_block("   ", title="Empty")
        captured = capsys.readouterr()
        # Empty content should not crash
        assert "Empty:" in captured.out

    def test_print_assistant(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_assistant("Hello world")
        captured = capsys.readouterr()
        assert "Assistant" in captured.out
        assert "Hello world" in captured.out

    def test_print_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_info("Model", "gpt-4o")
        captured = capsys.readouterr()
        assert "Model: gpt-4o" in captured.out

    def test_report_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_info("Something happened")
        captured = capsys.readouterr()
        assert "INFO: Something happened" in captured.out

    def test_report_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_success("All good")
        captured = capsys.readouterr()
        assert "All good" in captured.out

    def test_report_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_error("Something broke")
        captured = capsys.readouterr()
        assert "ERROR: Something broke" in captured.out

    def test_report_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_warning("Be careful")
        captured = capsys.readouterr()
        assert "Be careful" in captured.out

    def test_print_tool_result_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_tool_result("")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_tool_result_with_content(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_tool_result("Some result")
        captured = capsys.readouterr()
        assert "Tool Result:" in captured.out
        assert "Some result" in captured.out


class TestPrintToolCall:
    """Test print_tool_call with various argument types."""

    def test_print_tool_call_simple(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_tool_call("web_search", {"query": "python"})
        captured = capsys.readouterr()
        assert "Tool: web_search" in captured.out
        assert "query: python" in captured.out

    def test_print_tool_call_python_code(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_tool_call("execute_python", {"code": "print(1)"})
        captured = capsys.readouterr()
        assert "Tool: execute_python" in captured.out
        assert "code:" in captured.out
        assert "print(1)" in captured.out

    def test_print_tool_call_multiple_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_tool_call("search", {"query": "test", "limit": 10})
        captured = capsys.readouterr()
        assert "query: test" in captured.out
        assert "limit: 10" in captured.out


class TestFormatToolResultEdgeCases:
    """Test format_tool_result with edge cases."""

    def test_format_exec_result_no_stdout(self) -> None:
        result = ExecResult(stdout="", stderr="", exit_code=0)
        output = format_tool_result(result)
        assert "- stdout: (no output)" in output
        assert "- stderr: (no output)" in output
        assert "- exit_code: 0" in output

    def test_format_exec_result_with_stderr_only(self) -> None:
        result = ExecResult(stdout="", stderr="Error occurred", exit_code=1)
        output = format_tool_result(result)
        assert "- stdout: (no output)" in output
        assert "Error occurred" in output
        assert "- exit_code: 1" in output

    def test_format_search_result_no_snippet(self) -> None:
        items = [SearchResultItem(title="Test", url="https://example.com", snippet="")]
        result = SearchResult(query="test", results=items, result_count=1)
        output = format_tool_result(result)
        assert "Test" in output
        assert "https://example.com" in output
        assert "Snippet:" not in output

    def test_format_search_result_multiple_results(self) -> None:
        # Only the top result is shown to avoid cluttering the display.
        items = [
            SearchResultItem(title="First", url="https://first.com", snippet="First snippet"),
            SearchResultItem(title="Second", url="https://second.com", snippet="Second snippet"),
        ]
        result = SearchResult(query="multi", results=items, result_count=2)
        output = format_tool_result(result)
        assert "Results (2, showing top 1):" in output
        assert "1. First" in output
        assert "2. Second" not in output

    def test_format_search_result_truncates_long_snippet(self) -> None:
        # A very long snippet should be truncated to keep the display compact.
        long_snippet = "x" * 500
        items = [SearchResultItem(title="Long", url="https://long.com", snippet=long_snippet)]
        result = SearchResult(query="long", results=items, result_count=1)
        output = format_tool_result(result)
        # The snippet is truncated (300 chars) and marked with "..." at the end
        # of the snippet text.
        assert "x" * 300 + "..." in output

    def test_format_unknown_type(self) -> None:
        output = format_tool_result({"custom": "data"})
        assert '"custom": "data"' in output

    def test_format_none_result(self) -> None:
        output = format_tool_result(None)
        assert "null" in output or "None" in output
