"""Smoke test for the UI display module."""

from llm_cli_py.ui.display import (
    print_assistant,
    print_info,
    print_tool_call,
    print_tool_result,
    report_error,
)


def test_display_helpers_emit_expected_labels(capsys) -> None:
    print_assistant("Hello world")
    print_info("Model", "gpt-4o")
    print_tool_call("python", ["    code=print(1)"])
    print_tool_result(["Exit code: 0", "ok"])
    report_error("Something broke")
    out = capsys.readouterr().out

    assert "Assistant" in out and "Hello world" in out
    assert "Model: gpt-4o" in out
    assert "Executing tool: python" in out and "    code=print(1)" in out
    assert "Tool Result" in out and "Exit code: 0" in out
    assert "ERROR: Something broke" in out
