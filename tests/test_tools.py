"""Tests for the Python execution tool and the tool registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_cli_py.tools.python_exec import (
    PYTHON_TOOL_DESCRIPTION,
    PYTHON_TOOL_SCHEMA,
    _check_shell_true_with_list,
    execute_python,
)
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult, ToolError


def test_code_runs_and_stdout_is_captured() -> None:
    result = execute_python("print('hello world')", explanation="Print a greeting.")
    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "hello world"


def test_failing_code_reports_stderr_and_a_nonzero_exit_code() -> None:
    result = execute_python("raise ValueError('test error')", explanation="Show a failing run.")
    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert "ValueError" in result.stderr


def test_non_ascii_output_survives_the_subprocess_pipe() -> None:
    result = execute_python('print("\u65e5\u672c\u8a9e")', explanation="Check non-ASCII output.")
    assert isinstance(result, ExecResult)
    assert result.stdout.strip() == "\u65e5\u672c\u8a9e"


def test_shell_true_with_an_argv_list_is_refused() -> None:
    """``shell=True`` drops every list element after the first, so it is refused."""
    result = execute_python(
        'subprocess.run(["cmd", "--flag"], shell=True)', explanation="Show the refused shape."
    )
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
    assert isinstance(execute_python(code, explanation="Fetch a URL that has a query string."), ExecResult)


def test_a_run_is_not_cut_off_for_taking_its_time() -> None:
    """There is no limit on a run: code that takes a while still finishes."""
    result = execute_python(
        "import time\ntime.sleep(2)\nprint('finished')", explanation="Sleep, then report."
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "finished"


def test_the_description_has_no_time_limit_and_mentions_backgrounding() -> None:
    assert "no time limit" in PYTHON_TOOL_DESCRIPTION
    assert "background" in PYTHON_TOOL_DESCRIPTION


def test_the_schema_requires_an_explanation_as_well_as_the_code() -> None:
    """The reason for the call is not optional in what the model is shown."""
    properties = PYTHON_TOOL_SCHEMA["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"code", "explanation"}
    assert PYTHON_TOOL_SCHEMA["required"] == ["code", "explanation"]


def test_the_description_tells_the_model_the_explanation_is_shown() -> None:
    """``required`` alone does not say what the sentence is for; the text does."""
    assert "explanation" in PYTHON_TOOL_DESCRIPTION
    assert "after the code" in PYTHON_TOOL_DESCRIPTION


def test_an_explanation_does_not_change_what_runs() -> None:
    """It is the reason for the call, not an input to the code: same code, same output."""
    result = execute_python("print('hi')", explanation="Say hello.")

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "hi"


@pytest.mark.parametrize("explanation", ["", "   \n  "])
def test_a_call_without_an_explanation_is_refused(explanation: str, tmp_path: Path) -> None:
    """Required means required: the code is refused before it runs, as ``shell=True`` is."""
    marker = tmp_path / "written"
    result = execute_python(f"open({str(marker)!r}, 'w').write('x')", explanation=explanation)

    assert isinstance(result, ToolError)
    assert "explanation" in result.error
    assert "NOT run" in result.error
    assert not marker.exists()  # nothing ran: the refusal came first


def test_a_call_that_omits_the_explanation_is_refused_too() -> None:
    """A model that ignored the schema gets a usable sentence, not a ``TypeError``."""
    result = execute_python("print('hi')")  # type: ignore[call-arg]

    assert isinstance(result, ToolError)
    assert "explanation" in result.error


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
