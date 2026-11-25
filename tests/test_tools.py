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
from llm_cli_py.utils import terminal


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


def test_the_description_states_there_is_no_timeout_and_mentions_backgrounding() -> None:
    assert "no timeout" in PYTHON_TOOL_DESCRIPTION
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


# ── The child never gets the terminal ─────────────────────────────


def test_tool_code_cannot_read_what_the_user_types() -> None:
    """The child's stdin is not the terminal, so a prompt cannot be answered by it.

    Before this held, code that called ``input()`` read the user's next keystroke
    out from under the CLI's own prompt: the line typed at ``> `` went to the
    tool instead of to the model.
    """
    result = execute_python(
        "import sys\ndata = sys.stdin.read(10)\nprint(repr(data))",
        explanation="Check what the child's stdin is connected to.",
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "''"  # end of input: nothing to read


def test_tool_code_reading_stdin_returns_instead_of_waiting_for_a_keypress() -> None:
    """``input()`` fails at once rather than blocking the turn on a keystroke."""
    result = execute_python(
        "try:\n    input('prompt: ')\nexcept EOFError:\n    print('eof')",
        explanation="Show that a tool cannot wait for the keyboard.",
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("eof")  # the prompt came first


def test_tool_code_cannot_put_the_terminal_into_raw_mode() -> None:
    """Without a descriptor on the terminal there is no mode to change.

    A child that could do this left the terminal raw when it exited, and a raw
    terminal has ``ISIG`` off -- so Ctrl+C stopped being a signal, for the CLI
    and for the shell after it. ``tty.setraw`` fails here because the child has
    no terminal at all.
    """
    result = execute_python(
        "import os, tty\ntry:\n    tty.setraw(0)\nexcept Exception as e:\n    print(type(e).__name__)",
        explanation="Show that a tool has no terminal to change.",
    )

    assert isinstance(result, ExecResult)
    # `termios.error` (an OSError) from the ioctl, or ValueError for fd 0 being
    # /dev/null: either way the terminal is untouched.
    assert result.stdout.strip() in {"error", "OSError", "ValueError", "TermiosError"}


def test_a_tool_run_gives_the_terminal_back_even_if_it_was_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settings captured before the run are put back after it, however it ends.

    The child is kept away from the terminal, but the guarantee does not rest on
    that alone: a run of tool code starts and ends with the same terminal
    settings, so anything that did manage to change them cannot outlive the
    call.
    """
    calls: list[str] = []

    class _FakeTermios:
        """Enough of ``termios`` to see what is asked of it."""

        TCSANOW = 0
        error = OSError

        def __init__(self) -> None:
            self.state = ["cooked"]

        def tcgetattr(self, _fd: int) -> list[str]:
            calls.append("get")
            return list(self.state)

        def tcsetattr(self, _fd: int, _when: int, state: list[str]) -> None:
            calls.append(f"set {state[0]}")
            self.state = list(state)

    class _FakeProc:
        """A child that finishes at once, after leaving the terminal raw."""

        returncode = 0

        def __init__(self) -> None:
            fake.state = ["raw"]
            self.kwargs: dict[str, object] = {}

        def communicate(self) -> tuple[str, str]:
            return "", ""

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    fake = _FakeTermios()
    monkeypatch.setattr(terminal, "termios", fake)
    monkeypatch.setattr(terminal.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(terminal.sys.stdin, "fileno", lambda: 0, raising=False)
    monkeypatch.setattr(
        "llm_cli_py.tools.python_exec.subprocess.Popen",
        lambda *_a, **_k: _FakeProc(),
    )

    result = execute_python("print('x')", explanation="Run something.")

    assert isinstance(result, ExecResult)
    assert calls == ["get", "set cooked"]  # captured before, put back after
    assert fake.state == ["cooked"]


def test_no_terminal_means_nothing_to_save_or_restore() -> None:
    """A redirected stdin is not a terminal: every step is a no-op, never an error."""
    assert terminal.save() is None  # pytest's stdin is not a tty
    terminal.restore(None)  # must not raise
    with terminal.preserved():
        pass
