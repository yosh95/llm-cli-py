"""CLI entry point: argument handling, prompt assembly and the prompt session (no network)."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from prompt_toolkit.history import FileHistory

import llm_cli_py
from llm_cli_py.main import build_parser, initialize_tools
from llm_cli_py.main import main as main_module_main

VERSION_OUTPUT = "llm-cli-py 0.2.0"


def _run_as_module(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m llm_cli_py`` in a child process, with a clean env."""
    src_dir = Path(llm_cli_py.__file__).parent.parent
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_CLI_")}
    env["PYTHONPATH"] = str(src_dir)
    return subprocess.run(
        [sys.executable, "-m", "llm_cli_py", *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_execute_python_is_the_only_registered_tool() -> None:
    assert [t.name for t in initialize_tools().get_schemas()] == ["execute_python"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], {"sources": [], "prompt": [], "model": None, "api_key": None}),
        (["-m", "gpt-4o"], {"model": "gpt-4o"}),
        (["--api-url", "https://x/v1"], {"api_url": "https://x/v1"}),
        (["--api-key", "sk-1"], {"api_key": "sk-1"}),
        (["-s", "a", "-s", "b"], {"sources": ["a", "b"]}),
        (["What is 2+2?"], {"prompt": ["What is 2+2?"]}),
    ],
)
def test_parser_arguments(argv: list[str], expected: dict) -> None:
    args = build_parser().parse_args(argv)
    for key, value in expected.items():
        assert getattr(args, key) == value


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert VERSION_OUTPUT in capsys.readouterr().out


def test_module_entry_point_is_importable_without_running() -> None:
    """``python -m llm_cli_py`` discovers ``__main__``; importing it must be inert."""
    module = runpy.run_module("llm_cli_py.__main__", run_name="not_main")
    assert module["main"] is main_module_main  # the console script's own entry point


def test_python_dash_m_runs_the_cli() -> None:
    """The module entry point reports the same version as the console script."""
    result = _run_as_module("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == VERSION_OUTPUT


def test_python_dash_m_needs_no_console_script() -> None:
    """No launcher involved: the interpreter's own ``python`` is the only executable."""
    result = _run_as_module("-m", "gpt-4o")
    assert result.returncode == 1  # LLM_CLI_API_URL is not set in the child's env
    assert "LLM_CLI_API_URL" in result.stdout
    assert "Traceback" not in result.stderr


def test_main_exits_without_api_url(capsys, monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLI_API_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["llm-cli-py", "-m", "gpt-4o"])
    from llm_cli_py import main as main_module

    with pytest.raises(SystemExit):
        main_module.main()
    assert "LLM_CLI_API_URL" in capsys.readouterr().out


def test_read_prompt_adds_no_newline_after_ctrl_c(capsys) -> None:
    """prompt_toolkit closes the line as it aborts: a newline here would double it."""
    from llm_cli_py.session import interactive as interactive_mod

    prompt_session = MagicMock()
    prompt_session.prompt.side_effect = KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interactive_mod.read_prompt(prompt_session)

    assert capsys.readouterr().out == ""


def test_read_prompt_adds_no_newline_at_end_of_input(capsys) -> None:
    """Ctrl+D is closed the same way, so the shell prompt follows with no blank line."""
    from llm_cli_py.session import interactive as interactive_mod

    prompt_session = MagicMock()
    prompt_session.prompt.side_effect = EOFError

    with pytest.raises(EOFError):
        interactive_mod.read_prompt(prompt_session)

    assert capsys.readouterr().out == ""


def test_read_prompt_adds_nothing_after_enter(capsys) -> None:
    """A line finished with Enter is already terminated: no second newline."""
    from llm_cli_py.session import interactive as interactive_mod

    prompt_session = MagicMock()
    prompt_session.prompt.return_value = "hello"

    assert interactive_mod.read_prompt(prompt_session) == "hello"
    prompt_session.prompt.assert_called_once_with("> ")
    assert capsys.readouterr().out == ""


def _ctrl_z_through_the_prompt(monkeypatch, supported: bool, typed: str) -> tuple[str, int]:
    """Feed ``^Z`` + ``typed`` to a prompt session built with ``supported``.

    Returns the accepted line and how many times the session asked to be
    suspended. ``Application.suspend_to_background`` is where Ctrl+Z ends up, so
    it is replaced: actually suspending would put the test itself in the
    background.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from llm_cli_py.session import interactive as interactive_mod

    monkeypatch.setattr(interactive_mod, "SUSPEND_SUPPORTED", supported)

    with create_pipe_input() as pipe_input:
        prompt_session = interactive_mod.make_prompt_session(
            input_factory=lambda: pipe_input,
            output_factory=DummyOutput,
        )
        with patch.object(Application, "suspend_to_background") as suspend:
            pipe_input.send_text("\x1a" + typed + "\r")
            line = prompt_session.prompt("> ")

    return line, suspend.call_count


def test_ctrl_z_suspends_the_cli_where_the_platform_supports_it(monkeypatch) -> None:
    """Where there is a ``SIGTSTP`` to send, Ctrl+Z is the suspend key.

    That is how it behaves at a shell's own prompt: the process is stopped, and
    the line typed so far is still there after ``fg``. Suspending is the
    session's job -- this only checks that the key asks for it, once, and that
    the prompt carries on without the key ending up in the line.
    """
    line, suspends = _ctrl_z_through_the_prompt(monkeypatch, supported=True, typed="hello")

    assert suspends == 1
    assert line == "hello"


def test_ctrl_z_keeps_its_default_meaning_where_suspending_is_unsupported(monkeypatch) -> None:
    """Without ``SIGTSTP`` (Windows) the key is left alone.

    Nothing is suspended, and prompt_toolkit's own binding stands: Ctrl+Z
    inserts itself, so ``^Z`` + Enter still reaches the CLI as end-of-input.
    """
    line, suspends = _ctrl_z_through_the_prompt(monkeypatch, supported=False, typed="abc")

    assert suspends == 0
    assert line == "\x1aabc"


def test_suspending_is_enabled_exactly_where_the_platform_can_do_it() -> None:
    """The flag is the platform's answer, so the feature stays Unix-only."""
    import signal

    from llm_cli_py.session import interactive as interactive_mod

    assert interactive_mod.SUSPEND_SUPPORTED is hasattr(signal, "SIGTSTP")


def test_a_turn_typed_at_the_prompt_is_written_to_the_history_file(tmp_path, monkeypatch) -> None:
    """The file is written as turns are accepted, not only at exit."""
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from llm_cli_py.session import interactive as interactive_mod

    history_file = tmp_path / "history"
    monkeypatch.setenv(interactive_mod.ENV_PROMPT_HISTORY_FILE, str(history_file))

    with create_pipe_input() as pipe_input:
        prompt_session = interactive_mod.make_prompt_session(
            input_factory=lambda: pipe_input,
            output_factory=DummyOutput,
        )
        pipe_input.send_text("remember me\r")
        assert prompt_session.prompt("> ") == "remember me"

    assert "remember me" in history_file.read_text()


def test_the_history_keeps_the_newest_turns_only(tmp_path) -> None:
    """The cap bounds what is read back; the file keeps every turn."""

    from llm_cli_py.session import interactive as interactive_mod

    history_file = tmp_path / "history"
    stored = FileHistory(str(history_file))
    for turn in ("one", "two", "three"):
        stored.append_string(turn)

    history = interactive_mod.LimitedFileHistory(history_file, limit=2)

    assert list(history.load_history_strings()) == ["three", "two"]


def test_a_history_path_that_cannot_be_resolved_falls_back_to_memory(capsys, monkeypatch) -> None:
    """A ``~`` that cannot be expanded is reported; the run continues in memory."""
    from llm_cli_py import main as main_module
    from llm_cli_py.session import interactive as interactive_mod

    def unresolvable(_value: object) -> None:
        msg = "Could not determine home directory."
        raise RuntimeError(msg)

    monkeypatch.setenv(interactive_mod.ENV_PROMPT_HISTORY_FILE, "~/hist/turns")
    monkeypatch.setattr(interactive_mod, "clean_path", unresolvable)

    assert main_module._prompt_session_or_none() is None
    assert "history" in capsys.readouterr().out
