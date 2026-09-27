"""CLI entry point: argument handling and prompt assembly (no network)."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-m", "gpt-4o", "What is the capital of France?"], "What is the capital of France?"),
        (["-s", "context", "sum this up"], "context\nsum this up"),
        (["-s", "a", "-s", "b"], "a\nb"),
        (["-s", "https://example.com/page"], "https://example.com/page"),
        (["-s", "/etc/hostname"], "/etc/hostname"),
    ],
)
def test_prompt_is_passed_verbatim(argv: list[str], expected: str, monkeypatch) -> None:
    """``-s`` values and trailing words are one prompt: nothing is read or fetched."""
    monkeypatch.setenv("LLM_CLI_API_URL", "https://api.example.com/v1")
    monkeypatch.setattr("sys.argv", ["llm-cli-py", *argv])

    with patch("llm_cli_py.main.LlmApiClient"), patch("llm_cli_py.main.run_interactive") as run:
        from llm_cli_py import main as main_module

        main_module.main()

    assert run.call_args.args[1] == expected


def test_client_receives_the_resolved_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CLI_API_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_CLI_MODEL", "env-model")
    monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "be helpful")
    monkeypatch.setattr("sys.argv", ["llm-cli-py", "--api-key", "sk-1"])

    with patch("llm_cli_py.main.LlmApiClient") as client_cls, patch("llm_cli_py.main.run_interactive"):
        from llm_cli_py import main as main_module

        main_module.main()

    assert client_cls.call_args.kwargs == {
        "model": "env-model",
        "api_url": "https://api.example.com/v1",
        "api_key": "sk-1",
        "timeout": 300,
        "system_prompt": "be helpful",
    }


def test_end_of_input_ends_the_session(monkeypatch) -> None:
    """Ctrl+D (or Ctrl+Z then Enter on Windows) reaches the loop as EOF."""
    monkeypatch.setenv("LLM_CLI_API_URL", "https://api.example.com/v1")
    monkeypatch.setattr("sys.argv", ["llm-cli-py"])
    from llm_cli_py.session import interactive as interactive_mod

    with (
        patch("llm_cli_py.main.LlmApiClient"),
        patch.object(interactive_mod, "read_prompt", side_effect=EOFError),
    ):
        from llm_cli_py import main as main_module

        main_module.main()  # must return instead of looping forever


def test_ctrl_c_at_the_prompt_keeps_the_session_alive(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CLI_API_URL", "https://api.example.com/v1")
    monkeypatch.setattr("sys.argv", ["llm-cli-py"])
    from llm_cli_py.session import interactive as interactive_mod

    with (
        patch("llm_cli_py.main.LlmApiClient"),
        patch.object(interactive_mod, "read_prompt", side_effect=[KeyboardInterrupt, "hello", EOFError]),
        patch.object(interactive_mod, "handle_user_input") as handle,
    ):
        from llm_cli_py import main as main_module

        main_module.main()

    handle.assert_called_once()  # the interrupt only returned to the prompt


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


def test_the_prompt_has_no_history() -> None:
    """The CLI has no prompt history, so the arrow keys cannot replay a turn."""
    from prompt_toolkit.history import DummyHistory

    from llm_cli_py.session import interactive as interactive_mod

    assert isinstance(interactive_mod._make_prompt_session().history, DummyHistory)
