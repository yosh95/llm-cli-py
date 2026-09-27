"""CLI entry point: argument handling and prompt assembly (no network)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_cli_py.main import build_parser, initialize_tools


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
    assert "llm-cli-py 0.2.0" in capsys.readouterr().out


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


def test_read_prompt_ends_the_line_after_ctrl_c(capsys) -> None:
    """Ctrl+C does not send the newline Enter would: ``read_prompt`` prints it."""
    from llm_cli_py.session import interactive as interactive_mod

    with patch("builtins.input", side_effect=KeyboardInterrupt), pytest.raises(KeyboardInterrupt):
        interactive_mod.read_prompt()

    assert capsys.readouterr().out == "\n"


def test_read_prompt_ends_the_line_at_end_of_input(capsys) -> None:
    """Ctrl+D leaves the cursor on the ``> `` line too; the shell prompt follows it."""
    from llm_cli_py.session import interactive as interactive_mod

    with patch("builtins.input", side_effect=EOFError), pytest.raises(EOFError):
        interactive_mod.read_prompt()

    assert capsys.readouterr().out == "\n"


def test_read_prompt_adds_nothing_after_enter(capsys) -> None:
    """A line finished with Enter is already terminated: no second newline."""
    from llm_cli_py.session import interactive as interactive_mod

    with patch("builtins.input", return_value="hello"):
        assert interactive_mod.read_prompt() == "hello"

    assert capsys.readouterr().out == ""
