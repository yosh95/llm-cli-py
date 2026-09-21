"""Integration tests for the CLI entry point (no network, no TTY)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_cli_py.main import build_parser, initialize_tools


def test_default_tools_are_registered() -> None:
    assert "execute_python" in initialize_tools()
    assert [t.name for t in initialize_tools().get_schemas()] == ["execute_python"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], {"command": None, "sources": [], "model": None, "api_key": None}),
        (["-m", "gpt-4o"], {"model": "gpt-4o"}),
        (["--api-url", "https://x/v1"], {"api_url": "https://x/v1"}),
        (["--api-key", "sk-1"], {"api_key": "sk-1"}),
        (["-s", "a", "-s", "b"], {"sources": ["a", "b"]}),
        (["models"], {"command": "models"}),
    ],
)
def test_parser_arguments(argv: list[str], expected: dict) -> None:
    args = build_parser().parse_args(argv)
    for key, value in expected.items():
        assert getattr(args, key) == value


@pytest.mark.parametrize(("flag", "expected"), [("-V", "0.1.0"), ("--version", "llm-cli-py 0.1.0")])
def test_version_flag(capsys, flag: str, expected: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([flag])
    assert exc.value.code == 0
    assert expected in capsys.readouterr().out


def test_main_exits_without_api_url(capsys, monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLI_API_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["llm-cli-py", "-m", "gpt-4o"])
    from llm_cli_py import main as main_module

    with pytest.raises(SystemExit):
        main_module.main()
    assert "LLM_CLI_API_URL" in capsys.readouterr().out


def test_main_runs_without_api_key(capsys, monkeypatch) -> None:
    """The API key is optional (e.g. local Ollama): no warning is emitted."""
    monkeypatch.delenv("LLM_CLI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_CLI_API_URL", "https://api.example.com/v1")
    monkeypatch.setattr("sys.argv", ["llm-cli-py", "-m", "gpt-4o"])
    from llm_cli_py import main as main_module

    with patch("llm_cli_py.main.LlmApiClient"), patch("llm_cli_py.main.run_interactive"):
        main_module.main()
    assert "LLM_CLI_API_KEY" not in capsys.readouterr().out


def test_models_subcommand_lists_sorted_models(capsys, monkeypatch) -> None:
    monkeypatch.setenv("LLM_CLI_API_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_CLI_API_KEY", "key")
    monkeypatch.setattr("sys.argv", ["llm-cli-py", "models"])
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": [{"id": "model-b"}, {"id": "model-a"}]}

    with patch("llm_cli_py.commands.models.requests.get", return_value=resp) as get:
        from llm_cli_py import main as main_module

        main_module.main()

    out = capsys.readouterr().out
    assert out.index("model-a") < out.index("model-b")
    get.assert_called_once_with(
        "https://api.example.com/v1/models", headers={"Authorization": "Bearer key"}, timeout=30
    )
