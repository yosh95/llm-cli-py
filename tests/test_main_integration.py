"""Integration tests for the main CLI entry point (without network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_cli_py.main import build_parser, initialize_tools


class TestInitializeTools:
    """Test tool registry initialization."""

    def test_default_tools(self) -> None:
        registry = initialize_tools()
        assert "execute_python" in registry


class TestBuildParser:
    """Test argument parser already covered partially; here ensure main flow paths."""

    def test_parser_one_shot(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-s", "hello", "-m", "gpt-4o"])
        assert args.sources == ["hello"]
        assert args.model == "gpt-4o"


class TestMainIntegration:
    """Test main() with mocked network and interactive loop."""

    def test_main_exits_without_api_url(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LLM_CLI_API_URL is not set, main should error."""
        monkeypatch.delenv("LLM_CLI_API_URL", raising=False)
        monkeypatch.delenv("LLM_CLI_API_KEY", raising=False)
        monkeypatch.setattr("sys.argv", ["llm-cli-py", "-m", "gpt-4o"])
        from llm_cli_py import main as main_module

        with pytest.raises(SystemExit):
            main_module.main()
        captured = capsys.readouterr()
        assert "LLM_CLI_API_URL" in captured.out

    def test_main_works_without_api_key(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API key is optional (e.g., local Ollama). No warning should be shown."""
        monkeypatch.delenv("LLM_CLI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_CLI_API_URL", "https://api.example.com/v1")
        monkeypatch.setattr("sys.argv", ["llm-cli-py", "-m", "gpt-4o"])
        from llm_cli_py import main as main_module

        # Should not exit - API key is optional
        with (
            patch("llm_cli_py.main.LlmApiClient"),
            patch("llm_cli_py.main.run_interactive"),
        ):
            main_module.main()
        captured = capsys.readouterr()
        # No warning about missing API key
        assert "LLM_CLI_API_KEY" not in captured.out

    def test_models_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = {
            "LLM_CLI_API_KEY": "key",
            "LLM_CLI_API_URL": "https://api.example.com/v1",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "model-a"},
                {"id": "model-b"},
            ],
        }

        with (
            patch.dict("os.environ", env),
            patch("llm_cli_py.commands.models.requests.get", return_value=mock_resp) as mock_get,
            patch("sys.argv", ["llm-cli-py", "models"]),
        ):
            from llm_cli_py import main as main_module

            main_module.main()

        captured = capsys.readouterr()
        assert "model-a" in captured.out
        mock_get.assert_called_once_with(
            "https://api.example.com/v1/models",
            headers={"Authorization": "Bearer key"},
            timeout=30,
        )
