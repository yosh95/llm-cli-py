"""Tests for CLI argument parsing and subcommands."""

from __future__ import annotations

import pytest

from llm_cli_py.main import build_parser


class TestCliParser:
    """Test CLI argument parser."""

    def test_default_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None
        assert args.sources == []
        assert args.model is None
        assert args.disable_verifier is False
        assert args.plain_input is False

    def test_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-m", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_api_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--api-url", "https://custom.api.com/v1"])
        assert args.api_url == "https://custom.api.com/v1"

    def test_api_key(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--api-key", "sk-custom-key"])
        assert args.api_key == "sk-custom-key"

    def test_api_key_default_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.api_key is None

    def test_disable_verifier(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--disable-verifier"])
        assert args.disable_verifier is True

    def test_plain_input_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--plain-input"])
        assert args.plain_input is True

    def test_verifier_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--verifier-model", "gpt-4o-mini"])
        assert args.verifier_model == "gpt-4o-mini"

    def test_verifier_model_default_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.verifier_model is None

    def test_timeout_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--request-timeout",
                "60",
            ]
        )
        assert args.request_timeout == 60

    def test_sources(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-s", "hello", "-s", "world"])
        assert args.sources == ["hello", "world"]

    def test_models_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models"])
        assert args.command == "models"

    def test_base_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-D", "/custom/path"])
        assert args.base_dir == "/custom/path"

    def test_version_flag_short(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-V"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out

    def test_version_flag_long(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "llm-cli-py" in captured.out
        assert "0.1.0" in captured.out
