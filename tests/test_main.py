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

    def test_sources(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-s", "hello", "-s", "world"])
        assert args.sources == ["hello", "world"]

    def test_models_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["models"])
        assert args.command == "models"

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

    # ── OpenRouter aliases ─────────────────────────────────────────

    def test_openrouter_subcommand(self) -> None:
        """The full 'openrouter' subcommand still works."""
        parser = build_parser()
        args = parser.parse_args(["openrouter", "rankings"])
        assert args.command == "openrouter"
        assert args.or_command == "rankings"

    def test_openrouter_alias_o(self) -> None:
        """Short alias 'o' maps to openrouter."""
        parser = build_parser()
        args = parser.parse_args(["o", "rankings"])
        assert args.command == "o"
        assert args.or_command == "rankings"

    def test_openrouter_alias_no_subcommand(self) -> None:
        """Alias with no subcommand defaults to rankings + credits."""
        parser = build_parser()
        args = parser.parse_args(["o"])
        assert args.command == "o"
        assert args.or_command is None

    def test_openrouter_subcommand_short_aliases(self) -> None:
        """Subcommands support short aliases: r, c, m."""
        parser = build_parser()

        args = parser.parse_args(["o", "r"])
        assert args.or_command == "r"

        args = parser.parse_args(["o", "c"])
        assert args.or_command == "c"

        args = parser.parse_args(["o", "m", "anthropic/claude-sonnet"])
        assert args.or_command == "m"
        assert args.model_slug == "anthropic/claude-sonnet"

    def test_openrouter_subcommand_rank_alias(self) -> None:
        """Subcommand 'rank' is an alias for 'rankings'."""
        parser = build_parser()
        args = parser.parse_args(["o", "rank"])
        assert args.or_command == "rank"

    def test_openrouter_subcommand_credit_alias(self) -> None:
        """Subcommand 'credit' is an alias for 'credits'."""
        parser = build_parser()
        args = parser.parse_args(["o", "credit"])
        assert args.or_command == "credit"
