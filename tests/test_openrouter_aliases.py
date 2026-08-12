"""Tests for the OpenRouter subcommand (aliases and dispatch)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

# Add src to path so llm_cli_py is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_cli_py.commands.openrouter import (
    CREDITS_ALIASES,
    MODEL_ALIASES,
    OPENROUTER_ALIASES,
    RANKINGS_ALIASES,
    run_openrouter,
)


def make_args(or_command: str | None, model_slug: str | None = None) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for run_openrouter."""
    args = argparse.Namespace()
    args.or_command = or_command
    args.model_slug = model_slug
    return args


class TestConstants:
    """Test the alias constant definitions."""

    def test_openrouter_aliases(self) -> None:
        assert "or" in OPENROUTER_ALIASES
        assert "opr" in OPENROUTER_ALIASES
        assert "ort" in OPENROUTER_ALIASES

    def test_rankings_aliases(self) -> None:
        assert "r" in RANKINGS_ALIASES
        assert "rank" in RANKINGS_ALIASES

    def test_credits_aliases(self) -> None:
        assert "c" in CREDITS_ALIASES
        assert "credit" in CREDITS_ALIASES

    def test_model_aliases(self) -> None:
        assert "m" in MODEL_ALIASES


class TestRunOpenrouterDispatch:
    """Test run_openrouter dispatch logic with aliases."""

    def test_full_rankings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full 'rankings' subcommand calls show_rankings."""
        from llm_cli_py.commands import openrouter as mod

        mock = lambda: None  # noqa: E731
        monkeypatch.setattr(mod, "show_rankings", mock)
        monkeypatch.setattr(mod, "_get_api_key", lambda: "key")

        called = []
        monkeypatch.setattr(mod, "show_rankings", lambda: called.append("rankings"))
        run_openrouter(make_args("rankings"))
        assert called == ["rankings"]

    def test_alias_r_rankings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Alias 'r' calls show_rankings."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "_get_api_key", lambda: "key")
        monkeypatch.setattr(mod, "show_rankings", lambda: called.append("rankings"))
        run_openrouter(make_args("r"))
        assert called == ["rankings"]

    def test_alias_rank_rankings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Alias 'rank' calls show_rankings."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "_get_api_key", lambda: "key")
        monkeypatch.setattr(mod, "show_rankings", lambda: called.append("rankings"))
        run_openrouter(make_args("rank"))
        assert called == ["rankings"]

    def test_full_credits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full 'credits' subcommand calls show_credits."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "_get_api_key", lambda: "key")
        monkeypatch.setattr(mod, "show_credits", lambda: called.append("credits"))
        run_openrouter(make_args("credits"))
        assert called == ["credits"]

    def test_alias_c_credits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Alias 'c' calls show_credits."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "_get_api_key", lambda: "key")
        monkeypatch.setattr(mod, "show_credits", lambda: called.append("credits"))
        run_openrouter(make_args("c"))
        assert called == ["credits"]

    def test_alias_credit_credits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Alias 'credit' calls show_credits."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "_get_api_key", lambda: "key")
        monkeypatch.setattr(mod, "show_credits", lambda: called.append("credits"))
        run_openrouter(make_args("credit"))
        assert called == ["credits"]

    def test_full_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full 'model' subcommand calls show_model with slug."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "show_model", lambda s: called.append(s))
        run_openrouter(make_args("model", "openai/gpt-4o"))
        assert called == ["openai/gpt-4o"]

    def test_alias_m_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Alias 'm' calls show_model with slug."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "show_model", lambda s: called.append(s))
        run_openrouter(make_args("m", "openai/gpt-4o"))
        assert called == ["openai/gpt-4o"]

    def test_default_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No subcommand shows both rankings and credits."""
        from llm_cli_py.commands import openrouter as mod

        called = []
        monkeypatch.setattr(mod, "_get_api_key", lambda: "key")
        monkeypatch.setattr(mod, "show_rankings", lambda: called.append("rankings"))
        monkeypatch.setattr(mod, "show_credits", lambda: called.append("credits"))
        run_openrouter(make_args(None))
        assert called == ["rankings", "credits"]
