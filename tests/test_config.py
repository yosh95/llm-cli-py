"""Tests for configuration (env vars)."""

from __future__ import annotations

from llm_cli_py.consts import (
    ENV_API_KEY,
    ENV_API_URL,
    ENV_MODEL,
)


class TestConsts:
    """Test constants and path functions."""

    def test_env_var_names(self) -> None:
        assert ENV_API_KEY == "LLM_CLI_API_KEY"
        assert ENV_API_URL == "LLM_CLI_API_URL"
        assert ENV_MODEL == "LLM_CLI_MODEL"
