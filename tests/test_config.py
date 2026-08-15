"""Tests for configuration (env vars)."""

from __future__ import annotations

from llm_cli_py.consts import (
    ENV_API_KEY,
    ENV_API_URL,
    ENV_MODEL,
    ENV_PROXY_URL,
    ENV_VERIFIER_MODEL,
)


class TestConsts:
    """Test constants and path functions."""

    def test_env_var_names(self) -> None:
        assert ENV_API_KEY == "LLM_CLI_API_KEY"
        assert ENV_API_URL == "LLM_CLI_API_URL"
        assert ENV_MODEL == "LLM_CLI_MODEL"

    def test_verifier_model_env_var(self) -> None:
        assert ENV_VERIFIER_MODEL == "LLM_CLI_VERIFIER_MODEL"

    def test_proxy_url_env_var(self) -> None:
        assert ENV_PROXY_URL == "LLM_CLI_PROXY_URL"
