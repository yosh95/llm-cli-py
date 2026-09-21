"""Tests for configuration (env var names and defaults)."""

from llm_cli_py.consts import (
    DEFAULT_API_URL,
    ENV_API_KEY,
    ENV_API_URL,
    ENV_CHAT_LOG_APPEND,
    ENV_CHAT_LOG_FILE,
    ENV_MODEL,
)


def test_env_var_names_are_stable() -> None:
    assert (ENV_API_KEY, ENV_API_URL, ENV_MODEL) == (
        "LLM_CLI_API_KEY",
        "LLM_CLI_API_URL",
        "LLM_CLI_MODEL",
    )
    assert (ENV_CHAT_LOG_FILE, ENV_CHAT_LOG_APPEND) == (
        "LLM_CLI_CHAT_LOG_FILE",
        "LLM_CLI_CHAT_LOG_APPEND",
    )
    assert DEFAULT_API_URL.startswith("http")
