"""Tests for configuration (env var names and defaults)."""

from llm_cli_py.consts import (
    DEFAULT_API_URL,
    ENV_API_KEY,
    ENV_API_URL,
    ENV_CHAT_LOG_APPEND,
    ENV_CHAT_LOG_FILE,
    ENV_LOG_LEVEL,
    ENV_MODEL,
    ENV_SYSTEM_PROMPT,
    TRUTHY_VALUES,
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
    assert (ENV_SYSTEM_PROMPT, ENV_LOG_LEVEL) == ("LLM_CLI_SYSTEM_PROMPT", "LOG_LEVEL")


def test_default_api_url_is_only_an_example() -> None:
    """The URL is a documented example (help/error text), never an implicit default."""
    assert DEFAULT_API_URL.startswith("http")


def test_truthy_values_are_lower_case_and_shared() -> None:
    """Both the chat-log and other flags use one definition of "true"."""
    assert TRUTHY_VALUES == ("1", "true", "yes", "on")
