"""Constants for llm-cli-py."""

# ── Timeout constants (seconds) ────────────────────────────────────

DEFAULT_REQUEST_TIMEOUT: int = 300
"""Default timeout for LLM API requests (chat completions)."""

DEFAULT_CHILD_PYTHON: str = "python3"
"""Interpreter used by ``execute_python`` when the env var below is unset."""


# ── Environment variable names ─────────────────────────────────────

ENV_API_KEY = "LLM_CLI_API_KEY"
"""Environment variable for the LLM API key."""

ENV_API_URL = "LLM_CLI_API_URL"
"""Environment variable for the LLM API URL."""

ENV_MODEL = "LLM_CLI_MODEL"
"""Environment variable for the default LLM model."""

ENV_SYSTEM_PROMPT = "LLM_CLI_SYSTEM_PROMPT"
"""Environment variable holding the system prompt.

Read exactly once, by the composition root (``main``), and passed to the client
as a constructor argument -- it is seeded as the first conversation message.
When unset or empty, no system message is sent.
"""

DEFAULT_API_URL = "http://localhost:11434/v1"
"""Example LLM API base URL (OpenAI-compatible endpoint).

Shown in ``--help`` and in the "not configured" error; there is no implicit
default -- the API URL must be configured explicitly.
"""


# ── Logging ────────────────────────────────────────────────────────

ENV_LOG_LEVEL = "LOG_LEVEL"
"""Environment variable setting the root logger level (e.g. ``DEBUG``)."""
