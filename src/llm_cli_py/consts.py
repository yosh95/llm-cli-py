"""Constants for llm-cli-py."""
# ── Timeout constants (seconds) ────────────────────────────────────

DEFAULT_REQUEST_TIMEOUT: int = 300
"""Default timeout for LLM API requests (chat completions)."""

DEFAULT_MODEL_FETCH_TIMEOUT: int = 30
"""Default timeout for fetching model lists from providers."""


DEFAULT_URL_FETCH_TIMEOUT: int = 30
"""Default timeout for fetching URL content from CLI arguments."""


# ── Environment variable names ─────────────────────────────────────

ENV_API_KEY = "LLM_CLI_API_KEY"
"""Environment variable for the LLM API key."""

ENV_API_URL = "LLM_CLI_API_URL"
"""Environment variable for the LLM API URL."""

ENV_MODEL = "LLM_CLI_MODEL"
"""Environment variable for the default LLM model."""

DEFAULT_API_URL = "http://localhost:11434/v1"
"""Default LLM API base URL (OpenAI-compatible endpoint)."""


# ── Prompt history / chat log ─────────────────────────────────


ENV_PROMPT_HISTORY_FILE = "LLM_CLI_PROMPT_HISTORY_FILE"
"""Environment variable for the prompt history file.

When set, the interactive prompt history is persisted to this file across
invocations. When unset, history is kept only in memory for the current run.
"""

ENV_CHAT_LOG_FILE = "LLM_CLI_CHAT_LOG_FILE"
"""Environment variable for the session (chat) log file.

When set, the conversation (same content as ``/dump``, including per-message
timestamps) is written to this file after every message that is added to the
history -- not only when the session ends -- so the log survives a crash or an
abrupt termination. When unset, nothing is saved.
"""

ENV_CHAT_LOG_APPEND = "LLM_CLI_CHAT_LOG_APPEND"
"""Environment variable switching the chat log to append mode.

When set to a truthy value (``1``/``true``/``yes``/``on``), only newly added
messages are appended to the log file as ``[[message]]`` TOML tables instead of
rewriting the whole conversation. The file then keeps history across sessions
and stays valid TOML as a whole (useful with a per-day file name). When unset,
the log file always mirrors the current session.
"""
