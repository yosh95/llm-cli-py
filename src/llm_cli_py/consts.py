"""Constants and path management for llm-cli-py."""

from pathlib import Path

_BASE_DIR: Path | None = None


def get_base_dir() -> Path:
    """Get the base directory for config and logs."""
    global _BASE_DIR
    if _BASE_DIR is not None:
        return _BASE_DIR
    return Path.home() / ".llm-cli-py"


def set_base_dir(path: Path) -> None:
    """Set a custom base directory."""
    global _BASE_DIR
    _BASE_DIR = path


def history_file_path() -> Path:
    return get_base_dir() / "history.txt"


# ── Timeout constants (seconds) ────────────────────────────────────

DEFAULT_REQUEST_TIMEOUT: int = 300
"""Default timeout for LLM API requests (chat completions)."""

DEFAULT_VERIFIER_TIMEOUT: int = 60
"""Default timeout for verifier API requests."""

DEFAULT_WEB_SEARCH_TIMEOUT: int = 60
"""Default timeout for Brave Search API requests."""

DEFAULT_MODEL_FETCH_TIMEOUT: int = 30
"""Default timeout for fetching model lists from providers."""


DEFAULT_URL_FETCH_TIMEOUT: int = 30
"""Default timeout for fetching URL content from CLI arguments."""

DEFAULT_PYTHON_EXEC_TIMEOUT: int = 60
"""Default timeout for Python code execution in the subprocess."""


# ── Other application-wide defaults ────────────────────────────────

MAX_TOOL_ITERATIONS: int = 500
"""Maximum number of tool-call iterations per user request."""


# ── Environment variable names ─────────────────────────────────────

ENV_API_KEY = "LLM_CLI_API_KEY"
"""Environment variable for the LLM API key."""

ENV_API_URL = "LLM_CLI_API_URL"
"""Environment variable for the LLM API URL."""

ENV_MODEL = "LLM_CLI_MODEL"
"""Environment variable for the default LLM model."""

ENV_VERIFIER_MODEL = "LLM_CLI_VERIFIER_MODEL"
"""Environment variable for the verifier model (separate from the main LLM)."""

ENV_PROXY_URL = "LLM_CLI_PROXY_URL"
"""Environment variable for the proxy URL.
When set, both LLM API and Brave Search requests go through this proxy.
The proxy handles API key and model injection server-side.
"""

DEFAULT_API_URL = "http://localhost:11434/v1"
"""Default LLM API base URL (OpenAI-compatible endpoint)."""

# Brave Search API
ENV_BRAVE_API_KEY = "BRAVE_API_KEY"
"""Environment variable for the Brave Search API key."""

BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/llm/context"
"""Brave Search API endpoint (LLM Context API).

Returns pre-extracted web content (text, tables, code blocks) ranked by
relevance and ready for LLM consumption.
"""

BRAVE_SEARCH_MAX_RETRIES: int = 3
"""Maximum number of retries for Brave Search API requests."""
