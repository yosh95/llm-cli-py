"""Constants and path management for llm-cli-py."""

from pathlib import Path

_HISTORY_FILE = Path.home() / ".llm-cli-py-history"
"""Prompt-toolkit input history file (a dotfile in the user's home dir)."""


def history_file_path() -> Path:
    """Return the path to the prompt_toolkit input-history file.

    The only persisted state is the interactive prompt history, so it is
    stored directly as a dotfile in the user's home directory.
    """
    return _HISTORY_FILE


# ── Timeout constants (seconds) ────────────────────────────────────

DEFAULT_REQUEST_TIMEOUT: int = 300
"""Default timeout for LLM API requests (chat completions)."""

DEFAULT_VERIFIER_TIMEOUT: int = 30
"""Default timeout for verifier API requests."""

DEFAULT_WEB_SEARCH_TIMEOUT: int = 30
"""Default timeout for Brave Search API requests."""

DEFAULT_MODEL_FETCH_TIMEOUT: int = 30
"""Default timeout for fetching model lists from providers."""


DEFAULT_URL_FETCH_TIMEOUT: int = 30
"""Default timeout for fetching URL content from CLI arguments."""

DEFAULT_PYTHON_EXEC_TIMEOUT: int = 30
"""Default timeout for Python code execution in the subprocess."""


# ── Other application-wide defaults ────────────────────────────────

MAX_TOOL_ITERATIONS: int = 200
"""Maximum number of tool-call iterations per user request."""


# ── Environment variable names ─────────────────────────────────────

ENV_API_KEY = "LLM_CLI_API_KEY"
"""Environment variable for the LLM API key."""

ENV_API_URL = "LLM_CLI_API_URL"
"""Environment variable for the LLM API URL."""

ENV_MODEL = "LLM_CLI_MODEL"
"""Environment variable for the default LLM model."""

ENV_VERIFIER_MODEL = "LLM_CLI_VERIFIER_MODEL"
"""Verifier model (separate from the main LLM)."""

# ── Approval mode constants ──────────────────────────────────────
# Controls how tool calls are approved before execution:
#   APPROVAL_MODE_VERIFIER -- use the LLM-based verifier (default).
#   APPROVAL_MODE_MANUAL  -- no verifier; ask the human to approve every
#                            tool call (HITL / human-in-the-loop).
#   APPROVAL_MODE_AUTO    -- no verifier; auto-approve every tool call.
APPROVAL_MODE_VERIFIER = "verifier"
APPROVAL_MODE_MANUAL = "manual"
APPROVAL_MODE_AUTO = "auto"

APPROVAL_MODES = (APPROVAL_MODE_VERIFIER, APPROVAL_MODE_MANUAL, APPROVAL_MODE_AUTO)

"""Environment variable overriding the LLM API request timeout in seconds."""

ENV_DISABLE_REASONING = "LLM_CLI_DISABLE_REASONING"
"""Environment variable controlling whether model thinking/reasoning is disabled.

Set to "1"/"true" (default) to disable thinking/reasoning for all providers and
models. Set to "0"/"false" to keep reasoning enabled.
"""

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
