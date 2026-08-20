"""Constants for llm-cli-py."""

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


# ── Other application-wide defaults ────────────────────────────────

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

DEFAULT_API_URL = "http://localhost:11434/v1"
"""Default LLM API base URL (OpenAI-compatible endpoint)."""

# Brave Search API
ENV_BRAVE_SEARCH_API_KEY = "BRAVE_SEARCH_API_KEY"
"""Environment variable for the Brave Search API key."""

BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/llm/context"
"""Brave Search API endpoint (LLM Context API).

Returns pre-extracted web content (text, tables, code blocks) ranked by
relevance and ready for LLM consumption.
"""

BRAVE_SEARCH_MAX_RETRIES: int = 3
"""Maximum number of retries for Brave Search API requests."""
