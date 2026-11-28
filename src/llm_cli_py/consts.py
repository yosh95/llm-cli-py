"""Constants for llm-cli-py."""

DEFAULT_CHILD_PYTHON: str = "python3"
"""Interpreter used by ``execute_python`` when the env var below is unset."""


# ── Environment variable names ─────────────────────────────────────

ENV_API_KEY = "LLM_CLI_API_KEY"
"""Environment variable for the LLM API key."""

ENV_API_URL = "LLM_CLI_API_URL"
"""Environment variable for the LLM API URL."""

ENV_MODEL = "LLM_CLI_MODEL"
"""Environment variable for the default LLM model."""

ENV_LOG_DIR = "LLM_CLI_LOG_DIR"
"""Environment variable holding the directory conversations are written to.

Unset (the default) means no log is kept at all. When set, each run writes one
JSON Lines file there, named after the moment it started, holding the whole
conversation -- user turns, assistant replies, tool calls and tool results --
as one record per line, appended as the conversation grows.

One run, one file: starting the CLI again creates another file beside the ones
already there instead of rewriting them, so nothing logged before is read or
written back, and a new run starts from its project instructions (``LLM_CLI.md``)
alone. The directory is where a past run is looked up afterwards.
"""

ENV_PROMPT_HISTORY_FILE = "LLM_CLI_PROMPT_HISTORY_FILE"
"""Environment variable holding the file the prompt history (arrow keys) uses.

Unset (the default) keeps the prompt history in memory only: the arrow keys
still recall earlier turns of the current run, but nothing is written to disk
and nothing survives the session.
"""

DEFAULT_API_URL = "http://localhost:11434/v1"
"""Example LLM API base URL (OpenAI-compatible endpoint).

Shown in ``--help`` and in the "not configured" error; there is no implicit
default -- the API URL must be configured explicitly.
"""
