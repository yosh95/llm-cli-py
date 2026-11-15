"""Constants for llm-cli-py."""

# ── Timeout constants (seconds) ────────────────────────────────────

DEFAULT_REQUEST_TIMEOUT: int = 300
"""Default timeout for LLM API requests (chat completions)."""

DEFAULT_TOOL_TIMEOUT: int = 30
"""Default limit (30 seconds) on one ``execute_python`` run.

A tool call blocks the turn: the model waits for the result before it can do
anything else. A run that outlives this limit is killed -- with every process it
started -- and the timeout goes back to the model as that call's result, so a
runaway script ends the turn instead of hanging it. Work that is meant to take
longer belongs in the background (the tool description says so), where a later
call can check on it.
"""

DEFAULT_CHILD_PYTHON: str = "python3"
"""Interpreter used by ``execute_python`` when the env var below is unset."""


# ── Environment variable names ─────────────────────────────────────

ENV_API_KEY = "LLM_CLI_API_KEY"
"""Environment variable for the LLM API key."""

ENV_API_URL = "LLM_CLI_API_URL"
"""Environment variable for the LLM API URL."""

ENV_MODEL = "LLM_CLI_MODEL"
"""Environment variable for the default LLM model."""

ENV_LOG_FILE = "LLM_CLI_LOG_FILE"
"""Environment variable holding the file conversations are written to.

Unset (the default) means no log is kept at all. When set, the whole
conversation -- user turns, assistant replies, tool calls and tool results -- is
written there as JSON, and rewritten as it grows, so what is on disk is always
the conversation as it stood.

The file accumulates across runs: each run is one entry in its ``sessions``
array, so starting the CLI again adds a session rather than replacing the file.
Nothing in it is read back into the conversation, so a new run starts empty.
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
