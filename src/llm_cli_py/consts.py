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

ENV_LOG_FILE = "LLM_CLI_LOG_FILE"
"""Environment variable holding the file conversations are written to.

Unset (the default) means no log is kept at all. When set, the whole
conversation -- user turns, assistant replies, tool calls and tool results -- is
written there as JSON, and rewritten as it grows, so what is on disk is always
the conversation as it stood.

The file accumulates across runs: each run is one entry in its ``sessions``
array, so starting the CLI again adds a session rather than replacing the file.
Nothing in it is read back into the conversation, so a new run starts from its
project instructions (``LLM_CLI.md``) alone.
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

# ── Waiting for long work ──────────────────────────────────────────

ENV_JOBS_DIR = "LLM_CLI_JOBS_DIR"
"""Environment variable naming the directory background jobs live in.

Unset means a directory under the system temp dir
(``DEFAULT_JOBS_DIR_NAME``), which is enough for one machine and one user. A job
leaves its code, its output and its exit code there, so a run that outlives the
CLI can still be read and stopped by the next one.
"""

DEFAULT_JOBS_DIR_NAME = "llm-cli-py-jobs"
"""Directory name used under the system temp dir when the env var is unset."""

DEFAULT_WAIT_S: float = 600.0
"""How long a ``wait`` waits when the call does not say: ten minutes.

Ten minutes is chosen to cover the common long jobs -- a build, a test suite, a
server coming up -- without being so long that a mistyped condition costs an
hour of a turn. A call that needs longer says so; the number is the model's to
raise, not a cap.
"""

MAX_WAIT_S: float = 86_400.0
"""The longest ``wait`` accepted: one day.

Not a limit on how long work may run -- the job keeps running past it, and the
result says so -- but a sanity check on the number a call passes, so a typo
(``timeout_s: 60000000``) is refused instead of parking the turn for two years.
"""

DEFAULT_WAIT_POLL_S: float = 0.05
"""How often a ``wait`` looks at its condition, in seconds.

Twenty looks a second is below what a human notices and well below what a
filesystem ``stat`` costs, and the loop is in the tool: the model never writes a
sleep, so the cadence does not have to be a guess about the work.
"""

MAX_RETURNED_OUTPUT = 60_000
"""How much of a job's output is handed back to the model, in characters.

A job's output goes to a file, so it can be megabytes; what the model gets is
the tail of it, which is where an error or a final summary is. The result says
how big the whole thing was and that it was cut, so a reader knows to ask for
less or to read the file itself.
"""

MAX_RETURNED_LINES = 300
"""The most lines a ``tail_lines`` request is honoured with."""
