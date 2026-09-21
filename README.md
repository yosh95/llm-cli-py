# llm-cli-py

**Unified OpenAI-Compatible CLI for AI Agents (Python Edition)**

A small command-line client for any OpenAI-compatible LLM API, with streaming
output and a built-in Python-execution tool.

## Features

- **OpenAI-compatible** — works with any provider exposing `/chat/completions`
  (OpenAI, local servers, …)
- **Python execution** — one built-in tool, `execute_python`, which everything
  else (web search, file work, data analysis, …) is built on
- **No provider lock-in** — capabilities are described in `LLM_CLI_SYSTEM_PROMPT`,
  so the agent calls APIs itself via `execute_python`; switch providers by
  editing an env var
- **Interactive session** — persistent chat with history and slash commands
- **Always streaming** — answer tokens are rendered live as they arrive

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Or, without activating anything:

```bash
make install        # installs into .venv
make install-dev    # .venv + dev tools (pytest, ruff)
pipx install -e .   # global, independent of .venv
```

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package
(`sudo apt install python3-venv`).

## Usage

```bash
export LLM_CLI_API_URL="http://localhost:11434/v1"   # any OpenAI-compatible endpoint
export LLM_CLI_API_KEY="your-api-key"                # optional for local instances
export LLM_CLI_MODEL="gpt-4o"                        # optional, or use -m

llm-cli-py -m gpt-4o "What is the capital of France?"   # one-shot
llm-cli-py -m gpt-4o -s README.md "Summarize this file" # with file/URL input
llm-cli-py -m gpt-4o                                   # interactive
llm-cli-py models                                      # list models
```

### Slash Commands (Interactive Mode)

| Command | Description |
|---|---|
| `/help`, `/h` | Show help |
| `/info`, `/i` | Show session info |
| `/dump` | Dump conversation as TOML |
| `/quit`, `/q`, `/exit` | Exit session |

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_CLI_API_URL` | Base URL of the OpenAI-compatible API. Default: `http://localhost:11434/v1` |
| `LLM_CLI_API_KEY` | API key (optional for local instances). Overridden by `--api-key`. |
| `LLM_CLI_MODEL` | Default model. Overridden by `-m`. |
| `LLM_CLI_SYSTEM_PROMPT` | System prompt, read once at startup and seeded as the first message. When unset, none is sent. It can also describe extra capabilities (e.g. a search endpoint and the env var holding its key) that the agent calls via `execute_python`. |
| `LLM_CLI_PROMPT_HISTORY_FILE` | File to persist prompt input history across runs. |
| `LLM_CLI_CHAT_LOG_FILE` | File to write the conversation to (same as `/dump`, flushed after every message). |
| `LLM_CLI_CHAT_LOG_APPEND` | `1`/`true`/`yes`/`on` to append new messages instead of rewriting the file. |
| `LOG_LEVEL` | Root logger level (e.g. `DEBUG`, `INFO`). |
| `DEBUG_HTTP` | `1`/`true` for raw HTTP request/response debugging. |

### Example: Web Search Without a Search Tool

Describe the API in `LLM_CLI_SYSTEM_PROMPT` and the agent calls it itself:

```bash
export WEB_SEARCH_API_KEY="sk-..."
export LLM_CLI_SYSTEM_PROMPT='You are a helpful coding agent.
When you need web search, call the API below from execute_python:
- Endpoint: POST https://ollama.com/api/web_search
- Headers: Content-Type: application/json and Authorization: Bearer <value of WEB_SEARCH_API_KEY>
- Body: {"query": "...", "max_results": 5}
The key is available in the environment variable WEB_SEARCH_API_KEY.'
```

Prefer referencing the key by env var name (as above) — the prompt appears in
`/dump` and the chat log.

## Tools

`execute_python` is the only tool, by design: it runs Python code in a sandboxed
subprocess and returns the exit code plus stdout/stderr. Tool calls are always
executed automatically (no approval prompt).

## Development

```bash
make install-dev   # create .venv + pip install -e ".[dev]"
make test          # pytest -v
make check         # ruff check
make format        # ruff format
```

Dependencies live in `pyproject.toml` (runtime under `[project] dependencies`,
dev tools under the `dev` extra). There is no lock file: edit the lists and run
`python -m pip install -e ".[dev]"`. For reproducible installs, pin exact
versions with `==`.

## License

Apache-2.0
