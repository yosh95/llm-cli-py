# llm-cli-py

**Unified OpenAI-Compatible CLI for AI Agents (Python Edition)**

A command-line interface for interacting with any OpenAI-compatible LLM API, with built-in tool support for Python execution and web search.

## Features

- **OpenAI-Compatible API** — Works with any provider that supports the `/chat/completions` endpoint (OpenAI, Anthropic via OpenRouter, local instances, etc.)
- **Proxy Support** — Set `LLM_CLI_PROXY_URL` to route all requests through `llm_proxy.py`, which handles API key and model injection server-side
- **Web Search** — Built-in `web_search` tool using Brave Search API
- **Python Execution** — Built-in `execute_python` tool for running Python code
- **Interactive Session** — Persistent chat with history and slash commands
- **Tool Verification** — Optional LLM-based verifier to approve/reject tool calls
- **Markdown Rendering** — CJK-friendly Markdown output with Rich

## Quick Start

```bash
# Install
pip install -e .            # CLI only
pip install -e ".[proxy]"    # CLI + proxy server (includes aiohttp)

# Set environment variables
export LLM_CLI_API_URL="http://localhost:11434/v1"   # or any OpenAI-compatible endpoint
export LLM_CLI_API_KEY="your-api-key"                 # optional for local instances
export LLM_CLI_MODEL="gpt-4o"                         # optional, can use -m flag

# Run
llm-cli-py -m gpt-4o
```

### Using the Proxy

```bash
# On the proxy server (install with proxy extra first):
pip install -e "[proxy]"
export LLM_CLI_API_KEY="your-api-key"
export LLM_CLI_MODEL="gpt-4o"
export BRAVE_API_KEY="your-brave-api-key"
python proxy/llm_proxy.py

# On any client PC:
export LLM_CLI_PROXY_URL=http://<proxy-ip>:8080
# No API keys or model needed!
llm-cli-py
```

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_CLI_API_URL` | Base URL of the OpenAI-compatible API (e.g. `http://localhost:11434/v1`). Default: `http://localhost:11434/v1` |
| `LLM_CLI_API_KEY` | API key for the LLM endpoint. Optional for local instances. Can be overridden with `--api-key`. |
| `LLM_CLI_MODEL` | Default model to use (e.g. `gpt-4o`). Can be overridden with `-m`. |
| `LLM_CLI_VERIFIER_MODEL` | Separate model for tool call verification. Defaults to the main model. |
| `LLM_CLI_PROXY_URL` | Proxy URL. When set, both LLM API and Brave Search requests go through this proxy. The proxy handles API key and model injection server-side. |
| `LLM_CLI_REQUEST_TIMEOUT` | Override the LLM API request timeout in seconds (default 1800). Useful for cloud reasoning models that can take minutes before their first token. |
| `PROXY_MAX_BODY_SIZE` | Max accepted request body size in bytes on the proxy (default 100 MiB). Raise it if you get HTTP 413 `Content Too Large` on long conversations that accumulate large tool results. |
| `PROXY_PORT` | Port the proxy listens on (default `8080`). |
| `BRAVE_API_KEY` | Brave Search API key (required for `web_search` tool when not using proxy). |
| `OPENROUTER_API_KEY` | OpenRouter API key (required for the `openrouter` subcommand). |
| `SYSTEM_PROMPT` | Custom system prompt. When unset, a default prompt with today's date is used. |
| `LOG_LEVEL` | Set the root logger level (e.g. `DEBUG`, `INFO`). |
| `DEBUG_HTTP` | Set to `1`/`true` to enable raw HTTP request/response debugging. |

## Usage

```bash
# One-shot query
llm-cli-py -m gpt-4o "What is the capital of France?"

# With file input
llm-cli-py -m gpt-4o -s README.md "Summarize this file"

# Interactive mode
llm-cli-py -m gpt-4o

# Interactive mode with plain input() (no prompt_toolkit terminal handling)
# Useful when driven by an automation harness / non-tty stdin.
llm-cli-py -m gpt-4o --plain-input

# List available models
llm-cli-py models

# Override API URL and API key on the command line
llm-cli-py --api-url https://api.example.com/v1 --api-key sk-your-key -m gpt-4o

# Using proxy
llm-cli-py --proxy-url http://proxy-server:8080
```

### Slash Commands (Interactive Mode)

| Command | Description |
|---|---|
| `/help`, `/h` | Show help |
| `/quit`, `/q`, `/exit` | Exit session |
| `/clear`, `/c` | Clear conversation |
| `/info`, `/i` | Show session info |
| `/dump` | Dump conversation as TOML |
| `/verifier`, `/v` | Toggle verifier on/off |

## Interactive Input Backends

The interactive session normally uses `prompt_toolkit`, which provides history,
completion, and multiline editing. `prompt_toolkit` manipulates the terminal
directly (raw mode, alternate screen buffer, its own event loop). When an
external automation harness (e.g. HTB) drives the same tty, these can conflict
and leave the terminal in a broken state (no echo, unresponsive keyboard).

To avoid this:

- `--plain-input` — use plain `input()` instead of `prompt_toolkit`.
- **Auto-fallback** — if stdin is not a tty (piped / automated), the CLI
  automatically switches to plain `input()` regardless of the flag.

## Streaming

The CLI requests responses in streaming mode (`stream: true`). Reasoning and
answer tokens are rendered live as they arrive, so long-running thinking traces
(e.g. DeepSeek V4 Flash on Ollama Cloud, which can think for minutes before its
first answer token) are visible instead of appearing to hang.

- **Reasoning / thinking** tokens are streamed and shown under a
  `Reasoning (thinking process):` heading.
- **Answer** tokens stream under an `Assistant:` heading.
- **Tool calls** are buffered across chunks and only executed once their
  arguments are complete. If a provider emits a broken/truncated tool-call
  argument chunk, that turn is transparently re-requested in non-streaming
  mode so a well-formed call is obtained.
- The **verifier** model is also streamed: its reasoning / thinking is shown
  live under a `Verifier reasoning:` heading and its JSON verdict streams under
  a `Verifier:` heading before the approve/reject verdict is printed. If the
  verifier stream yields no content, it transparently falls back to a
  non-streaming request so verification always completes.
- The reasoning trace is round-tripped back to the API on assistant messages
  that include tool calls (required by DeepSeek V4, which otherwise rejects
  the next request with HTTP 400).

When routing through `llm_proxy.py`, the proxy relays the SSE stream to the
client live, so streaming works end-to-end over the LAN proxy as well.

If you need to disable streaming (e.g. to inspect raw non-streamed responses),
you can lower the request timeout or adjust the client; by default streaming
is always on.

## Tools

1. **`execute_python`** — Execute Python code in a sandboxed subprocess
2. **`web_search`** — Web search via Brave Search API (uses `BRAVE_API_KEY` env var, or proxy if `LLM_CLI_PROXY_URL` is set)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"
# Include proxy support if you also want to test the proxy:
pip install -e ".[proxy,dev]"

# Run tests
pytest

# Type check
mypy src

# Lint
ruff check src
```
