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
| `LLM_CLI_INCLUDE_REASONING` | Set to `0` or `false` to disable the reasoning parameter. |
| `LLM_CLI_HIDE_MODEL` | Set to `true` to redact model names in output (for demos). |
| `LLM_CLI_PROXY_URL` | Proxy URL. When set, both LLM API and Brave Search requests go through this proxy. The proxy handles API key and model injection server-side. |
| `PROXY_MAX_BODY_SIZE` | Max accepted request body size in bytes on the proxy (default 100 MiB). Raise it if you get HTTP 413 `Content Too Large` on long conversations that accumulate large tool results. |
| `BRAVE_API_KEY` | Brave Search API key (required for `web_search` tool when not using proxy). |

## Usage

```bash
# One-shot query
llm-cli-py -m gpt-4o "What is the capital of France?"

# With file input
llm-cli-py -m gpt-4o -s README.md "Summarize this file"

# Interactive mode
llm-cli-py -m gpt-4o

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
| `/quit`, `/q` | Exit session |
| `/clear`, `/c` | Clear conversation |
| `/info`, `/i` | Show session info |
| `/dump` | Dump conversation as TOML |
| `/verifier`, `/v` | Toggle verifier on/off |

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
