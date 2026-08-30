# llm-cli-py

**Unified OpenAI-Compatible CLI for AI Agents (Python Edition)**

A command-line interface for interacting with any OpenAI-compatible LLM API, with built-in tool support for Python execution and web search.

## Features

- **OpenAI-Compatible API** — Works with any provider that supports the `/chat/completions` endpoint (OpenAI, Anthropic via OpenRouter, local instances, etc.)
- **Web Search** — Built-in `web_search` tool using Brave Search API
- **Python Execution** — Built-in `execute_python` tool for running Python code
- **Interactive Session** — Persistent chat with history and slash commands
- **Automatic Tool Execution** — Tools run automatically without asking for user confirmation
- **Markdown Rendering** — CJK-friendly Markdown output with Rich

## Quick Start

```bash
# Install
pip install -e .

# Set environment variables
export LLM_CLI_API_URL="http://localhost:11434/v1"   # or any OpenAI-compatible endpoint
export LLM_CLI_API_KEY="your-api-key"                 # optional for local instances
export LLM_CLI_MODEL="gpt-4o"                         # optional, can use -m flag

# Run
llm-cli-py -m gpt-4o
```

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_CLI_API_URL` | Base URL of the OpenAI-compatible API (e.g. `http://localhost:11434/v1`). Default: `http://localhost:11434/v1` |
| `LLM_CLI_API_KEY` | API key for the LLM endpoint. Optional for local instances. Can be overridden with `--api-key`. |
| `LLM_CLI_MODEL` | Default model to use (e.g. `gpt-4o`). Can be overridden with `-m`. |
| `BRAVE_SEARCH_API_KEY` | Brave Search API key (required for the `web_search` tool). |
| `OPENROUTER_API_KEY` | OpenRouter API key (required for the `openrouter` subcommand). |
| `SYSTEM_PROMPT` | System prompt to send with every request. When unset or empty, no system prompt is sent (no default/date prompt is injected). |
| `LOG_LEVEL` | Set the root logger level (e.g. `DEBUG`, `INFO`). |
| `DEBUG_HTTP` | Set to `1`/`true` to enable raw HTTP request/response debugging. |
| `LLM_CLI_PROMPT_HISTORY_FILE` | Path to persist the interactive prompt input history across invocations. If unset, prompt history is kept only in memory for the current run. |
| `LLM_CLI_CHAT_LOG_FILE` | Path to write the full conversation (same content as `/dump`) when an interactive session ends. If unset, the conversation is not saved to disk. |

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
```

### OpenRouter Subcommand

The `openrouter` subcommand provides quick access to OpenRouter rankings, credits,
and model details. Short aliases make it even faster to type:

| Alias | Description |
|---|---|
| `o` | Short for `openrouter` |

Sub-subcommands also have short forms:

| Alias | Full | Description |
|---|---|---|
| `r`, `rank` | `rankings` | Model usage rankings |
| `c`, `credit` | `credits` | Credit / spending info |
| `m` | `model` | Model details (requires model slug) |

```bash
# Full command
llm-cli-py openrouter rankings
llm-cli-py openrouter credits
llm-cli-py openrouter model openai/gpt-4o

# Short aliases
llm-cli-py o r           # rankings
llm-cli-py o c           # credits
llm-cli-py o m openai/gpt-4o   # model details

# No subcommand → show both rankings and credits
llm-cli-py o
```

### Slash Commands (Interactive Mode)

| Command | Description |
|---|---|
| `/help`, `/h` | Show help |
| `/quit`, `/q`, `/exit` | Exit session |
| `/clear`, `/c` | Clear conversation |
| `/info`, `/i` | Show session info |
| `/dump` | Dump conversation as TOML |

## Interactive Input

The interactive session uses `prompt_toolkit`, which provides history,
completion, and multiline editing. The `PromptSession` is created exactly once
and used by the main chat loop, so
prompt_toolkit's terminal handling (raw mode, alternate screen buffer, its own
event loop) is initialized only once and never conflicts with itself.

- **Prompt history** — When `LLM_CLI_PROMPT_HISTORY_FILE` is set, your prompt
  input history is persisted to that file across invocations. If it is unset,
  history lives only in memory for the current run.
- **Session log** — When `LLM_CLI_CHAT_LOG_FILE` is set, the full conversation
  (the same content as the `/dump` command) is written to that file when the
  interactive session ends (via `/quit`, Ctrl+D, or otherwise). If it is unset,
  nothing is saved.

## Streaming

The CLI always requests responses in streaming mode (`stream: true`). Answer
tokens are rendered live as they arrive.

- **Answer** tokens stream under an `Assistant:` heading.
- **Tool calls** are buffered across chunks and only executed once their
  arguments are complete. If a provider emits a broken/truncated tool-call
  argument chunk, the call is surfaced with an explicit error and the turn
  exits (no silent non-streaming re-request).

## Tools

1. **`execute_python`** — Execute Python code in a sandboxed subprocess
2. **`web_search`** — Web search via Brave Search API (uses `BRAVE_SEARCH_API_KEY` env var)

## Tool Call Approval

Tool calls are **always executed automatically** without prompting the user for
confirmation. There is no manual/auto mode and no human-in-the-loop approval.

## Tool Display

When a tool runs, the CLI shows a one-line indicator
(`🔨 Executing tool: <name>...`), the tool parameters, then the tool result.
A horizontal rule (`─`) is drawn before the result block so the tool call and
its result are easy to tell apart:

- **Call** — `🔨 Executing tool: <name>...` followed by parameters.
- **Result** — a `───` rule, then the `Tool Result:` block.

Parameter and result display per tool:

- **`web_search`** — all parameters (the search query) are shown in full; the
  returned hits (title, URL, snippet) are printed in the result block.
- **`execute_python`** — the full `code` parameter is shown, and the result
  block includes the exit code plus stdout/stderr.
- Other tools show no parameters by default (result block still printed).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src
```
