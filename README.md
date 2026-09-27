# llm-cli-py

**Unified OpenAI-Compatible CLI for AI Agents (Python Edition)**

A small command-line client for any OpenAI-compatible LLM API, with a built-in
Python-execution tool. **Standard library only** -- no runtime dependencies.

## Features

- **OpenAI-compatible** — works with any provider exposing `/chat/completions`
  (OpenAI, local servers, …)
- **Python execution** — one built-in tool, `execute_python`, which everything
  else (web search, file work, data analysis, …) is built on
- **No provider lock-in** — capabilities are described in `LLM_CLI_SYSTEM_PROMPT`,
  so the agent calls APIs itself via `execute_python`; switch providers by
  editing an env var
- **Interactive session** — a plain `> ` prompt (`input()`), one turn per line
- **One-shot mode** — pass a prompt and the process answers once and exits
- **Zero dependencies** — `urllib.request`, `input()`, `subprocess`; nothing to install

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Or, without activating anything:

```bash
make install        # installs into .venv
make install-dev    # .venv + dev tools (pytest, ruff, mypy)
pipx install -e .   # global, independent of .venv
```

The runtime needs nothing else: no `requests`, no `prompt_toolkit`, no TOML
writer. Only the dev extra (pytest, ruff, mypy) installs anything.

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package
(`sudo apt install python3-venv`).

## Usage

```bash
export LLM_CLI_API_URL="http://localhost:11434/v1"   # required: any OpenAI-compatible endpoint
export LLM_CLI_API_KEY="your-api-key"                # optional for local instances
export LLM_CLI_MODEL="gpt-4o"                        # optional, or use -m

llm-cli-py -m gpt-4o "What is the capital of France?"   # one-shot prompt
llm-cli-py -m gpt-4o -s "Summarize this:" -s "$(cat README.md)"
llm-cli-py -m gpt-4o                                   # interactive
```

### How the prompt is assembled

`-s/--source` is a **pure text channel**: the value is sent to the model
verbatim -- no file is read, no URL is fetched. `-s` values each go on their own
line, then the trailing positional words on a final line, so

```bash
llm-cli-py -s "Context:" -s "$(cat notes.md)" "Summarize this"
```

sends `Context:
<notes>
Summarize this`. Read any file yourself with a shell
substitution (`"$(cat file)"`) or, for larger work, let the agent read it with
`execute_python`.

There are no slash commands, no prompt history and no stdin piping: every line
you type at the `> ` prompt is sent to the model as a new turn. A line starting
with `/` is ordinary prompt text, so nothing needs escaping.

End the session with end-of-input -- **Ctrl+D** on Linux/macOS, **Ctrl+Z then
Enter** on Windows (that is where the terminal reports end-of-file; Windows has
no Ctrl+D). A stray **Ctrl+C** at the prompt abandons the line being typed and
returns to a fresh prompt; `Ctrl+C` while a request or a tool is running
interrupts that operation instead. Neither key transmits what Enter would, so
the CLI prints the missing newline itself: whatever is written after a stray
Ctrl+C, and your shell prompt after Ctrl+D, start on a line of their own
instead of being appended to `> `.

With a prompt on the command line the CLI answers it first and then keeps
prompting; assistant answers and tool output go to stdout, so output can be
redirected while you type:

```bash
llm-cli-py -m gpt-4o "Write a haiku about JSON" > haiku.txt
```

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_CLI_API_URL` | Base URL of the OpenAI-compatible API. Required (e.g. `http://localhost:11434/v1`); can be given with `--api-url`. |
| `LLM_CLI_API_KEY` | API key (optional for local instances). Overridden by `--api-key`. |
| `LLM_CLI_MODEL` | Default model. Overridden by `-m`. |
| `LLM_CLI_SYSTEM_PROMPT` | System prompt, read once at startup and seeded as the first message. When unset, none is sent. It can also describe extra capabilities (e.g. a search endpoint and the env var holding its key) that the agent calls via `execute_python`. |
| `LLM_CLI_PYTHON_EXEC` | Interpreter used by `execute_python` (defaults to the CLI's own interpreter). |
| `LOG_LEVEL` | Root logger level (e.g. `DEBUG`, `INFO`). |

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

Prefer referencing the key by env var name (as above), not by value.

## Tools

`execute_python` is the only tool, by design: it runs Python code in a
subprocess and returns the exit code plus stdout/stderr. Its output is sent back
to the model, so the agent can read files, call APIs and run commands itself. Tool calls are always
executed automatically (no approval prompt). The child runs with the same
environment and file-system access as the CLI itself, and runs without a
timeout: interrupt it with Ctrl+C, which kills the code and everything it
spawned.

A tool must return an `ExecResult` or a `ToolError`; any other return value is
reported to the model as an explicit error rather than being silently treated as
output.

## Development

```bash
make install-dev   # create .venv + pip install -e ".[dev]"
make test          # pytest -v
make check         # ruff check
make format        # ruff format
make typecheck     # mypy
make check-all     # format + check + typecheck + test
make clean         # remove caches and build artifacts (keeps .venv)
make clean-all     # clean, and remove .venv too
```

`pyproject.toml` keeps `dependencies = []` on purpose: the package runs on the
standard library alone. Dev tools live under the `dev` extra; there is no lock
file, so pin exact versions with `==` if you need reproducible installs.

## License

Apache-2.0
