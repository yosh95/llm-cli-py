# llm-cli-py

**Unified OpenAI-Compatible CLI for AI Agents (Python Edition)**

A small command-line client for any OpenAI-compatible LLM API, with a built-in
Python-execution tool.

## Features

- **OpenAI-compatible** — works with any provider exposing `/chat/completions`
  (OpenAI, local servers, …)
- **Python execution** — one built-in tool, `execute_python`, which everything
  else (web search, file work, data analysis, …) is built on
- **No provider lock-in** — capabilities are described in `LLM_CLI_SYSTEM_PROMPT`,
  so the agent calls APIs itself via `execute_python`; switch providers by
  editing an env var
- **Interactive session** — a rich `> ` prompt powered by `prompt_toolkit`, one turn
  per line, kept open until you end it (Ctrl+D); a prompt on the command line is
  answered first, then the same prompt continues
- **Minimal dependencies** — `requests`, `prompt_toolkit`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Or, without activating anything:

```bash
make install        # pip install -e . into the current environment
make install-dev    # the same, plus the dev tools (pytest, ruff, mypy)
pipx install -e .   # global, independent of .venv
```

`make install` does not create a virtual environment: activate one first (as
above) or let `pipx` own it. `make clean` deletes `$(VENV)` (`.venv` by
default).

On Debian/Ubuntu, `python3 -m venv` needs the `python3-venv` package
(`sudo apt install python3-venv`).

### Two ways to start it

The install provides **two interchangeable entry points** -- same code, same
flags, same output:

| Command | What runs |
|---|---|
| `llm-cli-py` | The console script from `[project.scripts]`. `pip` writes a launcher next to it: a text script on Linux/macOS, a small **unsigned `.exe`** in `Scripts\` on Windows. |
| `python -m llm_cli_py` | The package's `__main__.py`, run by the interpreter you invoke. No launcher file is generated or executed. |

Use whichever you like; both are supported and tested. On Windows the second
one is worth knowing about: a freshly written unsigned `.exe` has no reputation
yet, so SmartScreen / Smart App Control can hold `llm-cli-py` back, while
`python -m llm_cli_py` starts from the already-trusted `python.exe` and is
never gated that way.

## Usage

```bash
export LLM_CLI_API_URL="http://localhost:11434/v1"   # required: any OpenAI-compatible endpoint
export LLM_CLI_API_KEY="your-api-key"                # optional for local instances
export LLM_CLI_MODEL="gpt-4o"                        # optional, or use -m

llm-cli-py -m gpt-4o "What is the capital of France?"   # answer, then keep prompting
llm-cli-py -m gpt-4o -s "Summarize this:" -s "$(cat README.md)"
llm-cli-py -m gpt-4o                                   # interactive

python -m llm_cli_py -m gpt-4o "What is the capital of France?"   # identical
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

There are no slash commands and no stdin piping: input is read from the terminal
by `prompt_toolkit` (a pipe is not a source of turns), the prompt keeps no
history -- the arrow keys do not bring an earlier line back -- and every line you
type at the `> ` prompt is sent to the model as a new turn. A line starting with
`/` is ordinary prompt text, so nothing needs escaping.

End the session with end-of-input -- **Ctrl+D** on Linux/macOS, **Ctrl+Z then
Enter** on Windows (that is where the terminal reports end-of-file; Windows has
no Ctrl+D). A stray **Ctrl+C** at the prompt abandons the line being typed and
returns to a fresh prompt; `Ctrl+C` while a request or a tool is running
interrupts that operation instead. Aborting the prompt already ends the line
`> ` opened, so the next output -- the rule of the following turn, or your shell
prompt after Ctrl+D -- follows the abandoned prompt directly, with no blank line
in between.

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

Code is also refused, before it runs, when it calls `subprocess` with
`shell=True` and an argv list: the shell executes only the first element and
silently ignores the rest, so the command would not do what the code says.

## Development

```bash
make install-dev   # pip install -e ".[dev]" (no venv is created)
make test          # pytest -v
make check         # ruff check
make format        # ruff format
make typecheck     # mypy
make check-all     # format + check + typecheck + test
make clean         # remove caches, build artifacts and .venv
make clean-all     # alias of clean
```

Run `make help` for the same list with one-line descriptions. The runtime
dependencies are `requests` and `prompt_toolkit` (declared in `pyproject.toml`);
everything else is the standard library. Dev tools live under the `dev` extra;
there is no lock file, so pin exact versions with `==` if you need reproducible
installs.

## License

Apache-2.0
