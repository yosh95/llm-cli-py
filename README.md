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
  answered first, then the same prompt continues. `prompt_toolkit`'s own key
  bindings are kept: the arrow keys recall earlier turns, Ctrl+X Ctrl+E opens the
  line in your editor, and Ctrl+Z suspends the CLI on Unix (where there is a
  `SIGTSTP` to send)
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
by `prompt_toolkit` (a pipe is not a source of turns), and every line you type at
the `> ` prompt is sent to the model as a new turn. A line starting with `/` is
ordinary prompt text, so nothing needs escaping.

The prompt is a full `prompt_toolkit` session, so its standard key bindings are
there:

- **Arrow keys** (and Ctrl+R) walk back through earlier turns. This is the
  prompt's own history, not the conversation -- recalling a line only puts it
  back on the line, and it is pressing Enter that sends it.
- **Ctrl+X Ctrl+E** opens the line in `$VISUAL` / `$EDITOR` and applies what you
  save as the turn, which is what makes a long, multi-line prompt pleasant to
  write. The editor is the usual `$VISUAL`, then `$EDITOR`, then the platform's
  list (nano, vi, ...).
- **Ctrl+Z** suspends the CLI -- on Unix, and there only. The process is stopped
  the way any job is, so the shell gets the terminal back (`[1]+ Stopped
  llm-cli-py`), and `fg` puts you back at the `> ` prompt with the half-typed
  line still on it. Windows has no `SIGTSTP` to send, so there the key keeps
  `prompt_toolkit`'s own meaning, which is the end-of-input below.

By default that history lives in memory: the arrow keys recall earlier turns of
the current run and nothing is written to disk. Set
`LLM_CLI_PROMPT_HISTORY_FILE=/path/to/history` and it is kept in that file
instead, so turns come back in later runs too; the newest 1000 are read back at
the prompt, older ones stay in the file unread. If the file cannot be opened,
the CLI reports it and carries on in memory.

End the session with end-of-input -- **Ctrl+D** on Linux/macOS, **Ctrl+Z then
Enter** on Windows (that is where the terminal reports end-of-file; Windows has
no Ctrl+D, and no `SIGTSTP` for Ctrl+Z to suspend with either). A stray
**Ctrl+C** at the prompt abandons the line being typed and returns to a fresh
prompt; `Ctrl+C` while a request or a tool is running interrupts that operation
instead. Aborting the prompt already ends the line `> ` opened, so the next
output -- the rule of the following turn, or your shell prompt after Ctrl+D --
follows the abandoned prompt directly, with no blank line in between.

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
| `LLM_CLI_PROMPT_HISTORY_FILE` | File the prompt history is kept in, so the arrow keys recall earlier turns across runs. Unset (the default) keeps the history in memory for the current run only. |
| `LLM_CLI_LOG_FILE` | File the whole conversation is logged to as JSON, tool calls and tool results included. Unset (the default) writes no log at all. |
| `LLM_CLI_PYTHON_EXEC` | Interpreter used by `execute_python` (defaults to the CLI's own interpreter). |

### Writing Paths in the Environment

Both `LLM_CLI_LOG_FILE` and `LLM_CLI_PROMPT_HISTORY_FILE` are read the way you
wrote them, without needing the shell to help:

- **`~`** is expanded against the home directory (`USERPROFILE` on Windows), so
  `~/logs/session.json` works.
- **Spaces need no escaping**: `C:\Users\me\My Logs\session.json` is fine, as
  is `~/My Logs/history`.
- **Surrounding quotes are dropped**, because `cmd.exe` keeps them in the value:
  `set LLM_CLI_LOG_FILE="C:\My Logs\log.json"` works, and so does the same line
  in PowerShell, or GitBash's `~/My Logs/log.json`. One pair, `"..."` or `'...'`,
  at either end.
- **Surrounding blanks are ignored.**
- **Parent directories are created**, in both cases.
- `$VAR` is *not* expanded -- that is the shell's job, so write
  `LLM_CLI_LOG_FILE=~/logs/$(date +%F).json` and let the shell do it.

A path that cannot be resolved (no home directory to expand `~` against) or
cannot be opened is reported, and that feature is dropped for the run: without a
usable history file the prompt keeps its history in memory, and without a usable
log file nothing is logged. Neither stops the session.

### Logging the Conversation

Set `LLM_CLI_LOG_FILE` and the whole conversation is written to that file as
JSON -- user turns, assistant replies, tool calls (name and arguments) and tool
results -- so a run can be read back afterwards:

```bash
LLM_CLI_LOG_FILE=~/.local/state/llm-cli/session.json llm-cli-py -m gpt-4o
```

The file is rewritten as the conversation grows, so what is on disk is always the
conversation as it stood, including after a run that ends badly: a request that
fails, a tool that hangs, a `kill`, or a Ctrl+C halfway through a turn all leave
everything recorded up to that point, as one parseable JSON document. (The
exception is a turn that is deliberately *discarded* -- a tool call whose
arguments did not parse, or an interrupted tool run -- which is rolled back out
of the conversation, and out of the log with it, because that is what the model
will be sent next.)

Writes are atomic (temporary file plus rename), so a reader never sees half a
document. Unset `LLM_CLI_LOG_FILE` -- the default -- and nothing is written at
all; `LLM_CLI_LOG_FILE=/dev/null` also turns it off. A path that cannot be
written is reported at startup and the run continues without a log.

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
