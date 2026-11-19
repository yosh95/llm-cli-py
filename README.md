# llm-cli-py

**Unified OpenAI-Compatible CLI for AI Agents (Python Edition)**

A small command-line client for any OpenAI-compatible LLM API, with a built-in
Python-execution tool.

## Features

- **OpenAI-compatible** — works with any provider exposing `/chat/completions`
  (OpenAI, local servers, …)
- **Python execution** — one built-in tool, `execute_python`, which everything
  else (web search, file work, data analysis, …) is built on
- **No provider lock-in** — capabilities are described in the prompt, so the
  agent calls APIs itself via `execute_python`; switch providers by editing an
  env var
- **Interactive session** — a rich `> ` prompt powered by `prompt_toolkit`, one turn
  per line, kept open until you end it (Ctrl+D); a prompt on the command line is
  answered first, then the same prompt continues. `prompt_toolkit`'s own key
  bindings are kept: the arrow keys recall earlier turns, Ctrl+X Ctrl+E opens the
  line in your editor, and Ctrl+Z suspends the CLI on Unix (where there is a
  `SIGTSTP` to send)
- **Time on screen** — the lines that record something (an answer, a report, a
  tool's result) are stamped with the wall-clock time, every rule carries the
  session's uptime, and a request or a tool run shows how long it has been running
- **Readable output** — answers are rendered as Markdown, code in a tool call is
  syntax-highlighted, results are coloured by outcome: on a terminal only, so a
  redirected transcript stays plain text
- **`/raw`** — type it at the prompt to print the whole conversation as plain
  Markdown instead of the rendered transcript, so the answer can be copied out
  of the terminal with its Markdown intact
- **No waits cut short** — a request or a tool run is never killed for taking
  its time: the status line says something is still running, and Ctrl+C is what
  ends a wait that should not go on
- **Few dependencies** — `requests`, `prompt_toolkit`, `rich`

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

There is no stdin piping: input is read from the terminal by `prompt_toolkit` (a
pipe is not a source of turns), and every line you type at the `> ` prompt is
sent to the model as a new turn. The one exception is `/raw` (below), which
shows the conversation again instead. Otherwise a line starting with `/` is
ordinary prompt text, so nothing needs escaping.

### `/raw`: the conversation as Markdown

Type `/raw` at the prompt and the whole conversation — your turns, the model's
answers, every tool call and tool result — is printed again as plain Markdown,
exactly as it stands:

````text
[09:32:10] ──────────────────────────────────────────────── [+00:03:21]
# Conversation (gpt-4o)

## user

この表をMarkdownでまとめて

## assistant

# 見出し

- 箇条書き
- **強調** と `コード`

## assistant

**Tool call:** `execute_python` (`call_1`)

```json
{
  "code": "print(1)"
}
```

## tool (call_1)

```json
{
  "stdout": "1\n",
  "stderr": "",
  "exit_code": 0
}
```
````

It is there for copying. The answers above it were *rendered* — bullets drawn as
`•`, code coloured, rules and status lines in between — and a screen full of
rendering is not the Markdown the model wrote. `/raw` prints the conversation
itself, unwrapped and unstyled, so selecting it and pasting it elsewhere gives
you the Markdown, tables and fences included:

- **No slash commands otherwise.** Only the bare word `/raw` is a command;
  `/raw please`, `/rawx` or `raw` are prompt text and are sent to the model.
- **No other change.** `/raw` is not a turn: nothing is sent, nothing is
  recorded, and the `LLM_CLI_LOG_FILE` log is left exactly as it was.
- **Long lines are not broken.** The text is written as it arrived and the
  terminal folds what does not fit on one screen line, so a long paragraph is
  still one line in the paste. (A terminal that wraps *and* re-indents what it
  wrapped — some Windows consoles — will still show the fold, but the text
  itself is untouched.)
- **Tool payloads are JSON code blocks**, indented, and a tool message names the
  call it answers (`## tool (call_1)`), so a call and its result can be matched
  after the structure is gone. Arguments that did not parse are shown as they
  arrived, fenced, rather than executed or rewritten.
- **Not on the command line.** `llm-cli-py /raw` sends `/raw` to the model as a
  prompt; the command belongs at the `> ` prompt. (A prompt given on the command
  line is then shown by `/raw` like any other message.)

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

### Time in the transcript

The lines that record something — a report, a tool's result — are stamped with
the wall-clock time they were written, and every rule carries, at its right end,
how long the session has been running:

```text
[09:32:10] ──────────────────────────────────────────────── [+00:00:00]
gpt-4o is thinking... 0.0s
gpt-4o is thinking... 2.5s
[09:32:13] ──────────────────────────────────────────────── [+00:00:02]
[Tool] Executing tool: execute_python...
Args:
  [code] import time; time.sleep(3); print("done")
[Tool] execute_python running... 0.0s
[Tool] execute_python running... 3.0s
[09:32:16] ──────────────────────────────────────────────── [+00:00:05]
[09:32:16] Tool Result:
    Exit code: 0
    [stdout]
      done
```

Progress is the exception. The status line of a running request or tool call, and
the block that announces the call, carry no stamp: they are about what is
happening now rather than about an event, and between them they already show what
the clock would say — the rule above the call was stamped when it started, and the
status line counts the seconds since. On a status line a stamp would be the one
thing on it that never needed to change while the line was redrawn.

A request to the model and a tool run are the two places where the CLI can go
quiet for a long time. Each shows a status line with the time elapsed, redrawn in
place ten times a second, and the last reading stays on screen as the operation's
duration. A counter that keeps moving is what tells a slow provider or a slow
script apart from a stuck one; the stamps on the surrounding lines then say when
it started and how long the gap really was.

The status line begins at the left edge and nothing sits to its left, so the
running frames and the finished reading start in the same column, and the ticking
number never moves anything else around.

A request has no time limit: no `timeout` is handed to `requests`, so the
connection and the answer are waited for as long as they take. The status line
keeps counting, which is what tells a slow model apart from a dead one, and
Ctrl+C ends the wait. Connection failures are still retried with the usual
backoff, and so are the statuses that ask for it (429 and 5xx), which is why one
of those can repeat the wait for the answer before it gives up.

Durations are measured on a monotonic clock, so a system time adjustment during
a long wait cannot produce a negative or absurd number. Stamps come from the wall
clock, so they line up with the log file, another terminal, or your watch.

When stdout is redirected to a file (or anything else that is not a terminal) the
line is not redrawn -- carriage returns in a file are unreadable -- so nothing is
written while the operation runs, and the same line, with the same duration, is
printed once when it finishes. Stamps and uptimes are unaffected.

### What a terminal gets, and what a file gets

Output is rendered with [`rich`](https://github.com/Textualize/rich), but only
where it helps. In a terminal:

- the assistant's answer is rendered as **Markdown** -- models write it, and
  headings, lists and fenced code read better as such;
- code in a tool call is **syntax-highlighted**, so the Python an agent is about
  to run looks like code -- and a line wider than the terminal continues on the
  next one instead of being cut at the right edge, so the URL a script fetches
  and the arguments after it stay readable;
- a tool result is **coloured by outcome**: a green `Exit code: 0`, a red failed
  one, `[stdout]`/`[stderr]` as labels -- and a line of its output that is wider
  than the terminal is folded onto the next row under the same label instead of
  running past the right edge, so the end of a long payload stays readable;
- `NO_COLOR=1` turns the colours off (the usual convention, and `rich` honours
  it).

Everywhere else -- redirected to a file, piped to `pager`/`grep`, captured by a
test -- the output is **plain text**, exactly as it was written: no escape
sequences, no `•` bullets in place of `-`, no padding around code. A transcript
saved to a file is its own source, and `llm-cli-py ... > answer.md` still writes
the answer the model wrote. Markup characters in a model's or a tool's own text
(`[INFO]`, `[1, 2]`) are never interpreted as styling.

## Environment Variables

| Variable | Description |
|---|---|
| `LLM_CLI_API_URL` | Base URL of the OpenAI-compatible API. Required (e.g. `http://localhost:11434/v1`); can be given with `--api-url`. |
| `LLM_CLI_API_KEY` | API key (optional for local instances). Overridden by `--api-key`. |
| `LLM_CLI_MODEL` | Default model. Overridden by `-m`. |
| `LLM_CLI_PROMPT_HISTORY_FILE` | File the prompt history is kept in, so the arrow keys recall earlier turns across runs. Unset (the default) keeps the history in memory for the current run only. |
| `LLM_CLI_LOG_FILE` | File every conversation is logged to as JSON, tool calls and tool results included. Each run is added to it as a new session, so earlier runs survive a restart (they are never loaded back into the conversation). Unset (the default) writes no log at all. |
| `LLM_CLI_PYTHON_EXEC` | Interpreter used by `execute_python` (defaults to the CLI's own interpreter). |

There is no system prompt: the CLI sends no system message, so the first request
of a run carries the user's own turn and nothing before it. Anything the agent
needs to know is written into the prompt itself (`-s` / positional arguments).

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

Set `LLM_CLI_LOG_FILE` and every conversation is written to that file as JSON --
user turns, assistant replies, tool calls (name and arguments) and tool results
-- so a run can be read back afterwards:

```bash
LLM_CLI_LOG_FILE=~/.local/state/llm-cli/session.json llm-cli-py -m gpt-4o
```

The log accumulates. Each run is one entry in the document's `sessions` array,
so starting the CLI again **adds a session instead of replacing the file**: every
run you have ever logged with that path stays there, in order, with the stamp it
started at:

```json
{
  "version": 3,
  "sessions": [
    { "started_at": "2026-11-12T21:40:03", "model": "gpt-4o", "conversation": [...] },
    { "started_at": "2026-11-13T09:12:45", "model": "gpt-4o", "conversation": [...] }
  ]
}
```

The sessions already in the file are read once at startup and written back
unchanged beside the current one. Reading them is all that happens to them:
**nothing from the log is loaded into the conversation**, so every run starts
empty and the model never sees a previous session's context. The file is where a
past run is looked up afterwards, not where the next one picks up.

Because the file only grows, a long-lived path is worth a look now and then (or
split by day: `LLM_CLI_LOG_FILE=~/logs/$(date +%F).json`, expanded by the shell).
A version 1 or 2 file -- one session at the top level -- is still read, and its
session is carried over as the first entry (`started_at` null: that run's start
was never recorded).

Within a session, the file is rewritten as the conversation grows, so what is on
disk is always that session as it stood, including after a run that ends badly: a
request that fails, a tool that times out, a `kill`, or a Ctrl+C halfway through a
turn all leave everything recorded up to that point, as one parseable JSON
document. (The exception is a turn that is deliberately *discarded* -- a tool
call whose arguments did not parse, or an interrupted tool run -- which is rolled
back out of the conversation, and out of the log with it, because that is what
the model will be sent next.)

Each message that took time to produce carries that time as `duration_ms`: the
`assistant` message the duration of the request that returned it, the `tool`
message the duration of the tool run. They are the same numbers the status line
showed. A session also totals them per kind, so a slow run can be accounted for
without adding up every message:

```json
{ "role": "assistant", "content": "", "tool_calls": [...], "duration_ms": 2502.6 },
{ "role": "tool", "content": "{\"stdout\": \"done\\n\"}", "tool_call_id": "c1", "duration_ms": 3012.2 },
...
"durations_ms": { "llm": 3505.0, "tool": 3012.2 }
```

User turns, and a call that never reached a tool carry no `duration_ms` --
nothing was measured for them -- and such a session has no
`durations_ms` either. The field lives in the log only: the request body is built
field by field, so nothing the API does not know about can be sent to it.

Writes are atomic (temporary file plus rename), so a reader never sees half a
document. Unset `LLM_CLI_LOG_FILE` -- the default -- and nothing is written at
all; `LLM_CLI_LOG_FILE=/dev/null` also turns it off. A path that cannot be
written is reported at startup and the run continues without a log, and a file
that is not a log at all (something else's, or one this version cannot read) is
reported and **left exactly as it is**: a file holding earlier runs is never
overwritten blind.

### Example: Web Search Without a Search Tool

Describe the API in the prompt itself and the agent calls it from
`execute_python`:

```bash
export WEB_SEARCH_API_KEY="sk-..."
llm-cli-py -s 'You are a helpful coding agent.
When you need web search, call the API below from execute_python:
- Endpoint: POST https://ollama.com/api/web_search
- Headers: Content-Type: application/json and Authorization: Bearer <value of WEB_SEARCH_API_KEY>
- Body: {"query": "...", "max_results": 5}
The key is available in the environment variable WEB_SEARCH_API_KEY.'
```

The prompt is a turn like any other, so it is part of the conversation and shows
up in `/raw` and in the log. To keep a standing instruction, put it in the first
line of every run (or in a shell function that passes it as `-s`). Prefer
referencing the key by env var name (as above), not by value.

## Tools

`execute_python` is the only tool, by design: it runs Python code in a
subprocess and returns the exit code plus stdout/stderr. Its output is sent back
to the model, so the agent can read files, call APIs and run commands itself. Tool calls are always
executed automatically (no approval prompt). The child runs with the same
environment and file-system access as the CLI itself.

A run has **no time limit**: the call returns when the code does, however long
that takes, and nothing is killed for taking its time. The call blocks the turn,
so a script that never ends keeps the turn waiting -- the running timer on the
status line shows it is still working, and Ctrl+C ends it: the child is killed
together with every process it spawned, and the prompt comes back with nothing
left running behind it. The tool description tells the model the same, and that
work which is not needed right away is better started detached
(`subprocess.Popen(..., start_new_session=True)` writing to a file) and checked
in a later call.

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
dependencies are `requests`, `prompt_toolkit` and `rich` (declared in
`pyproject.toml`);
everything else is the standard library. Dev tools live under the `dev` extra;
there is no lock file, so pin exact versions with `==` if you need reproducible
installs.

## License

Apache-2.0
