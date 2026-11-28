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
- **`LLM_CLI.md`** — a file in the directory you run it in (or in a parent of
  it) is read at startup and sent as the conversation's system message, the way
  Claude Code reads `CLAUDE.md`: standing instructions for the project, with no
  flag to type and no env var to set
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
- **A reason for every tool call** — the model has to send an `explanation`
  with `execute_python` (the schema requires it, and the tool refuses a call
  without one), and it is printed right after the code, so the transcript says
  why the script was run and not just what it was
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

The prompt is not the only thing the request carries: when the directory has an
`LLM_CLI.md`, its text is the conversation's system message and goes ahead of
this prompt (see [Project Instructions](#project-instructions-llm_climd)).
`-s` itself stays a pure text channel either way.

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
  "code": "print(1)",
  "explanation": "Check that Python runs here."
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
  recorded, and this run's `LLM_CLI_LOG_DIR` file is left exactly as it was.
- **The system message is in it.** When `LLM_CLI.md` supplied project
  instructions, they are the first section (`## system`), because they are the
  first thing the model was sent.
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
  [explanation] Time a three-second script to see the status line count.
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
| `LLM_CLI_LOG_DIR` | Directory every conversation is logged to, one JSON Lines file per run, tool calls and tool results included. Each run adds its own file, so earlier runs survive a restart (they are never read back into the conversation). Unset (the default) writes no log at all. |
| `LLM_CLI_PYTHON_EXEC` | Interpreter used by `execute_python` (defaults to the CLI's own interpreter). |

Only paths are configured by env var here: the project instructions are the
`LLM_CLI.md` file the run finds on disk (see
[Project Instructions](#project-instructions-llm_climd)), and there is no
variable that would point somewhere else instead.

There is no `LLM_CLI_SYSTEM_PROMPT` variable: the system message, when there is
one, is the project's `LLM_CLI.md` (below). Nothing else is injected, and with no
such file the first request of a run carries the user's own turn and nothing
before it. A one-off instruction belongs in the prompt itself (`-s` / positional
arguments).

### Project Instructions: `LLM_CLI.md`

Put an `LLM_CLI.md` in the directory you run the CLI in and it is loaded
automatically -- no flag, no env var. It is this CLI's `CLAUDE.md`: the place for
what is true of *this* project rather than of this request, such as the command
that runs the tests, the files not to touch, or how answers should be written.

```markdown
<!-- ~/work/myproject/LLM_CLI.md -->
- Run `make check-all` before saying a change is done.
- The generated files under `src/gen/` are not edited by hand.
- Answer in Japanese, with the code and paths as they are.
```

```bash
cd ~/work/myproject
llm-cli-py -m gpt-4o            # "Loading project instructions from .../LLM_CLI.md"
```

- **It is the system message.** The file is read once at startup and becomes the
  conversation's first message, with role `system`, ahead of every user turn
  (`-s` and the typed prompt alike) and sent again with each request. It is not
  merged into the first prompt: standing instructions are not a request, and a
  model treats a system message and a user turn differently.
- **The nearest file wins.** The current directory is checked first, then each
  parent up to the root, so running in `myproject/src/` still finds
  `myproject/LLM_CLI.md`. Exactly one file is used, and it is read once -- a run
  does not pick up edits made while it is open (start a new run for that).
- **No file, no message.** With no `LLM_CLI.md` anywhere above the directory,
  the conversation starts empty and the first request carries only your turn. An
  empty (or whitespace-only) file is the same as no file.
- **It is part of the conversation, so you can see it.** `/raw` shows it as the
  first section, and it is the first message record of the run in the
  `LLM_CLI_LOG_DIR` file. A turn that is discarded (a broken tool call, an
  interrupted tool run) is rolled back down to it -- the instructions are never
  discarded.
- **Read once, and bounded.** A file larger than 64,000 characters is cut at a
  line boundary and the truncation is reported: the text is sent with *every*
  request, so a file that has grown into a data dump would quietly eat the
  model's context.
- **A file that cannot be read is reported and skipped**, and the session
  carries on without it.

### Writing Paths in the Environment

Both `LLM_CLI_LOG_DIR` and `LLM_CLI_PROMPT_HISTORY_FILE` are read the way you
wrote them, without needing the shell to help:

- **`~`** is expanded against the home directory (`USERPROFILE` on Windows), so
  `~/state/llm-cli/runs` works.
- **Spaces need no escaping**: `C:\Users\me\My Logs\runs` is fine, as is
  `~/My Logs/history`.
- **Surrounding quotes are dropped**, because `cmd.exe` keeps them in the value:
  `set LLM_CLI_LOG_DIR="C:\My Logs\runs"` works, and so does the same line
  in PowerShell, or GitBash's `~/My Logs/runs`. One pair, `"..."` or `'...'`,
  at either end.
- **Surrounding blanks are ignored.**
- **Parent directories are created**, in both cases.
- `$VAR` is *not* expanded -- that is the shell's job, so write
  `LLM_CLI_PROMPT_HISTORY_FILE=~/state/history` and let the shell do it.

A path that cannot be resolved (no home directory to expand `~` against) or
cannot be opened is reported, and that feature is dropped for the run: without a
usable history file the prompt keeps its history in memory, and without a usable
log directory nothing is logged. Neither stops the session.

### Logging the Conversation

Set `LLM_CLI_LOG_DIR` and every conversation is written to a file in that
directory -- user turns, assistant replies, tool calls (name and arguments) and
tool results -- so a run can be read back afterwards:

```bash
LLM_CLI_LOG_DIR=~/.local/state/llm-cli/runs llm-cli-py -m gpt-4o
```

It is a **directory, one file per run**, not one file that grows. The file is
named after the moment the run started, so the directory reads as a timeline and
starting the CLI again **adds a file instead of rewriting the ones already
there**:

```text
~/.local/state/llm-cli/runs/
  2026-11-12T21-40-03.jsonl
  2026-11-13T09-12-45.jsonl
  2026-11-13T09-12-45-1.jsonl   ← a second run, same second
```

A run writes nothing back and reads nothing first: **nothing from a log is
loaded into the conversation**, so every run starts from its `LLM_CLI.md` alone
(or empty) and the model never sees a previous session's context. The directory
is where a past run is looked up afterwards, not where the next one picks up.
Because of that, the cost of logging does not depend on how much has been logged
-- a run does not parse the runs before it, and a file that is not one of this
version's own is not in the way (it is left as it is, not reported, not touched).

Each file is [JSON Lines](https://jsonlines.org/): one record per line, appended
as the conversation grows.

```json
{"type": "log", "version": 4}
{"type": "session", "started_at": "2026-11-13T09:12:45", "model": "gpt-4o"}
{"type": "message", "role": "user", "content": "compute this"}
{"type": "message", "role": "assistant", "content": "", "tool_calls": [...], "duration_ms": 2502.6}
{"type": "message", "role": "tool", "tool_call_id": "c1", "content": "{\"stdout\": \"42\\n\"}", "duration_ms": 3012.2}
{"type": "message", "role": "assistant", "content": "It is 42", "duration_ms": 811.0}
```

Why one line per record rather than one JSON document:

- **Appending is what happens.** A document has to be parsed and written whole on
  every turn, so logging a long run costs time proportional to the whole log,
  every turn. A line is one `write`.
- **A run that dies leaves a readable file.** A killed run can leave its last line
  truncated; every line before it is untouched, so the file reads up to there and
  one line is lost. A half-written document parses as nothing at all. `jq` stops
  at the damaged line, so read the records with `jq -R 'fromjson? // empty'` when
  that matters -- `select(...)` then filters as usual.
- **The file can be read while it is being written** -- `tail -f`, or `jq` on a
  run still going -- because a line is appended whole.
- **A discarded turn is cut off again.** A tool call whose arguments did not
  parse, or an interrupted tool run, is rolled back out of the conversation and
  the file is truncated back to the line before it, so what is on disk is still
  exactly the conversation the model will be sent next.

Each message that took time to produce carries that time as `duration_ms`: the
`assistant` message the duration of the request that returned it, the `tool`
message the duration of the tool run. They are the same numbers the status line
showed. There is no per-session total in the file -- it would be a line that has
to be rewritten as the run goes on, which is the one thing this layout avoids;
add them up when you want them:

```bash
jq -s '[.[] | select(.type == "message") | .duration_ms // empty] | add' runs/*.jsonl
```

User turns, and a call that never reached a tool, carry no `duration_ms` --
nothing was measured for them. The field lives in the log only: the request body
is built field by field, so nothing the API does not know about can be sent to it.

Writes are flushed line by line, so everything recorded is on disk as soon as it
is recorded, and a `kill` loses nothing that was already written. Unset
`LLM_CLI_LOG_DIR` -- the default -- and nothing is written at all. A path that
cannot be used is reported at startup and the run continues without a log, as is
a directory that stops accepting writes part-way through a run; the session
itself is unaffected either way.

The log file earlier versions wrote was one JSON document holding every run. It
is a different layout in a different place: this version does not read it and does
not write it, so pointing `LLM_CLI_LOG_DIR` at a new directory is all that is
needed, and the old file can be kept or deleted as you like. To carry it over,
one file per run, header line first:

```bash
mkdir -p ~/.local/state/llm-cli/runs
{ printf '%s\n' '{"type": "log", "version": 4}'
  jq -c '.sessions[] |
    {"type":"session","started_at":.started_at,"model":.model},
    (.conversation[] | {"type":"message"} + .)' old.json
} > ~/.local/state/llm-cli/runs/carried-over.jsonl
```

That puts every old run in one file. For a file per run, write each session's
records to a file of its own (`jq -c '.sessions[0] | ...'`, then `[1]`, ...), named
after its `started_at`.

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
up in `/raw` and in the log. For an instruction that should stand for the whole
project, put it in `LLM_CLI.md` instead: it is sent as the system message, and
nothing has to be typed at startup. Prefer referencing the key by env var name
(as above), not by value.

## Tools

`execute_python` is the only tool, by design: it runs Python code in a
subprocess and returns the exit code plus stdout/stderr. Its output is sent back
to the model, so the agent can read files, call APIs and run commands itself. Tool calls are always
executed automatically (no approval prompt). The child runs with the same
environment and file-system access as the CLI itself.

Every call carries two arguments, and the schema marks both as `required`:

| Argument | What it is |
|---|---|
| `code` | The Python to run. `print()` what the model needs to see. |
| `explanation` | One short sentence, written for you, saying why this code needs to run. |

The explanation is shown in the transcript **directly after the code** it is
about, and never inside anything the model controls: the order on screen is the
display's, not the JSON's, so a call that arrived with `explanation` first still
reads code-then-reason.

```text
[Tool] Executing tool: execute_python...
Args:
  [code]
    import pathlib
    print(len(list(pathlib.Path('src').rglob('*.py'))))
  [explanation] Count the Python files, to see how big the package is.
```

The requirement is enforced on both sides. The schema says `required`, so a
model reading the tool definition knows to send it; and the tool itself refuses a
call without one *before* the code runs, with the same kind of message it gives
for a `subprocess(..., shell=True)` list -- what was missing, and what to send
instead. The refusal is an ordinary tool result, so the turn is not lost: the
model reads it, sends the call again with the sentence added, and nothing has to
be retyped. (A `required` key alone is only a hint to the provider; some
OpenAI-compatible servers pass a call through without checking it.)

A run has **no time limit**: the call returns when the code does, however long
that takes, and nothing is killed for taking its time. The call blocks the turn,
so a script that never ends keeps the turn waiting -- the running timer on the
status line shows it is still working, and Ctrl+C ends it: the child is killed
together with every process it spawned, and the prompt comes back with nothing
left running behind it.

The child is given **no descriptor on the terminal**: its stdin is `/dev/null`,
and its stdout and stderr go to temporary files that are read back once the child
exits -- not to pipes. A pipe holds about 64 KiB and, worse, `communicate()`
reads to EOF: a process the code started and left running keeps the write end
open, so a call that said "start it and return" waited out the very process it
had detached. A file never blocks a writer and is read when the child is done, so
what a detached process does afterwards cannot hold the turn open. Tool code
cannot read the keys you type at the `> ` prompt, and it cannot change the
terminal's settings -- a child that put the terminal into raw mode used to leave
it that way, and a raw terminal has `ISIG` off, which means Ctrl+C is no longer a
signal for the CLI *or for the shell it returns to* (Ctrl+Z does not help either:
it is a byte too). To read input, put it in the code; to run something with your
keyboard, run it yourself. As a second line of defence the terminal's settings
are captured before the run and before the prompt loop, and put back afterwards,
however the run ends.

The tool description tells the model the same, and adds the one thing a model
gets wrong about a blocking call: the waiting is the call's job, so it should
**not** write a `time.sleep(...)` loop that re-checks the work, and should not
pass `timeout=` to `subprocess` -- a sleep and a timeout are both guesses at how
long the work takes, and the call simply returns when the code does. Work whose
result is not needed yet is started detached with its own output file
(`subprocess.Popen(..., stdout=open(...), start_new_session=True)`) and read in a
later call.

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
