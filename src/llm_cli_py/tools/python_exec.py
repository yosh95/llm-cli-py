"""Python execution tool - runs Python code in a subprocess."""

import ast
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..consts import (
    DEFAULT_CHILD_PYTHON,
    DEFAULT_WAIT_S,
    MAX_RETURNED_LINES,
    MAX_RETURNED_OUTPUT,
)
from ..utils import terminal
from . import jobs
from .types import ExecResult, ToolError, WaitInfo

ENV_PYTHON_EXEC = "LLM_CLI_PYTHON_EXEC"
"""Environment variable overriding the interpreter used to run tool code.

The CLI may be installed in an isolated environment (e.g. ``uv tool install``
or ``pipx``) whose interpreter is *not* the one a user's ``python3`` resolves
to. The tool code should run with the same interpreter as the CLI itself, so
``execute_python`` prefers this override first, then ``sys.executable``, and
only falls back to ``python3`` if the CLI process has no usable interpreter
(e.g. under some embedded/frozen builds).
"""

_CHILD_ENCODING = "utf-8"
"""Encoding the child's output is read and decoded with.

The CLI may run where the locale is not UTF-8 (cp932 on Japanese Windows, say),
so the child is told to write UTF-8 (``PYTHONIOENCODING``) and both ends of the
pipe use this. Undecodable bytes are replaced rather than raised: a tool result
that fails to decode is worse than one with a stray character in it.
"""

_CHILD_ERRORS = "replace"
"""How undecodable bytes in the child's output are handled (see above)."""

_CHILD_STDIN = subprocess.DEVNULL
"""What the child's stdin is opened on: nothing, so it never touches the terminal.

The child used to inherit this process's stdin, and with it the terminal: it
could read the user's keystrokes out from under the prompt, and it could put the
terminal into raw mode -- which it then left that way, because tool code has no
reason to know it changed anything. A raw terminal has ``ISIG`` off, so Ctrl+C
stops being a signal: neither the CLI nor the shell it returned to reacted to the
key any more, and Ctrl+Z (a byte too) was no help either.

``DEVNULL`` is an open file, so code that reads stdin sees end-of-input rather
than an error, and the child holds no descriptor on the terminal at all. That
last part is what makes the guarantee hold even for a run the CLI ends with
``SIGKILL``, where the child gets no chance to put anything back -- and none of
it is needed to run a script, whose input comes from the code, not the keyboard.
"""

_SHELL_META = {">", "<", "|", "2>&1", "2>", "1>", ">>", "2>>", ";", "&", "`", "$("}
"""Tokens that mean something to a shell but are inert inside an argv list.

Used only to make the refusal message concrete: the mere presence of one of
these is not by itself harmful (see ``_check_shell_true_with_list``).
"""


def _resolve_child_python() -> str:
    """Return the interpreter used to execute tool code.

    Preference order: ``LLM_CLI_PYTHON_EXEC`` override, the running interpreter
    (``sys.executable``), then ``python3`` on ``PATH``. The CLI runs tool code
    with *its own* interpreter so the tool sees the same installed packages.
    """
    override = os.environ.get(ENV_PYTHON_EXEC, "").strip()
    if override:
        return override
    if sys.executable:
        return sys.executable
    found = shutil.which(DEFAULT_CHILD_PYTHON)
    if found:
        return found
    msg = (
        "No Python interpreter available to run tool code: "
        "sys.executable is empty and 'python3' was not found on PATH."
    )
    raise RuntimeError(msg)


def _shell_dropped_argument(arguments: list[ast.expr]) -> str | None:
    """Name the argument a shell would ignore, for the refusal message, or ``None``.

    ``shell=True`` uses the command string -- the list's first element -- and
    ignores everything after it, so those are the arguments worth quoting back:
    preferring one carrying a shell metacharacter, whose loss is least obvious.
    ``None`` means there is no readable string to point at.
    """
    dropped = arguments[1:] or arguments
    literals = [
        element.value
        for element in dropped
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    return next(
        (text for text in literals if any(meta in text for meta in _SHELL_META)),
        literals[0] if literals else None,
    )


def _check_shell_true_with_list(code: str) -> str | None:
    """Return an explanation if ``code`` calls subprocess with ``shell=True`` and an argv list.

    That combination is almost always a mistake: ``shell=True`` hands ``argv[0]``
    to the shell as a command *string* and ignores the remaining list elements,
    so ``subprocess.run(["cmd", "2>&1"], shell=True)`` runs ``cmd`` alone -- the
    argument the code meant to pass is silently dropped. (The command string may
    also be a command the shell resolves from ``PATH`` rather than the program
    named in the list.) The process does not hang; it simply does not do what
    the call site says, which is silent enough to be worth refusing while the
    agent can still see the reason.

    What this check deliberately does not do: judge individual strings for shell
    metacharacters. ``&``, ``|`` and friends appear in perfectly ordinary
    arguments (URL query strings, log formats), and a list element is not run by
    a shell at all unless the whole command is one string.

    The list itself is not rejected: a code string is the only argument, exactly
    as the subprocess docs recommend.

    Returns:
        A human-readable refusal, or ``None`` when nothing of the sort is found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    class _Checker(ast.NodeVisitor):
        """Report the first refused ``subprocess`` call of the shape this check exists for."""

        def __init__(self) -> None:
            self.error: str | None = None

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr in ("run", "Popen", "call", "check_call", "check_output")
            ):
                has_shell_true = any(
                    kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in node.keywords
                )
                first_arg = node.args[0] if node.args else None
                if has_shell_true and isinstance(first_arg, ast.List):
                    dropped = _shell_dropped_argument(first_arg.elts)
                    shown = repr(dropped) if dropped is not None else "a non-literal argument"
                    lineno = getattr(node, "lineno", "?")
                    self.error = (
                        f"[L{lineno}] subprocess.{func.attr}(..., shell=True) was NOT run\n"
                        f"  Problem: shell=True takes one command *string*, so the "
                        f"remaining list items are ignored\n"
                        f"           ({shown} would be dropped or mishandled)\n"
                        f"  Result: the command runs differently from what the code says "
                        f"(it does not hang)\n"
                        f"  BAD: subprocess.run(['cmd', '--flag'], shell=True)\n"
                        f"  GOOD: subprocess.run(['cmd', '--flag'])\n"
                        f"        subprocess.run('cmd --flag', shell=True)  # "
                        f"single string, if a shell is really wanted\n"
                    )
                    return
            self.generic_visit(node)

    c = _Checker()
    c.visit(tree)
    return c.error


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill the child process and every process it spawned.

    ``proc`` was started with ``start_new_session=True``, so its pid is also the
    id of its own process group: one ``killpg`` takes down the child and all of
    its descendants. The direct child is signalled as well, in case it left the
    group (for example by calling ``setsid()`` itself). Failures are ignored --
    the process may already be gone.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid
    for target, kill in ((pgid, os.killpg), (proc.pid, os.kill)):
        with contextlib.suppress(OSError):
            kill(target, signal.SIGKILL)


def execute_python(
    code: str,
    explanation: str = "",
    wait: object = None,
) -> ExecResult | ToolError:
    """Execute Python code and, unless told not to, wait for it here.

    The child is started in its own session (``start_new_session=True``), so a
    Ctrl+C in the terminal is *not* delivered to the code being run -- it would
    otherwise never stop on its own. On ``KeyboardInterrupt`` the child's whole
    process group is killed first (the code plus everything it spawned) and the
    interrupt is then re-raised, so the prompt comes back with nothing left
    running in the background.

    The child is given no descriptor on the terminal (see ``_CHILD_STDIN``): its
    stdin is ``/dev/null``, and its output arrives over pipes. Tool code
    therefore cannot read what the user types at the prompt, cannot consume the
    Ctrl+C meant for the CLI, and cannot leave the terminal in raw mode -- where
    ``ISIG`` off means Ctrl+C stops being a signal at all, and the CLI (and the
    shell after it) ignores the key entirely. As a second line of defence the
    terminal's settings are captured before the run and put back after it,
    however the run ends (see ``llm_cli_py.utils.terminal``).

    ``wait`` decides what the call does about a run that takes a while, and it is
    deliberately an argument of this call rather than a tool of its own: a job
    started by one call and waited for by another costs a round trip for each
    step, and a model that has to remember to wait will sometimes not. Left
    unset, the call blocks until the code exits -- the whole of stdout and stderr
    in one result, one round trip, no files to read.

    With ``wait``, the work is the same work but the waiting is the tool's: the
    code runs as a job whose output goes to a file, and the call returns once a
    *condition* holds, on a budget it was given. Nothing sleeps to guess a
    duration. The shapes are:

    ``{"until": "exit"}``
        Return when the code exits, however long that takes (up to
        ``timeout_s``). This is the one to reach for instead of sleeping in a
        loop and re-running the check.
    ``{"until": "background"}``
        Start the code and return at once with its job id; read it later with
        the same ``wait: {"until": "exit"}`` and a job id, or by reading the
        output file the result names.
    ``{"until": "file", "path": ...}`` / ``{"until": "gone", "path": ...}``
        Return when that path appears / disappears. For "the artifact is built",
        "the lock is gone", or work started by a *previous* call.
    ``{"until": "regex", "path": ..., "pattern": ...}``
        Return when the pattern matches the tail of that file. For "the server
        printed Ready" -- a log line is the event a server actually publishes.
    ``{"until": "port", "port": N}``
        Return when something accepts a connection on localhost:N.

    Also accepted: ``timeout_s`` (default ``DEFAULT_WAIT_S``), ``tail_lines``
    (return only the last N lines of a long output) and ``name`` (a job id a
    later call can use).

    A condition about the world stops early when the job exits: the job cannot
    make the file appear any more, so the call reports that instead of sitting
    out the rest of its budget.

    The run itself still has no time limit, and ``wait`` is not one: it bounds
    how long *this call* waits, and a job that outlasts it keeps running -- the
    result says so, with the id to wait on again. Ctrl+C during the wait takes
    down the process group of whatever this call started, as before.

    Code that calls ``subprocess`` with ``shell=True`` and an argv list is
    refused before anything runs (see ``_check_shell_true_with_list``), because
    the shell would ignore everything after the first element.

    Args:
        code: The Python code to run.
        explanation: Why this code is being run, written for the user: one
            short sentence. Required, in the schema and here, because the reason
            is part of the call rather than optional commentary: the display
            prints it in the transcript directly after the code, so a reader sees
            why the script ran and not only that it did. It is empty by default
            only so that a call arriving without one is refused with a sentence
            the model can act on, instead of a ``TypeError`` about a signature it
            cannot see.

            What this function does with it is nothing: the display prints it
            from the call's own arguments. It is validated here anyway, so the
            requirement holds whichever layer a call comes through.

    Returns:
        ExecResult on completion, ToolError on failure. A result from a call
        that waited carries ``wait``: what it waited for, whether it happened,
        how long it took, and whether the job is still running.

    Raises:
        KeyboardInterrupt: If the user interrupts the execution (Ctrl+C).
    """
    # A call without a reason is refused before anything runs, with the fix
    # spelled out -- the same treatment as a shell=True argv list below. The
    # refusal is a tool result, so the turn survives: the model reads what was
    # missing, sends the call again with it, and nothing has to be typed twice.
    if not explanation.strip():
        return ToolError(
            error=(
                "execute_python was NOT run: 'explanation' is missing or empty.\n"
                "  It is required: one short sentence, written for the user, saying "
                "why this code needs to run --\n"
                "  it is shown in the transcript right after the code. Send the call "
                "again with it, e.g.\n"
                '  {"code": "...", "explanation": "Check which Python version is installed."}'
            )
        )

    # Refuse subprocess calls whose argv list would be ignored by the shell,
    # rather than running code that does not do what it looks like it does.
    refusal = _check_shell_true_with_list(code)
    if refusal:
        return ToolError(error=refusal)

    if wait is None:
        return _run_and_wait_for_it(code)
    return _run_and_wait_for_something(code, wait)


def _run_and_wait_for_it(code: str) -> ExecResult:
    """Run ``code`` and block until it exits -- the call with no ``wait``.

    This is the path a short script takes: one call, one round trip, the whole
    of stdout and stderr in the result. It is what every call did before ``wait``
    existed, and it stays the default because most code a model writes finishes
    in a moment, and a job's file round trip would only be slower for it.
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(code)

    proc: subprocess.Popen[str] | None = None
    try:
        # Whatever the child does to the terminal -- see ``_CHILD_STDIN`` for why
        # it should be able to do nothing at all -- the settings the user had are
        # the ones in force once the run is over, so a stray raw mode cannot
        # outlive the tool that set it.
        with terminal.preserved():
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.Popen(
                [_resolve_child_python(), str(tmp_path)],
                stdin=_CHILD_STDIN,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding=_CHILD_ENCODING,
                errors=_CHILD_ERRORS,
                env=env,
                start_new_session=True,
            )
            stdout, stderr = proc.communicate()
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)
    except KeyboardInterrupt:
        # Kill the child and everything it spawned, then let the interrupt
        # continue to the caller (the session loop silently returns to the
        # prompt). Waiting for the child here only waits for a process that has
        # just been SIGKILLed, so it cannot hold the prompt back.
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
            with contextlib.suppress(Exception):
                proc.wait()
        raise
    except Exception as e:
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
        return ExecResult(stdout="", stderr=str(e), exit_code=-1)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _run_and_wait_for_something(code: str, wait: object) -> ExecResult | ToolError:
    """Run ``code`` as a job and wait for ``wait`` -- the call that named a condition.

    The code goes to a job (see :mod:`llm_cli_py.tools.jobs`) and the waiting
    happens here, on a condition the call named. Three things are different from
    the blocking path, and each is the reason it exists:

    - the output goes to a **file**, not a pipe, so a job that prints megabytes
      cannot block itself and nothing is lost if this call is interrupted;
    - **how the job ended is a file** (``<id>.rc``), so a later call -- or a later
      run of the CLI -- can tell "still running" from "finished" without the pid,
      which is reused and gone once reaped;
    - the **polling is here**, at a cadence fixed in code, so no sleep is handed
      to the model to guess with.

    ``wait: {"job": "<id>"}`` refers to a job that already exists instead of
    running ``code``: that is the second half of a ``background`` start, and it
    costs one round trip rather than the two a separate wait tool would.

    A ``KeyboardInterrupt`` during the wait stops the job this call started (it is
    the same contract as the blocking path: Ctrl+C leaves nothing running behind
    the prompt) and re-raises, so the session rolls the turn back as usual.
    """
    try:
        condition = jobs.condition_from(wait)
        timeout_s = jobs.check_timeout(
            wait.get("timeout_s") if isinstance(wait, dict) else None, DEFAULT_WAIT_S
        )
    except (TypeError, ValueError) as e:
        return ToolError(error=f"execute_python was NOT run: {e}")
    tail_lines = jobs.tail_lines_from(wait)
    name = jobs.name_from(wait)
    existing = jobs.job_from(wait)
    stop = jobs.kill_from(wait)

    if existing is not None:
        return _follow_existing_job(existing, condition, timeout_s, tail_lines, stop)

    try:
        python = _resolve_child_python()
    except RuntimeError as e:
        return ToolError(error=str(e))

    started_at = time.monotonic()
    job: str | None = None
    try:
        with terminal.preserved():
            job = jobs.start(code, python, name=name, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if condition.kind in jobs.START_ONLY_KINDS:
                outcome = jobs.WaitOutcome(
                    outcome="background", elapsed_s=time.monotonic() - started_at, job=job
                )
            else:
                outcome = jobs.wait_for(job, condition, timeout_s, kill_on_give_up=stop)
    except KeyboardInterrupt:
        # The same contract as the blocking path: Ctrl+C leaves nothing of this
        # call's work running behind the prompt. A job this call did not start is
        # not ours to stop (see ``_follow_existing_job``), but this one is -- the
        # user's "stop" was aimed at the wait they were watching.
        if job is not None:
            with contextlib.suppress(Exception):
                jobs.kill(job)
        raise
    except OSError as e:
        return ExecResult(stdout="", stderr=str(e), exit_code=-1)

    return _job_result(job, condition, outcome, tail_lines)


def _follow_existing_job(
    job: str,
    condition: jobs.Condition,
    timeout_s: float,
    tail_lines: int | None,
    stop: bool,
) -> ExecResult | ToolError:
    """Wait for a job started by an earlier call, or just read how it stands.

    ``timeout_s: 0`` means "look once": that is the read -- is it running, how
    did it end, what has it printed -- and it is the same call with the budget
    turned down, rather than a fourth tool to remember.
    """
    if not jobs.exists(job):
        recent = jobs.known()
        hint = f"recent jobs: {', '.join(recent)}" if recent else "no jobs are running"
        return ToolError(error=f"No job named {job!r}. {hint}.")

    if stop:
        jobs.kill(job)

    if condition.kind in jobs.START_ONLY_KINDS or timeout_s == 0:
        # "Look once": a read, not a wait. The same call with the budget turned
        # down, rather than a separate tool to remember.
        outcome = jobs.status(job)
    else:
        outcome = jobs.wait_for(job, condition, timeout_s)

    record = jobs.meta(job)
    result = _job_result(job, condition, outcome, tail_lines)
    where = record.get("output")
    if isinstance(where, str) and not result.stdout.strip():
        # Nothing printed yet, or nothing in the tail: name the file so a reader
        # knows where the output will be, rather than guessing there is none.
        result.stderr = f"(job {job} writes its output to {where})"
    return result


def _exit_code_for(outcome: jobs.WaitOutcome) -> int:
    """Return the exit code to report for a call that waited.

    The number is the *call's* verdict, not the job's, because that is what
    ``ok`` colours and what a reader takes as "did that work". So:

    - the job has an exit code (it finished) -- report it, whatever it is: the
      code's own failure is the thing to see;
    - ``satisfied`` -- the condition held, which is what the call was for, so 0
      even when the job is still running (a server that said ``Ready`` is not a
      failure because it has not exited);
    - ``background`` -- nothing was waited for and the start worked: 0;
    - ``status`` -- running is fine, finished without an exit code is not;
    - ``ended`` / ``timeout`` -- no exit code was produced, so there is nothing
      to report but a failure: -1.

    That a call which ``ended`` still shows ``exit_code: 0`` is not a mistake:
    the code did exit 0, and hiding that would lose the only information there
    is about it. What says the call did not get its answer is ``wait.delivered``
    (see :class:`~llm_cli_py.tools.types.WaitInfo`), which is why the two are
    separate rather than squeezed into one number.
    """
    if outcome.exit_code is not None:
        return outcome.exit_code
    if outcome.outcome in {"satisfied", "background"}:
        return 0
    if outcome.outcome == "status":
        return 0 if outcome.running else -1
    return -1


def _job_result(
    job: str,
    condition: jobs.Condition,
    outcome: jobs.WaitOutcome,
    tail_lines: int | None,
) -> ExecResult:
    """Read a job's output and package the wait with it, as one result.

    Reading and waiting are one call on purpose: a result that said "the job
    finished" and left the output for another round trip would cost exactly the
    round trip this whole path exists to avoid. The tail is what comes back, and
    the result says how much was left behind so a reader can ask for less.
    """
    text, meta = jobs.tail(
        job,
        max_chars=MAX_RETURNED_OUTPUT,
        max_lines=min(tail_lines, MAX_RETURNED_LINES) if tail_lines else None,
    )
    if meta["truncated"]:
        # Say so in the text as well as in the counters: the model reads stdout,
        # and a silent cut is a wrong answer waiting to be given.
        text = (
            f"[output truncated: showing the last {len(text)} characters of "
            f"{meta['output_bytes']} bytes from job {job}; pass tail_lines to choose how much]\n{text}"
        )
    exit_code = _exit_code_for(outcome)
    return ExecResult(
        stdout=text,
        stderr="",
        exit_code=exit_code,
        wait=WaitInfo(
            job=job,
            condition=condition.describe(),
            outcome=outcome.outcome,
            elapsed_s=outcome.elapsed_s,
            running=outcome.running,
        ),
    )


# Tool schema definition
PYTHON_TOOL_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "The Python code to run. Use `print()` to produce output.",
        },
        # Second, and required: a call has to say why it is being made, and the
        # display prints that sentence after the code it explains.
        "explanation": {
            "type": "string",
            "description": (
                "Why this code needs to run, in one short sentence addressed to the user. "
                "It is shown in the transcript right after the code, so state the reason "
                "for the call, not a restatement of what the code does."
            ),
        },
        # Third, and optional: the default (no ``wait``) is the old behaviour --
        # block until the code exits -- so a call that says nothing new pays
        # nothing new. This is where a long run says how it is waited for.
        "wait": {
            "type": "object",
            "description": (
                "How to wait for code that takes a while. Omit it and the call blocks "
                "until the code exits, returning all of its output: right for anything "
                "that finishes in a moment. Instead of sleeping and re-checking, pass this "
                "and the call returns as soon as the condition holds -- one round trip, no "
                'guessed duration. Examples: {"until": "exit"} (wait for the code to '
                'finish, however long), {"until": "file", "path": "out/app"} (wait '
                'for an artifact), {"until": "regex", "path": "server.log", '
                '"pattern": "Ready"} (wait for a log line), {"until": "port", '
                '"port": 8080} (wait for a listener), {"until": "background"} (start '
                "it and return at once). Also takes timeout_s (default 600), tail_lines (return "
                "only the last N lines), name (a job id to give the code), job (wait for a job "
                "started earlier, instead of running code) and kill (stop that job)."
            ),
            "properties": {
                "until": {
                    "type": "string",
                    "enum": ["exit", "background", "file", "gone", "regex", "port"],
                    "description": "What to wait for. Default: exit.",
                },
                "path": {
                    "type": "string",
                    "description": "File for the file, gone and regex conditions.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regular expression for the regex condition.",
                },
                "port": {
                    "type": "integer",
                    "description": "Local port for the port condition.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": (
                        "How long this call waits, in seconds. Default 600. 0 looks once and "
                        "returns. The job is not stopped by this: the result says whether it "
                        "is still running."
                    ),
                },
                "tail_lines": {
                    "type": "integer",
                    "description": "Return only the last N lines of the job's output.",
                },
                "name": {
                    "type": "string",
                    "description": "A job id to give the code, so a later call can name it.",
                },
                "job": {
                    "type": "string",
                    "description": (
                        "Wait for an existing job with this id instead of running code: the "
                        "second half of a background start. The code argument is ignored."
                    ),
                },
                "kill": {
                    "type": "boolean",
                    "description": "With job: stop it (and everything it spawned) instead of waiting.",
                },
            },
        },
    },
    "required": ["code", "explanation"],
}

PYTHON_TOOL_DESCRIPTION = """\
Run Python code in a subprocess and return its stdout and stderr.

By default this call blocks until the process exits and has no timeout: it
returns when the run does, with everything the code printed. That is the right
call for anything that finishes in a moment, and most code does.

For work whose duration is not known in advance, pass `wait` -- and do NOT write
a `time.sleep(...)` loop that re-checks the work. A sleep is a guess at a
duration: too short spends a whole round trip to learn nothing, too long spends
the turn waiting on a wall that is not there, and the round trips themselves cost
more than the wait would have. `wait` hands the polling to the tool, which can
look twenty times a second for free:

  {"until": "exit", "timeout_s": 900}
      Returns when the code finishes. Use this for a build, a test suite, a
      download -- anything long. One call, and the result holds the output.
  {"until": "file", "path": "dist/app.whl"}
      Returns when that path exists: "wait until the artifact is there".
  {"until": "regex", "path": "server.log", "pattern": "^Ready"}
      Returns when that line appears in the log: "wait until the server is up".
  {"until": "port", "port": 8080}
      Returns when something on localhost:8080 accepts a connection.
  {"until": "background", "name": "build"}
      Starts the code and returns at once with a job id. Later, the same call
      with {"job": "build", "until": "exit"} waits for that job -- `job` refers to
      an id, `name` only gives one -- and its tail comes back with the result.
      {"job": "build", "timeout_s": 0} reads its state without waiting, and
      {"job": "build", "kill": true} stops it.

`timeout_s` bounds how long THIS CALL waits (default 600), and `tail_lines` asks
for only the end of a long output. A timeout does not stop the job: the result
says whether it is still running, and the same `job` waits for it again.
Because conditions are checked against the world, a job that exits with the
condition unmet returns at once instead of sitting out its budget.

Every call also carries an "explanation": one sentence, written for the user,
saying why this code needs to run. It is required, and it is shown in the
transcript right after the code -- so write the reason for the call rather than a
description of the code.
"""
