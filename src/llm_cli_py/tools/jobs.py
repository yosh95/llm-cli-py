"""Background jobs: work that outlives the call, and facts to wait on.

A tool call can start code and come back later, but only if something can say
*when*. That is what this module provides, and the reason it exists is a cost
measured in round trips: a turn that sleeps and re-checks spends a whole request
on a guess about how long the work takes, and a wrong guess is spent twice -- once
waiting too little, once waiting too much.

So the two things this module owns are:

``start``
    Run code in a job of its own. Its output goes to a *file* and its exit code
    to another, so nothing the job writes can block it and nothing the CLI does
    -- including exiting -- can lose it. The call returns at once.

``wait_for``
    Block until a *condition* holds: the job exited, a file appeared, a log line
    matched, a port accepted a connection. The polling lives here, at a cadence
    chosen in code, so the model never writes a sleep and never guesses a
    duration.

Why a launcher process, and why the exit code is a file
-------------------------------------------------------
A ``Popen`` handle is only good while the process that made it is alive. A job is
by definition read by a *later* call, possibly from a later run of the CLI, and
``kill(pid, 0)`` answers "does this pid exist" -- which is not the question: pids
are recycled, and a job's exit code is unrecoverable once reaped. So each job runs
under a small launcher that writes its exit code to ``<id>.rc`` when the child
finishes. "Is it done?" is then the existence of a file: a fact that survives the
CLI, cannot be confused by pid reuse, and reads the same before and after a
restart. The write goes through ``os.replace``, so a reader never sees a partial
number -- a torn ``.rc`` would report a wrong exit code, which is worse than the
wait it saves.

Why the output is a file rather than a pipe
-------------------------------------------
A pipe holds about 64 KiB. With nobody reading it -- and nobody can be reading it
while the CLI is waiting for the model -- a job that prints more blocks in
``write`` forever, looking exactly like a hung job. ``communicate()`` is not the
answer either: it reads to EOF, and EOF never arrives while any descendant holds
the write end, which is precisely what a detached child does. A file has neither
problem: writes never block, a reader can start anywhere, and the CLI can close
its descriptor immediately after starting the job, so no handle outlives the call
that made it.

Cleanup
-------
Jobs are files in one directory, and the directory is pruned of finished jobs
older than ``PRUNE_AFTER_S`` when it is first used in a run, so a machine that
runs the CLI for months does not accumulate logs. A job that is still running is
never pruned: it is not ours to delete while it is working.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ..consts import (
    DEFAULT_JOBS_DIR_NAME,
    DEFAULT_WAIT_POLL_S,
    ENV_JOBS_DIR,
    MAX_WAIT_S,
)

JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
"""What a job id may look like: a name a path can hold and a shell cannot mangle."""

UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
"""Characters replaced when a requested job name is turned into a job id."""

PRUNE_AFTER_S = 7 * 24 * 3600.0
"""How old a finished job must be before it is deleted, in seconds."""

REGEX_READ_LIMIT = 1 << 20
"""How much of a file a ``regex`` condition looks at: the last megabyte.

A condition asks about the *end* of a log -- "did it print Ready", "did the test
summary appear" -- and a build log can be hundreds of megabytes. Reading the tail
keeps a condition from costing more than the work it is watching.
"""

LAUNCHER = """\
import os, subprocess, sys

run, job, script = sys.argv[1], sys.argv[2], sys.argv[3]
rc = subprocess.call([sys.executable, script])
tmp = os.path.join(run, job + ".rc.tmp")
with open(tmp, "w", encoding="utf-8") as f:
    f.write(str(rc))
os.replace(tmp, os.path.join(run, job + ".rc"))
"""
"""The program each job runs under: run the code, then record how it ended.

Kept as text and written into the jobs directory at first use so the launcher is
versioned with this module rather than with whatever ``python3`` is on ``PATH``.
It writes only the exit code: the code's own output goes straight to the job's
output file, which the launcher's own stdout already is.
"""


# ── Conditions ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Condition:
    """Something that becomes true, and that a call can wait for.

    One flat object rather than a tree of nested specs: the model fills in
    ``until`` / ``path`` / ``pattern`` / ``port``, and a typo in one of those
    names is refused with the list of names that exist instead of silently
    waiting for nothing.

    Attributes:
        kind: ``exit``, ``file``, ``gone``, ``regex`` or ``port``.
        path: File the ``file``, ``gone`` and ``regex`` conditions look at.
        pattern: Regular expression the ``regex`` condition matches, with
            ``re.MULTILINE``, anywhere in the file.
        port: TCP port the ``port`` condition connects to on localhost.
    """

    kind: str
    path: str | None = None
    pattern: str | None = None
    port: int | None = None

    def describe(self) -> str:
        """Return how this condition is shown in a result and on a status line."""
        if self.kind == "exit":
            return "the job to exit"
        if self.kind == "file":
            return f"{self.path} to exist"
        if self.kind == "gone":
            return f"{self.path} to be gone"
        if self.kind == "regex":
            return f"/{self.pattern}/ in {self.path}"
        if self.kind == "port":
            return f"port {self.port} to accept a connection"
        return self.kind

    def met(self, job_finished: bool = False) -> bool:
        """Whether the condition holds right now.

        ``job_finished`` is supplied by the caller for the conditions that are
        about the world rather than about the job: a job that has exited can no
        longer make a file appear, so the wait does not have to sit out its
        timeout to learn that.
        """
        if self.kind == "exit":
            return job_finished
        if self.kind == "file":
            return self.path is not None and Path(self.path).expanduser().exists()
        if self.kind == "gone":
            return self.path is not None and not Path(self.path).expanduser().exists()
        if self.kind == "regex":
            return self._regex_matches()
        if self.kind == "port":
            return self._port_accepts()
        return False

    def _regex_matches(self) -> bool:
        """Whether the pattern matches anywhere in the tail of ``path``."""
        if not self.path or not self.pattern:
            return False
        try:
            with Path(self.path).open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - REGEX_READ_LIMIT))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return False  # not there yet: the same answer as "no match yet"
        with suppress(re.error):
            return re.search(self.pattern, text, re.MULTILINE) is not None
        return False  # an invalid pattern matches nothing; the caller says so

    def _port_accepts(self) -> bool:
        """Whether something on localhost accepts a TCP connection right now."""
        if self.port is None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                return True
        except OSError:
            return False


CONDITION_KEYS = {"until", "path", "pattern", "port", "timeout_s", "tail_lines", "name", "job", "kill"}
"""Every key a ``wait`` object may carry. Anything else is refused by name."""

CONDITION_KINDS = ("exit", "file", "gone", "regex", "port", "background")
"""Every value ``until`` may take. A closed list is what keeps a call from
waiting for something that cannot happen -- and then sitting out the timeout."""

START_ONLY_KINDS = ("background",)
"""Kinds that mean "do not wait at all", so they only make sense when starting.

Waiting for an existing job with ``until: "background"`` would be a request that
contradicts itself; it is refused rather than silently treated as "do nothing"."""


# ── Results of a wait ──────────────────────────────────────────────


@dataclass(frozen=True)
class WaitOutcome:
    """What a wait learned, before it is attached to an ``ExecResult``.

    Attributes:
        outcome: ``satisfied`` (the condition held), ``timeout`` (it did not,
            within the budget), ``ended`` (the job exited first, so the
            condition can no longer come true) or ``background`` (nothing was
            waited for: the job was only started).
        elapsed_s: How long the wait actually took.
        job: The job id, when this was about a job.
        running: Whether the job was still running when the wait returned.
        exit_code: The job's exit code, when it has finished.
    """

    outcome: str
    elapsed_s: float
    job: str | None = None
    running: bool = False
    exit_code: int | None = None


# ── The store ──────────────────────────────────────────────────────


def jobs_dir() -> Path:
    """Return the directory jobs live in, creating it if needed.

    Read from the environment on every call rather than at import, so a test (or
    an embedder) can point it somewhere else without reloading this module.
    """
    configured = os.environ.get(ENV_JOBS_DIR, "").strip()
    root = Path(configured).expanduser() if configured else Path(tempfile.gettempdir(), DEFAULT_JOBS_DIR_NAME)
    root.mkdir(parents=True, exist_ok=True)
    launcher = root / "_launcher.py"
    if not launcher.exists():
        launcher.write_text(LAUNCHER, encoding="utf-8")
    return root


def _pruned(root: Path) -> None:
    """Delete finished jobs older than ``PRUNE_AFTER_S``.

    Called once per process (see :func:`store`): a directory listing per tool
    call would be paid on every run to do nothing. Only jobs that have finished
    are touched -- a running job's files are its own -- and only names this
    module writes, so nothing else in the directory is a candidate.
    """
    now = time.time()
    for rc in root.glob("*.rc"):
        job = rc.stem
        try:
            if now - rc.stat().st_mtime < PRUNE_AFTER_S:
                continue
            if _still_running(root, job):
                continue
            for suffix in (".py", ".out", ".pid", ".rc", ".meta"):
                with suppress(OSError):
                    (root / f"{job}{suffix}").unlink()
        except OSError:
            continue  # a file that vanished under us is already gone


_PRUNED = False


def store() -> Path:
    """Return the jobs directory, having pruned it once for this process."""
    global _PRUNED
    root = jobs_dir()
    if not _PRUNED:
        _PRUNED = True
        _pruned(root)
    return root


def reset() -> None:
    """Forget that pruning has run, so the next ``store`` prunes again.

    Only tests need this (a test that wants to observe the pruning); production
    code has no reason to prune twice.
    """
    global _PRUNED
    _PRUNED = False


def _path(root: Path, job: str, suffix: str) -> Path:
    return root / f"{job}{suffix}"


def _read_rc(root: Path, job: str) -> int | None:
    """Return the job's exit code, or ``None`` while it has not written one.

    ``killed`` means the launcher was killed along with the job, so no exit code
    exists; the wait and the read report it as a failure without a number.
    """
    try:
        raw = _path(root, job, ".rc").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "killed":
        return -1
    try:
        return int(raw)
    except ValueError:
        return None


def _finished(root: Path, job: str) -> bool:
    return _path(root, job, ".rc").exists()


def _still_running(root: Path, job: str) -> bool:
    """Whether the job's process is alive.

    ``.rc`` answers first: it is written by the launcher when the child finishes,
    and a job that wrote one is done whatever ``kill(0)`` says. That matters
    because the launcher is a child of *this* process, and until it is reaped it
    is a zombie -- which ``kill(pid, 0)`` reports as alive. (``start`` reaps it in
    a thread for the same reason: a CLI that ran a hundred jobs would otherwise
    hold a hundred zombies.)

    When there is no ``.rc`` yet, the pid is the only thing to ask, and the
    answer is a guess: a reused pid would look like our job. It is used only for
    pruning and reporting, never for "is it done", so a wrong guess here costs at
    most a missed prune.
    """
    if _path(root, job, ".rc").exists():
        return False
    try:
        pid = int(_path(root, job, ".pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def exists(job: str) -> bool:
    """Whether a job with this id has ever been started here."""
    root = jobs_dir()
    return JOB_ID_PATTERN.match(job) is not None and _path(root, job, ".py").exists()


def job_id(requested: str | None) -> str:
    """Turn a requested name into a job id, or make one.

    A name the model chose is kept where it can be (so a later call can say
    ``job: "build"`` and mean this one), and anything a path could not hold is
    replaced rather than refused: the name is a convenience, and losing it is not
    worth failing a call over.
    """
    if not requested:
        return uuid.uuid4().hex[:10]
    cleaned = UNSAFE_IN_NAME.sub("-", str(requested)).strip(".-")[:64]
    if not cleaned or not JOB_ID_PATTERN.match(cleaned):
        return uuid.uuid4().hex[:10]
    return cleaned


def _unique(root: Path, job: str) -> str:
    """Return ``job``, or a free variant of it, so a running job is never overwritten.

    A name is the model's to choose and it will reuse one -- ``build`` on the next
    iteration of the same job. Writing over a job that is *still running* would
    truncate the output it is writing and replace the ``.rc`` a wait is looking
    for, so a name that is taken by live work gets a suffix instead. A name whose
    job has finished is reused as-is: re-running a finished job means the new run
    replaces the old report, which is what "run it again" means.
    """
    if not _path(root, job, ".py").exists() or not _still_running(root, job):
        return job
    for n in range(2, 100):
        candidate = f"{job}-{n}"
        if not _still_running(root, candidate):
            return candidate
    return f"{job}-{uuid.uuid4().hex[:6]}"


def _clear(root: Path, job: str) -> None:
    """Remove what an earlier run with the same id left behind.

    A name can be reused once its job has finished (see :func:`_unique`), and a
    stale ``.rc`` from that earlier run would say "finished" about the run that is
    just starting -- the wait would return before the code had done anything.
    The output is started fresh for the same reason: a reader asking for the
    output of *this* run should not get the last one's tail in front of it.

    Only called when nothing is running under the id, so there is no live
    descriptor to break.
    """
    for suffix in (".py", ".out", ".rc", ".rc.tmp", ".pid", ".meta"):
        with suppress(OSError):
            _path(root, job, suffix).unlink(missing_ok=True)


# ── Starting one ───────────────────────────────────────────────────


def start(
    code: str,
    python: str,
    *,
    name: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run ``code`` as a job and return its id, without waiting for it.

    The job's stdin is ``/dev/null`` and its output is a file, so it holds no
    descriptor on the terminal: it cannot steal a keystroke, cannot be left in
    raw mode, and cannot be broken by the CLI exiting. It is started in its own
    session, so it is also out of reach of the Ctrl+C that would otherwise kill
    it -- which is the point of a job, and the reason stopping one takes
    :func:`kill` rather than a signal aimed at the process group.

    Args:
        code: The Python code to run.
        python: Interpreter to run it with (the CLI's own, so the job sees the
            same packages).
        name: Optional name to make the job findable by; sanitised into an id.
        cwd: Working directory for the job; the CLI's own when ``None``.
        env: Environment for the job; the CLI's own (plus ``PYTHONIOENCODING``)
            when ``None``.

    Returns:
        The job id.
    """
    root = store()
    job = _unique(root, job_id(name))
    _clear(root, job)
    script = _path(root, job, ".py")
    script.write_text(code, encoding="utf-8")

    output = _path(root, job, ".out")
    with output.open("wb") as handle:
        handle.flush()
        child_env = dict(env if env is not None else os.environ)
        child_env["PYTHONIOENCODING"] = "utf-8"
        # ``stdout``/``stderr`` on the output file and no PIPE anywhere: nothing
        # the job prints can block it, and nothing it prints can be lost when
        # this process goes away (see the module docstring).
        proc = subprocess.Popen(
            [python, str(root / "_launcher.py"), str(root), job, str(script)],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=child_env,
            cwd=cwd,
            start_new_session=True,
        )
    # The handle is closed by the ``with`` above: the job keeps its own dup, and
    # the CLI keeps no descriptor on it.
    #
    # The launcher is reaped in a thread. Nobody waits for a job -- that is what a
    # job is -- but a child nobody waits for stays a zombie until this process
    # exits, and a zombie answers ``kill(pid, 0)`` as if it were alive: a hundred
    # jobs would be a hundred zombies that all look like running work. The thread
    # is a daemon, so it cannot hold the CLI open, and it does nothing but wait.
    threading.Thread(target=proc.wait, daemon=True).start()
    _path(root, job, ".pid").write_text(str(proc.pid), encoding="utf-8")
    _path(root, job, ".meta").write_text(
        json.dumps(
            {
                "job": job,
                "pid": proc.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cwd": cwd or str(Path.cwd()),
                "python": python,
                "output": str(output),
            }
        ),
        encoding="utf-8",
    )
    return job


# ── Reading one ────────────────────────────────────────────────────


def tail(job: str, *, max_chars: int, max_lines: int | None = None) -> tuple[str, dict[str, object]]:
    """Return the end of a job's output, plus how much was left behind.

    The tail rather than the head, because that is where a failure and a final
    summary are: a build that broke prints its error last, and the first
    thousand lines of a successful build say nothing. Reading is bounded at both
    ends -- a byte range from the end of the file, then whole lines, then whole
    characters -- so a job that printed gigabytes costs the same as one that
    printed a page.
    """
    root = jobs_dir()
    path = _path(root, job, ".out")
    try:
        size = path.stat().st_size
    except OSError:
        return "", {"output_bytes": 0, "truncated": False}

    window = max(max_chars * 4, 8192)
    with path.open("rb") as handle:
        handle.seek(max(0, size - window))
        raw = handle.read().decode("utf-8", errors="replace")

    text = raw
    if max_lines is not None and max_lines > 0:
        lines = text.splitlines(keepends=True)
        if len(lines) > max_lines:
            text = "".join(lines[-max_lines:])
    if len(text) > max_chars:
        text = text[-max_chars:]
    # A byte range can start mid-character and mid-line; a result should not
    # begin with half of either.
    text = text.lstrip("\n")
    truncated = len(text.encode("utf-8")) < size
    return text, {"output_bytes": size, "truncated": truncated}


def wait_for(
    job: str | None,
    condition: Condition,
    timeout_s: float,
    *,
    kill_on_give_up: bool = False,
) -> WaitOutcome:
    """Block until ``condition`` holds, the job ends, or the budget runs out.

    The loop is here, at a cadence fixed in code (``DEFAULT_WAIT_POLL_S``), for
    one reason: a model that sleeps is guessing a duration, and a guess that is
    too short costs a whole round trip to discover. The tool can afford to look
    twenty times a second; a turn cannot afford to ask twenty times.

    A condition about the world (a file, a line, a port) stops early when the job
    ends: the job can no longer make it true, so sitting out the remaining budget
    would only delay the report that says why.

    Args:
        job: The job being waited for, when there is one.
        condition: What to wait for.
        timeout_s: How long to wait, in seconds (``0`` means "look once").
        kill_on_give_up: Whether to stop the job when the budget runs out. Used
            for a job the same call started: a turn that gives up on work it
            began should not leave it running behind the prompt.

    Returns:
        What the wait learned.
    """
    root = jobs_dir()
    started = time.monotonic()
    deadline = started + max(0.0, timeout_s)

    while True:
        finished = job is not None and _finished(root, job)
        if condition.met(job_finished=finished):
            return WaitOutcome(
                outcome="satisfied",
                elapsed_s=time.monotonic() - started,
                job=job,
                running=job is not None and not finished,
                exit_code=_read_rc(root, job) if job else None,
            )

        if finished and condition.kind != "exit":
            # The job is over and the condition is about the world: it is not
            # going to come true now.
            return WaitOutcome(
                outcome="ended",
                elapsed_s=time.monotonic() - started,
                job=job,
                running=False,
                exit_code=_read_rc(root, job) if job else None,
            )

        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(DEFAULT_WAIT_POLL_S, deadline - now))

    if kill_on_give_up and job is not None:
        kill(job)
        return WaitOutcome(
            outcome="timeout",
            elapsed_s=time.monotonic() - started,
            job=job,
            running=False,
            exit_code=None,
        )
    return WaitOutcome(
        outcome="timeout",
        elapsed_s=time.monotonic() - started,
        job=job,
        running=job is not None and not _finished(root, job),
        exit_code=_read_rc(root, job) if job else None,
    )


def status(job: str) -> WaitOutcome:
    """Report a job's state without waiting for it.

    Used by a read that was not asked to wait: the caller learns whether the job
    is running, how it ended if it has, and where its output is -- which is the
    whole of what "is it done yet" needs, in one call.
    """
    root = jobs_dir()
    finished = _finished(root, job)
    return WaitOutcome(
        outcome="status",
        elapsed_s=0.0,
        job=job,
        running=not finished and _still_running(root, job),
        exit_code=_read_rc(root, job),
    )


def meta(job: str) -> dict[str, object]:
    """Return the record written when the job started (cwd, started_at, output path)."""
    root = jobs_dir()
    try:
        data = json.loads(_path(root, job, ".meta").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ── Stopping one ───────────────────────────────────────────────────


def kill(job: str) -> bool:
    """Kill a job and everything it spawned; return whether anything was signalled.

    The job was started with ``start_new_session=True``, so its pid is also its
    process group: one ``killpg`` takes the launcher, the code and every
    descendant of both. ``.rc`` is then written by us as ``killed``, because the
    launcher -- which would have written the real code -- is dead and a wait that
    found no ``.rc`` could not tell a killed job from a running one.
    """
    root = jobs_dir()
    with suppress(OSError, ValueError):
        pid = int(_path(root, job, ".pid").read_text(encoding="utf-8").strip())
        with suppress(OSError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        with suppress(OSError):
            os.kill(pid, signal.SIGKILL)
        _path(root, job, ".rc").write_text("killed", encoding="utf-8")
        return True
    return False


def check_timeout(value: object, default: float) -> float:
    """Coerce a call's ``timeout_s`` into a sane budget.

    Unset means ``default``; a value that is not a number, or is outside
    ``0..MAX_WAIT_S``, is refused by the caller rather than clamped: a budget the
    model did not ask for is a wait it cannot reason about, and a typo of one
    zero is better reported than obeyed.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"timeout_s must be a number of seconds, not {type(value).__name__}"
        raise TypeError(msg)
    seconds = float(value)
    if not 0 <= seconds <= MAX_WAIT_S:
        msg = f"timeout_s must be between 0 and {MAX_WAIT_S:g} (a day), got {seconds:g}"
        raise ValueError(msg)
    return seconds


def condition_from(wait: object) -> Condition:
    """Build a :class:`Condition` from a call's ``wait`` object.

    Accepts a bare string (``"exit"``) as shorthand for ``{"until": "exit"}``:
    the common case does not need an object, and a call that sends the string
    should not be refused over its shape.

    Raises:
        ValueError: If a key or an ``until`` value is not one this module knows,
            or a required field for that kind is missing. The message lists what
            is accepted, because the reader is a model that can act on it.
        TypeError: If ``timeout_s`` is not a number at all.
    """
    if isinstance(wait, str):
        wait = {"until": wait}
    if not isinstance(wait, dict):
        msg = 'wait must be an object like {"until": "exit"} or the string "exit"'
        raise TypeError(msg)

    unknown = set(wait) - CONDITION_KEYS
    if unknown:
        accepted = ", ".join(sorted(CONDITION_KEYS))
        msg = f"unknown key(s) in wait: {', '.join(sorted(unknown))}; accepted: {accepted}"
        raise ValueError(msg)

    until = wait.get("until", "exit")
    if until not in CONDITION_KINDS:
        msg = f"unknown until: {until!r}; accepted: {', '.join(CONDITION_KINDS)}"
        raise ValueError(msg)

    if until in {"file", "gone", "regex"}:
        path = wait.get("path")
        if not isinstance(path, str) or not path.strip():
            msg = f'until: "{until}" needs a "path"'
            raise ValueError(msg)
    else:
        path = None

    pattern = wait.get("pattern") if until == "regex" else None
    if until == "regex" and (not isinstance(pattern, str) or not pattern.strip()):
        msg = 'until: "regex" needs a "pattern" (a regular expression)'
        raise ValueError(msg)
    if isinstance(pattern, str):
        try:
            re.compile(pattern)
        except re.error as e:
            msg = f"pattern is not a valid regular expression: {e}"
            raise ValueError(msg) from e

    port = wait.get("port") if until == "port" else None
    if until == "port" and (isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536):
        msg = 'until: "port" needs a "port" number (1-65535)'
        raise ValueError(msg)

    return Condition(kind=until, path=path, pattern=pattern, port=port)


def tail_lines_from(wait: object) -> int | None:
    """Return a requested ``tail_lines``, or ``None`` when none was asked for."""
    if not isinstance(wait, dict):
        return None
    value = wait.get("tail_lines")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def name_from(wait: object) -> str | None:
    """Return a requested job ``name``, or ``None`` when none was asked for."""
    if not isinstance(wait, dict):
        return None
    value = wait.get("name")
    return value.strip() if isinstance(value, str) and value.strip() else None


def job_from(wait: object) -> str | None:
    """Return the id of an *existing* job this call is about, or ``None``.

    This is the key that separates "start this and wait for it" from "wait for
    the one that is already running": with ``job`` set, the code is not run at
    all, which is what a second call about the same work means and what makes
    ``name`` unambiguous -- ``name`` only ever *gives* a name, ``job`` only ever
    *refers* to one.
    """
    if not isinstance(wait, dict):
        return None
    value = wait.get("job")
    return value.strip() if isinstance(value, str) and value.strip() else None


def kill_from(wait: object) -> bool:
    """Whether the call asked for the job to be stopped."""
    return isinstance(wait, dict) and wait.get("kill") is True


def known(limit: int = 10) -> list[str]:
    """Return recent job ids, newest first -- what a call can refer to.

    Listed when a call names a job that is not there: the usual mistake is a
    name from an earlier turn (or an earlier run), and the answer "no such job"
    is only useful next to the names that do exist.
    """
    root = jobs_dir()
    entries: list[tuple[float, str]] = []
    for path in root.glob("*.pid"):
        try:
            entries.append((path.stat().st_mtime, path.stem))
        except OSError:
            continue
    entries.sort(reverse=True)
    return [job for _mtime, job in entries[:limit]]
