"""Tests for the Python execution tool and the tool registry."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from llm_cli_py.consts import ENV_JOBS_DIR
from llm_cli_py.tools.python_exec import (
    PYTHON_TOOL_DESCRIPTION,
    PYTHON_TOOL_SCHEMA,
    _check_shell_true_with_list,
    execute_python,
)
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult, ToolError
from llm_cli_py.utils import terminal


def test_code_runs_and_stdout_is_captured() -> None:
    result = execute_python("print('hello world')", explanation="Print a greeting.")
    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "hello world"


def test_failing_code_reports_stderr_and_a_nonzero_exit_code() -> None:
    result = execute_python("raise ValueError('test error')", explanation="Show a failing run.")
    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert "ValueError" in result.stderr


def test_non_ascii_output_survives_the_subprocess_pipe() -> None:
    result = execute_python('print("\u65e5\u672c\u8a9e")', explanation="Check non-ASCII output.")
    assert isinstance(result, ExecResult)
    assert result.stdout.strip() == "\u65e5\u672c\u8a9e"


def test_shell_true_with_an_argv_list_is_refused() -> None:
    """``shell=True`` drops every list element after the first, so it is refused."""
    result = execute_python(
        'subprocess.run(["cmd", "--flag"], shell=True)', explanation="Show the refused shape."
    )
    assert isinstance(result, ToolError)
    assert "shell=True" in result.error
    assert "--flag" in result.error  # names the argument the shell would drop


def test_shell_true_with_a_command_string_is_allowed() -> None:
    """A single command string is the shape ``shell=True`` actually wants."""
    assert _check_shell_true_with_list("subprocess.run('echo hi', shell=True)") is None


def test_metacharacters_in_an_argv_list_are_not_refused() -> None:
    """Regression: ``&`` in an ordinary URL is not a reason to refuse the call."""
    code = """import subprocess
subprocess.run(["curl", "-s", "https://api.example.com/search?a=1&b=2"])"""
    assert isinstance(execute_python(code, explanation="Fetch a URL that has a query string."), ExecResult)


def test_a_run_is_not_cut_off_for_taking_its_time() -> None:
    """There is no limit on a run: code that takes a while still finishes."""
    result = execute_python(
        "import time\ntime.sleep(2)\nprint('finished')", explanation="Sleep, then report."
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "finished"


def test_the_description_states_there_is_no_timeout_and_mentions_backgrounding() -> None:
    assert "no timeout" in PYTHON_TOOL_DESCRIPTION
    assert "background" in PYTHON_TOOL_DESCRIPTION


def test_the_description_tells_the_model_not_to_sleep_and_to_wait_instead() -> None:
    """The behaviour this exists for: a sleep loop costs round trips, ``wait`` does not.

    The old text told the model to start work detached and check the file in a
    later call, which is exactly the polling it read as "sleep, then look" -- the
    measured result was 59 of 134 tool calls containing a wait, and seven of them
    sleeping fifteen seconds or more. The instruction has to say the opposite.
    """
    assert "time.sleep" in PYTHON_TOOL_DESCRIPTION
    assert "do NOT" in PYTHON_TOOL_DESCRIPTION
    assert '"until": "exit"' in PYTHON_TOOL_DESCRIPTION
    assert "one round trip" in PYTHON_TOOL_DESCRIPTION.lower() or "round trip" in PYTHON_TOOL_DESCRIPTION


def test_the_schema_requires_an_explanation_as_well_as_the_code() -> None:
    """The reason for the call is not optional in what the model is shown."""
    properties = PYTHON_TOOL_SCHEMA["properties"]
    assert isinstance(properties, dict)
    # ``wait`` is advertised but optional: a call that omits it behaves exactly
    # as it did before it existed, and requiring it would make every short
    # script repeat a default.
    assert set(properties) == {"code", "explanation", "wait"}
    assert PYTHON_TOOL_SCHEMA["required"] == ["code", "explanation"]


def test_the_description_tells_the_model_the_explanation_is_shown() -> None:
    """``required`` alone does not say what the sentence is for; the text does."""
    assert "explanation" in PYTHON_TOOL_DESCRIPTION
    assert "after the code" in PYTHON_TOOL_DESCRIPTION


def test_an_explanation_does_not_change_what_runs() -> None:
    """It is the reason for the call, not an input to the code: same code, same output."""
    result = execute_python("print('hi')", explanation="Say hello.")

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "hi"


@pytest.mark.parametrize("explanation", ["", "   \n  "])
def test_a_call_without_an_explanation_is_refused(explanation: str, tmp_path: Path) -> None:
    """Required means required: the code is refused before it runs, as ``shell=True`` is."""
    marker = tmp_path / "written"
    result = execute_python(f"open({str(marker)!r}, 'w').write('x')", explanation=explanation)

    assert isinstance(result, ToolError)
    assert "explanation" in result.error
    assert "NOT run" in result.error
    assert not marker.exists()  # nothing ran: the refusal came first


def test_a_call_that_omits_the_explanation_is_refused_too() -> None:
    """A model that ignored the schema gets a usable sentence, not a ``TypeError``."""
    result = execute_python("print('hi')")  # type: ignore[call-arg]

    assert isinstance(result, ToolError)
    assert "explanation" in result.error


def test_tool_result_types_serialise() -> None:
    assert ExecResult(stdout="hello").to_dict() == {"stdout": "hello", "stderr": "", "exit_code": 0}
    assert ToolError(error="boom").to_dict() == {"error": "boom"}


def test_registry_registers_looks_up_and_advertises_a_tool() -> None:
    registry = ToolRegistry()
    registry.register("calc", "Calculate", {"type": "object"}, lambda **_: ExecResult())

    tool = registry.get("calc")
    assert tool is not None and tool.schema.name == "calc"
    assert "calc" in registry and "other" not in registry
    assert registry.get("missing") is None
    assert [s.name for s in registry.get_schemas()] == ["calc"]
    assert tool.schema.description == "Calculate"


# ── The child never gets the terminal ─────────────────────────────


def test_tool_code_cannot_read_what_the_user_types() -> None:
    """The child's stdin is not the terminal, so a prompt cannot be answered by it.

    Before this held, code that called ``input()`` read the user's next keystroke
    out from under the CLI's own prompt: the line typed at ``> `` went to the
    tool instead of to the model.
    """
    result = execute_python(
        "import sys\ndata = sys.stdin.read(10)\nprint(repr(data))",
        explanation="Check what the child's stdin is connected to.",
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "''"  # end of input: nothing to read


def test_tool_code_reading_stdin_returns_instead_of_waiting_for_a_keypress() -> None:
    """``input()`` fails at once rather than blocking the turn on a keystroke."""
    result = execute_python(
        "try:\n    input('prompt: ')\nexcept EOFError:\n    print('eof')",
        explanation="Show that a tool cannot wait for the keyboard.",
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("eof")  # the prompt came first


def test_tool_code_cannot_put_the_terminal_into_raw_mode() -> None:
    """Without a descriptor on the terminal there is no mode to change.

    A child that could do this left the terminal raw when it exited, and a raw
    terminal has ``ISIG`` off -- so Ctrl+C stopped being a signal, for the CLI
    and for the shell after it. ``tty.setraw`` fails here because the child has
    no terminal at all.
    """
    result = execute_python(
        "import os, tty\ntry:\n    tty.setraw(0)\nexcept Exception as e:\n    print(type(e).__name__)",
        explanation="Show that a tool has no terminal to change.",
    )

    assert isinstance(result, ExecResult)
    # `termios.error` (an OSError) from the ioctl, or ValueError for fd 0 being
    # /dev/null: either way the terminal is untouched.
    assert result.stdout.strip() in {"error", "OSError", "ValueError", "TermiosError"}


def test_a_tool_run_gives_the_terminal_back_even_if_it_was_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settings captured before the run are put back after it, however it ends.

    The child is kept away from the terminal, but the guarantee does not rest on
    that alone: a run of tool code starts and ends with the same terminal
    settings, so anything that did manage to change them cannot outlive the
    call.
    """
    calls: list[str] = []

    class _FakeTermios:
        """Enough of ``termios`` to see what is asked of it."""

        TCSANOW = 0
        error = OSError

        def __init__(self) -> None:
            self.state = ["cooked"]

        def tcgetattr(self, _fd: int) -> list[str]:
            calls.append("get")
            return list(self.state)

        def tcsetattr(self, _fd: int, _when: int, state: list[str]) -> None:
            calls.append(f"set {state[0]}")
            self.state = list(state)

    class _FakeProc:
        """A child that finishes at once, after leaving the terminal raw."""

        returncode = 0

        def __init__(self) -> None:
            fake.state = ["raw"]
            self.kwargs: dict[str, object] = {}

        def communicate(self) -> tuple[str, str]:
            return "", ""

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    fake = _FakeTermios()
    monkeypatch.setattr(terminal, "termios", fake)
    monkeypatch.setattr(terminal.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(terminal.sys.stdin, "fileno", lambda: 0, raising=False)
    monkeypatch.setattr(
        "llm_cli_py.tools.python_exec.subprocess.Popen",
        lambda *_a, **_k: _FakeProc(),
    )

    result = execute_python("print('x')", explanation="Run something.")

    assert isinstance(result, ExecResult)
    assert calls == ["get", "set cooked"]  # captured before, put back after
    assert fake.state == ["cooked"]


def test_no_terminal_means_nothing_to_save_or_restore() -> None:
    """A redirected stdin is not a terminal: every step is a no-op, never an error."""
    assert terminal.save() is None  # pytest's stdin is not a tty
    terminal.restore(None)  # must not raise
    with terminal.preserved():
        pass


# ── Waiting for work that takes a while ────────────────────────────


@pytest.fixture
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the job store at a tmp dir, so tests never touch the real one.

    The directory is read from the environment on every call (see
    ``jobs.jobs_dir``), so this needs no reload -- and a test that leaves a job
    running cannot leak into the next one.
    """
    from llm_cli_py.tools import jobs

    target = tmp_path / "jobs"
    monkeypatch.setenv(ENV_JOBS_DIR, str(target))
    jobs.reset()
    return target


@pytest.mark.usefixtures("jobs_dir")
def test_wait_exit_returns_one_result_with_the_whole_output() -> None:
    """``until: exit`` is the call that replaces "sleep, then check again".

    One call, one result: the outcome, how long it took, and the output. The
    point is not that it waits -- the default path waits too -- but that a code
    that outlives a single poll still costs exactly one round trip.
    """
    import time

    started = time.monotonic()
    result = execute_python(
        "import time\nprint('a', flush=True)\ntime.sleep(0.5)\nprint('b')",
        explanation="Show one call waiting for a slow script.",
        wait={"until": "exit"},
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip().splitlines() == ["a", "b"]
    assert result.wait is not None and result.wait.outcome == "satisfied"
    assert result.wait.running is False
    # The wait really happened, rather than the code having been run elsewhere:
    # the elapsed time is at least the script's own.
    assert result.wait.elapsed_s >= 0.4
    assert time.monotonic() - started >= 0.4


@pytest.mark.usefixtures("jobs_dir")
def test_the_result_of_a_wait_is_json_and_names_the_job() -> None:
    """What the model is sent has to carry the wait: it decides what to do next."""
    result = execute_python("print('x')", explanation="Show the payload.", wait={"until": "exit"})

    assert isinstance(result, ExecResult)
    payload = json.loads(result.as_tool_content())
    assert payload["stdout"] == "x\n"
    wait = payload["wait"]
    assert isinstance(wait, dict)
    assert wait["outcome"] == "satisfied"
    assert wait["job"]
    assert "the job to exit" in wait["condition"]


@pytest.mark.usefixtures("jobs_dir")
def test_background_start_returns_at_once_and_a_later_call_waits_for_it() -> None:
    """The two halves of a job: start it now, ask about it later.

    The second call refers to the job by id, so ``name`` keeps one meaning
    (it *gives* a name) and ``job`` the other (it *refers* to one).
    """
    started = execute_python(
        "import time\nprint('bg', flush=True)\ntime.sleep(0.6)",
        explanation="Start work that is not needed yet.",
        wait={"until": "background", "name": "bg"},
    )
    assert isinstance(started, ExecResult)
    assert started.wait is not None
    assert started.wait.outcome == "background"
    job = started.wait.job
    assert job is not None

    # A read: the same call with the budget turned down.
    read = execute_python("", explanation="See whether it is done.", wait={"job": job, "timeout_s": 0})
    assert isinstance(read, ExecResult)
    assert read.wait is not None
    assert read.wait.outcome == "status"

    waited = execute_python("", explanation="Wait for it to finish.", wait={"job": job, "until": "exit"})
    assert isinstance(waited, ExecResult)
    assert waited.wait is not None and waited.wait.outcome == "satisfied"
    assert "bg" in waited.stdout


def test_a_file_condition_returns_when_the_file_appears(jobs_dir: Path) -> None:
    """The artifact, not the time: what a build is actually waited on."""
    target = jobs_dir.parent / "artifact.txt"
    result = execute_python(
        f"import time\ntime.sleep(0.3)\nopen({str(target)!r}, 'w').write('x')\nimport time\ntime.sleep(3)",
        explanation="Wait for the artifact rather than for a guessed duration.",
        wait={"until": "file", "path": str(target), "timeout_s": 30},
    )

    assert isinstance(result, ExecResult)
    assert result.wait is not None and result.wait.outcome == "satisfied"
    # It returned on the file, not at the end of the script: the script sleeps
    # for three more seconds afterwards.
    assert result.wait.elapsed_s < 2.5
    # The call did what it was for, so it is a success -- even though the job it
    # was watching has not exited yet.
    assert result.exit_code == 0
    assert result.ok


def test_a_regex_condition_returns_on_a_log_line(jobs_dir: Path) -> None:
    """A server publishes "Ready" in its log; that line is the event."""
    log = jobs_dir.parent / "server.log"
    result = execute_python(
        (
            f"import time\nf = open({str(log)!r}, 'w')\n"
            "for i in range(3):\n    f.write('tick %d\\n' % i); f.flush(); time.sleep(0.2)\n"
            "f.write('Ready on 8080\\n'); f.flush(); time.sleep(3)"
        ),
        explanation="Wait for the server to say it is ready.",
        wait={"until": "regex", "path": str(log), "pattern": "^Ready", "timeout_s": 30},
    )

    assert isinstance(result, ExecResult)
    assert result.wait is not None and result.wait.outcome == "satisfied"
    assert result.wait.elapsed_s < 2.5
    assert result.ok  # the line matched: that was the call's whole purpose


def test_a_condition_stops_early_when_the_job_ends_first(jobs_dir: Path) -> None:
    """A job that exited cannot make the file appear: say so, do not sit out the budget.

    This is the failure a sleep loop cannot tell apart from "not yet": it waits
    the whole time and then reports the same nothing.
    """
    missing = jobs_dir.parent / "never.txt"
    result = execute_python(
        "raise SystemExit(0)",
        explanation="Wait for something the job will never produce.",
        wait={"until": "file", "path": str(missing), "timeout_s": 30},
    )

    assert isinstance(result, ExecResult)
    assert result.wait is not None
    assert result.wait.outcome == "ended"
    assert result.wait.elapsed_s < 5
    # The code exited 0 -- it succeeded at doing nothing -- but the call did not
    # get what it waited for, and the two are reported separately so neither is
    # mistaken for the other.
    assert result.exit_code == 0
    assert result.wait.delivered is False


@pytest.mark.usefixtures("jobs_dir")
def test_a_timeout_reports_that_the_job_is_still_running() -> None:
    """The budget bounds the call, not the work: the job keeps going."""
    result = execute_python(
        "import time\ntime.sleep(30)",
        explanation="Show a wait that runs out of budget.",
        wait={"until": "exit", "timeout_s": 0.3},
    )

    assert isinstance(result, ExecResult)
    assert result.wait is not None
    assert result.wait.outcome == "timeout"
    if result.wait.job:
        from llm_cli_py.tools import jobs

        jobs.kill(result.wait.job)


@pytest.mark.usefixtures("jobs_dir")
def test_an_unknown_job_is_an_error_that_lists_the_ones_that_exist() -> None:
    """A name from an earlier run is the usual mistake; say what does exist."""
    started = execute_python(
        "print('x')", explanation="Make a job.", wait={"until": "background", "name": "known"}
    )
    assert isinstance(started, ExecResult) and started.wait is not None

    result = execute_python("", explanation="Refer to a job that is not there.", wait={"job": "no-such-job"})

    assert isinstance(result, ToolError)
    assert "no-such-job" in result.error
    assert "known" in result.error  # the useful half of the message


def test_the_wait_object_is_validated_before_anything_runs(jobs_dir: Path) -> None:
    """A condition that cannot happen is refused with the list of ones that can."""
    marker = jobs_dir.parent / "ran"
    code = f"open({str(marker)!r}, 'w').write('x')"

    for wait, expected in [
        ({"until": "eventually"}, "unknown until"),
        ({"until": "exit", "timout_s": 5}, "unknown key"),
        ({"until": "exit", "timeout_s": "soon"}, "must be a number"),
        ({"until": "regex", "path": "x", "pattern": "[unclosed"}, "not a valid regular expression"),
        ({"until": "file"}, 'needs a "path"'),
        ({"until": "port", "port": 99999}, 'needs a "port"'),
        ({"until": "exit", "timeout_s": 999_999_999}, "must be between"),
    ]:
        result = execute_python(code, explanation="Show a refused wait.", wait=wait)
        assert isinstance(result, ToolError), wait
        assert expected in result.error, (wait, result.error)

    assert not marker.exists()  # nothing ran: the refusal came first


@pytest.mark.usefixtures("jobs_dir")
def test_the_string_shorthand_means_until_exit() -> None:
    """``wait: "exit"`` is the common case, and should not need an object."""
    result = execute_python("print('ok')", explanation="Use the shorthand.", wait="exit")

    assert isinstance(result, ExecResult)
    assert result.wait is not None and result.wait.outcome == "satisfied"


@pytest.mark.usefixtures("jobs_dir")
def test_tail_lines_returns_only_the_end_and_says_what_it_cut() -> None:
    """A long output is read from the end, and the cut is stated, not silent."""
    result = execute_python(
        "for i in range(200): print(i)",
        explanation="Show only the end of a long output.",
        wait={"until": "exit", "tail_lines": 3},
    )

    assert isinstance(result, ExecResult)
    body = [line for line in result.stdout.splitlines() if not line.startswith("[output truncated")]
    assert body[-3:] == ["197", "198", "199"]
    assert "output truncated" in result.stdout


@pytest.mark.usefixtures("jobs_dir")
def test_a_jobs_output_goes_to_a_file_so_a_flood_cannot_block_it() -> None:
    """A pipe holds ~64 KiB; a job that prints more would block forever on one.

    This is the bug the file exists for: ``communicate()`` on a pipe reads to
    EOF, and EOF never comes while a detached descendant holds the write end.
    With a file there is no write end to hold, and the job finishes.
    """
    result = execute_python(
        "import sys\nfor _ in range(2000): sys.stdout.write('y' * 100 + '\\n')\nprint('END')",
        explanation="Show that a big output cannot block the job.",
        wait={"until": "exit", "timeout_s": 60},
    )

    assert isinstance(result, ExecResult)
    assert result.wait is not None and result.wait.outcome == "satisfied"
    assert result.exit_code == 0
    assert "END" in result.stdout


@pytest.mark.usefixtures("jobs_dir")
def test_killing_a_job_stops_it_and_marks_it_killed() -> None:
    """A job left to run for an hour has to be stoppable, and its end has to be visible."""
    started = execute_python(
        "import time\ntime.sleep(300)",
        explanation="Start something to stop.",
        wait={"until": "background", "name": "stoppable"},
    )
    assert isinstance(started, ExecResult) and started.wait is not None
    job = started.wait.job
    assert job is not None

    from llm_cli_py.tools import jobs

    assert jobs.kill(job)
    assert jobs.status(job).running is False
    assert jobs.status(job).exit_code == -1  # "killed", not a claimed exit code


@pytest.mark.usefixtures("jobs_dir")
def test_a_running_jobs_files_are_not_overwritten_by_reusing_its_name() -> None:
    """``name: "build"`` twice means two jobs, not one with its output truncated."""
    first = execute_python(
        "import time\ntime.sleep(30)",
        explanation="Start a long job named build.",
        wait={"until": "background", "name": "build"},
    )
    assert isinstance(first, ExecResult) and first.wait is not None
    first_job = first.wait.job

    second = execute_python(
        "print('second')",
        explanation="Start another named build.",
        wait={"name": "build", "until": "exit"},
    )
    assert isinstance(second, ExecResult) and second.wait is not None

    assert first_job is not None and second.wait.job != first_job

    from llm_cli_py.tools import jobs

    # The first job is still there and still running: its output was not touched.
    assert jobs.status(first_job).running is True
    jobs.kill(first_job)


@pytest.mark.usefixtures("jobs_dir")
def test_reusing_a_finished_jobs_name_starts_a_fresh_job() -> None:
    """A stale ``.rc`` must not answer for the run that is just starting.

    The name of a finished job is reusable (re-running a build should not grow a
    suffix), and the obvious implementation gets this wrong: the old ``.rc`` is
    still on disk, so the wait returns "finished" before the new code has printed
    anything, and the reader is handed the previous run's output.
    """
    first = execute_python(
        "print('first')", explanation="Run it once.", wait={"until": "exit", "name": "reuse"}
    )
    assert isinstance(first, ExecResult) and first.wait is not None

    second = execute_python(
        "import time\ntime.sleep(0.4)\nprint('second')",
        explanation="Run it again under the same name.",
        wait={"until": "exit", "name": "reuse", "timeout_s": 20},
    )

    assert isinstance(second, ExecResult) and second.wait is not None
    assert second.wait.job == first.wait.job  # the name was reused, not suffixed
    assert second.wait.outcome == "satisfied"
    assert second.wait.elapsed_s >= 0.3  # it waited for *this* run
    assert second.stdout.strip() == "second"  # and this run's output, not the last one's


@pytest.mark.usefixtures("jobs_dir")
def test_a_finished_job_does_not_leave_a_process_that_looks_alive() -> None:
    """A job is a child of the CLI, so an unreaped one is a zombie that answers kill(0).

    Found while writing the reuse test: ``_still_running`` asked ``kill(pid, 0)``,
    which a zombie answers as if it were alive -- so a finished job read as
    running, and a name that was free took a suffix it did not need.
    """
    from llm_cli_py.tools import jobs

    ids = [
        execute_python("pass", explanation="Start a job.", wait={"until": "background"}).wait.job
        for _ in range(5)
    ]
    assert all(job is not None for job in ids)

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(jobs.status(job).running for job in ids if job):
        time.sleep(0.05)

    assert [job for job in ids if job and jobs.status(job).running] == []
    assert all(jobs.status(job).exit_code == 0 for job in ids if job)
