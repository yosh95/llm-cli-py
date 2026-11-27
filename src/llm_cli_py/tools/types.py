"""Typed result classes for tool execution.

A tool returns one of :class:`ExecResult` / :class:`ToolError`, and those two
classes own what the CLI does with such a value for the model and for the
transcript: ``as_tool_content`` produces the JSON string recorded for the model,
``as_display_lines`` the plain-text lines of the transcript, and ``ok`` says
whether the run succeeded -- which is what decides a colour in ``ui.display``,
without this module needing to know that colours exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


def _indent_lines(text: str, prefix: str) -> list[str]:
    """Split ``text`` into stripped lines, each indented with ``prefix``."""
    return [f"{prefix}{line}" for line in text.rstrip().splitlines()]


@dataclass(frozen=True)
class WaitInfo:
    """How a call waited, when it waited for anything.

    A tool call carries the answer to "is it done yet", and this is what it
    answered with. It is separate from ``stdout``/``stderr`` because the answer
    is about the *wait*, not about the code: a job can be started, waited for and
    read in one call, and three different facts matter -- what was waited for,
    whether it happened, and how long it took.

    Attributes:
        job: Id of the job this was about, when there was one.
        condition: What was waited for, in words (``"the job to exit"``).
        outcome: ``satisfied``, ``timeout``, ``ended``, ``status`` or
            ``background``.
        elapsed_s: How long the wait took, in seconds.
        running: Whether the job was still running when the call returned.
    """

    job: str | None
    condition: str
    outcome: str
    elapsed_s: float
    running: bool

    @property
    def delivered(self) -> bool:
        """Whether the call got what it waited for.

        Separate from the exit code, because the two answer different questions.
        A job can exit 0 without ever producing the file it was waited on for
        (``ended``): the code succeeded, the wait did not, and a reader that saw
        only ``exit_code: 0`` would take the call for a success. So the exit code
        stays what it is -- the code's own verdict, which is worth having -- and
        this says whether the *call* got its answer.
        """
        return self.outcome in {"satisfied", "background", "status"}

    def as_line(self) -> str:
        """Return the one-line summary shown under a result and sent to the model."""
        where = f"job {self.job}: " if self.job else ""
        took = f"{self.elapsed_s:.1f}s"
        if self.outcome == "background":
            return f"{where}started in the background, not waited for"
        if self.outcome == "satisfied":
            return f"{where}waited {took} for {self.condition} -- satisfied"
        if self.outcome == "ended":
            return f"{where}waited {took} for {self.condition} -- the job ended first, so it did not happen"
        if self.outcome == "status":
            state = "still running" if self.running else "finished"
            return f"{where}{state}"
        if self.outcome == "timeout":
            still = "still running" if self.running else "not running"
            return f"{where}waited {took} for {self.condition} -- timed out, job {still}"
        return f"{where}waited {took} for {self.condition} -- {self.outcome}"


@dataclass
class ExecResult:
    """Result of Python code execution in a subprocess.

    Fields:
        stdout: Standard output from the executed code.
        stderr: Standard error from the executed code.
        exit_code: Exit code (0 = success, non-zero = error, -1 = exception).
        wait: How the call waited, when it waited for anything (``None`` for a
            call that ran code and waited for nothing).
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    wait: WaitInfo | None = None

    @property
    def ok(self) -> bool:
        """Whether the code ran and exited zero."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, object]:
        """Return this result as a JSON-serialisable payload."""
        payload: dict[str, object] = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
        }
        if self.wait is not None:
            payload["wait"] = {
                "job": self.wait.job,
                "condition": self.wait.condition,
                "outcome": self.wait.outcome,
                "elapsed_s": round(self.wait.elapsed_s, 2),
                "running": self.wait.running,
                "delivered": self.wait.delivered,
            }
        return payload

    def as_display_lines(self) -> list[str]:
        """Return this result as display lines for the terminal."""
        lines = [f"Exit code: {self.exit_code}"]
        if self.wait is not None:
            # Above the output, not below it: whether the work finished is the
            # first thing a reader needs, and the output is read in that light.
            lines.append(f"[wait] {self.wait.as_line()}")
        if self.stdout.strip():
            lines.append("[stdout]")
            lines.extend(_indent_lines(self.stdout, "  "))
        if self.stderr.strip():
            lines.append("[stderr]")
            lines.extend(_indent_lines(self.stderr, "  "))
        return lines

    def as_tool_content(self) -> str:
        """Return the JSON string recorded as this turn's tool message."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ToolError:
    """Represents a tool execution error.

    When a tool encounters an error (missing API key, network failure,
    etc.), it returns a ToolError.
    """

    error: str

    @property
    def ok(self) -> bool:
        """A tool error is never a success."""
        return False

    def to_dict(self) -> dict[str, str]:
        """Return this error as a JSON-serialisable payload."""
        return {"error": self.error}

    def as_display_lines(self) -> list[str]:
        """Return this error as display lines for the terminal."""
        return [f"Error: {self.error}"]

    def as_tool_content(self) -> str:
        """Return the JSON string recorded as this turn's tool message."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


ToolResult = ExecResult | ToolError
"""Union type for all possible tool execution results."""


def normalise_tool_result(result: object) -> ToolResult:
    """Coerce whatever a tool returned into an :class:`ExecResult`/``ToolError``.

    Tools are declared as ``Callable[..., ToolResult]``; a tool returning some
    other type is a programming error. Rather than silently reinterpreting the
    value as a successful result (or letting a later attribute access crash on
    a string), it is turned into an explicit :class:`ToolError` so the model
    sees a clear, structured failure.
    """
    if isinstance(result, (ExecResult, ToolError)):
        return result
    return ToolError(error=f"Tool returned an invalid result type: {type(result).__name__}")
