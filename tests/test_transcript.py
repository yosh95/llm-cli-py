"""Tests for the conversation's two renderings: the log, and ``/raw``'s Markdown.

Most of these drive the real ``send`` path (only the HTTP transport is patched),
because what is being tested is that the conversation the client records -- and
nothing else -- is what lands in the file. The Markdown tests at the end write
the conversation by hand: what is being tested there is the layout, and the
shapes it has to survive (tool calls, tool results, fences inside the text).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_cli_py.models import ClientState, Message, Role
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.session.transcript import (
    LOG_FORMAT_VERSION,
    ConversationLog,
    message_record,
    render_markdown,
)
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult
from tests.conftest import completion, make_client, register, tool_call

LOG_ENV = "LLM_CLI_LOG_DIR"
SEND = "llm_cli_py.providers.llm_api.post_json"


def records(path: Path) -> list[dict]:
    """Read a run's file as its records, one per line (the format's whole point)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def messages(path: Path) -> list[dict]:
    """The message records of a run's file, in order, with ``type`` dropped."""
    return [{k: v for k, v in r.items() if k != "type"} for r in records(path) if r["type"] == "message"]


def run_files(directory: Path) -> list[Path]:
    """The run files in ``directory``, in name order (one per run, oldest first)."""
    return sorted(directory.glob("*.jsonl"))


def one_file(directory: Path) -> Path:
    """The single run file in ``directory``, asserting there is exactly one."""
    files = run_files(directory)
    assert len(files) == 1
    return files[0]


def write_turn(directory: Path, content: str, *, model: str = "m", **kwargs: object) -> ConversationLog:
    """A log holding one user turn, written through a client as the CLI writes it.

    Messages reach the log through ``remember`` (see
    :class:`~llm_cli_py.base.ConversationObserver`), so a test that wants a
    logged turn goes the same way rather than calling the observer itself.
    """
    log = ConversationLog(directory, **kwargs)  # type: ignore[arg-type]
    log.prepare()
    client = make_client(model)
    client.observe(log)
    client.remember(Message(role=Role.USER, content=content))
    return log


def logged_session(tmp_path: Path, *, started_at: datetime | None = None) -> tuple[ActiveSession, Path]:
    """A session against a fake transport, watched by a log in ``tmp_path``.

    ``tmp_path / "log"`` is the log directory, so the run's file can be found
    without knowing the name it was given.
    """
    directory = tmp_path / "log"
    client = make_client("gpt-4o")
    log = ConversationLog(directory, started_at=started_at)
    log.prepare()
    client.observe(log)
    return ActiveSession(client, SessionContext(tool_registry=ToolRegistry())), directory


class DeadHandle:
    """A file handle whose writes fail, as a full disk or a held-open file would.

    Substituted for the real handle rather than for ``_append``, so what is tested
    is the log's own handling of a failing write -- the report, the switch-off, and
    that the lines already on disk are still there.
    """

    def __init__(self, error: OSError) -> None:
        self.error = error
        self.closed = False

    def write(self, _data: bytes) -> int:
        raise self.error

    def flush(self) -> None:
        raise self.error

    def truncate(self, _size: int) -> int:
        raise self.error

    def seek(self, _offset: int) -> int:
        return 0

    def tell(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


# ── The directory: one file per run ────────────────────────────────


def test_without_the_env_var_no_log_is_written(monkeypatch, tmp_path) -> None:
    """Unset is the default, and means nothing is written anywhere."""
    from llm_cli_py import main as main_module

    monkeypatch.delenv(LOG_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    assert main_module._conversation_log_or_none() is None
    assert list(tmp_path.iterdir()) == []


def test_startup_leaves_the_runs_that_are_already_logged_alone(tmp_path, monkeypatch) -> None:
    """A new run adds a file; it does not touch the ones already in the directory."""
    from llm_cli_py import main as main_module

    directory = tmp_path / "log"
    write_turn(directory, "before", model="first", started_at=datetime(2026, 11, 12, 9, 0)).close()
    before = one_file(directory).read_text(encoding="utf-8")

    monkeypatch.setenv(LOG_ENV, str(directory))
    log = main_module._conversation_log_or_none()
    assert log is not None
    log.close()

    assert (directory / "20261112T090000.jsonl").read_text(encoding="utf-8") == before
    assert len(run_files(directory)) == 2


def test_a_new_run_starts_empty_and_only_writes_the_log(tmp_path, monkeypatch) -> None:
    """The log is a record, not the next run's context.

    Reading a log back into ``ClientState`` would make a run's history depend on
    whatever was logged last; instead the client starts empty and the log is only
    ever written.
    """
    from llm_cli_py import main as main_module

    directory = tmp_path / "log"
    write_turn(directory, "yesterday", model="first", started_at=datetime(2026, 11, 12, 9, 0)).close()
    monkeypatch.setenv(LOG_ENV, str(directory))

    log = main_module._conversation_log_or_none()
    client = make_client("m")
    assert log is not None
    assert client.state.conversation == []
    client.observe(log)
    client.remember(Message(role=Role.USER, content="hello"))
    log.close()

    assert [m.content for m in client.state.conversation] == ["hello"]
    assert [messages(path)[0]["content"] for path in run_files(directory)] == ["yesterday", "hello"]


def test_a_file_is_named_after_the_run_that_wrote_it(tmp_path) -> None:
    """The name says when the run started, so a directory reads as a timeline."""
    directory = tmp_path / "log"
    log = ConversationLog(directory, started_at=datetime(2026, 11, 12, 9, 30, 5))

    log.prepare()
    log.close()

    assert [p.name for p in directory.iterdir()] == ["20261112T093005.jsonl"]


def test_two_runs_started_in_the_same_second_get_two_files(tmp_path) -> None:
    """The name is only a starting point; a second run must not overwrite a first."""
    directory = tmp_path / "log"
    stamp = datetime(2026, 11, 12, 9, 0, 0)

    write_turn(directory, "first", model="one", started_at=stamp).close()
    write_turn(directory, "second", model="two", started_at=stamp).close()

    assert [p.name for p in run_files(directory)] == ["20261112T090000-1.jsonl", "20261112T090000.jsonl"]
    assert [messages(p)[0]["content"] for p in run_files(directory)] == ["second", "first"]


def test_starting_up_does_not_read_the_runs_already_there(tmp_path) -> None:
    """A startup with a directory full of runs reads none of them.

    The files that predate this run are left byte for byte as they were, which is
    what makes the cost of logging independent of how much has been logged.
    """
    directory = tmp_path / "log"
    write_turn(directory, "before", model="first", started_at=datetime(2026, 11, 12, 9, 0)).close()
    # Something this version would not know how to read: unparseable, and still
    # not in the way -- a run does not have to read its neighbours to add a file.
    (directory / "20261110T000000.jsonl").write_text("not json at all\n", encoding="utf-8")
    untouched = [p.read_bytes() for p in run_files(directory)]

    log = write_turn(directory, "after", started_at=datetime(2026, 11, 13, 9, 0))
    log.close()

    assert not log.disabled
    assert [p.name for p in run_files(directory)] == [
        "20261110T000000.jsonl",
        "20261112T090000.jsonl",
        "20261113T090000.jsonl",
    ]
    assert [p.read_bytes() for p in run_files(directory)[:2]] == untouched


def test_a_log_in_a_missing_directory_is_created(tmp_path) -> None:
    """The path is used as given; its directories are made for it."""
    directory = tmp_path / "runs" / "2026"

    write_turn(directory, "hi").close()

    assert messages(one_file(directory))[0]["content"] == "hi"


def test_a_second_run_adds_a_file_instead_of_replacing_the_first(tmp_path) -> None:
    """The point of the directory: starting the CLI again keeps the run before it.

    Nothing is read back into the conversation -- a run starts empty -- so the
    earlier session is in the directory only.
    """
    directory = tmp_path / "log"

    write_turn(directory, "yesterday", model="gpt-4o", started_at=datetime(2026, 11, 12, 9, 0)).close()
    write_turn(directory, "today", model="qwen3", started_at=datetime(2026, 11, 13, 9, 0)).close()

    files = run_files(directory)
    assert [p.name for p in files] == ["20261112T090000.jsonl", "20261113T090000.jsonl"]
    assert [messages(p)[0]["content"] for p in files] == ["yesterday", "today"]
    assert [records(p)[1]["model"] for p in files] == ["gpt-4o", "qwen3"]
    # The run that picked the directory up did not pick up its conversation.
    client = make_client("qwen3")
    assert [m.content for m in client.state.conversation] == []


# ── What a file holds ──────────────────────────────────────────────


def test_the_run_s_file_opens_with_a_header_and_its_session(tmp_path) -> None:
    """What a file holds is named at its top: the layout, then which run it is."""
    session, directory = logged_session(tmp_path, started_at=datetime(2026, 11, 12, 9, 0))

    with patch(SEND, return_value=completion("hi")):
        session.process_and_print("hello")

    first, second = records(one_file(directory))[:2]
    assert first == {"type": "log", "version": LOG_FORMAT_VERSION}
    assert second == {"type": "session", "started_at": "2026-11-12T09:00:00", "model": "gpt-4o"}


def test_the_log_holds_the_whole_conversation(tmp_path) -> None:
    """User turns, assistant replies, tool calls and tool results all appear."""
    session, directory = logged_session(tmp_path)
    register(session.ctx.tool_registry, "calc", lambda **_: ExecResult(stdout="42"))
    turns = [
        completion(tool_calls=[tool_call("c1", "calc", '{"code": "40+2"}')]),
        completion("It is 42"),
    ]

    with patch(SEND, side_effect=turns):
        session.process_and_print("compute this")

    path = one_file(directory)
    assert messages(path)[0] == {"role": "user", "content": "compute this"}
    assert [m["role"] for m in messages(path)] == ["user", "assistant", "tool", "assistant"]

    assistant = messages(path)[1]
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["tool_calls"][0]["function"]["name"] == "calc"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"code": "40+2"}

    tool = messages(path)[2]
    assert tool["tool_call_id"] == "c1"
    assert json.loads(tool["content"])["stdout"] == "42"
    assert messages(path)[3]["content"] == "It is 42"


def test_the_system_message_is_one_record_among_the_others(tmp_path) -> None:
    """:class:`~llm_cli_py.base.LlmClient` seeds the system message; the log keeps it.

    It is never passed through ``remember``, so the log has to pick it up from the
    conversation -- otherwise the first thing the model was sent would be the one
    record missing from the run.
    """
    directory = tmp_path / "log"
    log = ConversationLog(directory)
    log.prepare()
    client = make_client("gpt-4o", instructions="Be brief.\n")
    client.observe(log)
    client.remember(Message(role=Role.USER, content="hello"))
    log.close()

    assert [m["role"] for m in messages(one_file(directory))] == ["system", "user"]
    assert messages(one_file(directory))[0]["content"] == "Be brief.\n"


def test_a_message_record_omits_the_fields_that_do_not_apply() -> None:
    """A plain turn carries no ``tool_call_id``/``tool_calls`` keys at all."""
    assert message_record(Message(role=Role.USER, content="hi")) == {"role": "user", "content": "hi"}


# ── Appending, and rolling a turn back ──────────────────────────────


def test_the_log_is_written_as_the_run_goes_not_only_at_the_end(tmp_path) -> None:
    """A run that dies mid-turn still leaves what had been recorded.

    The tool looks at the log while it runs -- that is, between the assistant's
    tool call and its result -- and then dies, the shape of a run that ends badly
    halfway through a turn.
    """
    session, directory = logged_session(tmp_path)

    def die(**_kwargs: object) -> ExecResult:
        assert [m["role"] for m in messages(one_file(directory))] == ["user", "assistant"]
        raise KeyboardInterrupt

    register(session.ctx.tool_registry, "slow", die)
    turn = completion(tool_calls=[tool_call("c1", "slow", "{}")])

    with patch(SEND, return_value=turn), pytest.raises(KeyboardInterrupt):
        session.process_and_print("run it")

    # The interrupted turn is rolled back, and the file follows: the prompt
    # stays, the reply nobody could finish is gone.
    assert [m["role"] for m in messages(one_file(directory))] == ["user"]


def test_a_discarded_turn_disappears_from_the_log(tmp_path) -> None:
    """The log shows what the model will be sent, not what it tried once."""
    session, directory = logged_session(tmp_path)
    turn = completion(tool_calls=[tool_call("c1", "execute_python", '{"code": "pri')])

    with patch(SEND, return_value=turn):
        session.process_and_print("run it")

    assert [m["role"] for m in messages(one_file(directory))] == ["user"]


def test_a_rolled_back_turn_leaves_no_trace_of_its_bytes(tmp_path) -> None:
    """The file is truncated, not annotated: no marker, and no hole of NULs.

    Truncating and then appending from where the handle still stands would leave
    the bytes that were cut as a run of NULs; rolling a turn back means the file
    is what it would have been had the turn never happened.
    """
    directory = tmp_path / "log"
    log = ConversationLog(directory)
    log.prepare()
    client = make_client("gpt-4o")
    client.observe(log)
    client.remember(Message(role=Role.USER, content="hello"))

    client.remember(Message(role=Role.ASSISTANT, content="x" * 500))
    client.rollback_last_turn()
    client.remember(Message(role=Role.ASSISTANT, content="second answer"))
    log.close()

    path = one_file(directory)
    assert "\x00" not in path.read_text(encoding="utf-8")
    assert [m["content"] for m in messages(path)] == ["hello", "second answer"]


def test_a_log_whose_tail_was_cut_off_is_still_mostly_readable(tmp_path) -> None:
    """An append can only damage its last line; every line before it still parses.

    This is what the format trades the single-document guarantee for: a run killed
    mid-write leaves a truncated line, not an unparseable file.
    """
    directory = tmp_path / "log"
    write_turn(directory, "a turn that survived").close()
    path = one_file(directory)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "message", "role": "assis')  # killed mid-write

    parsed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    assert [r["type"] for r in parsed] == ["log", "session", "message"]
    assert parsed[-1]["content"] == "a turn that survived"


# ── A log that cannot be used ───────────────────────────────────────


def test_a_path_that_is_a_file_is_reported_not_written_into(tmp_path, capsys) -> None:
    """A directory the log needs is a directory; a file there is not one to clobber."""
    blocker = tmp_path / "log"
    blocker.write_text("a file, not a directory\n", encoding="utf-8")

    log = ConversationLog(blocker)
    log.prepare()
    client = make_client("m")
    client.observe(log)
    client.remember(Message(role=Role.USER, content="x"))

    assert log.disabled
    assert blocker.read_text(encoding="utf-8") == "a file, not a directory\n"
    out = capsys.readouterr().out
    assert "log" in out.lower()
    assert "not a directory" in out


def test_a_directory_that_cannot_be_written_to_stops_logging_without_the_turn(tmp_path, capsys) -> None:
    """A directory the OS refuses is reported at startup, and the run goes on."""
    directory = tmp_path / "log"
    directory.mkdir()

    def refuse(_self: object) -> Path:
        msg = "permission denied"
        raise OSError(msg)

    log = ConversationLog(directory)
    with patch.object(ConversationLog, "_file_for", refuse):
        log.prepare()
    client = make_client("m")
    client.observe(log)
    client.remember(Message(role=Role.USER, content="x"))  # must not raise

    assert log.disabled
    assert "permission denied" in capsys.readouterr().out


def test_a_write_that_fails_mid_run_disables_the_log_without_failing_the_turn(tmp_path, capsys) -> None:
    """A log that breaks halfway -- as on Windows when a file is held open -- costs the log only."""
    directory = tmp_path / "log"
    log = ConversationLog(directory)
    log.prepare()
    client = make_client("m")
    client.observe(log)
    client.remember(Message(role=Role.USER, content="kept"))

    log._handle = DeadHandle(OSError("the process cannot access the file"))  # type: ignore[assignment]
    client.remember(Message(role=Role.ASSISTANT, content="second"))  # must not raise

    assert log.disabled
    assert "log" in capsys.readouterr().out.lower()

    client.remember(Message(role=Role.ASSISTANT, content="third"))  # and it stays quiet afterwards
    assert capsys.readouterr().out == ""
    assert [m["content"] for m in messages(one_file(directory))] == ["kept"]  # still the last good write


def test_a_failed_rollback_disables_the_log_without_failing_the_turn(tmp_path, capsys) -> None:
    """A file that cannot be truncated is reported, and the session carries on."""
    directory = tmp_path / "log"
    log = ConversationLog(directory)
    log.prepare()
    client = make_client("m")
    client.observe(log)
    client.remember(Message(role=Role.USER, content="kept"))
    client.remember(Message(role=Role.ASSISTANT, content="dropped"))

    log._handle = DeadHandle(OSError("read-only file system"))  # type: ignore[assignment]
    client.rollback_last_turn()  # must not raise

    assert log.disabled
    assert "read-only file system" in capsys.readouterr().out


# ── The path ───────────────────────────────────────────────────────


def test_a_log_path_that_cannot_be_resolved_is_reported_not_raised(capsys, monkeypatch) -> None:
    """A ``~`` that cannot be expanded -- no home directory -- is not a crash."""
    from llm_cli_py import main as main_module

    def unresolvable(_value: object) -> None:
        msg = "Could not determine home directory."
        raise RuntimeError(msg)

    monkeypatch.setenv(LOG_ENV, "~/logs/session")
    monkeypatch.setattr("llm_cli_py.session.transcript.clean_path", unresolvable)

    assert main_module._conversation_log_or_none() is None
    assert "log" in capsys.readouterr().out


def test_the_log_path_is_read_the_way_a_shell_user_would_write_it(tmp_path, monkeypatch) -> None:
    """``~`` is expanded to the home directory; surrounding whitespace is dropped."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # expanduser on Windows

    assert ConversationLog("  ~/runs/log  ").directory == tmp_path / "runs" / "log"


def test_a_path_with_spaces_is_used_as_it_is(tmp_path) -> None:
    """A directory name with a space needs no escaping or quoting."""
    directory = tmp_path / "My Logs" / "run 1"

    write_turn(directory, "hi").close()

    assert messages(one_file(directory))[0]["content"] == "hi"


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quotes_a_windows_set_left_in_the_value_are_dropped(tmp_path, quote: str) -> None:
    r"""``set LLM_CLI_LOG_DIR="C:\My Logs\runs"`` must not create a quoted name.

    cmd.exe keeps the quotes in the value, unlike a POSIX shell or PowerShell,
    so a path with a space arrives wrapped. Dropping one surrounding pair is what
    makes such a path usable at all.
    """
    log = ConversationLog(f"{quote}{tmp_path / 'My Logs' / 'runs'}{quote}")

    log.prepare()
    log.close()

    assert log.directory == tmp_path / "My Logs" / "runs"


def test_the_log_path_is_resolved_before_anything_can_change_directory(tmp_path, monkeypatch) -> None:
    """A tool that changes directory cannot move where the log is written."""
    log = ConversationLog(tmp_path / "log")
    monkeypatch.chdir(tmp_path.parent)

    assert log.directory.is_absolute()
    assert log.directory == tmp_path / "log"


# ── Durations ──────────────────────────────────────────────────────


def test_each_measured_message_carries_its_duration(tmp_path: Path) -> None:
    """The log says how long the model took and how long the tool took.

    Durations are measured around the request and around the tool run, so they
    are the same numbers the status line showed, and they survive into the file
    the run leaves behind.
    """
    session, directory = logged_session(tmp_path)
    register(session.ctx.tool_registry, "calc", lambda **_: ExecResult(stdout="42"))
    turns = [
        completion(tool_calls=[tool_call("c1", "calc", '{"code": "40+2"}')]),
        completion("It is 42"),
    ]

    with patch(SEND, side_effect=turns):
        session.process_and_print("compute this")

    conversation = messages(one_file(directory))
    durations = {m["role"]: m.get("duration_ms") for m in conversation}
    assert durations["user"] is None  # nothing was measured to produce it
    assert isinstance(durations["assistant"], float)
    assert isinstance(durations["tool"], float)


def test_a_request_that_fails_leaves_no_half_measured_message(tmp_path: Path) -> None:
    """A failed request records nothing at all -- there is no reply to time."""
    session, directory = logged_session(tmp_path)

    with patch(SEND, side_effect=RuntimeError("boom")):
        session.process_and_print("hello")

    assert [m["role"] for m in messages(one_file(directory))] == ["user"]
    assert all("duration_ms" not in m for m in messages(one_file(directory)))


def test_a_failed_tool_still_records_how_long_it_ran(tmp_path: Path) -> None:
    """A tool that raised spent real time before it did; the log keeps that number."""
    session, directory = logged_session(tmp_path)
    tool = tool_call("c1", "flaky", "{}")

    def explode(**_kwargs: object) -> ExecResult:
        msg = "the API said no"
        raise RuntimeError(msg)

    register(session.ctx.tool_registry, "flaky", explode)
    # Two scripted replies: the tool call, then the answer. Returning the tool call
    # from every request would ask for it again for ever -- the agent loop keeps
    # going as long as the model asks for tools, and a mock that always asks hangs.
    turns = [completion(tool_calls=[tool]), completion("never mind")]
    with patch(SEND, side_effect=turns):
        session.process_and_print("run it")

    tool_record = next(m for m in messages(one_file(directory)) if m["role"] == "tool")
    assert isinstance(tool_record["duration_ms"], float)
    assert "the API said no" in tool_record["content"]


def test_the_duration_is_recorded_but_never_sent_to_the_model(tmp_path: Path) -> None:
    """The request body is built field by field, so a log-only field cannot leak."""

    class Spy:
        """Stand-in for ``post_json``: records the body, answers with a reply."""

        def __init__(self) -> None:
            self.body: dict = {}
            self.api_key = ""

        def __call__(self, _url: str, body: dict, *, api_key: str = "") -> dict:
            self.body = body
            self.api_key = api_key
            return completion("first").copy()

    spy = Spy()
    session, _directory = logged_session(tmp_path)

    with patch(SEND, side_effect=spy):
        session.process_and_print("hello")
        session.process_and_print("again")

    for message in spy.body["messages"]:
        assert "duration_ms" not in message


def test_the_request_duration_comes_from_the_client_s_own_timer() -> None:
    """A scripted timer makes the recorded duration exact instead of incidental."""
    from llm_cli_py.providers.llm_api import LlmApiClient
    from llm_cli_py.utils.clock import Timer

    readings = iter([100.0, 101.5])
    client = LlmApiClient(
        model="m",
        api_url="https://api.example.invalid/v1",
        timer_factory=lambda: Timer(lambda: next(readings, 101.5)),
    )

    with patch(SEND, return_value=completion("hi")):
        client.send("hello", [])

    assert client.state.conversation[-1].duration_ms == 1500.0


# ── The same conversation as Markdown (what ``/raw`` shows) ────────


def conversation(*messages: Message, model: str = "gpt-4o") -> ClientState:
    """A state holding ``messages``, as the session would have recorded them."""
    return ClientState(model=model, conversation=list(messages))


def tool_call_payload(call_id: str, name: str, arguments: str) -> dict:
    """One entry of a recorded assistant message's ``tool_calls``."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_the_markdown_holds_every_message_under_its_own_heading() -> None:
    """The document reads as a transcript: one section per message, in order."""
    state = conversation(
        Message(role=Role.USER, content="hello"),
        Message(role=Role.ASSISTANT, content="hi"),
    )

    assert render_markdown(state) == ("# Conversation (gpt-4o)\n\n## user\n\nhello\n\n## assistant\n\nhi\n")


def test_a_model_s_own_markdown_is_reproduced_not_re_rendered() -> None:
    """The text is what the model wrote: bullets, emphasis and fences untouched."""
    answer = "# Title\n\n- one\n- two\n\n**bold** and `code`\n"

    document = render_markdown(conversation(Message(role=Role.ASSISTANT, content=answer)))

    assert answer.rstrip("\n") in document
    assert "\u2022" not in document  # no rendered bullet
    assert "\x1b[" not in document  # and no escape sequences


def test_a_tool_call_and_its_result_are_linked_by_the_call_id() -> None:
    """A call carries the id; the result's heading names the call it answers."""
    state = conversation(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[tool_call_payload("c1", "execute_python", '{"code": "print(1)"}')],
        ),
        Message(role=Role.TOOL, content='{"stdout": "1\\n", "exit_code": 0}', tool_call_id="c1"),
    )

    document = render_markdown(state)

    assert "**Tool call:** `execute_python` (`c1`)" in document
    assert "## tool (c1)" in document
    # The arguments and the result are JSON, and they are laid out as such.
    assert '```json\n{\n  "code": "print(1)"\n}\n```' in document
    assert '"stdout": "1\\n"' in document


def test_a_message_with_no_text_does_not_leave_blank_sections() -> None:
    """An assistant turn that only asked for a tool has no body, and no gap either."""
    state = conversation(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[tool_call_payload("c1", "calc", "{}")],
        ),
        Message(role=Role.TOOL, content="", tool_call_id="c1"),
    )

    document = render_markdown(state)

    assert "\n\n\n" not in document
    assert document.endswith("## tool (c1)\n")


def test_a_fence_in_the_text_cannot_close_the_block_it_is_in() -> None:
    """Tool output that prints a fenced block gets a longer fence around it.

    Otherwise the text's own ```` ``` ```` would end the block, and the rest of
    the conversation -- the following sections included -- would be read as
    prose that had escaped the code.
    """
    state = conversation(
        Message(role=Role.TOOL, content="```\nprint('hi')\n```", tool_call_id="c1"),
    )

    document = render_markdown(state)

    assert document.rstrip("\n").endswith("````")  # a four-backtick fence
    assert "````\n```\nprint('hi')\n```\n````" in document


def test_arguments_that_do_not_parse_are_shown_as_they_arrived() -> None:
    """A truncated tool call is not executed -- but ``/raw`` still says what came.

    The arguments are fenced, not indented as JSON: what the model sent is the
    only thing there is, and re-writing it would be inventing a call.
    """
    state = conversation(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[tool_call_payload("c1", "execute_python", '{"code": "pri')],
        ),
    )

    document = render_markdown(state)

    assert '```\n{"code": "pri\n```' in document


def test_text_that_is_not_json_is_fenced_without_claiming_to_be_json() -> None:
    """An unknown tool's result is a sentence, not a payload."""
    state = conversation(Message(role=Role.TOOL, content="Tool 'ghost' not found", tool_call_id="c1"))

    document = render_markdown(state)

    assert "```\nTool 'ghost' not found\n```" in document
    assert "```json" not in document


def test_the_document_names_the_model_it_came_from() -> None:
    """Pasted into a file or another chat, the transcript still says who wrote it."""
    assert render_markdown(conversation(model="qwen3")).startswith("# Conversation (qwen3)\n")
    assert render_markdown(conversation(model="")).startswith("# Conversation\n")
