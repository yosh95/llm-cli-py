"""Tests for the conversation's two renderings: the log, and ``/raw``'s Markdown.

Most of these drive the real ``send`` path (only the HTTP transport is patched),
because what is being tested is that the conversation the client records -- and
nothing else -- is what lands in the file. The Markdown tests at the end write
the conversation by hand: what is being tested there is the layout, and the
shapes it has to survive (tool calls, tool results, fences inside the text).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_cli_py.models import ClientState, Message, Role
from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.session.transcript import ConversationLog, message_record, render, render_markdown
from llm_cli_py.tools.registry import ToolRegistry
from llm_cli_py.tools.types import ExecResult
from tests.conftest import completion, make_client, register, tool_call

LOG_ENV = "LLM_CLI_LOG_FILE"
SEND = "llm_cli_py.providers.llm_api.post_json"


def read_log(path: Path) -> dict:
    """Read a log file the CLI wrote."""
    return json.loads(path.read_text(encoding="utf-8"))


def logged_session(tmp_path: Path) -> tuple[ActiveSession, Path]:
    """A session against a fake transport, watched by a log in ``tmp_path``."""
    log_path = tmp_path / "log.json"
    client = make_client("gpt-4o", system_prompt="be helpful")
    client.observe(ConversationLog(log_path))
    return ActiveSession(client, SessionContext(tool_registry=ToolRegistry())), log_path


def test_without_the_env_var_no_log_is_written(monkeypatch) -> None:
    """Unset is the default, and means nothing is written anywhere."""
    from llm_cli_py import main as main_module

    monkeypatch.delenv(LOG_ENV, raising=False)

    assert main_module._conversation_log_or_none() is None


def test_the_log_holds_the_whole_conversation(tmp_path) -> None:
    """User turns, assistant replies, tool calls and tool results all appear."""
    session, log_path = logged_session(tmp_path)
    register(session.ctx.tool_registry, "calc", lambda **_: ExecResult(stdout="42"))
    turns = [
        completion(tool_calls=[tool_call("c1", "calc", '{"code": "40+2"}')]),
        completion("It is 42"),
    ]

    with patch(SEND, side_effect=turns):
        session.process_and_print("compute this")

    document = read_log(log_path)
    assert document["model"] == "gpt-4o"
    assert [m["role"] for m in document["conversation"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert document["conversation"][0]["content"] == "be helpful"
    assert document["conversation"][1]["content"] == "compute this"

    assistant = document["conversation"][2]
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["tool_calls"][0]["function"]["name"] == "calc"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"code": "40+2"}

    tool = document["conversation"][3]
    assert tool["tool_call_id"] == "c1"
    assert json.loads(tool["content"])["stdout"] == "42"
    assert document["conversation"][4]["content"] == "It is 42"


def test_the_log_is_written_as_the_run_goes_not_only_at_the_end(tmp_path) -> None:
    """A run that dies mid-turn still leaves what had been recorded.

    The tool looks at the log while it runs -- that is, between the assistant's
    tool call and its result -- and then dies, the shape of a run that ends badly
    halfway through a turn.
    """
    session, log_path = logged_session(tmp_path)

    def die(**_kwargs: object) -> ExecResult:
        assert [m["role"] for m in read_log(log_path)["conversation"]] == [
            "system",
            "user",
            "assistant",
        ]
        raise KeyboardInterrupt

    register(session.ctx.tool_registry, "slow", die)
    turn = completion(tool_calls=[tool_call("c1", "slow", "{}")])

    with patch(SEND, return_value=turn), pytest.raises(KeyboardInterrupt):
        session.process_and_print("run it")

    # The interrupted turn is rolled back, and the file follows: the prompt
    # stays, the reply nobody could finish is gone.
    assert [m["role"] for m in read_log(log_path)["conversation"]] == ["system", "user"]


def test_a_discarded_turn_disappears_from_the_log(tmp_path) -> None:
    """The log shows what the model will be sent, not what it tried once."""
    session, log_path = logged_session(tmp_path)
    turn = completion(tool_calls=[tool_call("c1", "execute_python", '{"code": "pri')])

    with patch(SEND, return_value=turn):
        session.process_and_print("run it")

    assert [m["role"] for m in read_log(log_path)["conversation"]] == ["system", "user"]


def test_a_message_record_omits_the_fields_that_do_not_apply() -> None:
    """A plain turn carries no ``tool_call_id``/``tool_calls`` keys at all."""
    from llm_cli_py.models import Message, Role

    assert message_record(Message(role=Role.USER, content="hi")) == {"role": "user", "content": "hi"}


def test_a_log_in_a_missing_directory_is_created(tmp_path) -> None:
    """The path is used as given; its directories are made for it."""
    log = ConversationLog(tmp_path / "runs" / "2026" / "log.json")

    log.save(make_client("m").state)

    assert read_log(tmp_path / "runs" / "2026" / "log.json")["model"] == "m"


def test_a_failed_write_leaves_the_previous_log_intact(tmp_path, monkeypatch, capsys) -> None:
    """The log on disk is always a whole document, never a half-written one."""
    log = ConversationLog(tmp_path / "log.json")
    log.save(make_client("first").state)

    def explode(_state: object) -> str:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("llm_cli_py.session.transcript.render", explode)
    log.save(make_client("second").state)

    assert read_log(tmp_path / "log.json")["model"] == "first"  # still readable
    assert [p.name for p in tmp_path.iterdir()] == ["log.json"]  # no debris left
    assert "disk full" in capsys.readouterr().out


def test_render_is_json_with_the_conversation_and_its_version() -> None:
    """The document is self-describing: a version, the model, the messages."""
    client = make_client("m", system_prompt="be brief")

    document = json.loads(render(client.state))

    assert document["version"] >= 1
    assert document["conversation"] == [{"role": "system", "content": "be brief"}]


def test_a_path_that_is_not_a_regular_file_is_written_to_directly() -> None:
    """``/dev/null`` is a sensible way to switch the log off; it must not fail."""
    null_device = Path("/dev/null")
    if not null_device.exists():  # pragma: no cover - Windows
        pytest.skip("/dev/null is a POSIX device")

    ConversationLog(null_device).save(make_client("m").state)  # must not raise


def test_a_write_that_fails_mid_run_disables_the_log_without_failing_the_turn(
    tmp_path, monkeypatch, capsys
) -> None:
    """A log that breaks halfway -- as on Windows when a file is held open -- costs the log only."""
    log = ConversationLog(tmp_path / "log.json")
    log.save(make_client("first").state)

    def explode(_state: object) -> str:
        msg = "the process cannot access the file"
        raise OSError(msg)

    monkeypatch.setattr("llm_cli_py.session.transcript.render", explode)
    log.save(make_client("second").state)  # must not raise

    assert log.disabled
    assert "log" in capsys.readouterr().out.lower()

    log.save(make_client("third").state)  # and it stays quiet afterwards
    assert capsys.readouterr().out == ""
    assert read_log(tmp_path / "log.json")["model"] == "first"  # still the last good write


def test_a_log_path_that_cannot_be_resolved_is_reported_not_raised(capsys, monkeypatch) -> None:
    """A ``~`` that cannot be expanded -- no home directory -- is not a crash."""
    from llm_cli_py import main as main_module

    def unresolvable(_value: object) -> None:
        msg = "Could not determine home directory."
        raise RuntimeError(msg)

    monkeypatch.setenv(LOG_ENV, "~/logs/session.json")
    monkeypatch.setattr("llm_cli_py.session.transcript.clean_path", unresolvable)

    assert main_module._conversation_log_or_none() is None
    assert "log" in capsys.readouterr().out


def test_the_log_path_is_read_the_way_a_shell_user_would_write_it(tmp_path, monkeypatch) -> None:
    """``~`` is expanded to the home directory; surrounding whitespace is dropped."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # expanduser on Windows

    assert ConversationLog("  ~/runs/log.json  ").path == tmp_path / "runs" / "log.json"


def test_a_path_with_spaces_is_used_as_it_is(tmp_path) -> None:
    """A directory name with a space needs no escaping or quoting."""
    log = ConversationLog(tmp_path / "My Logs" / "session 1.json")

    log.save(make_client("m").state)

    assert read_log(tmp_path / "My Logs" / "session 1.json")["model"] == "m"


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quotes_a_windows_set_left_in_the_value_are_dropped(tmp_path, quote: str) -> None:
    r"""``set LLM_CLI_LOG_FILE="C:\My Logs\log.json"`` must not create a quoted name.

    cmd.exe keeps the quotes in the value, unlike a POSIX shell or PowerShell,
    so a path with a space arrives wrapped. Dropping one surrounding pair is what
    makes such a path usable at all.
    """
    log = ConversationLog(f"{quote}{tmp_path / 'My Logs' / 'log.json'}{quote}")

    log.save(make_client("m").state)

    assert log.path == tmp_path / "My Logs" / "log.json"
    assert read_log(tmp_path / "My Logs" / "log.json")["model"] == "m"


def test_the_log_path_is_resolved_before_anything_can_change_directory(tmp_path, monkeypatch) -> None:
    """A tool that changes directory cannot move where the log is written."""
    log = ConversationLog(tmp_path / "log.json")
    monkeypatch.chdir(tmp_path.parent)

    assert log.path.is_absolute()
    assert log.path == tmp_path / "log.json"


def test_each_measured_message_carries_its_duration(tmp_path: Path) -> None:
    """The log says how long the model took and how long the tool took.

    Durations are measured around the request and around the tool run, so they
    are the same numbers the status line showed, and they survive into the file
    the run leaves behind.
    """
    session, log_path = logged_session(tmp_path)
    register(session.ctx.tool_registry, "calc", lambda **_: ExecResult(stdout="42"))
    turns = [
        completion(tool_calls=[tool_call("c1", "calc", '{"code": "40+2"}')]),
        completion("It is 42"),
    ]

    with patch(SEND, side_effect=turns):
        session.process_and_print("compute this")

    conversation = read_log(log_path)["conversation"]
    durations = {m["role"]: m.get("duration_ms") for m in conversation}
    assert durations["user"] is None  # nothing was measured to produce it
    assert isinstance(durations["assistant"], float)
    assert isinstance(durations["tool"], float)

    totals = read_log(log_path)["durations_ms"]
    assert set(totals) == {"llm", "tool"}
    assert totals["tool"] == conversation[3]["duration_ms"]


def test_a_request_that_fails_leaves_no_half_measured_message(tmp_path: Path) -> None:
    """A failed request records nothing at all -- there is no reply to time."""
    session, log_path = logged_session(tmp_path)

    with patch(SEND, side_effect=RuntimeError("boom")):
        session.process_and_print("hello")

    assert [m["role"] for m in read_log(log_path)["conversation"]] == ["system", "user"]
    assert "durations_ms" not in read_log(log_path)


def test_a_failed_tool_still_records_how_long_it_ran(tmp_path: Path) -> None:
    """A tool that raised spent real time before it did; the log keeps that number."""
    session, log_path = logged_session(tmp_path)
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

    tool_record = next(m for m in read_log(log_path)["conversation"] if m["role"] == "tool")
    assert isinstance(tool_record["duration_ms"], float)
    assert "the API said no" in tool_record["content"]


def test_the_duration_is_recorded_but_never_sent_to_the_model(tmp_path: Path) -> None:
    """The request body is built field by field, so a log-only field cannot leak."""

    class Spy:
        """Stand-in for ``post_json``: records the body, answers with a reply."""

        def __init__(self) -> None:
            self.body: dict = {}
            self.api_key = ""

        def __call__(self, _url: str, body: dict, _timeout: int, *, api_key: str = "") -> dict:
            self.body = body
            self.api_key = api_key
            return completion("first").copy()

    spy = Spy()
    session, _log_path = logged_session(tmp_path)

    with patch(SEND, side_effect=spy):
        session.process_and_print("hello")
        session.process_and_print("again")

    for message in spy.body["messages"]:
        assert "duration_ms" not in message


def test_a_log_without_measured_messages_has_no_durations_section() -> None:
    """Version 1 logs stay readable, and an unmeasured log is written as before."""
    document = json.loads(render(make_client("m", system_prompt="be brief").state))

    assert document["version"] == 2
    assert "durations_ms" not in document
    assert document["conversation"] == [{"role": "system", "content": "be brief"}]


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
        Message(role=Role.SYSTEM, content="be brief"),
        Message(role=Role.USER, content="hello"),
        Message(role=Role.ASSISTANT, content="hi"),
    )

    assert render_markdown(state) == (
        "# Conversation (gpt-4o)\n\n## system\n\nbe brief\n\n## user\n\nhello\n\n## assistant\n\nhi\n"
    )


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
