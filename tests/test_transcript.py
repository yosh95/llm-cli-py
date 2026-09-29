"""Tests for the optional conversation log (``LLM_CLI_LOG_FILE``).

These drive the real ``send`` path (only the HTTP transport is patched), because
what is being tested is that the conversation the client records -- and nothing
else -- is what lands in the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_cli_py.session.session import ActiveSession, SessionContext
from llm_cli_py.session.transcript import ConversationLog, message_record, render
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
