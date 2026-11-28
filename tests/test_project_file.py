"""Tests for ``LLM_CLI.md``: finding it, reading it, and sending it as system.

Two things are checked here, in that order: which file is found from a directory
(the search is what makes the file a default rather than a flag), and how what it
holds reaches the model (as the conversation's one system message, ahead of every
turn, never as a turn of its own).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from llm_cli_py import project_file
from llm_cli_py.models import Role
from llm_cli_py.project_file import FILENAME, MAX_CHARS, find, load
from tests.conftest import completion, make_client

SEND = "llm_cli_py.providers.llm_api.post_json"


def write(directory: Path, text: str = "Run make check.\n") -> Path:
    """Write ``text`` as the project instructions in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    path.write_text(text, encoding="utf-8")
    return path


# ── Finding the file ──────────────────────────────────────────────


def test_finds_the_file_in_the_current_directory(tmp_path) -> None:
    path = write(tmp_path)
    assert find(tmp_path) == path


def test_finds_the_file_in_a_parent_directory(tmp_path) -> None:
    """Running in a subdirectory of a project still finds that project's file."""
    path = write(tmp_path)
    deep = tmp_path / "src" / "llm_cli_py"
    deep.mkdir(parents=True)
    assert find(deep) == path


def test_the_nearest_file_wins(tmp_path) -> None:
    """A closer file is the project you are in; the one above is not consulted."""
    write(tmp_path, "from the top\n")
    nearer = write(tmp_path / "sub", "from the subdirectory\n")
    assert find(tmp_path / "sub") == nearer


def test_no_file_anywhere_is_none(tmp_path) -> None:
    assert find(tmp_path) is None


def test_a_directory_named_like_the_file_is_not_the_file(tmp_path) -> None:
    (tmp_path / FILENAME).mkdir()
    assert find(tmp_path) is None


# ── Loading it ────────────────────────────────────────────────────


def test_load_returns_the_text_and_where_it_came_from(tmp_path) -> None:
    path = write(tmp_path, "  Be brief.\n\n")
    loaded = load(tmp_path)

    assert loaded is not None
    assert loaded.path == path
    assert loaded.text == "Be brief.\n"  # surrounding blanks dropped, the text kept


def test_load_without_a_file_is_none(tmp_path) -> None:
    assert load(tmp_path) is None


def test_an_empty_file_states_nothing_so_it_loads_as_none(tmp_path) -> None:
    """A file that is only whitespace adds no system message to the conversation."""
    write(tmp_path, "\n  \n")
    assert load(tmp_path) is None


def test_a_file_too_large_to_send_is_cut_at_a_line_boundary(tmp_path, capsys) -> None:
    """Past the cap the file is truncated before it is sent, and that is reported."""
    line = "x" * 99 + "\n"
    write(tmp_path, line * (MAX_CHARS // 100 + 10))

    loaded = load(tmp_path)

    assert loaded is not None
    assert len(loaded.text) <= MAX_CHARS
    assert loaded.text.endswith("\n")  # cut between lines, not through one
    assert "longer than" in capsys.readouterr().out


def test_an_unreadable_file_is_reported_and_ignored(tmp_path, capsys) -> None:
    """Losing the instructions must not lose the session."""
    path = tmp_path / FILENAME
    path.write_text("instructions\n", encoding="utf-8")

    with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
        assert load(tmp_path) is None

    assert "denied" in capsys.readouterr().out


def test_load_defaults_to_the_current_directory(tmp_path, monkeypatch) -> None:
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    loaded = load()

    assert loaded is not None
    assert loaded.path == tmp_path / FILENAME


# ── Sending it: the system message ────────────────────────────────


def test_instructions_are_sent_as_the_system_message_ahead_of_every_turn() -> None:
    client = make_client("gpt-4o", instructions="Run make check.\n")

    with patch(SEND, return_value=completion("ok")) as post:
        client.send("hello", [])

    assert post.call_args.args[1]["messages"] == [
        {"role": "system", "content": "Run make check.\n"},
        {"role": "user", "content": "hello"},
    ]
    assert client.state.conversation[0].role is Role.SYSTEM


def test_the_system_message_is_sent_again_with_each_later_turn() -> None:
    """It is part of the conversation, so it is not just the first request's."""
    client = make_client("gpt-4o", instructions="Be brief.\n")

    with patch(SEND, return_value=completion("one")) as post:
        client.send("hello", [])
        client.send("again", [])

    messages = post.call_args.args[1]["messages"]
    assert messages[0] == {"role": "system", "content": "Be brief.\n"}
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_an_empty_instruction_sends_no_system_message_at_all() -> None:
    client = make_client("gpt-4o", instructions="")
    assert client.state.conversation == []


def test_a_discarded_turn_stops_at_the_system_message() -> None:
    """Rolling a broken turn back never reaches the instructions at the head."""
    from llm_cli_py.models import Message

    client = make_client("gpt-4o", instructions="Be brief.\n")
    client.remember(Message(role=Role.USER, content="hello"))
    client.remember(Message(role=Role.ASSISTANT, content="hi"))

    client.rollback_last_turn()

    assert [(m.role, m.content) for m in client.state.conversation] == [
        (Role.SYSTEM, "Be brief.\n"),
        (Role.USER, "hello"),
    ]


def test_a_rollback_with_nothing_but_the_instructions_keeps_them() -> None:
    """Nothing to undo is a no-op -- and the head of the conversation stays."""
    client = make_client("gpt-4o", instructions="Be brief.\n")

    client.rollback_last_turn()

    assert [(m.role, m.content) for m in client.state.conversation] == [(Role.SYSTEM, "Be brief.\n")]


def test_the_instructions_appear_in_raw_as_the_first_section() -> None:
    """/raw shows what the model was sent, the system message included."""
    from llm_cli_py.models import Message
    from llm_cli_py.session.transcript import render_markdown

    client = make_client("gpt-4o", instructions="Be brief.\n")
    client.remember(Message(role=Role.USER, content="hello"))

    markdown = render_markdown(client.state)

    assert "## system\n\nBe brief." in markdown
    assert markdown.index("## system") < markdown.index("## user")


def test_the_instructions_are_the_first_record_in_the_log(tmp_path) -> None:
    """The system message is the run's opening record, not a preamble kept apart."""
    from llm_cli_py.models import Message
    from llm_cli_py.session.transcript import ConversationLog

    log = ConversationLog(tmp_path / "log")
    log.prepare()
    client = make_client("gpt-4o", instructions="Be brief.\n")
    client.observe(log)
    client.remember(Message(role=Role.USER, content="hello"))
    log.close()

    path = log.path
    assert path is not None
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    logged = [{k: v for k, v in record.items() if k != "type"} for record in records[2:]]
    assert logged == [
        {"role": "system", "content": "Be brief.\n"},
        {"role": "user", "content": "hello"},
    ]


def test_the_instructions_are_not_counted_as_a_measured_duration() -> None:
    """Nothing was measured to produce the instructions, so their record has no time."""
    from llm_cli_py.models import Message
    from llm_cli_py.session.transcript import message_record

    client = make_client("gpt-4o", instructions="Be brief.\n")
    client.remember(Message(role=Role.ASSISTANT, content="hi", duration_ms=10.0))

    system, assistant = client.state.conversation
    assert "duration_ms" not in message_record(system)
    assert message_record(assistant)["duration_ms"] == 10.0


def test_the_module_reads_the_file_named_by_the_constant() -> None:
    assert project_file.FILENAME == "LLM_CLI.md"
