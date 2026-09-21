"""Tests for the OpenAI-compatible API client: request building and retries."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_cli_py.models import DataSource, LlmResponse, Message, Role, ToolSchema
from tests.conftest import (
    TEST_URL,
    make_client,
    stream_response,
    text_stream,
    tool_call_chunk,
)

# ── Request building ──────────────────────────────────────────────


def test_system_prompt_is_seeded_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "You are a test assistant.")
    client = make_client("gpt-4o")
    client._state.conversation.append(Message(role=Role.USER, content="Hello"))

    messages = client._build_messages()
    assert messages[0] == {"role": "system", "content": "You are a test assistant."}
    assert messages[1] == {"role": "user", "content": "Hello"}


def test_system_prompt_is_read_once_at_init(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "Startup prompt.")
    client = make_client("gpt-4o")
    monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "Changed after init.")
    assert client._build_messages()[0]["content"] == "Startup prompt."


@pytest.mark.parametrize("value", [None, ""])
def test_no_system_message_when_unset_or_empty(monkeypatch, value) -> None:
    monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
    if value is not None:
        monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", value)
    client = make_client("gpt-4o")
    client._state.conversation = [Message(role=Role.USER, content="Hello")]
    assert [m["role"] for m in client._build_messages()] == ["user"]


def test_timestamps_are_recorded_but_never_sent(monkeypatch) -> None:
    """Timestamps are local metadata for the log/dump, not part of the request."""
    monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "sys")
    client = make_client("gpt-4o")
    client._append_user_messages([DataSource(text="Hello")])
    client._record_assistant(LlmResponse(text="Hi there"))
    client._state.conversation.append(Message(role=Role.TOOL, content="42", tool_call_id="call_1"))

    system, user, assistant = client._state.conversation[:3]
    assert all(m.timestamp for m in (system, user, assistant))
    assert datetime.fromisoformat(user.timestamp)
    messages = client._build_messages()
    assert messages and json.dumps(messages, ensure_ascii=False).count("timestamp") == 0


def test_tool_result_and_tool_call_entries_have_openai_shape() -> None:
    client = make_client("gpt-4o")
    client._state.conversation = [
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "python"}}],
        ),
        Message(role=Role.TOOL, content="42", tool_call_id="call_1"),
    ]
    assert client._build_messages() == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "python"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ]


def test_build_request_advertises_tools_and_streaming() -> None:
    client = make_client("gpt-4o")
    schema = ToolSchema(name="python", description="Run Python", parameters={"type": "object"})
    body = client._build_request([{"role": "user", "content": "hi"}], [schema])
    assert body["model"] == "gpt-4o"
    assert body["stream"] is True
    assert body["tools"][0]["function"]["name"] == "python"


def test_source_types_are_labelled_in_the_user_turn() -> None:
    client = make_client()
    client._append_user_messages(
        [
            DataSource(text="plain"),
            DataSource(text="body", source_type="file"),
            DataSource(text="page", source_type="url"),
        ]
    )
    content = client._state.conversation[-1].content
    assert "[File content]:\nbody" in content
    assert "[URL content]:\npage" in content


def test_send_posts_to_chat_completions(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
    client = make_client("gpt-4o")
    with patch(
        "llm_cli_py.utils.http.requests.Session.post", return_value=stream_response(text_stream("Answer"))
    ) as post:
        result = client.send([DataSource(text="Question")], [])

    assert result.text == "Answer"
    args, kwargs = post.call_args
    assert args[0] == f"{TEST_URL}/chat/completions"
    assert kwargs["json"]["stream"] is True
    assert client._state.conversation[0].role == Role.USER


def test_replayed_tool_call_arguments_are_json_encoded() -> None:
    client = make_client()
    resp = stream_response([tool_call_chunk('{"code": "print(1)"}'), "data: [DONE]"])
    with patch("llm_cli_py.utils.http.requests.Session.post", return_value=resp):
        client.send([DataSource(text="run it")], [])

    function = client._state.conversation[-1].tool_calls[0]["function"]
    assert function["arguments"] == '{"code": "print(1)"}'


# ── Retries ───────────────────────────────────────────────────────


def test_retries_on_rate_limit_then_succeeds() -> None:
    client = make_client("gpt-4o")
    rate_limited = MagicMock(status_code=429)
    rate_limited.text = "rate limit exceeded"

    with (
        patch(
            "llm_cli_py.utils.http.requests.Session.post",
            side_effect=[rate_limited, stream_response(text_stream("OK"))],
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep") as sleep,
    ):
        assert client.send([DataSource(text="Hi")], []).text == "OK"

    assert post.call_count == 2 and sleep.call_count == 1


def test_timeout_is_retried_then_raised() -> None:
    client = make_client("gpt-4o")
    with (
        patch(
            "llm_cli_py.utils.http.requests.Session.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep"),
        pytest.raises(requests.exceptions.Timeout),
    ):
        client.send([DataSource(text="Hi")], [])

    assert post.call_count == 3


def test_context_manager_and_close_are_idempotent() -> None:
    with make_client("gpt-4o") as client:
        assert client.api_url == TEST_URL
    client.close()
    client.close()  # must not raise
