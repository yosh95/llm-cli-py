"""Tests for the OpenAI-compatible API client (request body and response parsing)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_cli_py.models import Message, Role, ToolSchema
from llm_cli_py.utils.http import HttpError
from tests.conftest import make_client


def _completion(text: str | None = "Answer", tool_calls: list[dict] | None = None) -> dict:
    message: dict[str, object] = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def _tool_call(arguments: str = '{"code": "print(1)"}') -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "execute_python", "arguments": arguments},
    }


def test_request_targets_chat_completions_with_the_prompt_and_tools() -> None:
    client = make_client("gpt-4o")
    schema = ToolSchema(name="execute_python", description="Run Python", parameters={"type": "object"})

    with patch("llm_cli_py.providers.llm_api.post_json", return_value=_completion()) as post:
        result = client.send("hello", [schema])

    url, body = post.call_args.args[0], post.call_args.args[1]
    assert url == "https://api.example.invalid/v1/chat/completions"
    assert body["model"] == "gpt-4o"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["tools"][0]["function"]["name"] == "execute_python"
    assert "stream" not in body
    assert post.call_args.kwargs["api_key"] == "k"
    assert result.text == "Answer"


def test_the_system_prompt_is_seeded_as_the_first_message() -> None:
    client = make_client(system_prompt="You are a test assistant.")
    with patch("llm_cli_py.providers.llm_api.post_json", return_value=_completion()) as post:
        client.send("hello", [])

    assert post.call_args.args[1]["messages"][0] == {
        "role": "system",
        "content": "You are a test assistant.",
    }


def test_the_answer_is_recorded_in_the_conversation() -> None:
    client = make_client()
    with patch("llm_cli_py.providers.llm_api.post_json", return_value=_completion("Hi there")):
        client.send("hello", [])

    assert [(m.role.value, m.content) for m in client._state.conversation] == [
        ("user", "hello"),
        ("assistant", "Hi there"),
    ]


def test_tool_calls_are_parsed_and_replayed_as_json_strings() -> None:
    client = make_client()
    with patch("llm_cli_py.providers.llm_api.post_json", return_value=_completion(None, [_tool_call()])):
        result = client.send("run it", [])

    (tc,) = result.tool_calls
    assert (tc.id, tc.name, tc.arguments, tc.parse_error) == (
        "call_1",
        "execute_python",
        {"code": "print(1)"},
        None,
    )
    calls = client._state.conversation[-1].tool_calls
    assert calls is not None and calls[0]["function"]["arguments"] == '{"code": "print(1)"}'


def test_unparseable_tool_arguments_are_reported_as_a_parse_error() -> None:
    client = make_client()
    with patch(
        "llm_cli_py.providers.llm_api.post_json",
        return_value=_completion(None, [_tool_call('{"code": "pri')]),
    ):
        result = client.send("run it", [])

    (tc,) = result.tool_calls
    assert tc.arguments == {} and tc.parse_error == '{"code": "pri'


def test_a_failing_request_propagates_the_http_error() -> None:
    client = make_client()
    with (
        patch("llm_cli_py.providers.llm_api.post_json", side_effect=HttpError("401 Client Error")),
        pytest.raises(HttpError),
    ):
        client.send("hello", [])


def test_tool_result_messages_use_the_openai_shape() -> None:
    client = make_client()
    client._state.conversation = [Message(role=Role.TOOL, content="42", tool_call_id="call_1")]
    assert client._build_messages() == [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]
