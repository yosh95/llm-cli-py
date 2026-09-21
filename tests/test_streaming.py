"""Tests for SSE parsing and the streaming send() path."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_cli_py.models import DataSource
from tests.conftest import make_client, sse_chunk, stream_response, text_stream, tool_call_chunk


def test_text_stream_is_accumulated_and_streamed_live() -> None:
    client = make_client()
    chunks = [_sse for _sse in text_stream("Hello ", "world")]
    deltas: list[str] = []
    result = client._parse_stream_response(stream_response(chunks), on_text=deltas.append)

    assert result.text == "Hello world"
    assert deltas == ["Hello ", "world"]
    assert result.tool_calls == []


def test_tool_call_arguments_are_buffered_across_chunks() -> None:
    client = make_client()
    resp = stream_response(
        [
            tool_call_chunk('{"code": "pri'),
            tool_call_chunk('nt(1)"}'),
            sse_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ]
    )
    result = client._parse_stream_response(resp)
    assert [(tc.id, tc.name, tc.arguments, tc.parse_error) for tc in result.tool_calls] == [
        ("call_1", "python", {"code": "print(1)"}, None)
    ]


def test_truncated_tool_call_sets_parse_error_and_empty_arguments() -> None:
    client = make_client()
    result = client._parse_stream_response(
        stream_response([tool_call_chunk('{"code": "pri'), "data: [DONE]"])
    )
    (tc,) = result.tool_calls
    assert tc.arguments == {}
    assert tc.parse_error == '{"code": "pri'


def test_send_requests_a_stream_and_records_the_answer(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
    client = make_client()
    with patch(
        "llm_cli_py.providers.llm_api.post_with_retries", return_value=stream_response(text_stream("Hi"))
    ) as post:
        result = client.send([DataSource(text="Hello")], [])

    assert post.call_args[0][2]["stream"] is True
    assert result.text == "Hi"
    assert [m.role.value for m in client.state.conversation] == ["user", "assistant"]


def test_send_notifies_state_change_once_per_message(monkeypatch) -> None:
    """The client signals history changes so the log can be flushed per message."""
    monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
    client = make_client()
    counts: list[int] = []
    client.state.on_change = lambda: counts.append(len(client.state.conversation))

    with patch(
        "llm_cli_py.providers.llm_api.post_with_retries", return_value=stream_response(text_stream("Hi"))
    ):
        client.send([DataSource(text="Hello")], [])

    assert counts == [1, 2]  # user turn (before the request), then the answer


@pytest.mark.parametrize(
    ("arguments", "expect_raw"),
    [
        ('{"code": "print(1)"}', False),
        ('{"code": "pri', True),
    ],
)
def test_send_surfaces_tool_calls_without_a_second_request(arguments: str, expect_raw: bool) -> None:
    """Exactly one request is made; broken calls are returned, never retried non-streaming."""
    client = make_client()
    resp = stream_response(
        [tool_call_chunk(arguments), sse_chunk({}, finish_reason="tool_calls"), "data: [DONE]"]
    )

    with patch("llm_cli_py.providers.llm_api.post_with_retries", return_value=resp) as post:
        result = client.send([DataSource(text="Run it")], [])

    assert post.call_count == 1
    (tc,) = result.tool_calls
    assert (tc.parse_error is not None) is expect_raw
