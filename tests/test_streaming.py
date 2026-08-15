"""Tests for streaming chat completions (SSE parsing and send())."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from llm_cli_py.models import DataSource
from llm_cli_py.providers.llm_api import LlmApiClient


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> str:
    """Build a single SSE data line from a delta dict."""
    return json.dumps({"choices": [{"delta": delta, "finish_reason": finish_reason}]})


def _make_stream_response(chunks: list[str]) -> MagicMock:
    """Build a mock requests.Response that yields SSE data lines."""
    resp = MagicMock()
    resp.iter_lines.return_value = [c.encode("utf-8") for c in chunks]
    return resp


class TestParseStreamResponse:
    """Test the SSE chunk parser."""

    def test_text_stream(self) -> None:
        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        resp = _make_stream_response(
            [
                _chunk({"content": "Hello "}),
                _chunk({"content": "world"}),
                _chunk({}, finish_reason="stop"),
                "data: [DONE]",
            ]
        )
        text_parts: list[str] = []
        result = client._parse_stream_response(resp, on_text=text_parts.append)
        assert result.text == "Hello world"
        assert result.finish_reason == "stop"
        assert result.tool_calls == []
        assert text_parts == ["Hello ", "world"]

    def test_reasoning_stream_both_field_names(self) -> None:
        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        # reasoning_content (DeepSeek/Ollama) and reasoning (OpenRouter)
        resp = _make_stream_response(
            [
                _chunk({"reasoning_content": "Step 1"}),
                _chunk({"reasoning": "Step 2"}),
                _chunk({"content": "Answer"}),
                _chunk({}, finish_reason="stop"),
                "data: [DONE]",
            ]
        )
        reason_parts: list[str] = []
        result = client._parse_stream_response(resp, on_reasoning=reason_parts.append)
        assert result.reasoning == "Step 1\nStep 2"
        assert result.text == "Answer"
        assert reason_parts == ["Step 1", "Step 2"]

    def test_tool_call_arguments_buffered_and_parsed(self) -> None:
        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        # arguments split across chunks as JSON fragments
        resp = _make_stream_response(
            [
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "python", "arguments": '{"code": "pri'},
                            }
                        ]
                    }
                ),
                _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'nt(1)"}'}}]}),
                _chunk({}, finish_reason="tool_calls"),
                "data: [DONE]",
            ]
        )
        result = client._parse_stream_response(resp)
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_1"
        assert tc.name == "python"
        assert tc.arguments == {"code": "print(1)"}
        assert result.finish_reason == "tool_calls"

    def test_broken_tool_call_arguments_fall_back_to_raw(self) -> None:
        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        # arguments never complete to valid JSON
        resp = _make_stream_response(
            [
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "python", "arguments": '{"code": "pri'},
                            }
                        ]
                    }
                ),
                _chunk({}, finish_reason="tool_calls"),
                "data: [DONE]",
            ]
        )
        result = client._parse_stream_response(resp)
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert isinstance(tc.arguments, dict)
        assert "raw" in tc.arguments


class TestSendStreaming:
    """Test send() in streaming mode, including the fallback."""

    def test_send_streaming_requests_stream_true(self) -> None:
        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        stream_resp = _make_stream_response(
            [
                _chunk({"content": "Hi"}),
                _chunk({}, finish_reason="stop"),
                "data: [DONE]",
            ]
        )
        stream_resp.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}
        stream_resp.status_code = 200

        with patch("llm_cli_py.providers.llm_api.post_with_retries", return_value=stream_resp) as mock_post:
            result = client.send([DataSource(text="Hello")], [], stream=True)

        # post_with_retries(session, url, json_body, timeout)
        json_body = mock_post.call_args[0][2]
        assert json_body["stream"] is True
        assert result.text == "Hi"
        assert client._state.conversation[-1].content == "Hi"

    def test_send_streaming_tool_call_success(self) -> None:
        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        stream_resp = _make_stream_response(
            [
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "python", "arguments": '{"code": "print(1)"}'},
                            }
                        ]
                    }
                ),
                _chunk({}, finish_reason="tool_calls"),
                "data: [DONE]",
            ]
        )
        stream_resp.status_code = 200

        with patch("llm_cli_py.providers.llm_api.post_with_retries", return_value=stream_resp):
            result = client.send([DataSource(text="Run it")], [], stream=True)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "python"
        assert result.tool_calls[0].arguments == {"code": "print(1)"}

    def test_reasoning_round_trip_in_build_messages(self) -> None:
        """DeepSeek V4 requires reasoning to be replayed with tool calls."""
        from llm_cli_py.models import Message, Role

        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        client._state.conversation = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "python", "arguments": '{"code": "print(1)"}'},
                    }
                ],
                reasoning="thinking trace here",
            )
        ]
        messages = client._build_messages()
        assistant = messages[-1]
        assert assistant["reasoning"] == "thinking trace here"
        assert assistant["tool_calls"] is not None

    def test_build_messages_omits_reasoning_when_absent(self) -> None:
        from llm_cli_py.models import Message, Role

        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        client._state.conversation = [Message(role=Role.ASSISTANT, content="plain answer")]
        messages = client._build_messages()
        assistant = messages[-1]
        assert "reasoning" not in assistant

    def test_send_streaming_broken_tool_call_returned_without_fallback(self) -> None:
        """Broken tool calls are returned as-is (no non-streaming fallback)."""
        client = LlmApiClient("m", "https://api.example.com/v1", "k")
        broken_stream = _make_stream_response(
            [
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "python", "arguments": '{"code": "pri'},
                            }
                        ]
                    }
                ),
                _chunk({}, finish_reason="tool_calls"),
                "data: [DONE]",
            ]
        )
        broken_stream.status_code = 200

        with patch("llm_cli_py.providers.llm_api.post_with_retries", return_value=broken_stream) as mock_post:
            result = client.send([DataSource(text="Run it")], [], stream=True)

        # Exactly one request, and the broken call is surfaced for the caller.
        assert mock_post.call_count == 1
        assert mock_post.call_args_list[0][0][2]["stream"] is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"raw": '{"code": "pri'}
