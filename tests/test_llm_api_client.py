"""Tests for the OpenAI-compatible chat API client.

Covers request building, streaming response parsing, retry behaviour,
and context manager.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_cli_py.models import DataSource, Message, Role, ToolSchema
from llm_cli_py.providers.llm_api import LlmApiClient


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> str:
    """Build a single SSE data line from a delta dict."""
    return json.dumps({"choices": [{"delta": delta, "finish_reason": finish_reason}]})


def _make_stream_response(chunks: list[str]) -> MagicMock:
    """Build a mock requests.Response that yields SSE data lines."""
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = [c.encode("utf-8") for c in chunks]
    return resp


class TestLlmApiClient:
    """Test the OpenAI-compatible chat API client."""

    def test_build_messages_omits_system_prompt_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        client._state.conversation = [Message(role=Role.USER, content="Hello")]
        messages = client._build_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_build_messages_uses_system_prompt_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "You are a test assistant.")
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        client._state.conversation.append(Message(role=Role.USER, content="Hello"))
        messages = client._build_messages()

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a test assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_system_prompt_snapshotted_at_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_CLI_SYSTEM_PROMPT is read once at client init; later env changes are ignored."""
        monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "Startup prompt.")
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        assert client.state.system_prompt == "Startup prompt."
        monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "Changed after init.")
        client._state.conversation.append(Message(role=Role.USER, content="Hello"))

        messages = client._build_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Startup prompt."
        assert messages[1]["role"] == "user"

    def test_build_messages_omits_system_prompt_when_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CLI_SYSTEM_PROMPT", "")
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        client._state.conversation = [Message(role=Role.USER, content="Hello")]
        messages = client._build_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_build_request_with_tools(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        schema = ToolSchema(
            name="python",
            description="Run Python",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )
        body = client._build_request([{"role": "user", "content": "hi"}], [schema])

        assert body["model"] == "gpt-4o"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["stream"] is True
        assert body["tools"][0]["function"]["name"] == "python"

    def test_send_appends_user_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_CLI_SYSTEM_PROMPT", raising=False)
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        stream_resp = _make_stream_response(
            [
                _chunk({"content": "Answer"}),
                _chunk({}, finish_reason="stop"),
                "data: [DONE]",
            ]
        )

        with patch("llm_cli_py.utils.http.requests.Session.post", return_value=stream_resp) as mock_post:
            result = client.send([DataSource(text="Question")], [])

        assert result.text == "Answer"
        assert len(client._state.conversation) == 2
        assert client._state.conversation[0].role == Role.USER
        assert "Question" in client._state.conversation[0].content

        mock_post.assert_called_once()
        _args, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["stream"] is True
        assert _args[0] == "https://api.example.com/v1/chat/completions"

    def test_send_retries_on_rate_limit(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        mock_fail = MagicMock()
        mock_fail.status_code = 429
        mock_fail.text = "rate limit exceeded"

        stream_resp = _make_stream_response(
            [
                _chunk({"content": "OK"}),
                _chunk({}, finish_reason="stop"),
                "data: [DONE]",
            ]
        )

        with (
            patch(
                "llm_cli_py.utils.http.requests.Session.post",
                side_effect=[mock_fail, stream_resp],
            ) as mock_post,
            patch("llm_cli_py.utils.http.time.sleep") as mock_sleep,
        ):
            result = client.send([DataSource(text="Hi")], [])

        assert result.text == "OK"
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1

    def test_send_retries_on_timeout_and_fails(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )

        with (
            patch(
                "llm_cli_py.utils.http.requests.Session.post",
                side_effect=requests.exceptions.Timeout("timed out"),
            ) as mock_post,
            patch("llm_cli_py.utils.http.time.sleep"),
            pytest.raises(requests.exceptions.Timeout),
        ):
            client.send([DataSource(text="Hi")], [])

        assert mock_post.call_count == 3

    def test_send_tool_call_arguments_are_json_encoded(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
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

        with patch("llm_cli_py.utils.http.requests.Session.post", return_value=stream_resp):
            client.send([DataSource(text="run it")], [])

        appended = client._state.conversation[-1]
        assert appended.tool_calls is not None
        function = appended.tool_calls[0]["function"]
        assert isinstance(function, dict)
        assert function["arguments"] == '{"code": "print(1)"}'

    def test_context_manager_closes_session(self) -> None:
        with LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        ) as client:
            assert client._api_url == "https://api.example.com/v1"

    def test_close_idempotent(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        client.close()
        client.close()  # should not raise
