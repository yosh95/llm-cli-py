"""Tests for the OpenAI-compatible chat API client.

Covers request building, response parsing, retry behaviour, and context manager.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_cli_py.models import DataSource, Message, Role, ToolSchema
from llm_cli_py.providers.llm_api import LlmApiClient


class TestLlmApiClient:
    """Test the OpenAI-compatible chat API client."""

    def test_build_messages_with_default_system_prompt(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        client._state.conversation = [Message(role=Role.USER, content="Hello")]
        messages = client._build_messages()

        assert messages[0]["role"] == "system"
        assert "Today's actual date is:" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_build_messages_respects_system_prompt_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SYSTEM_PROMPT", "You are a test assistant.")
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        messages = client._build_messages()

        assert messages[0]["content"] == "You are a test assistant."

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
        assert body["stream"] is False
        assert body["tools"][0]["function"]["name"] == "python"

    def test_parse_text_response(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        result = client._parse_response(
            {
                "choices": [
                    {
                        "message": {"content": "Hi!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

        assert result.text == "Hi!"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert client._state.token_usage.total_tokens == 15

    def test_parse_tool_call_response(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        result = client._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "python",
                                        "arguments": {"code": "print(1)"},
                                    },
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "python"
        assert result.tool_calls[0].arguments == {"code": "print(1)"}

    def test_parse_tool_call_response_missing_id_is_synthesized(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        result = client._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "python", "arguments": {"code": "print(1)"}}}
                            ],
                        },
                    }
                ],
            }
        )

        assert result.tool_calls[0].id == "call_0"

    def test_send_appends_user_message(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
        }

        with patch("llm_cli_py.utils.http.requests.Session.post", return_value=mock_resp) as mock_post:
            result = client.send([DataSource(text="Question")], [])

        assert result.text == "Answer"
        assert len(client._state.conversation) == 2
        assert client._state.conversation[0].role == Role.USER
        assert "Question" in client._state.conversation[0].content

        mock_post.assert_called_once()
        _args, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["stream"] is False
        assert _args[0] == "https://api.example.com/v1/chat/completions"

    def test_send_retries_on_rate_limit(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        mock_fail = MagicMock()
        mock_fail.status_code = 429
        mock_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("HTTP 429")

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
        }

        with (
            patch(
                "llm_cli_py.utils.http.requests.Session.post", side_effect=[mock_fail, mock_ok]
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
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "python", "arguments": {"code": "print(1)"}},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        with patch("llm_cli_py.utils.http.requests.Session.post", return_value=mock_resp):
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

    def test_build_request_disables_reasoning_for_openrouter(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://openrouter.ai/api/v1",
            api_key="key",
        )
        body = client._build_request([{"role": "user", "content": "hi"}], [])
        assert body["reasoning"] == {"enabled": False}

    def test_build_request_disables_reasoning_for_ollama(self) -> None:
        client = LlmApiClient(
            model="qwen3",
            api_url="https://ollama.com/v1",
            api_key="key",
        )
        body = client._build_request([{"role": "user", "content": "hi"}], [])
        assert body["reasoning_effort"] == "none"
        assert body["think"] is False

    def test_build_request_keeps_reasoning_when_enabled(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://openrouter.ai/api/v1",
            api_key="key",
            disable_reasoning=False,
        )
        body = client._build_request([{"role": "user", "content": "hi"}], [])
        assert "reasoning" not in body

    def test_build_request_respects_env_disable_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CLI_DISABLE_REASONING", "0")
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://openrouter.ai/api/v1",
            api_key="key",
        )
        body = client._build_request([{"role": "user", "content": "hi"}], [])
        assert "reasoning" not in body


class TestLlmResponse:
    """Test LlmResponse integration with the client."""

    def test_parse_response_with_reasoning(self) -> None:
        client = LlmApiClient(
            model="gpt-4o",
            api_url="https://api.example.com/v1",
            api_key="key",
        )
        result = client._parse_response(
            {
                "choices": [
                    {
                        "message": {"content": "Final", "reasoning": "I thought..."},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

        assert result.text == "Final"
        assert result.reasoning == "I thought..."
