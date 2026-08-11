"""Tests for the verifier module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from llm_cli_py.models import ToolCall
from llm_cli_py.verifier import Verifier, _extract_json_object


class TestVerifier:
    """Test Verifier functionality."""

    def test_initialization(self) -> None:
        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )
        assert verifier.enabled is True

    def test_disable_verifier(self) -> None:
        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )
        assert verifier.enabled is True

        verifier.set_enabled(False)
        assert verifier.enabled is False

        verifier.set_enabled(True)
        assert verifier.enabled is True

    def test_disabled_verifier_auto_approves(self) -> None:
        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )
        verifier.set_enabled(False)

        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is True
        assert reason == "Verifier disabled"

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_approved(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"approved": true, "reason": "Approved - safe operation"}'}}]
        }
        mock_post.return_value = mock_response

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )

        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print('hello')", "explanation": "Testing"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is True
        assert "Approved" in reason

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_rejected(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": ('{"approved": false, "reason": "Potentially dangerous operation"}')}}
            ]
        }
        mock_post.return_value = mock_response

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )

        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "import os; os.remove('/etc/passwd')"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is False
        assert "dangerous" in reason

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_accepts_fenced_json(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '```json\n{"approved": true, "reason": "read-only op"}\n```'}}
            ]
        }
        mock_post.return_value = mock_response

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="tencent/hy3:free",
        )
        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is True
        assert "read-only" in reason

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_accepts_prose_wrapped_json(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Sure, here is my decision:\n"
                            '{"approved": false, "reason": "deletes a file"}\n'
                            "Hope that helps."
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="tencent/hy3:free",
        )
        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "import os; os.remove('/tmp/x')"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is False
        assert "deletes a file" in reason

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_json_parse_error_sends_to_hitl(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is not valid JSON at all"}}]
        }
        mock_post.return_value = mock_response

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )

        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is False
        assert "confirm manually" in reason.lower()

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_api_error_safe_fallback(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = Exception("API Error")

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )

        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is False
        assert "confirm manually" in reason.lower()

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_timeout_reports_timeout(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )

        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is False
        assert "did not respond within" in reason.lower()
        assert "confirm manually" in reason.lower()

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_other_request_error_reports_failure(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.exceptions.ConnectionError("boom")

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )

        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is False
        assert "request failed" in reason.lower()
        assert "confirm manually" in reason.lower()

    def test_default_not_configured(self) -> None:
        verifier = Verifier()
        assert verifier.is_configured is False
        assert verifier.enabled is True

    def test_not_configured_rejects(self) -> None:
        verifier = Verifier()
        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is False
        assert "not configured" in reason.lower()

    def test_configured_property(self) -> None:
        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )
        assert verifier.is_configured is True

    def test_not_configured_with_disabled_passes(self) -> None:
        verifier = Verifier()
        verifier.set_enabled(False)
        tool_call = ToolCall(
            id="call_1",
            name="python",
            arguments={"code": "print(1)"},
        )
        approved, reason = verifier.verify(tool_call, [])
        assert approved is True
        assert reason == "Verifier disabled"

    def test_default_constructor_no_args(self) -> None:
        verifier = Verifier()
        assert verifier.model == ""
        assert verifier.enabled is True
        assert verifier.is_configured is False

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_streams_content(self, mock_post: MagicMock) -> None:
        import json as _json

        def chunk(delta: dict[str, str]) -> str:
            return f"data: {_json.dumps({'choices': [{'delta': delta}]})}"

        stream_resp = MagicMock()
        stream_resp.status_code = 200
        stream_resp.iter_lines.return_value = [
            # Reasoning deltas are received from the provider but, with reasoning
            # display removed, they are ignored by the verifier.
            chunk({"reasoning_content": "Checking "}).encode(),
            chunk({"reasoning_content": "safety"}).encode(),
            chunk({"content": '{"approved": true, "reason": "read only"}'}).encode(),
            b"data: [DONE]",
        ]
        mock_post.return_value = stream_resp

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )
        tool_call = ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})
        content: list[str] = []
        approved, reason = verifier.verify(
            tool_call,
            [],
            on_content=content.append,
        )

        assert approved is True
        assert reason == "read only"
        assert content == ['{"approved": true, "reason": "read only"}']
        # Request was sent with stream=True
        assert mock_post.call_args.kwargs["json"]["stream"] is True

    @patch("llm_cli_py.utils.http.requests.Session.post")
    def test_verify_stream_falls_back_when_empty(self, mock_post: MagicMock) -> None:
        # Stream yields nothing -> fall back to a non-streaming parse.
        empty_stream = MagicMock()
        empty_stream.status_code = 200
        empty_stream.iter_lines.return_value = [b"data: [DONE]"]

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [{"message": {"content": '{"approved": true, "reason": "fallback ok"}'}}]
        }
        mock_post.side_effect = [empty_stream, ok_response]

        verifier = Verifier(
            api_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )
        tool_call = ToolCall(id="call_1", name="python", arguments={"code": "print(1)"})
        approved, reason = verifier.verify(tool_call, [], on_content=lambda _d: None)
        assert approved is True
        assert reason == "fallback ok"
        # first call stream=True, fallback stream=False
        assert mock_post.call_args_list[0].kwargs["json"]["stream"] is True
        assert mock_post.call_args_list[1].kwargs["json"]["stream"] is False


class TestExtractJsonObject:
    """Test the resilient JSON extraction used by the verifier."""

    def test_plain_json(self) -> None:
        out = _extract_json_object('{"approved": true, "reason": "ok"}')
        assert out == {"approved": True, "reason": "ok"}

    def test_code_fence(self) -> None:
        text = '```json\n{"approved": false, "reason": "danger"}\n```'
        out = _extract_json_object(text)
        assert out == {"approved": False, "reason": "danger"}

    def test_code_fence_no_lang(self) -> None:
        text = '```\n{"approved": true, "reason": "fine"}\n```'
        out = _extract_json_object(text)
        assert out == {"approved": True, "reason": "fine"}

    def test_prose_around_json(self) -> None:
        text = (
            "Here is my assessment:\n"
            '{"approved": false, "reason": "writes a file"}\n'
            "Let me know if you need more detail."
        )
        out = _extract_json_object(text)
        assert out == {"approved": False, "reason": "writes a file"}

    def test_nested_braces_in_prose(self) -> None:
        text = 'Sure! { "approved": true, "reason": "read-only {safe}" } done'
        out = _extract_json_object(text)
        assert out == {"approved": True, "reason": "read-only {safe}"}

    def test_no_json(self) -> None:
        assert _extract_json_object("not json at all") is None

    def test_empty(self) -> None:
        assert _extract_json_object("") is None
        assert _extract_json_object("   ") is None
