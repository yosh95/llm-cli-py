"""Tests for the HTTP POST helper using requests (no network: the transport is patched)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_cli_py.utils.http import HttpError, post_json


def _mock_response(
    status_code: int = 200,
    reason: str = "OK",
    json_data: object = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason = reason
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json.dumps(json_data)
    else:
        resp.text = text
        resp.json.side_effect = ValueError("No JSON")
    return resp


def test_post_sends_json_with_the_api_key_and_returns_the_body() -> None:
    mock_resp = _mock_response(200, json_data={"ok": True})
    with patch("llm_cli_py.utils.http.requests.post", return_value=mock_resp) as post:
        assert post_json("https://api.example.com/v1/chat/completions", {"q": 1}, api_key="sk-1") == {
            "ok": True
        }

    post.assert_called_once_with(
        "https://api.example.com/v1/chat/completions",
        json={"q": 1},
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer sk-1",
        },
        # No timeout: the transport waits for the answer as long as it takes.
    )


def test_error_body_is_surfaced_in_the_exception() -> None:
    error_resp = _mock_response(
        401,
        "Unauthorized",
        json_data={"error": {"message": "bad credentials", "code": 20015}},
    )

    with (
        patch("llm_cli_py.utils.http.requests.post", return_value=error_resp) as post,
        patch("llm_cli_py.utils.http.time.sleep"),
        pytest.raises(HttpError) as excinfo,
    ):
        post_json("https://api.example.com/v1/chat/completions", {})

    assert post.call_count == 1  # a 401 is not retried
    assert "bad credentials" in str(excinfo.value)


def test_transient_failure_is_retried_then_the_last_error_is_raised() -> None:
    error_resp = _mock_response(503, "Unavailable", json_data={"error": {"message": "down"}})
    ok_resp = _mock_response(200, json_data={"ok": True})

    with (
        patch(
            "llm_cli_py.utils.http.requests.post",
            side_effect=[error_resp, ok_resp],
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep") as sleep,
    ):
        assert post_json("https://api.example.com/v1/chat/completions", {}) == {"ok": True}

    assert post.call_count == 2 and sleep.call_count == 1


def test_connection_error_is_retried() -> None:
    ok_resp = _mock_response(200, json_data={"ok": True})

    with (
        patch(
            "llm_cli_py.utils.http.requests.post",
            side_effect=[requests.ConnectionError("refused"), ok_resp],
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep") as sleep,
    ):
        assert post_json("https://api.example.com/v1/chat/completions", {}) == {"ok": True}

    assert post.call_count == 2 and sleep.call_count == 1


def test_a_transport_timeout_is_reported_not_retried() -> None:
    """No limit of ours produces this, so it is reported rather than waited on again."""
    with (
        patch(
            "llm_cli_py.utils.http.requests.post",
            side_effect=requests.ReadTimeout("connection went quiet"),
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep") as sleep,
        pytest.raises(HttpError) as excinfo,
    ):
        post_json("https://api.example.com/v1/chat/completions", {})

    assert post.call_count == 1
    assert sleep.call_count == 0
    assert "connection went quiet" in str(excinfo.value)


def test_no_timeout_is_handed_to_the_transport() -> None:
    """Nothing limits the wait: every attempt asks without a deadline."""
    ok_resp = _mock_response(200, json_data={"ok": True})

    with (
        patch(
            "llm_cli_py.utils.http.requests.post",
            side_effect=[requests.ConnectionError("refused"), ok_resp],
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep"),
    ):
        post_json("https://api.example.com/v1/chat/completions", {})

    assert post.call_count == 2
    for call in post.call_args_list:
        assert "timeout" not in call.kwargs
