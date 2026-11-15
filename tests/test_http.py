"""Tests for the HTTP POST helper using requests (no network: the transport is patched)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_cli_py.consts import DEFAULT_CONNECT_TIMEOUT
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
        assert post_json("https://api.example.com/v1/chat/completions", {"q": 1}, 30, api_key="sk-1") == {
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
        # The connect phase has its own, much shorter limit: a host that does not
        # answer must say so in seconds instead of inheriting the answer's limit.
        timeout=(DEFAULT_CONNECT_TIMEOUT, 30),
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
        post_json("https://api.example.com/v1/chat/completions", {}, 30)

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
        assert post_json("https://api.example.com/v1/chat/completions", {}, 30) == {"ok": True}

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
        assert post_json("https://api.example.com/v1/chat/completions", {}, 30) == {"ok": True}

    assert post.call_count == 2 and sleep.call_count == 1


def test_a_read_timeout_is_not_retried() -> None:
    """A reply that missed its deadline is gone; asking again is the same wall."""
    with (
        patch(
            "llm_cli_py.utils.http.requests.post",
            side_effect=requests.ReadTimeout("read timed out"),
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep") as sleep,
        pytest.raises(HttpError) as excinfo,
    ):
        post_json("https://api.example.com/v1/chat/completions", {}, 120)

    assert post.call_count == 1  # not 3: the wait is paid once, not three times
    assert sleep.call_count == 0
    assert "120s" in str(excinfo.value)
    assert "NOT retried" in str(excinfo.value)


def test_a_connect_timeout_is_retried() -> None:
    """Connecting costs nothing to attempt again, so it still gets all attempts."""
    ok_resp = _mock_response(200, json_data={"ok": True})

    with (
        patch(
            "llm_cli_py.utils.http.requests.post",
            side_effect=[requests.ConnectTimeout("connect timed out"), ok_resp],
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep") as sleep,
    ):
        assert post_json("https://api.example.com/v1/chat/completions", {}, 120) == {"ok": True}

    assert post.call_count == 2 and sleep.call_count == 1


def test_every_attempt_uses_the_connect_and_read_pair() -> None:
    """Both limits reach the transport on every attempt, retries included."""
    ok_resp = _mock_response(200, json_data={"ok": True})

    with (
        patch(
            "llm_cli_py.utils.http.requests.post",
            side_effect=[requests.ConnectionError("refused"), ok_resp],
        ) as post,
        patch("llm_cli_py.utils.http.time.sleep"),
    ):
        post_json("https://api.example.com/v1/chat/completions", {}, 120)

    for call in post.call_args_list:
        assert call.kwargs["timeout"] == (DEFAULT_CONNECT_TIMEOUT, 120)
