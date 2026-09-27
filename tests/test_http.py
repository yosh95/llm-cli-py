"""Tests for the stdlib HTTP POST helper (no network: the transport is patched)."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from llm_cli_py.utils.http import HttpError, post_json


class _Response:
    """A minimal ``urlopen`` result: a context manager with ``read()``."""

    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _http_error(code: int, reason: str, body: object) -> urllib.error.HTTPError:
    raw = json.dumps(body).encode("utf-8")
    return urllib.error.HTTPError(
        url="https://api.example.com/v1/chat/completions",
        code=code,
        msg=reason,
        hdrs=MagicMock(),  # type: ignore[arg-type]
        fp=MagicMock(read=lambda: raw),
    )


def test_post_sends_json_with_the_api_key_and_returns_the_body() -> None:
    with patch(
        "llm_cli_py.utils.http.urllib.request.urlopen", return_value=_Response({"ok": True})
    ) as urlopen:
        assert post_json("https://api.example.com/v1/chat/completions", {"q": 1}, 30, api_key="sk-1") == {
            "ok": True
        }

    request = urlopen.call_args.args[0]
    assert request.method == "POST"
    assert json.loads(request.data) == {"q": 1}
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") == "Bearer sk-1"


def test_error_body_is_surfaced_in_the_exception() -> None:
    error = _http_error(401, "Unauthorized", {"error": {"message": "bad credentials", "code": 20015}})

    with (
        patch("llm_cli_py.utils.http.urllib.request.urlopen", side_effect=error) as urlopen,
        patch("llm_cli_py.utils.http.time.sleep"),
        pytest.raises(HttpError) as excinfo,
    ):
        post_json("https://api.example.com/v1/chat/completions", {}, 30)

    assert urlopen.call_count == 1  # a 401 is not retried
    assert "bad credentials" in str(excinfo.value)


def test_transient_failure_is_retried_then_the_last_error_is_raised() -> None:
    error = _http_error(503, "Unavailable", {"error": {"message": "down"}})

    with (
        patch(
            "llm_cli_py.utils.http.urllib.request.urlopen",
            side_effect=[error, _Response({"ok": True})],
        ) as urlopen,
        patch("llm_cli_py.utils.http.time.sleep") as sleep,
    ):
        assert post_json("https://api.example.com/v1/chat/completions", {}, 30) == {"ok": True}

    assert urlopen.call_count == 2 and sleep.call_count == 1
