"""Tests for the shared HTTP retry utility."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_cli_py.utils.http import post_with_retries


class TestPostWithRetries:
    """Test the shared POST-with-retry utility."""

    def test_success_first_try(self) -> None:
        session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        session.post.return_value = mock_resp

        result = post_with_retries(session, "https://example.com/api", {"q": 1}, 30)
        assert result is mock_resp
        session.post.assert_called_once()

    def test_retries_on_429(self) -> None:
        session = MagicMock(spec=requests.Session)
        mock_fail = MagicMock()
        mock_fail.status_code = 429
        mock_ok = MagicMock()
        mock_ok.status_code = 200

        session.post.side_effect = [mock_fail, mock_ok]

        with patch("llm_cli_py.utils.http.time.sleep") as mock_sleep:
            result = post_with_retries(session, "https://example.com/api", {"q": 1}, 30)

        assert result is mock_ok
        assert session.post.call_count == 2
        assert mock_sleep.call_count == 1

    def test_retries_on_500(self) -> None:
        session = MagicMock(spec=requests.Session)
        mock_fail = MagicMock()
        mock_fail.status_code = 503
        mock_ok = MagicMock()
        mock_ok.status_code = 200

        session.post.side_effect = [mock_fail, mock_ok]

        with patch("llm_cli_py.utils.http.time.sleep"):
            result = post_with_retries(session, "https://example.com/api", {"q": 1}, 30)

        assert result is mock_ok

    def test_retries_on_timeout(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.post.side_effect = requests.exceptions.Timeout("timed out")

        with (
            patch("llm_cli_py.utils.http.time.sleep"),
            pytest.raises(requests.exceptions.Timeout),
        ):
            post_with_retries(session, "https://example.com/api", {"q": 1}, 30, max_retries=3)

        assert session.post.call_count == 3

    def test_retries_on_connection_error(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.post.side_effect = requests.exceptions.ConnectionError("refused")

        with (
            patch("llm_cli_py.utils.http.time.sleep"),
            pytest.raises(requests.exceptions.ConnectionError),
        ):
            post_with_retries(session, "https://example.com/api", {"q": 1}, 30, max_retries=2)

        assert session.post.call_count == 2

    def test_non_retryable_http_error_raises_immediately(self) -> None:
        session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.reason = "Unauthorized"
        mock_resp.text = '{"error": {"message": "bad credentials"}}'
        session.post.return_value = mock_resp

        # Non-retryable HTTPError should be raised immediately without retry
        with (
            patch("llm_cli_py.utils.http.time.sleep") as mock_sleep,
            pytest.raises(requests.exceptions.HTTPError) as excinfo,
        ):
            post_with_retries(session, "https://example.com/api", {"q": 1}, 30, max_retries=3)

        assert session.post.call_count == 1
        mock_sleep.assert_not_called()
        # Provider error body should be surfaced in the exception message
        assert "bad credentials" in str(excinfo.value)

    def test_nonn2xx_error_body_is_surfaced(self) -> None:
        session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.reason = "Bad Request"
        mock_resp.text = '{"error": {"message": "Invalid request parameters.", "code": 20015}}'
        session.post.return_value = mock_resp

        with pytest.raises(requests.exceptions.HTTPError) as excinfo:
            post_with_retries(session, "https://example.com/api", {"q": 1}, 30, max_retries=3)

        assert "Invalid request parameters" in str(excinfo.value)
        assert "(code: 20015)" in str(excinfo.value)

    def test_all_retries_exhausted_raises_last_exception(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.post.side_effect = requests.exceptions.Timeout("always timeout")

        with (
            patch("llm_cli_py.utils.http.time.sleep"),
            pytest.raises(requests.exceptions.Timeout, match="always timeout"),
        ):
            post_with_retries(session, "https://example.com/api", {"q": 1}, 30, max_retries=3)

        assert session.post.call_count == 3

    def test_custom_max_retries(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.post.side_effect = requests.exceptions.Timeout("timeout")

        with (
            patch("llm_cli_py.utils.http.time.sleep"),
            pytest.raises(requests.exceptions.Timeout),
        ):
            post_with_retries(session, "https://example.com/api", {"q": 1}, 30, max_retries=5)

        assert session.post.call_count == 5
