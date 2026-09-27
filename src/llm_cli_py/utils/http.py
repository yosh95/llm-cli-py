"""Minimal HTTP POST helper for the OpenAI-compatible chat endpoint.

Standard library only: the CLI makes exactly one kind of request (a JSON
``POST`` to ``/chat/completions``), so a full HTTP client library is not
needed. This module keeps the two behaviours that matter for that single
request: retrying transient failures, and surfacing the provider's error body
instead of a bare status line.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
"""HTTP statuses treated as transient (retried with exponential backoff)."""

_ERROR_DETAIL_MAX_LEN = 800
"""Maximum length of a raw error body quoted back to the user."""


class HttpError(Exception):
    """HTTP request failure, optionally carrying the provider's error body."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _error_detail(raw: str, max_len: int = _ERROR_DETAIL_MAX_LEN) -> str:
    """Extract a human-readable error detail from an HTTP response body.

    Many providers (DeepSeek, Ollama) return a JSON error body whose
    ``error.message`` field contains the actual diagnostic; quoting it makes the
    failure diagnosable from the terminal alone, so it is surfaced first and a
    truncated raw body is only the fallback.
    """
    if not raw:
        return ""

    try:
        data: Any = json.loads(raw)
    except (ValueError, TypeError):
        data = None

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or "")
            code = err.get("code")
            if code is not None:
                detail = f"{detail} (code: {code})".strip()
            if detail:
                return detail
        elif isinstance(err, str) and err:
            return err
        message = data.get("message")
        if message:
            return str(message)

    body = raw.strip()
    if len(body) > max_len:
        body = body[:max_len] + "\u2026"
    return body


def _http_error(status: int, reason: str, raw: str) -> HttpError:
    """Build an ``HttpError`` whose message includes the provider's detail."""
    kind = "Client Error" if 400 <= status < 500 else "Server Error"
    message = f"{status} {kind}: {reason}"
    detail = _error_detail(raw)
    if detail:
        message = f"{message} - {detail}"
    return HttpError(message, status=status)


def post_json(
    url: str,
    json_body: dict[str, Any],
    timeout: int,
    max_retries: int = 3,
    *,
    api_key: str = "",
) -> dict[str, Any]:
    """POST ``json_body`` to ``url`` and return the decoded JSON response.

    Retries transient failures (429/5xx, timeouts, connection errors) with
    exponential backoff starting at 1 second; other HTTP errors (e.g. 400, 401)
    are raised immediately, because retrying them cannot help.

    Args:
        url: Target URL.
        json_body: The JSON request body.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of attempts (default 3).
        api_key: Optional API key, sent as ``Authorization: Bearer <key>``.

    Returns:
        The parsed JSON response body.

    Raises:
        HttpError: On a non-retryable error, or after the retries are exhausted.
    """
    data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )

    last_error: HttpError | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                return payload
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            error = _http_error(e.code, e.reason or "", raw)
            if e.code not in RETRYABLE_STATUS:
                raise error from e
            last_error = error
        except (urllib.error.URLError, TimeoutError) as e:
            # Connection refused/DNS failure (URLError) and read timeouts
            # (TimeoutError) are both worth another attempt.
            last_error = HttpError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON response from {url}: {e}"
            raise HttpError(msg) from e

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    assert last_error is not None
    raise last_error
