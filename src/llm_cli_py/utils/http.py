"""Minimal HTTP POST helper for the OpenAI-compatible chat endpoint.

This module provides a helper for making JSON POST requests to
``/chat/completions`` using ``requests``. It keeps the two behaviours that
matter for that single request: retrying transient failures, and surfacing the
provider's error body instead of a bare status line.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
"""HTTP statuses treated as transient (retried with exponential backoff)."""

_ERROR_DETAIL_MAX_LEN = 800
"""Maximum length of a raw error body quoted back to the user."""

_MAX_ATTEMPTS = 3
"""How many times a request is attempted in total (1 try + 2 retries)."""


class HttpError(Exception):
    """HTTP request failure whose message carries the provider's error body."""


def _error_detail(raw: str) -> str:
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
    if len(body) > _ERROR_DETAIL_MAX_LEN:
        body = body[:_ERROR_DETAIL_MAX_LEN] + "\u2026"
    return body


def _http_error(status: int, reason: str, raw: str) -> HttpError:
    """Build an ``HttpError`` whose message includes the provider's detail."""
    kind = "Client Error" if 400 <= status < 500 else "Server Error"
    message = f"{status} {kind}: {reason}"
    detail = _error_detail(raw)
    if detail:
        message = f"{message} - {detail}"
    return HttpError(message)


def post_json(
    url: str,
    json_body: dict[str, Any],
    timeout: int,
    *,
    api_key: str = "",
) -> dict[str, Any]:
    """POST ``json_body`` to ``url`` and return the decoded JSON response.

    Retries transient failures (429/5xx, timeouts, connection errors) with
    exponential backoff starting at 1 second; other HTTP errors (e.g. 400, 401)
    are raised immediately, because retrying them cannot help. The attempt
    limit is fixed (see ``_MAX_ATTEMPTS``): this is the one request the CLI
    makes, and it is not configurable from the outside.

    Args:
        url: Target URL.
        json_body: The JSON request body.
        timeout: Request timeout in seconds.
        api_key: Optional API key, sent as ``Authorization: Bearer <key>``.

    Returns:
        The parsed JSON response body.

    Raises:
        HttpError: On a non-retryable error, or after the retries are exhausted.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: HttpError | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.post(
                url,
                json=json_body,
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code >= 400:
                error = _http_error(resp.status_code, resp.reason or "", resp.text)
                if resp.status_code not in RETRYABLE_STATUS:
                    raise error
                last_error = error
            else:
                try:
                    payload: dict[str, Any] = resp.json()
                except ValueError as e:
                    msg = f"Invalid JSON response from {url}: {e}"
                    raise HttpError(msg) from e
                else:
                    return payload
        except (requests.ConnectionError, requests.Timeout) as e:
            # Connection refused/DNS failure and read timeouts
            # are both worth another attempt.
            last_error = HttpError(f"Request failed: {e}")

        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(2**attempt)

    assert last_error is not None
    raise last_error
