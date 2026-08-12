"""Shared HTTP retry utility for POST requests with transient-failure retry."""

from __future__ import annotations

import time
from typing import Any

import requests


def _error_detail_from_response(resp: requests.Response, max_len: int = 800) -> str:
    """Extract a human-readable error detail from an HTTP response body.

    Many providers (OpenRouter, DeepSeek, Ollama) return a JSON error body
    whose ``error.message`` field contains the actual diagnostic (e.g. "The
    reasoning_content in the thinking mode must be passed back to the API").
    ``raise_for_status()`` throws that detail away, so we surface it here to
    make the failure diagnosable from the terminal alone.
    """
    if resp is None:
        return ""

    try:
        raw = resp.text
    except requests.exceptions.RequestException:
        return "(response body could not be read)"

    if not raw:
        return ""

    # Prefer structured JSON error fields.
    try:
        import json

        data = json.loads(raw)
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            code = err.get("code")
            detail = str(msg) if msg else ""
            if code is not None:
                detail = f"{detail} (code: {code})".strip()
                if detail.startswith("(code:"):
                    detail = f"code: {code}"
            if detail:
                return detail
        elif isinstance(err, str) and err:
            return err
        if isinstance(data, dict):
            msg = data.get("message")
            if msg:
                return str(msg)
    except (ValueError, TypeError):
        pass

    # Fall back to a truncated raw body.
    body = raw.strip()
    if len(body) > max_len:
        body = body[:max_len] + "…"
    return body


def raise_for_status_with_detail(resp: requests.Response) -> None:
    """Raise ``requests.HTTPError`` for a non-2xx response, including body detail.

    Mirrors ``resp.raise_for_status()`` but enriches the exception message with
    the provider's error body so the real cause is visible to the user.
    """
    if resp is None or 200 <= resp.status_code < 300:
        return

    detail = _error_detail_from_response(resp)
    reason = resp.reason or ""
    base = f"{resp.status_code} Client Error: {reason}"
    if not (400 <= resp.status_code < 500):
        base = f"{resp.status_code} Server Error: {reason}"
    if detail:
        base = f"{base} — {detail}"

    raise requests.exceptions.HTTPError(base, response=resp)


def post_with_retries(
    session: requests.Session,
    url: str,
    json_body: dict[str, Any],
    timeout: int,
    max_retries: int = 3,
    *,
    stream: bool = False,
) -> requests.Response:
    """POST with retry on transient failures (429, 5xx, timeout, connection errors).

    Uses exponential backoff starting at 1 second.
    Non-retryable HTTP errors (e.g. 400, 401, 403, 404) are raised immediately.

    Args:
        session: ``requests.Session`` to use for the request.
        url: Target URL.
        json_body: JSON request body.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of attempts (default 3).

    Returns:
        The successful ``requests.Response``.

    Raises:
        requests.exceptions.RequestException: On non-retryable errors or after
            exhausting retries. The message includes the server's error body
            when available, so provider-side diagnostics are visible.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = session.post(url, json=json_body, timeout=timeout, stream=stream)
            if resp.status_code in (429, 500, 502, 503, 504):
                detail = _error_detail_from_response(resp)
                msg = f"HTTP {resp.status_code}"
                if detail:
                    msg = f"{msg} — {detail}"
                last_exception = requests.exceptions.HTTPError(msg, response=resp)
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue
            raise_for_status_with_detail(resp)
            return resp
        except requests.exceptions.Timeout as exc:
            last_exception = exc
        except requests.exceptions.ConnectionError as exc:
            last_exception = exc
        except requests.exceptions.HTTPError:
            # Non-retryable HTTP errors are surfaced immediately
            raise

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    if isinstance(last_exception, Exception):
        raise last_exception
    raise RuntimeError("Request failed after retries")
