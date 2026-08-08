"""Shared HTTP retry utility for POST requests with transient-failure retry."""

from __future__ import annotations

import time
from typing import Any

import requests


def post_with_retries(
    session: requests.Session,
    url: str,
    json_body: dict[str, Any],
    timeout: int,
    max_retries: int = 3,
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
        requests.exceptions.RequestException: On non-retryable errors or after exhausting retries.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = session.post(url, json=json_body, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exception = requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code}",
                    response=resp,
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue
            resp.raise_for_status()
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
