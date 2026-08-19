"""Web Search tool using Brave Search API (LLM Context endpoint)."""

import os
import time

import requests

from ..consts import (
    BRAVE_SEARCH_API_URL,
    BRAVE_SEARCH_MAX_RETRIES,
    DEFAULT_WEB_SEARCH_TIMEOUT,
    ENV_BRAVE_SEARCH_API_KEY,
    ENV_PROXY_URL,
)
from .types import SearchResult, SearchResultItem, ToolError


def web_search(
    query: str,
) -> SearchResult | ToolError:
    """Search the web using Brave Search API (LLM Context endpoint).

    If ``LLM_CLI_PROXY_URL`` is set, the request is sent to ``{proxy_url}/web_search``
    so the proxy can inject the API key server-side.

    Otherwise, uses ``BRAVE_SEARCH_API_KEY`` environment variable for authentication
    and sends the request directly to the Brave Search API.

    Uses the LLM Context API endpoint (``/res/v1/llm/context``) which returns
    pre-extracted web content (text, tables, code blocks) ranked by relevance
    and ready for LLM consumption.

    Implements retry with exponential backoff for rate limits.

    Args:
        query: The search query to submit to the web search API.

    Returns:
        SearchResult on success, ToolError on failure.
    """
    proxy_url = os.environ.get(ENV_PROXY_URL, "").strip()

    if proxy_url:
        # Use proxy for search
        api_url = proxy_url.rstrip("/") + "/web_search"
        headers = {"Content-Type": "application/json"}
        use_get = False
    else:
        api_url = BRAVE_SEARCH_API_URL
        api_key = os.environ.get(ENV_BRAVE_SEARCH_API_KEY, "").strip()
        if not api_key:
            return ToolError(
                error=f"{ENV_BRAVE_SEARCH_API_KEY} is not set. "
                "Please set this environment variable to use the web search tool."
            )
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        use_get = True

    last_error: str | None = None
    max_retries = BRAVE_SEARCH_MAX_RETRIES

    for attempt in range(1, max_retries + 1):
        try:
            if use_get:
                resp = requests.get(
                    api_url,
                    params={"q": query},
                    headers=headers,
                    timeout=DEFAULT_WEB_SEARCH_TIMEOUT,
                )
            else:
                resp = requests.post(
                    api_url,
                    json={"query": query},
                    headers=headers,
                    timeout=DEFAULT_WEB_SEARCH_TIMEOUT,
                )

            if resp.status_code in (429, 503):
                last_error = f"HTTP {resp.status_code}: Rate limited or unavailable"
                if attempt < max_retries:
                    time.sleep(2**attempt)
                continue

            resp.raise_for_status()
            data = resp.json()

            results = []
            # Brave LLM Context API response format:
            # {
            #   "grounding": {
            #     "generic": [
            #       {"url": "...", "title": "...", "snippets": ["...", "..."]},
            #       ...
            #     ]
            #   },
            #   "sources": { "url": {"title": "...", "hostname": "...", "age": ...} }
            # }
            generic_results = data.get("grounding", {}).get("generic", [])
            for item in generic_results:
                snippets = item.get("snippets", [])
                # Join multiple snippets into a single snippet string
                snippet = "\n".join(snippets) if snippets else ""
                results.append(
                    SearchResultItem(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=snippet,
                    )
                )

            return SearchResult(
                query=query,
                results=results,
                result_count=len(results),
            )

        except requests.RequestException as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(2**attempt)
            continue

    error_msg = (
        f"Search failed after {max_retries} attempts: {last_error}\n"
        "Try re-running web_search with a simpler, more specific query, "
        "or retry later if this looks like a transient network/rate-limit issue."
    )
    return ToolError(error=error_msg)


# Tool schema definition
WEB_SEARCH_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to submit to the web search API.",
        },
    },
    "required": ["query"],
}

WEB_SEARCH_DESCRIPTION = (
    "Search the web using Brave Search API (LLM Context endpoint). "
    "Returns title, URL, and extracted content snippets optimized for LLM consumption."
)
