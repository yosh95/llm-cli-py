"""OpenRouter web search server tool.

OpenRouter's ``openrouter:web_search`` server tool lets any model run a
real-time web search during a request. Unlike a user-defined tool, the
provider executes the search server-side and feeds the results back to the
model automatically - no client-side implementation is required.

This module only *advertises* the tool to the API. When the model decides
to call it, OpenRouter handles the execution; the CLI simply continues the
agent loop (the next request carries the search results already injected by
the provider).

Reference: https://openrouter.ai/docs/guides/features/server-tools/web-search
"""

from __future__ import annotations

# Tool name as advertised to the OpenRouter API.
OPENROUTER_WEB_SEARCH_TOOL_NAME = "openrouter:web_search"
"""Name of the OpenRouter web search server tool."""

OPENROUTER_WEB_SEARCH_DESCRIPTION = (
    "Search the web for real-time information. The model decides when to use "
    "this tool; OpenRouter executes the search server-side and returns the "
    "results to the model automatically."
)

# Optional parameters accepted by the server tool. All are optional; the
# provider applies sensible defaults (engine=auto, max_results=5, ...).
OPENROUTER_WEB_SEARCH_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to run.",
        },
        "engine": {
            "type": "string",
            "enum": ["auto", "native", "exa", "firecrawl", "parallel", "perplexity"],
            "description": "Search engine to use. Default: auto.",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 25,
            "description": "Maximum results per search. Default: 5.",
        },
        "max_uses": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of searches for this request.",
        },
        "max_total_results": {
            "type": "integer",
            "minimum": 1,
            "description": "Cap on total results across all searches (cost/context control).",
        },
        "search_context_size": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Amount of context to fetch per result.",
        },
        "allowed_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Restrict search to these domains.",
        },
        "excluded_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exclude these domains from search.",
        },
    },
    "required": ["query"],
}
