"""OpenRouter web search server tool.

OpenRouter's ``openrouter:web_search`` server tool lets any model run a
real-time web search during a request. Unlike a user-defined tool, the
provider executes the search server-side and feeds the results back to the
model automatically - no client-side implementation is required.

This module only *advertises* the tool to the API. The tool is sent in its
minimal form - ``{"type": "openrouter:web_search"}`` - and OpenRouter applies
sensible defaults (e.g. ``engine: auto``, ``max_results: 5``). No JSON schema
is needed because the model never has to construct the call arguments: when
the model decides to search, OpenRouter handles the execution and injects the
results into the request automatically; the CLI simply continues the agent
loop.

Reference: https://openrouter.ai/docs/guides/features/server-tools/web-search
"""

from __future__ import annotations

# Tool name as advertised to the OpenRouter API. This is the entire tool
# specification - the request only needs ``{"type": this}``.
OPENROUTER_WEB_SEARCH_TOOL_NAME = "openrouter:web_search"
"""Name of the OpenRouter web search server tool."""

OPENROUTER_WEB_SEARCH_PARAMETERS: dict[str, object] = {}
"""Optional server-tool parameters.

Empty by default so the tool is advertised in its minimal form
(``{"type": "openrouter:web_search"}``) and the provider applies its defaults.
To tune behaviour (e.g. ``engine``, ``max_results``, ``max_uses``), add keys
here and they are sent as the tool's ``parameters`` in the request. See
https://openrouter.ai/docs/guides/features/server-tools/web-search
"""
