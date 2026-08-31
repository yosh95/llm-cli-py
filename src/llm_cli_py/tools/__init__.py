"""Tools package."""

from __future__ import annotations

from .python_exec import PYTHON_TOOL_DESCRIPTION, PYTHON_TOOL_SCHEMA, execute_python
from .registry import Tool, ToolRegistry
from .types import ExecResult, ToolError, ToolResult
from .web_search import (
    OPENROUTER_WEB_SEARCH_DESCRIPTION,
    OPENROUTER_WEB_SEARCH_SCHEMA,
    OPENROUTER_WEB_SEARCH_TOOL_NAME,
)

__all__ = [
    "OPENROUTER_WEB_SEARCH_DESCRIPTION",
    "OPENROUTER_WEB_SEARCH_SCHEMA",
    "OPENROUTER_WEB_SEARCH_TOOL_NAME",
    "PYTHON_TOOL_DESCRIPTION",
    "PYTHON_TOOL_SCHEMA",
    "ExecResult",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "execute_python",
]
