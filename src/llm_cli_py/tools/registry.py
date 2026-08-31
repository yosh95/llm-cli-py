"""Tool registry for managing available tools."""

from __future__ import annotations

from collections.abc import Callable

from ..models import ToolSchema
from .types import ToolResult

ToolFunc = Callable[..., ToolResult]


class Tool:
    """Represents a registered tool with its schema and implementation.

    Attributes:
        server_tool: When True, the tool is executed by the provider
            (e.g. OpenRouter server tools such as ``openrouter:web_search``)
            instead of by this application. ``func`` is None in that case and
            the schema is emitted verbatim (``{"type": "openrouter:web_search"}``).
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, object],
        func: ToolFunc | None,
        *,
        server_tool: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.server_tool = server_tool

    @property
    def schema(self) -> ToolSchema:
        if self.server_tool:
            # Server tools are advertised verbatim (no wrapping in a
            # "function" object); the provider executes them server-side.
            return ToolSchema(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
                server_tool=True,
            )
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolRegistry:
    """Registry for tools that the LLM can call."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, object],
        func: ToolFunc | None,
        *,
        server_tool: bool = False,
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name.
            description: Tool description.
            parameters: JSON Schema for the tool's parameters.
            func: Callable implementing the tool. Must be None when
                ``server_tool`` is True (the provider executes it).
            server_tool: When True, the tool is executed by the provider
                (OpenRouter server tools) instead of this application.
        """
        self._tools[name] = Tool(name, description, parameters, func, server_tool=server_tool)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        """Remove a registered tool by name.

        Raises KeyError if the tool is not registered.
        """
        del self._tools[name]

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    def get_schemas(self) -> list[ToolSchema]:
        """Get all tool schemas for API requests."""
        return [t.schema for t in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """Get sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)
