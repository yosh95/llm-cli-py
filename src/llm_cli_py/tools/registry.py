"""Tool registry for managing available tools."""

from __future__ import annotations

from collections.abc import Callable

from ..models import ToolSchema
from .types import ToolResult

ToolFunc = Callable[..., ToolResult]


class Tool:
    """Represents a registered tool with its schema and implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, object],
        func: ToolFunc,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    @property
    def schema(self) -> ToolSchema:
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
        func: ToolFunc,
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name.
            description: Tool description.
            parameters: JSON Schema for the tool's parameters.
            func: Callable implementing the tool.
        """
        self._tools[name] = Tool(name, description, parameters, func)

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
