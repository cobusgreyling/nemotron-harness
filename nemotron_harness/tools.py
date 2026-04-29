"""
Tool registry and execution for Nemotron Harness.

Provides a generic ToolRegistry that maps tool definitions to handler
functions, so any application can register domain-specific tools and
the harness executes them uniformly.
"""

import json
from typing import Callable


class ToolRegistry:
    """Registry mapping tool names to OpenAI-format definitions and handlers.

    Usage:
        registry = ToolRegistry()

        @registry.register(
            name="search_web",
            description="Search the web for a query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        )
        def search_web(query: str) -> str:
            return f"Results for: {query}"

        # Or register without the decorator:
        registry.add("my_tool", definition, handler_fn)
    """

    def __init__(self):
        self._definitions: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
    ):
        """Decorator to register a tool handler with its definition.

        Args:
            name: Tool function name.
            description: Human-readable description for the model.
            parameters: JSON Schema for the tool's parameters.
        """
        definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._definitions[name] = definition

        def decorator(fn: Callable) -> Callable:
            self._handlers[name] = fn
            return fn
        return decorator

    def add(self, name: str, definition: dict, handler: Callable):
        """Register a tool with an explicit definition dict and handler.

        Args:
            name: Tool function name.
            definition: Full OpenAI function-calling format definition.
            handler: Callable that accepts keyword arguments matching the schema.
        """
        self._definitions[name] = definition
        self._handlers[name] = handler

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a registered tool by name.

        Args:
            name: Tool function name.
            arguments: Parsed arguments dict from the model's tool call.

        Returns:
            String result from the handler, or an error message.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}"
        try:
            result = handler(**arguments)
            return str(result)
        except Exception as e:
            return f"Tool error ({name}): {e}"

    def execute_from_raw(self, name: str, raw_arguments: str) -> str:
        """Execute a tool from raw JSON arguments string.

        Args:
            name: Tool function name.
            raw_arguments: JSON string of arguments.

        Returns:
            String result from the handler.
        """
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return f"Failed to parse arguments for {name}"
        return self.execute(name, arguments)

    @property
    def definitions(self) -> list[dict]:
        """Return all tool definitions in OpenAI function-calling format."""
        return list(self._definitions.values())

    @property
    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._definitions.keys())

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, name: str) -> bool:
        return name in self._definitions
