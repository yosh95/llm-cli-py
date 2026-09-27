"""Session handler: one turn of chat, plus the tool executions it asks for."""

from __future__ import annotations

from .. import ui
from ..base import LlmClient
from ..models import ClientState, Message, Role, ToolCall
from ..tools.registry import ToolRegistry
from ..tools.types import ToolResult, normalise_tool_result


def _append_tool_message(state: ClientState, content: str, tool_call_id: str) -> None:
    """Add a tool result to the conversation.

    Tool output is part of the record the model reasons over, exactly like a
    user or assistant message.
    """
    state.conversation.append(Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id))


class SessionContext:
    """Context holding shared resources for a session."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
    ) -> None:
        self.tool_registry = tool_registry


class ActiveSession:
    """Manages an active chat session with LLM interaction and tool execution."""

    def __init__(
        self,
        client: LlmClient,
        ctx: SessionContext,
    ) -> None:
        self.client = client
        self.ctx = ctx

    def process_and_print(self, prompt: str) -> None:
        """Main processing loop: send to LLM, handle tool calls, display results."""
        if not self.client.state.model:
            ui.display.report_info("No model specified locally.")

        current_prompt = prompt

        while True:
            tool_schemas = self.ctx.tool_registry.get_schemas()
            model = self.client.state.model
            display_model = model if model else "LLM"
            ui.display.print_rule()
            ui.display.print_thinking(display_model)

            try:
                response = self.client.send(current_prompt, tool_schemas)
            except Exception as e:
                ui.display.report_error(f"LLM request failed: {e}")
                break
            current_prompt = ""

            if response.text:
                ui.display.print_assistant(response.text)

            if not response.tool_calls:
                break

            # A tool call whose arguments were truncated/corrupted (JSON did not
            # parse) cannot be executed safely. The turn is only usable once
            # every tool call in it has a result, so drop the half-finished reply
            # and leave the agent loop, returning the user to the prompt.
            if self._has_broken_tool_call(response.tool_calls):
                ui.display.report_error(
                    "A tool call had truncated (unparseable) arguments, so it "
                    "was NOT executed. That turn was discarded -- please try again."
                )
                self.client.rollback_last_turn()
                break

            try:
                self._handle_tool_calls(response.tool_calls)
            except KeyboardInterrupt:
                # Ctrl+C while a tool is running leaves the same half-finished
                # reply (tool_calls whose results were never appended); drop it
                # before the interrupt returns the user to the prompt.
                self.client.rollback_last_turn()
                raise

    @staticmethod
    def _has_broken_tool_call(tool_calls: list[ToolCall]) -> bool:
        """Return True if any tool call has unparseable (truncated) arguments."""
        return any(tc.parse_error is not None for tc in tool_calls)

    @staticmethod
    def _format_tool_arguments(arguments: dict[str, object]) -> list[str]:
        """Format tool-call arguments as the indented lines shown under ``Args:``.

        Short values stay on a single ``[name] value`` line so simple calls read
        at a glance. Values that span several lines (``code`` being the obvious
        case) get a ``[name]`` label of their own and are reproduced verbatim
        underneath, so the code keeps the indentation it was written with
        instead of being mangled into one long line.
        """
        lines: list[str] = []
        for name, value in arguments.items():
            text = value if isinstance(value, str) else str(value)
            # Code is almost always written with a trailing newline; dropping it
            # first keeps the block from ending on an empty indented line and
            # lets one-line code stay on a single ``[name] value`` line.
            value_lines = text.rstrip("\n").split("\n")
            if len(value_lines) == 1:
                lines.append(f"  [{name}] {value_lines[0]}")
                continue
            lines.append(f"  [{name}]")
            lines.extend(f"    {line}" for line in value_lines)
        return lines

    def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Execute tool calls automatically (no user confirmation)."""
        for tc in tool_calls:
            ui.display.print_tool_call(tc.name, self._format_tool_arguments(tc.arguments))

            tool = self.ctx.tool_registry.get(tc.name)
            if tool is None:
                message = f"Tool '{tc.name}' not found"
                ui.display.report_error(message)
                _append_tool_message(self.client.state, message, tc.id)
                continue

            try:
                result = tool.func(**tc.arguments)
            except Exception as e:
                ui.display.report_error(f"Tool '{tc.name}' failed: {e}")
                _append_tool_message(self.client.state, str(e), tc.id)
                continue

            # A tool that returned something other than ExecResult/ToolError is
            # a programming error; normalise it into a ToolError instead of
            # silently treating the value as a successful result.
            normalised: ToolResult = normalise_tool_result(result)
            ui.display.print_tool_result(normalised.as_display_lines())
            _append_tool_message(self.client.state, normalised.as_tool_content(), tc.id)
