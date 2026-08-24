"""Session handler for interactive and one-shot chat."""

from __future__ import annotations

import json
import uuid

from .. import ui
from ..base import LlmClient
from ..models import DataSource, LlmResponse, Message, Role, ToolCall, ToolSchema
from ..tools.registry import ToolRegistry
from ..tools.types import ExecResult, SearchResult, ToolError
from .stream_state import StreamState


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
        self.trace_id = str(uuid.uuid4())

    def process_and_print(self, data: list[DataSource]) -> None:
        """Main processing loop: send to LLM, handle tool calls, display results."""
        if not self.client.state.model:
            ui.display.report_info("No model specified locally.")

        current_data = data

        while True:
            tool_schemas = self.ctx.tool_registry.get_schemas()
            model = self.client.state.model
            display_model = model if model else "LLM"
            ui.display.print_rule()
            print(f"\U0001f914 {display_model} is thinking...")

            stream_state = StreamState()

            try:
                response = self._send_streamed(
                    current_data,
                    tool_schemas,
                    stream_state,
                )
            except Exception as e:
                ui.display.report_error(f"LLM request failed: {e}")
                break
            current_data = []

            self._finalize_streamed(stream_state, response)

            if not response.tool_calls:
                break

            # A tool call whose arguments are truncated/corrupted (JSON did not
            # parse) cannot be executed safely. Print what we already streamed,
            # surface an explicit error, and leave the agent loop so the user
            # gets back to the prompt.
            if self._has_broken_tool_call(response.tool_calls):
                ui.display.report_error(
                    "A tool call had truncated (unparseable) arguments, so it "
                    "was NOT executed. Please try again."
                )
                self._drop_broken_tool_calls_from_history(response.tool_calls)
                break

            self._handle_tool_calls(response.tool_calls)
            current_data = []

    def _send_streamed(
        self,
        data: list[DataSource],
        tool_schemas: list[ToolSchema],
        state: StreamState,
    ) -> LlmResponse:
        """Send a streaming turn, displaying answer deltas live."""

        def on_text(delta: str) -> None:
            if not state.answer_open:
                ui.display.stream_start("\U0001f600 Assistant:")
                state.answer_open = True
            ui.display.stream_text(delta)

        return self.client.send(
            data,
            tool_schemas,
            on_text=on_text,
        )

    def _finalize_streamed(self, state: StreamState, response: LlmResponse) -> None:
        """Close open streaming blocks and show any missed output.

        If a provider produced no deltas, display the accumulated response once here.
        """
        if state.answer_open:
            ui.display.stream_end()
        elif response.text:
            ui.display.print_assistant(response.text)

    @staticmethod
    def _has_broken_tool_call(tool_calls: list[ToolCall]) -> bool:
        """Return True if any tool call has unparseable (truncated) arguments."""
        return any(isinstance(tc.arguments, dict) and "raw" in tc.arguments for tc in tool_calls)

    def _drop_broken_tool_calls_from_history(self, tool_calls: list[ToolCall]) -> None:
        """Remove the just-recorded broken tool calls from conversation history.

        Leaving truncated tool_calls on the last assistant message makes the next
        request carry assistant tool_calls with no matching tool result, which
        some OpenAI-compatible APIs reject with HTTP 400. Drop them so the user
        can simply retry with a clean assistant message.
        """
        broken_ids = {tc.id for tc in tool_calls if isinstance(tc.arguments, dict) and "raw" in tc.arguments}
        for msg in reversed(self.client.state.conversation):
            if msg.role != Role.ASSISTANT or not msg.tool_calls:
                continue
            msg.tool_calls = [tc for tc in msg.tool_calls if tc.get("id") not in broken_ids]
            if not msg.tool_calls:
                msg.tool_calls = None
            break

    def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Execute tool calls automatically (no user confirmation)."""
        for tc in tool_calls:
            ui.display.print_rule()
            print(f"\U0001f680 Executing tool: {tc.name}...")

            tool = self.ctx.tool_registry.get(tc.name)
            if not tool:
                ui.display.report_error(f"Tool '{tc.name}' not found")
                self.client.state.conversation.append(
                    Message(
                        role=Role.TOOL,
                        content=f"Tool '{tc.name}' not found",
                        tool_call_id=tc.id,
                    )
                )
                continue

            try:
                result = tool.func(**tc.arguments)

                if isinstance(result, ToolError):
                    content_str = result.error
                elif isinstance(result, (ExecResult, SearchResult)):
                    content_str = json.dumps(result.to_dict(), ensure_ascii=False)
                else:
                    content_str = json.dumps(result, ensure_ascii=False) if result is not None else ""

                self.client.state.conversation.append(
                    Message(
                        role=Role.TOOL,
                        content=content_str,
                        tool_call_id=tc.id,
                    )
                )

            except Exception as e:
                ui.display.report_error(f"Tool '{tc.name}' failed: {e}")
                self.client.state.conversation.append(
                    Message(
                        role=Role.TOOL,
                        content=str(e),
                        tool_call_id=tc.id,
                    )
                )
