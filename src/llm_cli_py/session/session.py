"""Session handler for interactive and one-shot chat."""

from __future__ import annotations

import json

from .. import ui
from ..base import LlmClient
from ..models import ClientState, DataSource, LlmResponse, Message, Role, ToolCall, ToolSchema
from ..tools.registry import ToolRegistry
from ..tools.types import ExecResult, ToolError
from ..utils.timeutil import now_iso
from .stream_state import StreamState


def _append_tool_message(state: ClientState, content: str, tool_call_id: str) -> None:
    """Add a tool result to the conversation and persist the history immediately.

    Tool output is part of the record, so it is timestamped and pushed to the
    chat log right away (same as user/assistant messages).
    """
    state.conversation.append(
        Message(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            timestamp=now_iso(),
        )
    )
    state.notify_changed()


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
            print(f"{display_model} is thinking...")

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
                ui.display.stream_start("Assistant:")
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
        return any(tc.parse_error is not None for tc in tool_calls)

    def _drop_broken_tool_calls_from_history(self, tool_calls: list[ToolCall]) -> None:
        """Remove the just-recorded broken tool calls from conversation history.

        Leaving truncated tool_calls on the last assistant message makes the next
        request carry assistant tool_calls with no matching tool result, which
        some OpenAI-compatible APIs reject with HTTP 400. Drop them so the user
        can simply retry with a clean assistant message.
        """
        broken_ids = {tc.id for tc in tool_calls if tc.parse_error is not None}
        for msg in reversed(self.client.state.conversation):
            if msg.role != Role.ASSISTANT or not msg.tool_calls:
                continue
            msg.tool_calls = [tc for tc in msg.tool_calls if tc.get("id") not in broken_ids]
            if not msg.tool_calls:
                msg.tool_calls = None
            break

    @staticmethod
    def _format_tool_argument(name: str, value: object) -> list[str]:
        """Format one tool-call argument as indented display lines.

        Short values stay on a single ``[name] value`` line so simple calls
        read at a glance. Values that span several lines (``code`` being the
        obvious case) get a ``[name]`` label of their own and are reproduced
        verbatim underneath, so the code keeps the indentation it was written
        with instead of being mangled into one long line. The caller prints
        the ``Args:`` header that opens the block.
        """
        text = value if isinstance(value, str) else str(value)
        # Code is almost always written with a trailing newline; dropping it
        # first keeps the block from ending on an empty indented line and lets
        # one-line code stay on a single ``[name] value`` line.
        lines = text.rstrip("\n").split("\n")
        if len(lines) == 1:
            return [f"  [{name}] {lines[0]}"]

        formatted = [f"  [{name}]"]
        formatted.extend(f"    {line}" for line in lines)
        return formatted

    @classmethod
    def _format_tool_arguments(cls, arguments: dict[str, object]) -> list[str] | None:
        """Format tool-call parameters for terminal display (all shown in full)."""
        if not arguments:
            return None

        lines: list[str] = []
        for name, value in arguments.items():
            lines.extend(cls._format_tool_argument(name, value))
        return lines

    @staticmethod
    def _format_tool_result(content_str: str) -> list[str]:
        """Format tool execution result for terminal display.

        Parses the JSON result and returns a list of display lines.
        """
        if not content_str:
            return ["(empty result)"]

        try:
            data = json.loads(content_str)
        except json.JSONDecodeError:
            return [content_str]

        lines: list[str] = []

        if "error" in data and "stdout" not in data:
            # ToolError
            lines.append(f"Error: {data['error']}")
        elif "stdout" in data:
            # ExecResult
            ec = data.get("exit_code", 0)
            lines.append(f"Exit code: {ec}")
            stdout = str(data.get("stdout", ""))
            if stdout.strip():
                lines.append("[stdout]")
                for line in stdout.rstrip().splitlines():
                    lines.append(f"  {line}")
            stderr_val = str(data.get("stderr", ""))
            if stderr_val.strip():
                lines.append("[stderr]")
                for line in stderr_val.rstrip().splitlines():
                    lines.append(f"  {line}")
        else:
            lines.append(content_str)

        return lines

    def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Execute tool calls automatically (no user confirmation)."""
        for tc in tool_calls:
            tool = self.ctx.tool_registry.get(tc.name)

            ui.display.print_tool_call(
                tc.name,
                self._format_tool_arguments(tc.arguments) or [],
            )

            if not tool:
                ui.display.report_error(f"Tool '{tc.name}' not found")
                _append_tool_message(
                    self.client.state,
                    f"Tool '{tc.name}' not found",
                    tc.id,
                )
                continue

            try:
                result = tool.func(**tc.arguments)

                if isinstance(result, ToolError):
                    content_str = result.error
                elif isinstance(result, ExecResult):
                    content_str = json.dumps(result.to_dict(), ensure_ascii=False)
                else:
                    content_str = json.dumps(result, ensure_ascii=False) if result is not None else ""

                # Display the tool execution result
                result_lines = self._format_tool_result(content_str)
                ui.display.print_tool_result(result_lines)

                _append_tool_message(self.client.state, content_str, tc.id)

            except Exception as e:
                ui.display.report_error(f"Tool '{tc.name}' failed: {e}")
                _append_tool_message(self.client.state, str(e), tc.id)
