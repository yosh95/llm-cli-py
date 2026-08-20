"""Session handler for interactive and one-shot chat."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import tomli_w

from .. import ui
from ..base import LlmClient
from ..consts import (
    APPROVAL_MODE_AUTO,
    APPROVAL_MODE_MANUAL,
    APPROVAL_MODE_VERIFIER,
)
from ..models import DataSource, LlmResponse, Message, Role, ToolCall, ToolSchema
from ..tools.registry import ToolRegistry
from ..tools.types import ExecResult, SearchResult, ToolError
from ..verifier import Verifier
from .prompt import prompt
from .stream_state import StreamState


class SessionContext:
    """Context holding shared resources for a session."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        verifier: Verifier | None = None,
        approval_mode: str = APPROVAL_MODE_VERIFIER,
    ) -> None:
        self.tool_registry = tool_registry
        self.verifier = verifier
        # Approval strategy for tool calls:
        #   "verifier" -> use the LLM-based verifier (default).
        #   "manual"   -> no verifier; prompt the human for every tool call (HITL).
        #   "auto"     -> no verifier; auto-approve every tool call.
        if approval_mode not in (APPROVAL_MODE_VERIFIER, APPROVAL_MODE_MANUAL, APPROVAL_MODE_AUTO):
            approval_mode = APPROVAL_MODE_VERIFIER
        self.approval_mode = approval_mode


class ActiveSession:
    """Manages an active chat session with LLM interaction and tool execution."""

    def __init__(
        self,
        client: LlmClient,
        ctx: SessionContext,
        log_file: str | Path | None = None,
    ) -> None:
        self.client = client
        self.ctx = ctx
        self.trace_id = str(uuid.uuid4())
        # Optional file that receives the LLM conversation history (+ tool
        # call logs) in the same TOML format as /dump.
        # It is overwritten at the end of every turn and every React-loop
        # iteration so that progress survives an abnormal exit.
        self._log_file = Path(log_file).expanduser() if log_file else None

    def process_and_print(self, data: list[DataSource]) -> None:
        """Main processing loop: send to LLM, handle tool calls, display results."""
        if not self.client.state.model:
            ui.display.report_info("No model specified locally.")

        if (
            self.ctx.approval_mode == APPROVAL_MODE_VERIFIER
            and self.ctx.verifier
            and self.ctx.verifier.enabled
            and not self.ctx.verifier.is_configured
        ):
            ui.display.report_error(
                "Verifier is enabled but not configured. "
                "Tool calls cannot proceed.\n"
                "  Use --approval-mode manual/auto to skip the verifier, "
                "or disable it with /v off."
            )
            return

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
            self._write_log()  # capture this React-loop turn

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
                self._write_log()
                break

            self._handle_tool_calls(response.tool_calls)
            self._write_log()  # tool results now recorded
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

    def _write_log(self) -> None:
        """Write the current conversation history to the log file.

        Produces the same TOML content as the ``/dump`` command (role/content).
        The whole file is overwritten on every call so that, in case of an
        abnormal exit at any point, the most recent state is on disk. Writes
        are best-effort: a failure is reported but does not stop the session.
        """
        if not self._log_file:
            return
        data: dict[str, object] = {
            "message": [
                {
                    "role": m.role.value,
                    "content": m.content,
                }
                for m in self.client.state.conversation
            ]
        }
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.write_text(
                tomli_w.dumps(data).replace("\\n", "\n"),
                encoding="utf-8",
            )
        except Exception as e:  # pragma: no cover - defensive
            ui.display.report_warning(f"Failed to write log file '{self._log_file}': {e}")

    def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Execute tool calls (with verification and optional user confirmation)."""
        for tc in tool_calls:
            ui.display.print_tool_call(tc.name, tc.arguments)

            approved = True
            reason = ""

            if self.ctx.approval_mode == APPROVAL_MODE_MANUAL:
                # No verifier: ask the human to approve every tool call (HITL).
                ui.display.print_rule()
                raw_input_text = prompt(f"\U0001f91a Approve tool call '{tc.name}'? [Y/n] ").strip()
                user_confirmation = raw_input_text.lower()
                if user_confirmation in ("", "y", "yes"):
                    ui.display.report_info("User approved tool call (HITL).")
                else:
                    ui.display.report_info("Tool call skipped per user decision.")
                    self.client.state.conversation.append(
                        Message(
                            role=Role.TOOL,
                            content=f"User declined tool call '{tc.name}'.",
                            tool_call_id=tc.id,
                        )
                    )
                    continue

            elif self.ctx.approval_mode == APPROVAL_MODE_AUTO:
                # No verifier: auto-approve everything.
                ui.display.report_info(f"Auto-approved '{tc.name}' (--approval-mode auto).")

            else:  # APPROVAL_MODE_VERIFIER (default)
                if self.ctx.verifier and self.ctx.verifier.enabled:
                    verifier_model = self.ctx.verifier.model if self.ctx.verifier.model else "LLM"
                    ui.display.print_rule()
                    print(f"\U0001f50d {verifier_model} checking...")

                    # Only pass the most relevant context to the verifier:
                    #  1. the assistant's explanation of why it wants to run the
                    #     tool (the tool's purpose), and
                    #  2. the tool call content (function name + arguments).
                    # Earlier history is intentionally omitted to keep the
                    # verifier focused on the current call and its rationale.
                    ctx_messages: list[dict[str, str]] = []
                    for m in reversed(self.client.state.conversation):
                        if m.role != Role.ASSISTANT or not m.tool_calls:
                            continue
                        ctx_messages.append({"role": "assistant", "content": m.content or ""})
                        # Reconstruct the tool call data (OpenAI format:
                        #   name/arguments live under "function").
                        calls = []
                        for call in m.tool_calls:
                            func = call.get("function", {})
                            calls.append(f"{func.get('name', 'unknown')}({func.get('arguments', '')})")
                        ctx_messages.append(
                            {
                                "role": "assistant",
                                "content": "Tool to execute: " + "; ".join(calls),
                            }
                        )
                        break

                    # Stream the verifier's response live, mirroring how the main
                    # LLM turn is displayed.
                    verifier_state = StreamState()

                    def on_content(delta: str, _st: StreamState = verifier_state) -> None:
                        if not _st.answer_open:
                            ui.display.stream_start("Verifier:")
                            _st.answer_open = True
                        ui.display.stream_text(delta)

                    approved, reason = self.ctx.verifier.verify(
                        tc,
                        ctx_messages,
                        on_content=on_content,
                    )

                    if verifier_state.answer_open:
                        ui.display.stream_end()

                    ui.display.print_rule()
                    if not approved:
                        ui.display.report_warning(f"Verifier rejected '{tc.name}'.")
                        raw_input_text = prompt("Execute anyway? [Y/n or feedback] ").strip()
                        user_confirmation = raw_input_text.lower()
                        if user_confirmation in ("", "y", "yes"):
                            ui.display.report_info("User override: executing tool call.")
                        else:
                            ui.display.report_info("Tool call skipped per user decision.")
                            self.client.state.conversation.append(
                                Message(
                                    role=Role.TOOL,
                                    content=f"User declined after verifier rejection: {reason}",
                                    tool_call_id=tc.id,
                                )
                            )
                            if user_confirmation not in ("n", "no"):
                                self.client.state.conversation.append(
                                    Message(
                                        role=Role.USER,
                                        content=(
                                            f"[User feedback on verifier rejection for tool "
                                            f"'{tc.name}'] {raw_input_text}"
                                        ),
                                    )
                                )
                            continue
                    else:
                        ui.display.report_success(f"Verifier approved '{tc.name}'.")

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
                result_str = ui.display.format_tool_result(result)
                # Print tool results to the terminal. Web search results are
                # shortened by format_tool_result (top result + truncated
                # snippet) to avoid cluttering the display, while the full
                # results are still passed to the LLM via conversation history.
                ui.display.print_tool_result(result_str)

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
