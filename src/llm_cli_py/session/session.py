"""Session handler for interactive and one-shot chat."""

from __future__ import annotations

import json
import uuid

from .. import ui
from ..base import LlmClient
from ..consts import (
    APPROVAL_MODE_AUTO,
    APPROVAL_MODE_MANUAL,
    APPROVAL_MODE_VERIFIER,
    MAX_TOOL_ITERATIONS,
)
from ..models import DataSource, LlmResponse, Message, Role, ToolCall, ToolSchema
from ..tools.registry import ToolRegistry
from ..tools.types import ExecResult, SearchResult, ToolError
from ..verifier import Verifier
from .input_backend import InputBackend, PlainInputBackend
from .stream_state import StreamState


class SessionContext:
    """Context holding shared resources for a session."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        verifier: Verifier | None = None,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
        backend: InputBackend | None = None,
        approval_mode: str = APPROVAL_MODE_VERIFIER,
    ) -> None:
        self.tool_registry = tool_registry
        self.verifier = verifier
        self.max_tool_iterations = max_tool_iterations
        # Approval strategy for tool calls:
        #   "verifier" -> use the LLM-based verifier (default).
        #   "manual"   -> no verifier; prompt the human for every tool call (HITL).
        #   "auto"     -> no verifier; auto-approve every tool call.
        if approval_mode not in (APPROVAL_MODE_VERIFIER, APPROVAL_MODE_MANUAL, APPROVAL_MODE_AUTO):
            approval_mode = APPROVAL_MODE_VERIFIER
        self.approval_mode = approval_mode
        # Backend used for interactive confirmations (e.g. verifier override).
        # Defaults to a plain input() so that automation / non-tty runs never
        # trigger prompt_toolkit's terminal manipulation unexpectedly.
        self.backend = backend if backend is not None else PlainInputBackend()


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

    @property
    def token_usage(self) -> tuple[int, int, int]:
        """Return (prompt_tokens, completion_tokens, total_tokens)."""
        usage = self.client.state.token_usage
        return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens

    def process_and_print(self, data: list[DataSource]) -> None:
        """Main processing loop: send to LLM, handle tool calls, display results."""
        if not self.client.state.model:
            ui.display.report_info("No model specified locally. The proxy will inject the model server-side.")

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
        iteration = 0

        while True:
            iteration += 1
            if iteration > self.ctx.max_tool_iterations:
                ui.display.report_error(
                    f"Reached maximum tool-call iterations ({self.ctx.max_tool_iterations}). "
                    "Stopping to prevent infinite loops."
                )
                break

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
            # gets back to the prompt. (Previous behaviour silently re-requested
            # non-streaming, which desynced the screen from history.)
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
            stream=True,
            on_text=on_text,
        )

    def _finalize_streamed(self, state: StreamState, response: LlmResponse) -> None:
        """Close open streaming blocks and show any missed (non-streamed) output.

        A provider may return a non-streaming fallback (e.g. re-requested tool
        calls) or an error body that produced no deltas; display it once here.
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
        """Execute tool calls (with verification and optional user confirmation)."""
        for tc in tool_calls:
            ui.display.print_tool_call(tc.name, tc.arguments)

            approved = True
            reason = ""

            if self.ctx.approval_mode == APPROVAL_MODE_MANUAL:
                # No verifier: ask the human to approve every tool call (HITL).
                ui.display.print_rule()
                raw_input_text = self.ctx.backend.prompt(
                    f"\U0001f91a Approve tool call '{tc.name}'? [Y/n] "
                ).strip()
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

                    ctx_messages = [
                        {"role": m.role.value, "content": m.content}
                        for m in self.client.state.conversation[-5:]
                    ]

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
                        raw_input_text = self.ctx.backend.prompt(
                            f"\U0001f91a Verifier rejected '{tc.name}'. Execute anyway? [Y/n or feedback] "
                        ).strip()
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
