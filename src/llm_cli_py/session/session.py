"""Session handler for interactive and one-shot chat."""

from __future__ import annotations

import json
import uuid

from prompt_toolkit.shortcuts import prompt as pt_prompt

from .. import ui
from ..base import LlmClient
from ..consts import MAX_TOOL_ITERATIONS
from ..models import DataSource, Message, Role, ToolCall
from ..tools.registry import ToolRegistry
from ..tools.types import ExecResult, SearchResult, ToolError
from ..verifier import Verifier


class SessionContext:
    """Context holding shared resources for a session."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        verifier: Verifier | None = None,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        self.tool_registry = tool_registry
        self.verifier = verifier
        self.max_tool_iterations = max_tool_iterations


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

    def switch_client(self, new_client: LlmClient) -> None:
        """Switch to a different LLM client, preserving conversation."""
        old_state = self.client.state
        new_client.state.conversation = old_state.conversation
        self.client = new_client

    @property
    def token_usage(self) -> tuple[int, int, int]:
        """Return (prompt_tokens, completion_tokens, total_tokens)."""
        usage = self.client.state.token_usage
        return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens

    def process_and_print(self, data: list[DataSource]) -> None:
        """Main processing loop: send to LLM, handle tool calls, display results."""
        if not self.client.state.model:
            ui.display.report_info("No model specified locally. The proxy will inject the model server-side.")

        if self.ctx.verifier and self.ctx.verifier.enabled and not self.ctx.verifier.is_configured:
            ui.display.report_error(
                "Verifier is enabled but not configured. "
                "Tool calls cannot proceed.\n"
                "  Use --disable-verifier startup flag, or disable it with /v off."
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

            try:
                response = self.client.send(current_data, tool_schemas)
            except Exception as e:
                ui.display.report_error(f"LLM request failed: {e}")
                break
            current_data = []

            if response.text or response.reasoning:
                if response.reasoning:
                    ui.display.print_reasoning(response.reasoning)
                if response.text:
                    ui.display.print_assistant(response.text)

            if not response.tool_calls:
                break

            self._handle_tool_calls(response.tool_calls)
            current_data = []

    def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        """Execute tool calls (with verification and optional user confirmation)."""
        for tc in tool_calls:
            ui.display.print_tool_call(tc.name, tc.arguments)

            if self.ctx.verifier and self.ctx.verifier.enabled:
                verifier_model = self.ctx.verifier.model if self.ctx.verifier.model else "LLM"
                ui.display.print_rule()
                print(f"\U0001f50d {verifier_model} checking...")
                ctx_messages = [
                    {"role": m.role.value, "content": m.content} for m in self.client.state.conversation[-5:]
                ]
                approved, reason = self.ctx.verifier.verify(tc, ctx_messages)
                ui.display.print_rule()
                if not approved:
                    ui.display.report_warning(f"Verifier rejected '{tc.name}': {reason}")

                    raw_input_text = pt_prompt("\u2753 Execute anyway? [Y/n or feedback] ").strip()
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
                    ui.display.report_success(f"Verifier approved '{tc.name}': {reason}")

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
