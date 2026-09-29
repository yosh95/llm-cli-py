"""OpenAI-compatible chat API client (POST {base_url}/chat/completions)."""

from __future__ import annotations

import json
from typing import Any

from ..base import LlmClient
from ..consts import DEFAULT_REQUEST_TIMEOUT
from ..models import LlmResponse, Message, Role, ToolCall, ToolCallPayload, ToolSchema
from ..utils.http import post_json


class LlmApiClient(LlmClient):
    """Client for an OpenAI-compatible ``/chat/completions`` endpoint.

    One request per turn: the full answer (or the tool calls) is requested in a
    single non-streaming call and displayed when it arrives. No HTTP session is
    kept between turns -- each request opens and closes its own connection.
    """

    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: str | None = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        *,
        system_prompt: str = "",
    ) -> None:
        super().__init__(model, system_prompt=system_prompt)
        # The one endpoint this client talks to is fixed at construction, so the
        # base URL is never needed again (and a trailing "/" is harmless).
        self._endpoint = api_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key or ""
        self._timeout = timeout

    def _build_messages(self) -> list[dict[str, object]]:
        """Build the messages array for the API request.

        The conversation history is the one and only source of the request body:
        only fields the API understands are emitted.
        """
        messages: list[dict[str, object]] = []

        # The system prompt is seeded into the conversation at client
        # initialization (see LlmClient.__init__) and simply replayed here
        # with the rest of the history.
        for msg in self._state.conversation:
            entry: dict[str, object] = {"role": msg.role.value}

            if msg.role == Role.TOOL:
                entry["tool_call_id"] = msg.tool_call_id or ""
                entry["content"] = msg.content
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                entry["content"] = msg.content or None
                entry["tool_calls"] = msg.tool_calls
            else:
                entry["content"] = msg.content

            messages.append(entry)

        return messages

    def _build_request(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[ToolSchema],
    ) -> dict[str, object]:
        """Build the request body for the API call.

        Args:
            messages: The message array to send.
            tool_schemas: Tool schemas to advertise to the model.
        """
        body: dict[str, object] = {
            "model": self._state.model,
            "messages": messages,
        }

        if tool_schemas:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": ts.name,
                        "description": ts.description,
                        "parameters": ts.parameters,
                    },
                }
                for ts in tool_schemas
            ]

        return body

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Convert the response's ``tool_calls`` array into :class:`ToolCall` values.

        ``arguments`` arrives as a JSON-encoded string. A call whose string does
        not parse (broken or truncated by the provider) is returned with
        ``parse_error`` set and empty ``arguments``: it must never be executed,
        and the caller is responsible for surfacing the failure.
        """
        tool_calls: list[ToolCall] = []
        for index, call in enumerate(raw_calls):
            fn = call.get("function") or {}
            args_raw = fn.get("arguments") or ""
            parse_error: str | None = None
            try:
                arguments: dict[str, Any] = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                arguments = {}
                parse_error = args_raw
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{index}",
                    name=fn.get("name") or "",
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
        return tool_calls

    def _parse_response(self, payload: dict[str, Any]) -> LlmResponse:
        """Turn a chat-completion response body into an :class:`LlmResponse`."""
        choices = payload.get("choices") or []
        if not choices:
            return LlmResponse()
        message = choices[0].get("message") or {}
        content = message.get("content")
        return LlmResponse(
            text=content or None,
            tool_calls=self._parse_tool_calls(message.get("tool_calls") or []),
        )

    def _build_user_message(self, prompt: str) -> None:
        """Append the user turn carrying ``prompt`` verbatim to the conversation."""
        if prompt.strip():
            self.remember(Message(role=Role.USER, content=prompt.strip()))

    def _record_assistant(self, result: LlmResponse) -> None:
        """Append the assistant response (text/tool_calls) to the history."""
        if not (result.text or result.tool_calls):
            return
        tool_calls_data: list[ToolCallPayload] | None = (
            [
                ToolCallPayload(
                    id=tc.id,
                    type="function",
                    function={
                        "name": tc.name,
                        # OpenAI-compatible APIs require arguments to be a
                        # JSON-encoded string, not a nested object, when the
                        # assistant's tool call is replayed back in later requests.
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                )
                for tc in result.tool_calls
            ]
            if result.tool_calls
            else None
        )

        self.remember(
            Message(
                role=Role.ASSISTANT,
                content=result.text or "",
                tool_calls=tool_calls_data,
            )
        )

    def send(
        self,
        prompt: str,
        tool_schemas: list[ToolSchema],
    ) -> LlmResponse:
        """Send one chat request to the OpenAI-compatible ``/chat/completions`` endpoint.

        Args:
            prompt: The user's prompt text for this turn, sent verbatim.
            tool_schemas: Tool schemas to advertise (always sent, since this CLI
                enables ``execute_python`` by default).

        Returns:
            An :class:`LlmResponse`. If a tool call's ``arguments`` string does
            not parse as JSON, its ``ToolCall.parse_error`` is set and
            ``arguments`` is empty; the caller surfaces the failure.
        """
        self._build_user_message(prompt)

        messages = self._build_messages()
        body = self._build_request(messages, tool_schemas)
        payload = post_json(
            self._endpoint,
            body,
            self._timeout,
            api_key=self._api_key,
        )

        result = self._parse_response(payload)
        self._record_assistant(result)
        return result
