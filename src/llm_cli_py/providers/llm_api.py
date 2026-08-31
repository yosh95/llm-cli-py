"""OpenAI-compatible chat API client implementation (POST {api_url}/chat/completions)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import requests

from ..base import LlmClient
from ..consts import DEFAULT_REQUEST_TIMEOUT
from ..models import DataSource, LlmResponse, Message, Role, ToolCall, ToolSchema
from ..utils.http import post_with_retries


def _get_system_prompt() -> str:
    """Return the system prompt for every request.

    Reads the ``SYSTEM_PROMPT`` environment variable. When unset, no system
    prompt is sent (no default/date prompt is injected).
    """
    return os.environ.get("SYSTEM_PROMPT", "")


class LlmApiClient(LlmClient):
    """Client for OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: str | None = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        super().__init__(model)
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout
        self._session = requests.Session()
        # Do not reuse keep-alive connections. This CLI makes one sequential chat
        # request per turn (no parallel assets), so keep-alive saves nothing while
        # leaving a stale idle pool connection vulnerable to silent drops in the
        # network path (NAT/proxy/cloud-LB), which makes the first request after a
        # long idle hang until timeout. Close the connection each request instead.
        self._session.headers["Connection"] = "close"
        if self._api_key:
            self._session.headers.update(
                {
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                }
            )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    @property
    def api_url(self) -> str:
        """Return the configured API base URL."""
        return self._api_url

    def __enter__(self) -> LlmApiClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build the messages array for the API request."""
        messages: list[dict[str, Any]] = []

        sys_prompt = _get_system_prompt()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        for msg in self._state.conversation:
            entry: dict[str, Any] = {"role": msg.role.value}

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
        messages: list[dict[str, Any]],
        tool_schemas: list[ToolSchema],
    ) -> dict[str, Any]:
        """Build the request body for the API call.

        Args:
            messages: The message array to send.
            tool_schemas: Tool schemas to advertise to the model.
        """
        body: dict[str, Any] = {
            "model": self._state.model,
            "messages": messages,
            "stream": True,
        }

        if tool_schemas:
            tools = []
            for ts in tool_schemas:
                if ts.server_tool:
                    # Provider-executed server tool (e.g. openrouter:web_search):
                    # executed by the provider server-side. Sent in its minimal
                    # form (``{"type": "openrouter:web_search"}``) so no
                    # function wrapper or JSON schema is sent; the provider
                    # applies its own defaults. Optional parameters are only
                    # attached when configured.
                    server_tool_entry = {"type": ts.name}
                    if ts.parameters:
                        server_tool_entry["parameters"] = ts.parameters
                    tools.append(server_tool_entry)
                else:
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": ts.name,
                                "description": ts.description,
                                "parameters": ts.parameters,
                            },
                        }
                    )
            body["tools"] = tools

        return body

    def _parse_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Parse a single streaming ``choices[0].delta`` chunk.

        Returns a dict with the delta fields (``content``, ``tool_calls``)
        plus ``finish_reason``.
        """
        choices = chunk.get("choices") or []
        if not choices:
            return {}
        choice = choices[0]
        delta = choice.get("delta") or {}
        parsed: dict[str, Any] = {}
        content = delta.get("content")
        if content:
            parsed["content"] = content
        tc = delta.get("tool_calls")
        if tc:
            parsed["tool_calls"] = tc
        parsed["finish_reason"] = choice.get("finish_reason")
        return parsed

    def _parse_stream_response(
        self,
        response: requests.Response,
        on_text: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        """Consume an SSE stream from ``requests.Response``.

        - Text deltas are printed via ``on_text`` as they arrive (live).
        - Tool-call arguments are buffered per ``index`` and only executed
          after the whole call is complete. If a buffered ``arguments`` string
          does not parse as JSON (broken/malformed chunk), a ``ToolCall`` with
          ``{"raw": ...}`` arguments is returned so the caller can fall back.

        Returns an :class:`LlmResponse` with the fully accumulated result.
        """
        text_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None

        for raw_line_bytes in response.iter_lines():
            if not raw_line_bytes:
                continue
            raw_line = raw_line_bytes.decode("utf-8", errors="replace")
            # SSE: each event is "data: <json>" (possibly blank / comment lines)
            if raw_line.startswith("data:"):
                payload = raw_line[len("data:") :].strip()
            elif raw_line.startswith(":"):
                continue  # SSE comment line
            else:
                payload = raw_line

            if not payload or payload == "[DONE]":
                continue

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            delta = self._parse_stream_chunk(chunk)
            if not delta:
                continue

            if delta.get("finish_reason"):
                # Keep the raw finish_reason: "tool_calls" (user-defined tools)
                # and "server_tool_calls" (OpenRouter server tools) both mean
                # "the model asked for a tool and the loop must continue".
                finish_reason = delta["finish_reason"]

            content = delta.get("content")
            if content:
                text_parts.append(content)
                if on_text:
                    on_text(content)

            for tc in delta.get("tool_calls") or []:
                index = tc.get("index", 0)
                entry = tool_calls_map.setdefault(
                    index,
                    {"id": tc.get("id") or "", "name": "", "arguments": ""},
                )
                fn = tc.get("function") or {}
                if tc.get("id"):
                    entry["id"] = tc["id"]
                if fn.get("name"):
                    entry["name"] = fn["name"]
                # arguments stream as JSON fragments -> concatenate
                if fn.get("arguments"):
                    entry["arguments"] += fn["arguments"]

        tool_calls: list[ToolCall] = []
        for index in sorted(tool_calls_map):
            entry = tool_calls_map[index]
            args_raw = entry["arguments"]
            try:
                arguments: Any = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                # Malformed / truncated chunk. Keep the raw string so the
                # caller can surface the failure.
                arguments = {"raw": args_raw}
            tool_calls.append(
                ToolCall(
                    id=entry["id"] or f"call_{index}",
                    name=entry["name"],
                    arguments=arguments,
                )
            )

        result = LlmResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        return result

    def _append_user_messages(self, data: list[DataSource]) -> None:
        """Append user messages from ``data`` to the conversation state."""
        user_content = ""
        for ds in data:
            if ds.source_type == "text":
                user_content += ds.text + "\n"
            elif ds.source_type == "file":
                user_content += "[File content]:\n" + ds.text + "\n"
            elif ds.source_type == "url":
                user_content += "[URL content]:\n" + ds.text + "\n"
        if user_content.strip():
            self._state.conversation.append(Message(role=Role.USER, content=user_content.strip()))

    def _record_assistant(self, result: LlmResponse) -> None:
        """Append the assistant response (text/reasoning/tool_calls) to history."""
        if not (result.text or result.tool_calls):
            return
        tool_calls_data: list[dict[str, object]] | None = None
        if result.tool_calls:
            tool_calls_data = []
            for tc in result.tool_calls:
                tool_calls_data.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            # OpenAI-compatible APIs require arguments to be a
                            # JSON-encoded string, not a nested object, when the
                            # assistant's tool call is replayed back in later requests.
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                )

        self._state.conversation.append(
            Message(
                role=Role.ASSISTANT,
                content=result.text or "",
                tool_calls=tool_calls_data,
            )
        )

    def send(
        self,
        data: list[DataSource],
        tool_schemas: list[ToolSchema],
        on_text: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        """Send a streaming chat request to the OpenAI-compatible ``/chat/completions`` endpoint.

        Args:
            data: User input sources for this turn.
            tool_schemas: Tool schemas to advertise (always sent, since this CLI
                enables ``execute_python`` by default).
            on_text: Optional callback invoked with each text delta (streaming).

        Returns:
            An :class:`LlmResponse`. Tool-call arguments are buffered until
            complete. If a streamed tool call does not parse as JSON (broken
            chunks), a :class:`ToolCall` whose ``arguments`` is ``{"raw": ...}``
            is returned as-is; the caller is responsible for surfacing the
            failure (no silent non-streaming fallback).
        """
        self._append_user_messages(data)

        messages = self._build_messages()
        body = self._build_request(messages, tool_schemas)

        resp = post_with_retries(
            self._session,
            self._api_url + "/chat/completions",
            body,
            self._timeout,
            stream=True,
        )

        result = self._parse_stream_response(
            resp,
            on_text=on_text,
        )

        self._record_assistant(result)
        return result
