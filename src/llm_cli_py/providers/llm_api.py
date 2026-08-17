"""OpenAI-compatible chat API client implementation (POST {api_url}/chat/completions)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date
from typing import Any

import requests

from ..base import LlmClient
from ..consts import DEFAULT_REQUEST_TIMEOUT
from ..models import DataSource, LlmResponse, Message, Role, ToolCall, ToolSchema
from ..utils.http import post_with_retries


def _get_default_system_prompt() -> str:
    """Generate the default system prompt with today's actual date injected."""
    today = date.today()
    return (
        f"Today's actual date is: {today.isoformat()}. "
        "When the user asks for current information, use today's date as reference."
    )


def _detect_provider(api_url: str) -> str:
    """Classify an API base URL into a known provider family.

    Used to decide how to express "thinking off" on an OpenAI-compatible
    ``/chat/completions`` endpoint, since providers disagree on the field.
    """
    url = (api_url or "").lower()
    if "openrouter.ai" in url:
        return "openrouter"
    if "api.openai.com" in url:
        return "openai"
    if "localhost" in url or "127.0.0.1" in url or "::1" in url or "11434" in url or "ollama.com" in url:
        return "ollama"
    return "generic"


def provider_thinking_off_payload(api_url: str) -> tuple[dict[str, Any], bool]:
    """Return ``(extra_request_fields, supported)`` to request "thinking off".

    - OpenRouter normalises thinking across upstreams via its unified
      ``reasoning: {"effort": "none"}`` — fully disables thinking there.
    - Modern Ollama maps ``reasoning_effort: "none"`` on its OpenAI-compatible
      endpoint to ``think: false`` (it does *not* accept ``think`` on
      ``/v1/chat/completions`` nor the ``minimal`` level). Older Ollama simply
      ignores the field, which is harmless.
    - OpenAI itself cannot fully disable o-series reasoning; ``low`` merely
      reduces it. Anything else would 400.
    - Unknown/generic endpoints get no field so we never corrupt the request;
      ``supported=False`` lets the caller warn the user.
    """
    provider = _detect_provider(api_url)
    if provider == "openrouter":
        return {"reasoning": {"effort": "none"}}, True
    if provider == "ollama":
        return {"reasoning_effort": "none"}, True
    if provider == "openai":
        # Reduces but does not fully disable thinking on reasoning models.
        return {"reasoning_effort": "low"}, True
    return {}, False


def _get_system_prompt() -> str:
    """Return the system prompt for every request.

    Uses the ``SYSTEM_PROMPT`` environment variable if set,
    otherwise generates a default prompt with today's date.
    """
    env_prompt = os.environ.get("SYSTEM_PROMPT")
    if env_prompt is not None:
        return env_prompt
    return _get_default_system_prompt()


class LlmApiClient(LlmClient):
    """Client for OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: str | None = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        thinking: bool = True,
    ) -> None:
        super().__init__(model)
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout
        # Whether the model is allowed to emit reasoning/thinking tokens.
        # When False, a provider-appropriate "thinking off" parameter is added
        # to every request so slow thinking traces (and their extra tokens) are
        # skipped where the provider supports it.
        self._thinking = thinking
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
        messages.append({"role": "system", "content": sys_prompt})

        for msg in self._state.conversation:
            entry: dict[str, Any] = {"role": msg.role.value}

            if msg.role == Role.TOOL:
                entry["tool_call_id"] = msg.tool_call_id or ""
                entry["content"] = msg.content
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                entry["content"] = msg.content or None
                entry["tool_calls"] = msg.tool_calls
                # DeepSeek V4 requires the assistant reasoning trace to be
                # round-tripped alongside a tool call, or the next request
                # fails with HTTP 400. Include it under the provider-agnostic
                # "reasoning" key so both DeepSeek/Ollama and OpenRouter work.
                if msg.reasoning:
                    entry["reasoning"] = msg.reasoning
            else:
                entry["content"] = msg.content

            messages.append(entry)

        return messages

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[ToolSchema],
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request body for the API call.

        Args:
            messages: The message array to send.
            tool_schemas: Tool schemas to advertise to the model.
            stream: When True, request a Server-Sent Events (SSE) token stream.
        """
        body: dict[str, Any] = {
            "model": self._state.model,
            "messages": messages,
            "stream": stream,
        }

        if tool_schemas:
            tools = []
            for ts in tool_schemas:
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

        if not self._thinking:
            fields, _ = provider_thinking_off_payload(self._api_url)
            body.update(fields)

        return body

    def _parse_response(self, response_data: dict[str, Any]) -> LlmResponse:
        """Parse an OpenAI-compatible ``/chat/completions`` response into an LlmResponse."""
        choices = response_data.get("choices") or []
        if not choices:
            return LlmResponse()

        choice = choices[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")

        text = message.get("content") or None
        reasoning = message.get("reasoning") or None
        tool_calls_raw = message.get("tool_calls") or []

        tool_calls = []
        for i, tc in enumerate(tool_calls_raw):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    name=func.get("name", ""),
                    arguments=args,
                )
            )

        usage = response_data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        if prompt_tokens or completion_tokens:
            self._state.token_usage.prompt_tokens += prompt_tokens
            self._state.token_usage.completion_tokens += completion_tokens
            self._state.token_usage.total_tokens += prompt_tokens + completion_tokens

        return LlmResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning=reasoning,
        )

    def _parse_stream_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Parse a single streaming ``choices[0].delta`` chunk.

        Returns a dict with the delta fields (``content``, ``reasoning``,
        ``tool_calls``) plus ``finish_reason``. Both ``reasoning_content``
        (DeepSeek/Ollama) and ``reasoning`` (OpenRouter) delta field names are
        recognised, and accumulated by the caller.
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
        # Reasoning trace: providers disagree on the field name.
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if reasoning:
            parsed["reasoning"] = reasoning
        tc = delta.get("tool_calls")
        if tc:
            parsed["tool_calls"] = tc
        parsed["finish_reason"] = choice.get("finish_reason")
        return parsed

    def _parse_stream_response(
        self,
        response: requests.Response,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        """Consume an SSE stream from ``requests.Response``.

        - Text deltas are printed via ``on_text`` as they arrive (live).
        - Reasoning deltas are printed via ``on_reasoning`` as they arrive.
        - Tool-call arguments are buffered per ``index`` and only executed
          after the whole call is complete. If a buffered ``arguments`` string
          does not parse as JSON (broken/malformed chunk), a ``ToolCall`` with
          ``{"raw": ...}`` arguments is returned so the caller can fall back.

        Returns an :class:`LlmResponse` with the fully accumulated result.
        """
        reasoning_parts: list[str] = []
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
                finish_reason = delta["finish_reason"]

            content = delta.get("content")
            if content:
                text_parts.append(content)
                if on_text:
                    on_text(content)

            reasoning = delta.get("reasoning")
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_reasoning:
                    on_reasoning(reasoning)

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
                # caller can fall back (e.g. re-request non-streaming).
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
            reasoning="\n".join(reasoning_parts) or None,
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
                reasoning=result.reasoning,
            )
        )

    def send(
        self,
        data: list[DataSource],
        tool_schemas: list[ToolSchema],
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        """Send a chat request to the OpenAI-compatible ``/chat/completions`` endpoint.

        Args:
            data: User input sources for this turn.
            tool_schemas: Tool schemas to advertise (always sent, since this CLI
                enables ``web_search`` and ``execute_python`` by default).
            stream: When True, consume the SSE token stream and surface text /
                reasoning deltas live via ``on_text`` / ``on_reasoning``.
            on_text: Optional callback invoked with each text delta (streaming).
            on_reasoning: Optional callback invoked with each reasoning delta.

        Returns:
            An :class:`LlmResponse`. In streaming mode, tool-call arguments are
            buffered until complete. If a streamed tool call does not parse as
            JSON (broken chunks), a :class:`ToolCall` whose ``arguments`` is
            ``{"raw": ...}`` is returned as-is; the caller is responsible for
            surfacing the failure (no silent non-streaming fallback).
        """
        self._append_user_messages(data)

        messages = self._build_messages()
        body = self._build_request(messages, tool_schemas, stream=stream)

        resp = post_with_retries(
            self._session,
            self._api_url + "/chat/completions",
            body,
            self._timeout,
            stream=stream,
        )

        if stream:
            result = self._parse_stream_response(
                resp,
                on_text=on_text,
                on_reasoning=on_reasoning,
            )
            # A streamed tool call whose JSON arguments were truncated mid-flight
            # is returned as-is (a ToolCall with {"raw": ...}). We deliberately do
            # NOT silently re-request non-streaming here: doing so replaced the
            # already-displayed streamed text with a regenerated response, making
            # the answer look cut off and desyncing the screen from conversation
            # history. The caller decides how to surface a broken tool call.
        else:
            result = self._parse_response(resp.json())

        self._record_assistant(result)
        return result
