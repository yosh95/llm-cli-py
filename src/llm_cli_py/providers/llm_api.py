"""OpenAI-compatible chat API client implementation (POST {api_url}/chat/completions)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date
from typing import Any

import requests

from ..base import LlmClient
from ..consts import DEFAULT_REQUEST_TIMEOUT, ENV_DISABLE_REASONING
from ..models import DataSource, LlmResponse, Message, Role, ToolCall, ToolSchema
from ..utils.http import post_with_retries


def _get_default_system_prompt() -> str:
    """Generate the default system prompt with today's actual date injected."""
    today = date.today()
    return (
        f"Today's actual date is: {today.isoformat()}. "
        "When the user asks for current information, use today's date as reference."
    )


def should_disable_reasoning(override: bool | None = None) -> bool:
    """Return whether model thinking/reasoning should be disabled.

    Priority: explicit ``override`` > ``LLM_CLI_DISABLE_REASONING`` env var
    (default "1" = disabled). Returns True when reasoning should be turned off.
    """
    if override is not None:
        return override
    val = os.environ.get(ENV_DISABLE_REASONING, "1").strip().lower()
    return val not in ("0", "false", "no", "off", "")


def reasoning_disable_params(api_url: str) -> dict[str, Any]:
    """Return provider-appropriate request params to disable thinking/reasoning.

    The provider is detected from the API base URL so the correct parameter is
    sent regardless of which model is used:

    - OpenRouter: ``reasoning: {"enabled": false, "effort": "none"}``
    - Ollama / Ollama Cloud (OpenAI-compatible ``/v1`` endpoint):
      ``reasoning_effort: "none"`` (plus ``think: false`` for native API)
    - Generic OpenAI-compatible endpoint: ``reasoning_effort: "none"``
    """
    url = (api_url or "").lower()
    if "openrouter" in url:
        return {"reasoning": {"enabled": False, "effort": "none"}}
    if "ollama" in url or "11434" in url:
        return {"reasoning_effort": "none", "think": False}
    return {"reasoning_effort": "none"}


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
        disable_reasoning: bool | None = None,
    ) -> None:
        super().__init__(model)
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout
        self._disable_reasoning = disable_reasoning
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

        # Disable thinking/reasoning by default (provider-aware). Can be turned
        # back on via --enable-reasoning / LLM_CLI_DISABLE_REASONING=0.
        if should_disable_reasoning(self._disable_reasoning):
            body.update(reasoning_disable_params(self._api_url))

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

    def _post_non_streaming(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[ToolSchema],
    ) -> LlmResponse:
        """Send a non-streaming request and parse the complete response."""
        body = self._build_request(messages, tool_schemas, stream=False)
        resp = post_with_retries(
            self._session,
            self._api_url + "/chat/completions",
            body,
            self._timeout,
        )
        return self._parse_response(resp.json())

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
            buffered until complete; if a buffered tool-call does not parse as
            JSON (broken chunks), the call is transparently re-requested in
            non-streaming mode to obtain a well-formed tool call.
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
            # If any streamed tool call has broken/truncated JSON arguments,
            # re-request that turn non-streaming to get a well-formed call.
            # This is a fallback for LLMs that occasionally split chunks badly.
            if result.tool_calls and any(
                isinstance(tc.arguments, dict) and "raw" in tc.arguments for tc in result.tool_calls
            ):
                result = self._post_non_streaming(messages, tool_schemas)
        else:
            result = self._parse_response(resp.json())

        self._record_assistant(result)
        return result
