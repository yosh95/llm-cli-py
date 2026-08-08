"""OpenAI-compatible chat API client implementation (POST {api_url}/chat/completions)."""

from __future__ import annotations

import json
import os
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
    ) -> None:
        super().__init__(model)
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout
        self._session = requests.Session()
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
            else:
                entry["content"] = msg.content

            messages.append(entry)

        return messages

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[ToolSchema],
    ) -> dict[str, Any]:
        """Build the request body for the API call."""
        body: dict[str, Any] = {
            "model": self._state.model,
            "messages": messages,
            "stream": False,
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

    def send(
        self,
        data: list[DataSource],
        tool_schemas: list[ToolSchema],
    ) -> LlmResponse:
        """Send a chat request to the OpenAI-compatible ``/chat/completions`` endpoint."""
        # Append user messages from data to conversation state
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

        messages = self._build_messages()
        body = self._build_request(messages, tool_schemas)

        resp = post_with_retries(
            self._session,
            self._api_url + "/chat/completions",
            body,
            self._timeout,
        )
        result = self._parse_response(resp.json())

        if result.text or result.tool_calls:
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

            msg = Message(
                role=Role.ASSISTANT,
                content=result.text or "",
                tool_calls=tool_calls_data,
                reasoning=result.reasoning,
            )
            self._state.conversation.append(msg)

        return result
