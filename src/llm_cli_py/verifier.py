"""LLM-based verifier for tool call safety.

Uses the same API endpoint/key/model as the main LLM,
but sends requests without tools so the LLM acts purely
as a verifier with just the conversation context.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import requests

from .consts import DEFAULT_VERIFIER_TIMEOUT
from .models import ToolCall
from .providers.llm_api import reasoning_disable_params, should_disable_reasoning
from .utils.http import post_with_retries

VERIFIER_SYSTEM_PROMPT = (
    "You are a security verifier. Your task is to examine tool calls "
    "proposed by an AI assistant and determine if they are safe.\n\n"
    "You will receive:\n"
    "1. The conversation context (recent messages)\n"
    "2. A proposed tool call (function name and arguments)\n\n"
    "Respond with a JSON object containing:\n"
    '- "approved": true or false\n'
    '- "reason": A short explanation of your decision\n\n'
    "Approve ONLY if ALL of the following are true:\n"
    "- The operation does NOT modify files, data, or system state\n"
    "- The operation does NOT read or write sensitive/personal information\n"
    "- The operation does NOT send data to external services, "
    "EXCEPT for simple web searches where the query does not contain "
    "sensitive or personal information\n"
    "- The operation is read-only and has no side effects\n\n"
    "If ANY of these conditions are violated, set approved to false.\n"
    "When in doubt, set approved to false.\n"
    "If approved is false, the user will be prompted to confirm manually, "
    "so it is safe to reject anything uncertain."
    "The reply MUST be a single raw JSON object and nothing else. "
    "Do NOT wrap it in Markdown code fences (```json ... ```) and do NOT add "
    "any text, greetings, or commentary before or after the JSON."
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from an LLM response, tolerating common noise."""
    if not text:
        return None

    candidate = text.strip()

    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        slice_ = candidate[start : end + 1]
        try:
            parsed = json.loads(slice_)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    return None


class Verifier:
    """LLM-based verifier for tool call safety.

    Uses the same API endpoint/key/model as the main LLM,
    but sends requests WITHOUT tools in the request so the LLM
    acts purely as a verifier with just the conversation context.
    """

    def __init__(
        self,
        api_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout: int = DEFAULT_VERIFIER_TIMEOUT,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._enabled = True
        self._session = requests.Session()
        if self._api_key:
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
            )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> Verifier:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def is_configured(self) -> bool:
        # If no model is set, the proxy is expected to inject it server-side.
        return bool(self._api_url)

    @property
    def model(self) -> str:
        return self._model

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def _request(self, messages: list[dict[str, Any]], *, stream: bool) -> Any:
        """POST a chat-completions request to the verifier endpoint.

        NO tools parameter is sent - the verifier works without tools.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        # Disable thinking/reasoning for the verifier too (provider-aware).
        if should_disable_reasoning():
            body.update(reasoning_disable_params(self._api_url))

        return post_with_retries(
            self._session,
            f"{self._api_url}/chat/completions",
            body,
            self._timeout,
            stream=stream,
        )

    @staticmethod
    def _read_content(resp: Any) -> str:
        """Extract the assistant content from a non-streaming response."""
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])

    def _consume_stream(
        self,
        resp: Any,
        on_reasoning: Callable[[str], None] | None,
        on_content: Callable[[str], None] | None,
    ) -> str:
        """Consume an SSE token stream, surfacing reasoning/content deltas.

        Returns the fully accumulated assistant content (used to parse the
        verifier's JSON verdict after the stream completes).
        """
        content_parts: list[str] = []
        for raw_line_bytes in resp.iter_lines():
            if not raw_line_bytes:
                continue
            raw_line = raw_line_bytes.decode("utf-8", errors="replace")
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

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            # Reasoning trace: providers disagree on the field name.
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning and on_reasoning:
                on_reasoning(reasoning)

            content = delta.get("content")
            if content:
                content_parts.append(content)
                if on_content:
                    on_content(content)

        return "".join(content_parts)

    def verify(
        self,
        tool_call: ToolCall,
        conversation_context: list[dict[str, Any]],
        on_reasoning: Callable[[str], None] | None = None,
        on_content: Callable[[str], None] | None = None,
    ) -> tuple[bool, str]:
        """Verify a tool call. Returns (approved, reason).

        When ``on_reasoning`` or ``on_content`` is provided, the verifier
        request is streamed (SSE) and each reasoning / content delta is
        surfaced live through those callbacks, mirroring the main LLM path.
        If no callback is given, a plain non-streaming request is used.
        """
        if not self._enabled:
            return True, "Verifier disabled"

        if not self.is_configured:
            return False, "Verifier is not configured. Use --disable-verifier."

        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        ]

        for msg in conversation_context[-5:]:
            messages.append(msg)

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Please verify this tool call:\n\n"
                    f"Tool: {tool_call.name}\n"
                    f"Arguments: {json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)}\n\n"
                    "Respond with a JSON object containing "
                    '"approved" (boolean) and "reason" (string).'
                ),
            }
        )

        try:
            # Stream when callbacks want live output; any call provides an
            # immediate non-streaming fallback so verification always finishes.
            stream = on_reasoning is not None or on_content is not None
            resp = self._request(messages, stream=stream)
            if stream:
                content = self._consume_stream(resp, on_reasoning, on_content)
                if not content:
                    resp = self._request(messages, stream=False)
                    content = self._read_content(resp)
            else:
                content = self._read_content(resp)

            result = _extract_json_object(content)
            if result is None:
                return (
                    False,
                    "Verifier response was not valid JSON. Please confirm manually.",
                )
            approved = result.get("approved", False)
            reason = result.get("reason", "No reason provided")
            return bool(approved), str(reason)

        except requests.exceptions.Timeout:
            return (
                False,
                f"Verifier did not respond within {self._timeout} seconds. Please confirm manually.",
            )
        except requests.exceptions.RequestException as exc:
            return (
                False,
                f"Verifier request failed ({exc}). Please confirm manually.",
            )
        except Exception as exc:
            return (
                False,
                f"Verifier could not complete verification ({exc}). Please confirm manually.",
            )
