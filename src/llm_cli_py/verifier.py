"""LLM-based verifier for tool call safety.

Uses the same API endpoint/key/model as the main LLM,
but sends requests without tools so the LLM acts purely
as a verifier with just the conversation context.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from .consts import DEFAULT_VERIFIER_TIMEOUT
from .models import ToolCall
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

    def verify(
        self,
        tool_call: ToolCall,
        conversation_context: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Verify a tool call. Returns (approved, reason)."""
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
            resp = post_with_retries(
                self._session,
                f"{self._api_url}/chat/completions",
                {
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    # NO tools parameter - verifier works without tools
                },
                self._timeout,
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

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
