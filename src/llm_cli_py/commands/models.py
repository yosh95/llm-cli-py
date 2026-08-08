"""Models subcommand - list available models from the API."""

from __future__ import annotations

import requests

from ..consts import DEFAULT_MODEL_FETCH_TIMEOUT
from ..ui import display as ui_display


def run_models(api_url: str, api_key: str) -> None:
    """Fetch and display available models."""
    api_url = api_url.rstrip("/")

    ui_display.report_info("Fetching available models...")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(
            f"{api_url}/models",
            headers=headers,
            timeout=DEFAULT_MODEL_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible format: { "data": [{"id": "model-name", ...}, ...] }
        # Also support alternative format: { "models": [{"name": "model-name", ...}, ...] }
        raw_list = data.get("data") or data.get("models") or []
        if not raw_list:
            ui_display.report_info("No models returned from API.")
            return

        print("Available Models")
        print()
        # Sort models alphabetically by name (ascending)
        raw_list.sort(key=lambda x: x.get("id") or x.get("name", ""))
        for m in raw_list:
            name = m.get("id") or m.get("name", "")
            print(f"  {name}")
        print()
        ui_display.report_info(f"Total: {len(raw_list)} models")

    except requests.RequestException as e:
        ui_display.report_error(f"Failed to fetch models: {e}")
    except Exception as e:
        ui_display.report_error(f"Unexpected error: {e}")
