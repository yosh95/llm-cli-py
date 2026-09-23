"""Models subcommand - list available models from the API."""

from __future__ import annotations

import requests

from ..consts import DEFAULT_MODEL_FETCH_TIMEOUT
from ..ui import display as ui_display
from ..utils.http import get_with_detail


def _model_name(entry: dict[str, object]) -> str:
    """Return the display name of one model entry (OpenAI or Ollama shape)."""
    name = entry.get("id") or entry.get("name") or ""
    return str(name)


def run_models(api_url: str, api_key: str) -> None:
    """Fetch and display available models."""
    api_url = api_url.rstrip("/")

    ui_display.report_info("Fetching available models...")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with requests.Session() as session:
            resp = get_with_detail(
                session,
                f"{api_url}/models",
                DEFAULT_MODEL_FETCH_TIMEOUT,
                headers=headers,
            )
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
        names = sorted(_model_name(m) for m in raw_list)
        for name in names:
            print(f"  {name}")
        print()
        ui_display.report_info(f"Total: {len(names)} models")

    except requests.RequestException as e:
        ui_display.report_error(f"Failed to fetch models: {e}")
    except (ValueError, TypeError, AttributeError) as e:
        ui_display.report_error(f"Unexpected response from API: {e}")
