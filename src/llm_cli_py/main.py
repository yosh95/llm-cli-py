"""Main CLI entry point for llm-cli-py."""

from __future__ import annotations

import argparse
import http.client
import logging
import os
import sys
from pathlib import Path

import requests

from . import __version__
from .consts import (
    DEFAULT_API_URL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_URL_FETCH_TIMEOUT,
    ENV_API_KEY,
    ENV_API_URL,
    ENV_MODEL,
)
from .models import DataSource
from .providers.llm_api import LlmApiClient
from .session.interactive import run_interactive
from .session.session import ActiveSession, SessionContext
from .tools.python_exec import PYTHON_TOOL_DESCRIPTION, PYTHON_TOOL_SCHEMA, execute_python
from .tools.registry import ToolRegistry
from .ui import display as ui_display


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-cli-py",
        description="Unified OpenAI-Compatible CLI for AI Agents (Python Edition)",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "-s",
        "--source",
        action="append",
        dest="sources",
        default=[],
        help="Input sources (text, file paths, URLs). Can be specified multiple times.",
    )
    parser.add_argument(
        "-m",
        "--model",
        help=f"Model to use (e.g., gpt-4o). Also read from {ENV_MODEL} env var.",
    )
    parser.add_argument(
        "--api-url",
        help=f"API URL. Overrides {ENV_API_URL} env var. Default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--api-key",
        help=f"API key. Overrides {ENV_API_KEY} env var.",
    )
    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("models", help="List available models from the API")

    return parser


def initialize_tools() -> ToolRegistry:
    """Initialize and register all tools.

    Returns:
        A ToolRegistry with all built-in tools registered.
    """
    registry = ToolRegistry()

    registry.register(
        "execute_python",
        PYTHON_TOOL_DESCRIPTION,
        PYTHON_TOOL_SCHEMA,
        execute_python,
    )

    return registry


def main() -> None:
    """Main entry point."""
    # ── Logging configuration ─────────────────────────────────────────
    # LOG_LEVEL env var sets the root logger level (e.g., LOG_LEVEL=DEBUG, LOG_LEVEL=INFO)
    # This is a general-purpose control; any library's debug logs will be shown.
    _log_level_str = os.environ.get("LOG_LEVEL", "").upper()
    if _log_level_str:
        _numeric_level = getattr(logging, _log_level_str, None)
        if _numeric_level is not None:
            logging.basicConfig()
            logging.getLogger().setLevel(_numeric_level)

    # DEBUG_HTTP env var specifically enables raw HTTP request/response debugging
    # (sets http.client debuglevel and urllib3 logger to DEBUG)
    if os.environ.get("DEBUG_HTTP", "").lower() in ("1", "true"):
        if not _log_level_str:
            logging.basicConfig()
        http.client.HTTPConnection.debuglevel = 1
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").propagate = True

    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve API URL / API key ─────────────────────────────────
    # Priority: 1) --api-url, 2) LLM_CLI_API_URL env
    api_url = (args.api_url or os.environ.get(ENV_API_URL, "")).strip()
    if not api_url:
        ui_display.report_error(
            f"{ENV_API_URL} is not set. "
            "Please set it:\n"
            f"  export {ENV_API_URL}=http://localhost:11434/v1\n"
            "Or use the --api-url CLI flag."
        )
        sys.exit(1)

    # API key is optional (e.g., local Ollama instances do not require one)
    # Priority: 1) --api-key, 2) LLM_CLI_API_KEY env
    api_key = (args.api_key or os.environ.get(ENV_API_KEY, "")).strip()

    # ── Handle subcommands ─────────────────────────────────────────
    if args.command == "models":
        from .commands.models import run_models

        run_models(api_url, api_key)
        return

    # ── Resolve model ──────────────────────────────────────────────
    # Priority: 1) -m/--model, 2) LLM_CLI_MODEL env
    model = args.model or os.environ.get(ENV_MODEL, "")

    # ── Request timeout ────────────────────────────────────────────
    # Built-in default is used (no CLI flag / env override).
    request_timeout = DEFAULT_REQUEST_TIMEOUT

    # ── Initialize tools ───────────────────────────────────────────
    tool_registry = initialize_tools()

    # ── Initialize LLM client and run session ────────────────────────
    with LlmApiClient(
        model=model,
        api_url=api_url,
        api_key=api_key,
        timeout=request_timeout,
    ) as client:
        ctx = SessionContext(
            tool_registry=tool_registry,
        )
        session = ActiveSession(client, ctx)

        initial_sources: list[DataSource] = []
        for src in args.sources:
            path = Path(src)
            if path.exists() and path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                    initial_sources.append(DataSource(text=content, source_type="file"))
                except Exception as e:
                    ui_display.report_warning(f"Failed to read file '{src}': {e}")
            elif src.startswith(("http://", "https://")):
                try:
                    resp = requests.get(src, timeout=DEFAULT_URL_FETCH_TIMEOUT)
                    resp.raise_for_status()
                    initial_sources.append(DataSource(text=resp.text, source_type="url"))
                except Exception as e:
                    ui_display.report_warning(f"Failed to fetch URL '{src}': {e}")
            else:
                initial_sources.append(DataSource(text=src, source_type="text"))

        run_interactive(
            session,
            initial_sources if initial_sources else None,
        )


if __name__ == "__main__":
    main()
