"""Main CLI entry point for llm-cli-py.

This module is the composition root: it reads the environment, builds the
client/session/tool graph and hands control to the interactive loop. Component
configuration (model, API URL/key, system prompt) is passed in explicitly
rather than read from ``os.environ`` inside the components themselves.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys

from . import __version__
from .consts import (
    DEFAULT_API_URL,
    DEFAULT_REQUEST_TIMEOUT,
    ENV_API_KEY,
    ENV_API_URL,
    ENV_LOG_LEVEL,
    ENV_MODEL,
    ENV_SYSTEM_PROMPT,
)
from .providers.llm_api import LlmApiClient
from .session.interactive import run_interactive
from .session.session import ActiveSession, SessionContext
from .sources import build_prompt
from .tools import PYTHON_TOOL_DESCRIPTION, PYTHON_TOOL_SCHEMA, ToolRegistry, execute_python
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
        help="Prompt text, passed to the model verbatim. Can be repeated.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        metavar="PROMPT",
        help="Prompt text (equivalent to a trailing -s argument).",
    )
    parser.add_argument(
        "-m",
        "--model",
        help=f"Model to use (e.g., gpt-4o). Also read from {ENV_MODEL} env var.",
    )
    parser.add_argument(
        "--api-url",
        help=f"API URL. Overrides {ENV_API_URL} env var (e.g. {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--api-key",
        help=f"API key. Overrides {ENV_API_KEY} env var.",
    )
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


def _configure_logging() -> None:
    """Apply the ``LOG_LEVEL`` environment switch."""
    log_level_str = os.environ.get(ENV_LOG_LEVEL, "").upper()
    if log_level_str:
        numeric_level = getattr(logging, log_level_str, None)
        if numeric_level is not None:
            logging.basicConfig()
            logging.getLogger().setLevel(numeric_level)


def _configure_streams() -> None:
    """Never die on an unencodable character: replace instead of raising.

    Keeps the CLI usable on consoles whose encoding (e.g. cp932 on Japanese
    Windows) cannot represent every emitted character, and removes any
    dependency on PYTHONIOENCODING being set in the environment.
    """
    for stream in (sys.stdout, sys.stderr):
        # getattr keeps this working on streams that are not TextIOWrapper
        # (e.g. captured/embedded streams) without a type error.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(Exception):
            reconfigure(errors="replace")


def main() -> None:
    """Main entry point."""
    _configure_streams()
    _configure_logging()

    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve API URL / API key ─────────────────────────────────
    # Priority: 1) --api-url, 2) LLM_CLI_API_URL env
    api_url = (args.api_url or os.environ.get(ENV_API_URL, "")).strip()
    if not api_url:
        ui_display.report_error(
            f"{ENV_API_URL} is not set. "
            "Please set it:\n"
            f"  export {ENV_API_URL}={DEFAULT_API_URL}\n"
            "Or use the --api-url CLI flag."
        )
        sys.exit(1)

    # API key is optional (e.g., local Ollama instances do not require one)
    # Priority: 1) --api-key, 2) LLM_CLI_API_KEY env
    api_key = (args.api_key or os.environ.get(ENV_API_KEY, "")).strip()

    # ── Resolve model, system prompt and prompt text ───────────────
    # Priority for the model: 1) -m/--model, 2) LLM_CLI_MODEL env
    model = args.model or os.environ.get(ENV_MODEL, "")
    # Read once here (startup snapshot) and pass down; see LlmClient.__init__.
    system_prompt = os.environ.get(ENV_SYSTEM_PROMPT, "")
    # -s values and trailing words are all prompt text, sent verbatim.
    prompt = build_prompt(args.sources, args.prompt)

    # ── Initialize tools ───────────────────────────────────────────
    tool_registry = initialize_tools()

    # ── Initialize LLM client and run session ────────────────────────
    # Built-in request timeout is used (no CLI flag / env override).
    client = LlmApiClient(
        model=model,
        api_url=api_url,
        api_key=api_key,
        timeout=DEFAULT_REQUEST_TIMEOUT,
        system_prompt=system_prompt,
    )
    ctx = SessionContext(
        tool_registry=tool_registry,
    )
    session = ActiveSession(client, ctx)
    run_interactive(session, prompt)


if __name__ == "__main__":
    main()
