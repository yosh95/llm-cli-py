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
from .commands.openrouter import add_subparser as add_openrouter_subparser
from .commands.openrouter import run_openrouter
from .consts import (
    APPROVAL_MODE_AUTO,
    APPROVAL_MODE_MANUAL,
    APPROVAL_MODE_VERIFIER,
    DEFAULT_API_URL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_URL_FETCH_TIMEOUT,
    DEFAULT_VERIFIER_TIMEOUT,
    ENV_API_KEY,
    ENV_API_URL,
    ENV_MODEL,
    ENV_PROXY_URL,
    ENV_REQUEST_TIMEOUT,
    ENV_VERIFIER_MODEL,
    MAX_TOOL_ITERATIONS,
)
from .models import DataSource
from .providers.llm_api import LlmApiClient
from .session.input_backend import PlainInputBackend
from .session.interactive import run_interactive
from .session.session import ActiveSession, SessionContext
from .tools.python_exec import PYTHON_TOOL_DESCRIPTION, PYTHON_TOOL_SCHEMA, execute_python
from .tools.registry import ToolRegistry
from .tools.web_search import WEB_SEARCH_DESCRIPTION, WEB_SEARCH_SCHEMA, web_search
from .ui import display as ui_display
from .verifier import Verifier


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
    parser.add_argument(
        "--proxy-url",
        help=f"Proxy URL. Overrides {ENV_PROXY_URL} env var. "
        "When set, both LLM API and Brave Search requests go through this proxy. "
        "The proxy handles API key and model injection server-side.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--approval-mode",
        choices=[APPROVAL_MODE_VERIFIER, APPROVAL_MODE_MANUAL, APPROVAL_MODE_AUTO],
        default=APPROVAL_MODE_VERIFIER,
        help=(
            "How tool calls are approved before execution: "
            f"'{APPROVAL_MODE_VERIFIER}' (default) uses the LLM-based verifier; "
            f"'{APPROVAL_MODE_MANUAL}' skips the verifier and asks the human to "
            "approve every tool call (HITL); "
            f"'{APPROVAL_MODE_AUTO}' skips the verifier and auto-approves everything."
        ),
    )
    parser.add_argument(
        "--verifier-model",
        help=f"Model to use for the verifier. Overrides {ENV_VERIFIER_MODEL} env var. "
        "Defaults to the main LLM model if not specified.",
    )
    parser.add_argument(
        "--max-tool-iterations",
        type=int,
        default=MAX_TOOL_ITERATIONS,
        help="Maximum number of tool-call iterations per user request (default: %(default)s)",
    )
    parser.add_argument(
        "--plain-input",
        action="store_true",
        help="Use plain input() instead of prompt_toolkit. "
        "Avoids terminal manipulation (raw mode, alternate screen) that can "
        "conflict with automation harnesses and non-tty stdin. "
        "Also enabled automatically when stdin is not a tty.",
    )

    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Disable model thinking/reasoning for all providers and models (default: enabled).",
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Keep model thinking/reasoning enabled. Overrides "
        "--disable-reasoning and the LLM_CLI_DISABLE_REASONING env var.",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("models", help="List available models from the API")

    add_openrouter_subparser(subparsers)

    return parser


def initialize_tools() -> ToolRegistry:
    """Initialize and register all tools."""
    registry = ToolRegistry()

    registry.register(
        "execute_python",
        PYTHON_TOOL_DESCRIPTION,
        PYTHON_TOOL_SCHEMA,
        execute_python,
    )

    registry.register(
        "web_search",
        WEB_SEARCH_DESCRIPTION,
        WEB_SEARCH_SCHEMA,
        web_search,
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

    # ── Resolve API URL / Proxy URL ──────────────────────────────
    # Priority: 1) --proxy-url, 2) LLM_CLI_PROXY_URL env
    proxy_url = (args.proxy_url or os.environ.get(ENV_PROXY_URL, "")).strip()

    # When proxy is set, it overrides the API URL and Brave Search URL.
    # The proxy handles API key and model injection server-side.
    if proxy_url:
        # Set env var so tools (web_search) can pick it up
        os.environ[ENV_PROXY_URL] = proxy_url
        api_url = proxy_url.rstrip("/")
        api_key = ""  # Proxy injects the API key
        ui_display.report_info(f"Using proxy: {proxy_url}")
    else:
        # Priority: 1) --api-url, 2) LLM_CLI_API_URL env
        api_url = (args.api_url or os.environ.get(ENV_API_URL, "")).strip()
        if not api_url:
            ui_display.report_error(
                f"Neither {ENV_PROXY_URL} nor {ENV_API_URL} is set. "
                "Please set one of them:\n"
                f"  export {ENV_PROXY_URL}=http://<proxy-ip>:8080   (for proxy mode)\n"
                f"  export {ENV_API_URL}=http://localhost:11434/v1   (for direct API access)\n"
                "Or use the corresponding --proxy-url / --api-url CLI flags."
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

    if args.command in ("openrouter", "or", "opr", "ort"):
        run_openrouter(args)
        return

    # ── Resolve model ──────────────────────────────────────────────
    # Priority: 1) -m/--model, 2) LLM_CLI_MODEL env
    # If neither is set, the model is left empty — the proxy will inject
    # the model name server-side via its own LLM_CLI_MODEL env var.
    model = args.model or os.environ.get(ENV_MODEL, "")

    # ── Resolve request timeout ─────────────────────────────────────
    # Priority: 1) --request-timeout flag (already parsed with a default),
    #            2) LLM_CLI_REQUEST_TIMEOUT env var, 3) built-in default.
    request_timeout = args.request_timeout
    env_timeout = os.environ.get(ENV_REQUEST_TIMEOUT, "").strip()
    if env_timeout.isdigit():
        request_timeout = int(env_timeout)

    # ── Initialize tools ───────────────────────────────────────────
    tool_registry = initialize_tools()

    # ── Resolve verifier model ─────────────────────────────────────
    # Priority: 1) --verifier-model, 2) LLM_CLI_VERIFIER_MODEL env, 3) main model
    verifier_model = args.verifier_model or os.environ.get(ENV_VERIFIER_MODEL) or model

    # ── Initialize verifier ─────────────────────────────────────────
    verifier = Verifier(
        api_url=api_url,
        api_key=api_key,
        model=verifier_model,
        timeout=DEFAULT_VERIFIER_TIMEOUT,
    )
    # The verifier is only used when approval_mode == "verifier". For "manual"
    # and "auto" it is left disabled at the session level (see SessionContext).
    if args.approval_mode == APPROVAL_MODE_VERIFIER:
        verifier.set_enabled(True)
        ui_display.report_info("Approval mode: verifier (LLM-based verification).")
    else:
        verifier.set_enabled(False)
        if args.approval_mode == APPROVAL_MODE_MANUAL:
            ui_display.report_info("Approval mode: manual (HITL - you approve every tool call).")
        else:
            ui_display.report_info("Approval mode: auto (all tool calls auto-approved).")

    # ── Resolve reasoning toggle ────────────────────────────────────
    # Priority: --enable-reasoning > --disable-reasoning > env var (default off)
    disable_reasoning: bool | None = None
    if args.enable_reasoning:
        disable_reasoning = False
    elif args.disable_reasoning:
        disable_reasoning = True

    # ── Initialize LLM client and run session ────────────────────────
    with (
        LlmApiClient(
            model=model,
            api_url=api_url,
            api_key=api_key,
            timeout=request_timeout,
            disable_reasoning=disable_reasoning,
        ) as client,
        verifier,
    ):
        # Use plain input when requested or when stdin is not a tty
        # (automation harnesses / piped input). prompt_toolkit manipulates the
        # terminal directly and can leave it in a broken state when driven by
        # an external harness, so fall back to plain input() in that case.
        plain_input = args.plain_input or not sys.stdin.isatty()
        if plain_input:
            ui_display.report_info("Using plain input() (prompt_toolkit disabled).")

        ctx = SessionContext(
            tool_registry=tool_registry,
            verifier=verifier,
            max_tool_iterations=args.max_tool_iterations,
            backend=PlainInputBackend(),
            approval_mode=args.approval_mode,
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
            plain_input=plain_input,
        )


if __name__ == "__main__":
    main()
