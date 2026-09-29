"""Main CLI entry point for llm-cli-py.

This module is the composition root: it reads the environment, builds the
client/session/tool graph and hands control to the interactive loop. Component
configuration (model, API URL/key, system prompt) is passed in explicitly
rather than read from ``os.environ`` inside the components themselves.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

from prompt_toolkit import PromptSession

from . import __version__
from .consts import (
    DEFAULT_API_URL,
    DEFAULT_REQUEST_TIMEOUT,
    ENV_API_KEY,
    ENV_API_URL,
    ENV_LOG_FILE,
    ENV_MODEL,
    ENV_SYSTEM_PROMPT,
)
from .models import ClientState
from .providers.llm_api import LlmApiClient
from .session.interactive import make_prompt_session, run_interactive
from .session.session import ActiveSession, SessionContext
from .session.transcript import ConversationLog
from .sources import build_prompt
from .tools import PYTHON_TOOL_DESCRIPTION, PYTHON_TOOL_SCHEMA, ToolRegistry, execute_python
from .ui import display as ui_display
from .utils.fileio import UNUSABLE_PATH


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


def _prompt_session_or_none() -> PromptSession[str] | None:
    """Build the prompt session, or report why not and do without.

    This is the one step that can fail for reasons outside the CLI's control:
    ``LLM_CLI_PROMPT_HISTORY_FILE`` may name a file whose directory cannot be
    created, or lie in a read-only place. History is a convenience -- a prompt
    without it still works -- so the failure is reported and the caller falls
    back to the default session.
    """
    try:
        return make_prompt_session()
    except UNUSABLE_PATH as e:
        ui_display.report_error(f"Could not open the prompt history file: {e}")
        ui_display.report_info("Continuing in memory; the arrow keys will not outlive this run.")
        return None


def _conversation_log_or_none() -> ConversationLog | None:
    """Build the log named by ``LLM_CLI_LOG_FILE``, or ``None`` when unset.

    Unset is the default and means no file is written at all. When set, the file
    is touched once here, so a path that cannot be written says so at startup
    rather than at the end of the first turn; the log then disables itself and
    the run continues without it.
    """
    configured = os.environ.get(ENV_LOG_FILE, "").strip()
    if not configured:
        return None

    try:
        log = ConversationLog(configured)
        # An empty conversation with no model: this creates the file (and its
        # directory), so a path that cannot be written says so now rather than
        # at the end of the first turn.
        log.save(ClientState())
    except UNUSABLE_PATH as e:
        ui_display.report_error(f"Could not use the conversation log {configured!r}: {e}")
        ui_display.report_info("Continuing without a log.")
        return None

    return None if log.disabled else log


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


def _configure_streams() -> None:
    """Never die on an unencodable character: replace instead of raising.

    Keeps the CLI usable on consoles whose encoding (e.g. cp932 on Japanese
    Windows) cannot represent every emitted character, and removes any
    dependency on PYTHONIOENCODING being set in the environment. Streams that
    cannot be reconfigured (captured/embedded ones have no ``reconfigure``) are
    left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]


def main() -> None:
    """Main entry point."""
    _configure_streams()

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
    # ── Optional conversation log ──────────────────────────────────
    # Unset LLM_CLI_LOG_FILE means no file; when set, the conversation is
    # rewritten there as it grows, tool calls included, so an interrupted run
    # still leaves what had been recorded.
    log = _conversation_log_or_none()
    if log is not None:
        client.observe(log)

    session = ActiveSession(client, ctx)
    run_interactive(session, prompt, _prompt_session_or_none())


if __name__ == "__main__":
    main()
