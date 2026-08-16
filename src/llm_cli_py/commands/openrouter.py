"""OpenRouter subcommand - show rankings, credits, and model details from OpenRouter API."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, cast

import requests

from ..consts import DEFAULT_MODEL_FETCH_TIMEOUT
from ..ui import display as ui_display

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
"""Environment variable for the OpenRouter API key."""

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
"""OpenRouter API base URL."""

# Command aliases for the openrouter subcommand.
OPENROUTER_ALIASES = ["or", "opr", "ort"]
"""Short aliases for the `openrouter` subcommand name."""

# Short aliases for the openrouter sub-subcommands.
RANKINGS_ALIASES = ["r", "rank"]
"""Short aliases for the `rankings` subcommand."""

CREDITS_ALIASES = ["c", "credit"]
"""Short aliases for the `credits` subcommand."""

MODEL_ALIASES = ["m"]
"""Short aliases for the `model` subcommand."""


def _get_api_key() -> str:
    """Get the OpenRouter API key from environment or exit with error."""
    key = os.environ.get(OPENROUTER_API_KEY_ENV, "").strip()
    if not key:
        ui_display.report_error(
            f"{OPENROUTER_API_KEY_ENV} is not set. "
            "Please set it to your OpenRouter API key.\n"
            "Get one at: https://openrouter.ai/settings/keys"
        )
        sys.exit(1)
    return key


def _headers() -> dict[str, str]:
    """Build common headers for OpenRouter API requests."""
    return {"Authorization": f"Bearer {_get_api_key()}"}


def _fetch_json(
    url: str, timeout: int = DEFAULT_MODEL_FETCH_TIMEOUT, require_auth: bool = True
) -> dict[str, Any]:
    """Fetch JSON from a URL with common error handling.

    If *require_auth* is True (default), the API key must be set.
    If False, the request is made without authentication (for public endpoints).
    """
    headers = _headers() if require_auth else {}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())
    except requests.RequestException as e:
        ui_display.report_error(f"OpenRouter API request failed: {e}")
        sys.exit(1)


# ── Rankings ─────────────────────────────────────────────────────────


def _fmt_tokens(token_count: int) -> str:
    """Format a token count into a human-readable string with units."""
    if token_count >= 1_000_000_000:
        return f"{token_count / 1_000_000_000:.2f}B"
    elif token_count >= 1_000_000:
        return f"{token_count / 1_000_000:.2f}M"
    elif token_count >= 1_000:
        return f"{token_count / 1_000:.1f}K"
    else:
        return str(token_count)


def show_rankings() -> None:
    """Fetch and display model rankings from OpenRouter."""
    url = f"{OPENROUTER_API_BASE}/datasets/rankings-daily"
    ui_display.report_info("Fetching OpenRouter model rankings...")

    data = _fetch_json(url)
    raw_list = data.get("data") or []
    meta = data.get("meta", {})

    if not raw_list:
        ui_display.report_info("No ranking data returned from OpenRouter.")
        return

    as_of = meta.get("as_of", "unknown")

    print("\nOpenRouter Model Rankings")
    print(f"  As of:  {as_of}")
    print()

    # Group by date
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_list:
        by_date[item["date"]].append(item)

    # Show only the latest date
    latest_date = sorted(by_date.keys(), reverse=True)[0]
    items = by_date[latest_date]
    # Sort by total_tokens descending (other pinned last)
    items.sort(
        key=lambda x: (
            1 if x.get("model_permaslug") == "other" else 0,
            -int(x.get("total_tokens", 0)),
        ),
    )
    # Show top 10
    top_items = items[:10]
    print(f"  [{latest_date}] (top {len(top_items)})")
    for i, item in enumerate(top_items, 1):
        slug = item.get("model_permaslug", "?")
        tokens = int(item.get("total_tokens", 0))
        fmt_tokens = _fmt_tokens(tokens)
        if slug == "other":
            print(f"    {i:2d}. (other models)           {fmt_tokens:>8s} tokens")
        else:
            print(f"    {i:2d}. {slug:<50s} {fmt_tokens:>8s} tokens")
    print()

    ui_display.report_info("Source: OpenRouter (openrouter.ai/rankings)")


# ── Credits ──────────────────────────────────────────────────────────


def show_credits() -> None:
    """Fetch and display OpenRouter credit information."""
    # 1) Credits endpoint
    credits_url = f"{OPENROUTER_API_BASE}/credits"
    ui_display.report_info("Fetching OpenRouter credit info...")

    credits_data = _fetch_json(credits_url)
    creds = credits_data.get("data", {})

    total_credits = float(creds.get("total_credits", 0))
    total_usage = float(creds.get("total_usage", 0))
    balance = total_credits - total_usage

    print()
    print("OpenRouter Credits")
    print(f"  Total purchased:  ${total_credits:.2f}")
    print(f"  Total used:       ${total_usage:.2f}")
    print(f"  Balance:          ${balance:.2f}")

    # 2) Key endpoint (for rate limits / usage breakdown)
    key_url = f"{OPENROUTER_API_BASE}/key"
    try:
        key_data = _fetch_json(key_url)
        kd = key_data.get("data", {})

        label = kd.get("label", "")
        if label:
            print(f"  Key label:        {label}")

        limit = kd.get("limit")
        limit_remaining = kd.get("limit_remaining")
        if limit is not None and limit_remaining is not None:
            print(f"  Key credit limit: ${float(limit):.2f}")
            print(f"  Remaining:        ${float(limit_remaining):.2f}")

        # Daily/weekly/monthly spend
        usage_daily = kd.get("usage_daily")
        usage_weekly = kd.get("usage_weekly")
        usage_monthly = kd.get("usage_monthly")
        if any(v is not None for v in (usage_daily, usage_weekly, usage_monthly)):
            print()
            print("  Spend (current period):")
            if usage_daily is not None:
                print(f"    Daily:   ${float(usage_daily):.2f}")
            if usage_weekly is not None:
                print(f"    Weekly:  ${float(usage_weekly):.2f}")
            if usage_monthly is not None:
                print(f"    Monthly: ${float(usage_monthly):.2f}")

        is_free_tier = kd.get("is_free_tier")
        if is_free_tier is not None:
            print(f"  Free tier:        {is_free_tier}")

    except SystemExit:
        pass  # _fetch_json exits on error, but we already showed credits info
    except Exception as e:
        ui_display.report_warning(f"Could not fetch key details: {e}")

    print()


# ── Model details ────────────────────────────────────────────────────


def _fmt_price(price_str: str | None) -> str:
    """Format a price string (USD per token) to a readable per-million rate."""
    if price_str is None:
        return "N/A"
    try:
        per_token = float(price_str)
        per_million = per_token * 1_000_000
        if per_million == 0:
            return "Free"
        return f"${per_million:.6f}/M tokens"
    except (ValueError, TypeError):
        return str(price_str)


def _fmt_date(timestamp: int | None) -> str:
    """Format a Unix timestamp to a readable date string."""
    if timestamp is None:
        return "N/A"
    try:
        return datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return str(timestamp)


def show_model(model_slug: str) -> None:
    """Fetch and display details for a single OpenRouter model."""
    # The endpoint is public - no auth needed
    url = f"{OPENROUTER_API_BASE}/model/{model_slug}"
    ui_display.report_info(f"Fetching details for model '{model_slug}'...")

    data = _fetch_json(url, require_auth=False)
    model = data.get("data", {})

    if not model or not model.get("id"):
        ui_display.report_error(f"Model '{model_slug}' not found.")
        sys.exit(1)

    print()
    print(f"  Model: {model.get('name', '?')}")
    print(f"  ID:    {model.get('id', '?')}")
    print()

    # ── Description ──
    desc = model.get("description", "")
    if desc:
        print("  Description:")
        # Word-wrap at ~80 chars
        import textwrap

        for line in desc.split("\n"):
            for wrapped in textwrap.wrap(line, width=76):
                print(f"    {wrapped}")
        print()

    # ── Architecture (modality) ──
    arch = model.get("architecture") or {}
    modality = arch.get("modality", "N/A")
    tokenizer = arch.get("tokenizer", "N/A")
    input_mods = arch.get("input_modalities") or []
    output_mods = arch.get("output_modalities") or []
    print(f"  Modality:          {modality}")
    if input_mods:
        print(f"  Input modalities:  {', '.join(input_mods)}")
    if output_mods:
        print(f"  Output modalities: {', '.join(output_mods)}")
    print(f"  Tokenizer:         {tokenizer}")

    # ── Context & Limits ──
    context_length = model.get("context_length")
    if context_length:
        print(f"  Context length:    {context_length:,} tokens")

    top_provider = model.get("top_provider") or {}
    max_completion = top_provider.get("max_completion_tokens")
    if max_completion:
        print(f"  Max completion:    {max_completion:,} tokens")

    per_request = model.get("per_request_limits")
    if per_request and isinstance(per_request, dict):
        prl = per_request.get("prompt_tokens") or per_request.get("max_prompt_tokens")
        if prl:
            print(f"  Max prompt (req):  {prl:,} tokens")

    # ── Pricing ──
    pricing = model.get("pricing") or {}
    print()
    print("  Pricing:")
    print(f"    Prompt:              {_fmt_price(pricing.get('prompt'))}")
    print(f"    Completion:          {_fmt_price(pricing.get('completion'))}")
    if pricing.get("input_cache_read"):
        print(f"    Input cache read:    {_fmt_price(pricing.get('input_cache_read'))}")
    if pricing.get("input_cache_write"):
        print(f"    Input cache write:   {_fmt_price(pricing.get('input_cache_write'))}")
    if pricing.get("web_search"):
        print(f"    Web search:          ${float(pricing['web_search']):.4f} / request")
    if pricing.get("image"):
        print(f"    Image:               ${float(pricing['image']):.4f} / image")

    # Pricing overrides
    overrides = pricing.get("overrides")
    if overrides and isinstance(overrides, list):
        print()
        print("    Pricing overrides:")
        for override in overrides:
            conditions = []
            if override.get("min_prompt_tokens"):
                conditions.append(f"prompt > {override['min_prompt_tokens']:,} tokens")
            utc_start = override.get("utc_start")
            utc_end = override.get("utc_end")
            if utc_start is not None and utc_end is not None:
                conditions.append(f"UTC {utc_start}-{utc_end}")
            cond_str = ", ".join(conditions) if conditions else "special"
            prompt_p = _fmt_price(override.get("prompt"))
            comp_p = _fmt_price(override.get("completion"))
            print(f"      [{cond_str}] prompt={prompt_p}, completion={comp_p}")

    # ── Supported parameters ──
    params = model.get("supported_parameters")
    if params:
        print()
        print(f"  Supported parameters: {', '.join(params)}")

    # ── Knowledge cutoff & Created ──
    cutoff = model.get("knowledge_cutoff")
    if cutoff:
        print(f"  Knowledge cutoff:  {cutoff}")
    created = model.get("created")
    if created:
        print(f"  Created:           {_fmt_date(created)}")

    # ── Benchmarks ──
    benchmarks = model.get("benchmarks")
    if benchmarks:
        print()
        print("  Benchmarks:")
        # Design Arena
        da = benchmarks.get("design_arena")
        if da and isinstance(da, list):
            for entry in da:
                cat = entry.get("category", "?")
                elo = entry.get("elo", "?")
                wr = entry.get("win_rate", "?")
                rank = entry.get("rank", "?")
                print(f"    Design Arena [{cat}]:  ELO={elo},  Win Rate={wr}%,  Rank={rank}")
        # Artificial Analysis
        aa = benchmarks.get("artificial_analysis")
        if aa and isinstance(aa, list):
            for entry in aa:
                name = entry.get("display_name") or entry.get("name", "?")
                ai = entry.get("agentic_index")
                ci = entry.get("coding_index")
                ii = entry.get("intelligence_index")
                print(f"    Artificial Analysis [{name}]:", end="")
                if ai:
                    print(f"  Agentic={ai}", end="")
                if ci:
                    print(f"  Coding={ci}", end="")
                if ii:
                    print(f"  Intelligence={ii}", end="")
                print()

    # ── Top provider info ──
    print()
    print("  Top provider:")
    print(f"    Context length:    {top_provider.get('context_length', 'N/A'):,}")
    print(f"    Max completion:    {top_provider.get('max_completion_tokens', 'N/A')}")
    print(f"    Moderated:         {top_provider.get('is_moderated', 'N/A')}")

    # ── Links ──
    links = model.get("links") or {}
    details_url = links.get("details", "")
    if details_url:
        print(f"  Endpoints:         https://openrouter.ai{details_url}")
    print(f"  OpenRouter page:   https://openrouter.ai/models/{model_slug}")

    print()


# ── Entry point ──────────────────────────────────────────────────────


def run_openrouter(args: argparse.Namespace) -> None:
    """Dispatch to the appropriate subcommand."""
    cmd = args.or_command
    if cmd in ("model", *MODEL_ALIASES):
        if not args.model_slug:
            ui_display.report_error(
                "No model slug specified. Usage: llm-cli-py openrouter model <model-slug>"
            )
            sys.exit(1)
        show_model(args.model_slug)
    elif cmd in ("rankings", *RANKINGS_ALIASES):
        _get_api_key()
        show_rankings()
    elif cmd in ("credits", *CREDITS_ALIASES):
        _get_api_key()
        show_credits()
    else:
        # show both rankings and credits (require API key)
        _get_api_key()
        show_rankings()
        print()
        show_credits()


def add_subparser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Add the 'openrouter' subcommand (with aliases) to the CLI parser."""
    parser = subparsers.add_parser(
        "openrouter",
        aliases=OPENROUTER_ALIASES,
        help="Show OpenRouter rankings, credits, and model details",
        description="Display OpenRouter model rankings, credit information, and model details.",
    )
    parser.add_argument(
        "or_command",
        nargs="?",
        choices=[
            "rankings",
            *RANKINGS_ALIASES,
            "credits",
            *CREDITS_ALIASES,
            "model",
            *MODEL_ALIASES,
        ],
        help=(
            "Subcommand: 'rankings' (model rankings), 'credits' (credit info), "
            "'model' (model details). "
            "Short aliases: r/rank, c/credit, m. "
            "Omit to show both rankings and credits."
        ),
    )
    parser.add_argument(
        "model_slug",
        nargs="?",
        default=None,
        help="Model slug (e.g. 'openai/gpt-4o', 'anthropic/claude-sonnet-5-20260630'). "
        "Required when subcommand is 'model'.",
    )
    parser.set_defaults(func=run_openrouter)
