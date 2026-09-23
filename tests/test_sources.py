"""Tests for resolving ``-s/--source`` values (files, URLs, literal text)."""

from __future__ import annotations

from pathlib import Path

from llm_cli_py.models import DataSource
from llm_cli_py.sources import _fetch_url, _read_file, resolve_sources


def _warn_collector() -> tuple[list[str], object]:
    warnings: list[str] = []

    def warn(message: str) -> None:
        warnings.append(message)

    return warnings, warn


def _fake_reader(path: Path, src: str, warn) -> DataSource | None:  # noqa: ARG001
    """A file reader that reports failure instead of touching the filesystem."""
    warn(f"Failed to read file '{src}': boom")
    return None


def test_plain_text_and_file_and_url_are_dispatched(tmp_path: Path) -> None:
    file = tmp_path / "input.txt"
    file.write_text("file body", encoding="utf-8")
    warnings, warn = _warn_collector()

    def fake_fetcher(src: str, warn) -> DataSource | None:  # noqa: ARG001
        return DataSource(text="page", source_type="url")

    sources = resolve_sources(
        [str(file), "https://example.com/x", "just text"],
        warn=warn,
        url_fetcher=fake_fetcher,
    )

    assert [(s.source_type, s.text) for s in sources] == [
        ("file", "file body"),
        ("url", "page"),
        ("text", "just text"),
    ]
    assert warnings == []


def test_file_read_failure_is_warned_and_skipped(tmp_path: Path) -> None:
    file = tmp_path / "input.txt"
    file.write_text("body", encoding="utf-8")
    warnings, warn = _warn_collector()

    sources = resolve_sources([str(file)], warn=warn, file_reader=_fake_reader)

    assert sources == []
    assert warnings == [f"Failed to read file '{file}': boom"]


def test_url_fetch_failure_is_warned_and_skipped() -> None:
    warnings, warn = _warn_collector()

    def failing_fetcher(src: str, warn) -> DataSource | None:
        warn(f"Failed to fetch URL '{src}': boom")
        return None

    sources = resolve_sources(["https://example.com"], warn=warn, url_fetcher=failing_fetcher)

    assert sources == []
    assert warnings == ["Failed to fetch URL 'https://example.com': boom"]


def test_empty_input_resolves_to_nothing() -> None:
    warnings, warn = _warn_collector()
    assert resolve_sources([], warn=warn) == []
    assert warnings == []


def test_read_file_reads_utf8(tmp_path: Path) -> None:
    file = tmp_path / "uni.txt"
    file.write_text("こんにちは", encoding="utf-8")
    _warnings, warn = _warn_collector()
    source = _read_file(file, str(file), warn)
    assert source == DataSource(text="こんにちは", source_type="file")


def test_fetch_url_reports_the_provider_error_body(monkeypatch) -> None:
    """A failing fetch surfaces the provider body through the shared HTTP helper."""
    from llm_cli_py.utils import http as http_mod

    class _Resp:
        status_code = 404
        reason = "Not Found"
        text = '{"error": {"message": "no such page"}}'

    monkeypatch.setattr(http_mod.requests.Session, "get", lambda *_a, **_k: _Resp())
    warnings, warn = _warn_collector()

    assert _fetch_url("https://example.com/missing", warn) is None
    assert "404" in warnings[0]
    assert "no such page" in warnings[0]
