"""Resolve ``-s/--source`` arguments into :class:`DataSource` values.

Split out of ``main`` so the file/URL/text dispatch (and its failure
reporting) is a plain function that can be tested without a terminal, a
network or an argv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import requests

from .consts import DEFAULT_URL_FETCH_TIMEOUT
from .models import DataSource
from .utils.http import get_with_detail


class WarnSink(Protocol):
    """Minimal sink for the warnings emitted while resolving sources."""

    def __call__(self, message: str) -> None: ...


class FileReader(Protocol):
    """Reads ``src`` (an existing path) into a data source, or ``None`` on failure."""

    def __call__(self, path: Path, src: str, warn: WarnSink) -> DataSource | None: ...


class UrlFetcher(Protocol):
    """Fetches ``src`` (an http/https URL) into a data source, or ``None`` on failure."""

    def __call__(self, src: str, warn: WarnSink) -> DataSource | None: ...


def _read_file(path: Path, src: str, warn: WarnSink) -> DataSource | None:
    """Read ``src`` as a file, reporting and swallowing read failures."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        warn(f"Failed to read file '{src}': {e}")
        return None
    return DataSource(text=content, source_type="file")


def _fetch_url(src: str, warn: WarnSink) -> DataSource | None:
    """Fetch ``src`` over HTTP(S), reporting and swallowing fetch failures."""
    try:
        with requests.Session() as session:
            resp = get_with_detail(session, src, DEFAULT_URL_FETCH_TIMEOUT)
            text = resp.text
    except requests.RequestException as e:
        warn(f"Failed to fetch URL '{src}': {e}")
        return None
    return DataSource(text=text, source_type="url")


def resolve_sources(
    sources: list[str],
    *,
    warn: WarnSink,
    file_reader: FileReader = _read_file,
    url_fetcher: UrlFetcher = _fetch_url,
) -> list[DataSource]:
    """Turn raw ``-s`` values into data sources.

    Each value is interpreted in order: an existing file (read as UTF-8), an
    ``http(s)://`` URL (fetched), and otherwise the literal text itself. File
    and URL failures are reported through ``warn`` and skipped, so one bad
    argument never aborts the run.

    ``file_reader``/``url_fetcher`` are injectable to keep the dispatch itself
    testable without touching the filesystem or the network.
    """
    resolved: list[DataSource] = []
    for src in sources:
        path = Path(src)
        if path.exists() and path.is_file():
            source = file_reader(path, src, warn)
        elif src.startswith(("http://", "https://")):
            source = url_fetcher(src, warn)
        else:
            source = DataSource(text=src, source_type="text")
        if source is not None:
            resolved.append(source)
    return resolved
