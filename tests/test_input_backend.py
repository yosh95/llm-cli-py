"""Tests for the input backend abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_cli_py.session.input_backend import (
    InputBackend,
    PlainInputBackend,
    PromptToolkitBackend,
    create_backend,
)
from llm_cli_py.session.interactive import SlashCommandCompleter, _build_key_bindings


class TestPlainInputBackend:
    """Plain input() should not touch prompt_toolkit and read a line."""

    def test_prompt_reads_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: list[str] = []

        def fake_input(text: str) -> str:
            received.append(text)
            return "hello"

        monkeypatch.setattr("builtins.input", fake_input)
        backend = PlainInputBackend()
        result = backend.prompt("> ")
        assert result == "hello"
        # The prompt text is forwarded to input()
        assert received == ["> "]

    def test_is_input_backend(self) -> None:
        backend = PlainInputBackend()
        assert isinstance(backend, InputBackend)


class TestPromptToolkitBackend:
    """prompt_toolkit backend requires a history path."""

    def test_create_backend_plain(self, tmp_path: Path) -> None:
        backend = create_backend(tmp_path / "hist.txt", plain=True)
        assert isinstance(backend, PlainInputBackend)

    def test_create_backend_prompt_toolkit(self, tmp_path: Path) -> None:
        backend = create_backend(
            tmp_path / "hist.txt",
            plain=False,
            completer=SlashCommandCompleter(),
            bindings=_build_key_bindings(),
        )
        assert isinstance(backend, PromptToolkitBackend)

    def test_create_backend_prompt_toolkit_creates_history(self, tmp_path: Path) -> None:
        hist = tmp_path / "sub" / "hist.txt"
        create_backend(hist, plain=False)
        assert hist.exists()
