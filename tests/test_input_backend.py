"""Tests for the shared prompt_toolkit session."""

from __future__ import annotations

import pytest

from llm_cli_py.session.prompt import (
    SlashCommandCompleter,
    build_key_bindings,
)


class TestPromptSession:
    """The prompt_toolkit session should be created exactly once."""

    def test_get_prompt_session_returns_same_instance(self) -> None:
        with pytest.MonkeyPatch.context() as mp:
            # Force a fresh module state so we can assert singleness.
            import llm_cli_py.session.prompt as prompt_mod

            mp.setattr(prompt_mod, "_session", None)
            # Replace PromptSession with a fake so the test never touches a
            # real terminal. On Windows, creating a real PromptSession without
            # a console raises NoConsoleScreenBufferError (e.g. in CI or when
            # pytest runs in a subprocess without a TTY).
            mp.setattr(prompt_mod, "PromptSession", _FakePromptSession)
            first = prompt_mod.get_prompt_session()
            second = prompt_mod.get_prompt_session()
            assert first is second
            # The lazy factory must have been called exactly once.
            assert _FakePromptSession.call_count == 1

    def test_build_key_bindings(self) -> None:
        kb = build_key_bindings()
        assert kb is not None

    def test_slash_command_completer(self) -> None:
        completer = SlashCommandCompleter()
        assert completer is not None
        assert "/help" in completer._commands  # noqa: SLF001


class _FakePromptSession:
    """Minimal stand-in for prompt_toolkit.PromptSession."""

    call_count = 0

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        _FakePromptSession.call_count += 1
