"""Tests for prompt history selection in the shared prompt_toolkit session."""

from __future__ import annotations

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory, InMemoryHistory

from llm_cli_py.session import prompt as prompt_mod
from llm_cli_py.session.prompt import SlashCommandCompleter, build_key_bindings, get_prompt_session


@pytest.fixture(autouse=True)
def _reset_prompt_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompt_mod, "_session", None)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, InMemoryHistory), ("{tmp}/history.log", FileHistory)],
)
def test_history_backend_follows_env(tmp_path, monkeypatch, env_value, expected) -> None:
    monkeypatch.delenv("LLM_CLI_PROMPT_HISTORY_FILE", raising=False)
    if env_value is not None:
        monkeypatch.setenv("LLM_CLI_PROMPT_HISTORY_FILE", env_value.format(tmp=tmp_path))

    assert isinstance(get_prompt_session().history, expected)


def test_session_is_created_once() -> None:
    prompt_mod._session = None
    assert get_prompt_session() is get_prompt_session()


def test_key_bindings_build() -> None:
    assert build_key_bindings() is not None


def test_completer_offers_the_canonical_command_spellings() -> None:
    completer = SlashCommandCompleter()
    completions = list(completer.get_completions(Document("/h"), CompleteEvent()))
    assert "/help" in [c.text for c in completions]


def test_completer_stays_silent_for_plain_text_and_arguments() -> None:
    completer = SlashCommandCompleter()
    assert list(completer.get_completions(Document("hello"), CompleteEvent())) == []
    assert list(completer.get_completions(Document("/help "), CompleteEvent())) == []
