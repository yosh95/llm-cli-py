"""Tests for prompt history selection in the shared prompt_toolkit session."""

from __future__ import annotations

import pytest

from llm_cli_py.session import prompt as prompt_mod
from llm_cli_py.session.prompt import get_prompt_session


@pytest.fixture(autouse=True)
def _reset_prompt_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the lazily-created shared PromptSession between tests."""
    monkeypatch.setattr(prompt_mod, "_session", None)


class TestPromptHistory:
    """The prompt history uses the file from LLM_CLI_PROMPT_HISTORY_FILE when set,
    otherwise an in-memory history."""

    def test_uses_file_history_when_env_set(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.history import FileHistory

        history_file = tmp_path / "history.log"
        monkeypatch.setenv("LLM_CLI_PROMPT_HISTORY_FILE", str(history_file))

        session = get_prompt_session()

        assert isinstance(session.history, FileHistory)

    def test_uses_memory_history_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prompt_toolkit.history import InMemoryHistory

        monkeypatch.delenv("LLM_CLI_PROMPT_HISTORY_FILE", raising=False)

        session = get_prompt_session()

        assert isinstance(session.history, InMemoryHistory)
