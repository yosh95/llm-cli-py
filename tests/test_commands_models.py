"""Tests for the ``models`` subcommand's response handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from llm_cli_py.commands.models import run_models


def _resp(payload, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def test_openai_style_model_list_is_sorted_and_printed(capsys) -> None:
    payload = {"data": [{"id": "model-b"}, {"id": "model-a"}]}
    with patch("llm_cli_py.commands.models.requests.Session.get", return_value=_resp(payload)):
        run_models("https://api.example.com/v1", "key")

    out = capsys.readouterr().out
    assert out.index("model-a") < out.index("model-b")
    assert "Total: 2 models" in out


def test_ollama_style_models_key_is_supported(capsys) -> None:
    payload = {"models": [{"name": "llama3"}, {"name": "gemma"}]}
    with patch("llm_cli_py.commands.models.requests.Session.get", return_value=_resp(payload)):
        run_models("https://api.example.com/v1", "")

    out = capsys.readouterr().out
    assert "gemma" in out and "llama3" in out
    assert out.index("gemma") < out.index("llama3")


def test_empty_model_list_is_reported(capsys) -> None:
    with patch("llm_cli_py.commands.models.requests.Session.get", return_value=_resp({"data": []})):
        run_models("https://api.example.com/v1", "")

    assert "No models returned from API." in capsys.readouterr().out


def test_http_error_is_reported_with_provider_detail(capsys) -> None:
    resp = _resp({}, status_code=401)
    resp.reason = "Unauthorized"
    resp.text = '{"error": {"message": "bad credentials"}}'
    with patch("llm_cli_py.commands.models.requests.Session.get", return_value=resp):
        run_models("https://api.example.com/v1", "key")

    out = capsys.readouterr().out
    assert "Failed to fetch models" in out
    assert "bad credentials" in out


def test_malformed_body_is_reported_as_unexpected_response(capsys) -> None:
    resp = _resp({"weird": "shape"})
    with patch("llm_cli_py.commands.models.requests.Session.get", return_value=resp):
        run_models("https://api.example.com/v1", "")

    out = capsys.readouterr().out
    assert "No models returned from API." in out or "Unexpected response" in out


def test_request_exception_is_reported(capsys) -> None:
    with patch(
        "llm_cli_py.commands.models.requests.Session.get",
        side_effect=requests.exceptions.ConnectionError("refused"),
    ):
        run_models("https://api.example.com/v1", "")

    assert "Failed to fetch models" in capsys.readouterr().out
