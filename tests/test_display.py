"""Tests for the UI display module.

Covers print functions and formatting.
"""

from __future__ import annotations

import pytest

from llm_cli_py.ui.display import (
    print_assistant,
    print_block,
    print_info,
    report_error,
    report_info,
    report_success,
    report_warning,
)


class TestPrintFunctions:
    """Test display print functions."""

    def test_print_block_empty_content(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_block("   ", title="Empty")
        captured = capsys.readouterr()
        # Empty content should not crash
        assert "Empty:" in captured.out

    def test_print_assistant(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_assistant("Hello world")
        captured = capsys.readouterr()
        assert "Assistant" in captured.out
        assert "Hello world" in captured.out

    def test_print_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_info("Model", "gpt-4o")
        captured = capsys.readouterr()
        assert "Model: gpt-4o" in captured.out

    def test_report_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_info("Something happened")
        captured = capsys.readouterr()
        assert "INFO: Something happened" in captured.out

    def test_report_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_success("All good")
        captured = capsys.readouterr()
        assert "All good" in captured.out

    def test_report_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_error("Something broke")
        captured = capsys.readouterr()
        assert "ERROR: Something broke" in captured.out

    def test_report_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        report_warning("Be careful")
        captured = capsys.readouterr()
        assert "Be careful" in captured.out
