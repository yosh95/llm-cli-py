"""Tests for tool result serialisation, parsing and rendering."""

from llm_cli_py.tools.types import (
    ExecResult,
    ToolError,
    normalise_tool_result,
    parse_tool_result,
    render_tool_result,
)


class TestRoundTrip:
    def test_exec_result_round_trips(self) -> None:
        result = ExecResult(stdout="out", stderr="err", exit_code=2)
        assert parse_tool_result(_dumped(result)) == result

    def test_tool_error_round_trips(self) -> None:
        error = ToolError(error="boom")
        assert parse_tool_result(_dumped(error)) == error


def _dumped(result: ExecResult | ToolError) -> str:
    import json

    return json.dumps(result.to_dict())


class TestParse:
    def test_plain_text_is_none(self) -> None:
        assert parse_tool_result("just a string") is None

    def test_non_object_json_is_none(self) -> None:
        assert parse_tool_result("[1, 2, 3]") is None

    def test_unrelated_object_is_none(self) -> None:
        assert parse_tool_result('{"unrelated": 1}') is None


class TestNormalise:
    def test_valid_results_pass_through(self) -> None:
        result = ExecResult(stdout="ok")
        assert normalise_tool_result(result) is result
        error = ToolError(error="x")
        assert normalise_tool_result(error) is error

    def test_string_result_becomes_a_tool_error(self) -> None:
        normalised = normalise_tool_result("error happened")
        assert isinstance(normalised, ToolError)
        assert "str" in normalised.error


class TestRender:
    def test_empty_result_is_labelled(self) -> None:
        assert render_tool_result("") == ["(empty result)"]

    def test_exec_result_lines(self) -> None:
        lines = render_tool_result(_dumped(ExecResult(stdout="hello")))
        assert lines == ["Exit code: 0", "[stdout]", "  hello"]

    def test_tool_error_lines(self) -> None:
        assert render_tool_result(_dumped(ToolError(error="boom"))) == ["Error: boom"]

    def test_plain_text_is_shown_verbatim(self) -> None:
        assert render_tool_result("not json") == ["not json"]

    def test_text_with_an_error_word_is_not_mistaken_for_a_tool_error(self) -> None:
        """Regression: a bare string containing "error" must not be read as an error dict."""
        assert render_tool_result("an error happened") == ["an error happened"]
