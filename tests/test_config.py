"""Tests for configuration (env vars)."""

from __future__ import annotations

from pathlib import Path

from llm_cli_py.consts import (
    ENV_API_KEY,
    ENV_API_URL,
    ENV_INCLUDE_REASONING,
    ENV_MODEL,
    ENV_PROXY_URL,
    ENV_VERIFIER_MODEL,
    get_base_dir,
    set_base_dir,
)


class TestConsts:
    """Test constants and path functions."""

    def test_get_base_dir_default(self) -> None:
        from llm_cli_py import consts

        consts._BASE_DIR = None
        base = get_base_dir()
        assert base == Path.home() / ".llm-cli-py"

    def test_set_base_dir(self) -> None:
        custom = Path("/tmp/test-llm-cli-py")
        set_base_dir(custom)
        assert get_base_dir() == custom
        set_base_dir(Path.home() / ".llm-cli-py")

    def test_path_functions(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            set_base_dir(Path(tmpdir))
            from llm_cli_py.consts import config_dir, log_dir

            assert config_dir() == Path(tmpdir)
            assert log_dir() == Path(tmpdir) / "logs"

    def test_env_var_names(self) -> None:
        assert ENV_API_KEY == "LLM_CLI_API_KEY"
        assert ENV_API_URL == "LLM_CLI_API_URL"
        assert ENV_MODEL == "LLM_CLI_MODEL"

    def test_verifier_model_env_var(self) -> None:
        assert ENV_VERIFIER_MODEL == "LLM_CLI_VERIFIER_MODEL"

    def test_include_reasoning_env_var(self) -> None:
        assert ENV_INCLUDE_REASONING == "LLM_CLI_INCLUDE_REASONING"

    def test_proxy_url_env_var(self) -> None:
        assert ENV_PROXY_URL == "LLM_CLI_PROXY_URL"

    def test_max_tool_iterations_is_500(self) -> None:
        from llm_cli_py.consts import MAX_TOOL_ITERATIONS

        assert MAX_TOOL_ITERATIONS == 500
