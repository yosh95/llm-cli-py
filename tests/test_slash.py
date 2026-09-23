"""Tests for the single slash-command declaration table."""

from llm_cli_py.session.slash import SLASH_COMMANDS, find_spec, help_rows

EXPECTED_ALIASES = {
    "help": {"help", "h"},
    "quit": {"quit", "q", "exit"},
    "info": {"info", "i"},
    "dump": {"dump"},
}


def test_every_declared_command_is_findable_by_all_its_aliases() -> None:
    for spec in SLASH_COMMANDS:
        for name in spec.names:
            assert find_spec(name) is spec


def test_aliases_match_the_documented_set() -> None:
    for spec in SLASH_COMMANDS:
        assert set(spec.names) == EXPECTED_ALIASES[spec.names[0]]


def test_unknown_command_is_none() -> None:
    assert find_spec("nope") is None


def test_canonical_and_help_rows_are_rendered_from_the_same_source() -> None:
    rows = dict(help_rows())
    assert rows["/help, /h"] == "Show this help message"
    assert set(rows) == {", ".join("/" + n for n in spec.names) for spec in SLASH_COMMANDS}
