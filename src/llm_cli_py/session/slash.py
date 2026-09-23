"""Single source of truth for slash commands.

The command list is defined once here and consumed by both the interactive
dispatcher and the prompt-toolkit completer, so help text, completion and
behaviour can never drift apart (they used to live in two hand-maintained
dictionaries).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashSpec:
    """Declaration of one slash command.

    Attributes:
        names: All accepted aliases, canonical name first.
        description: One-line description shown by ``/help`` and completion.
    """

    names: tuple[str, ...]
    description: str

    @property
    def canonical(self) -> str:
        """The preferred spelling, e.g. ``/help``."""
        return "/" + self.names[0]


SLASH_COMMANDS: tuple[SlashSpec, ...] = (
    SlashSpec(("help", "h"), "Show this help message"),
    SlashSpec(("quit", "q", "exit"), "Exit the session"),
    SlashSpec(("info", "i"), "Show session info (API URL, model, tools)"),
    SlashSpec(("dump",), "Dump conversation history as TOML to stdout"),
)


def find_spec(cmd: str) -> SlashSpec | None:
    """Return the spec accepting the alias ``cmd`` (without the leading ``/``)."""
    for spec in SLASH_COMMANDS:
        if cmd in spec.names:
            return spec
    return None


def help_rows() -> list[tuple[str, str]]:
    """Return ``(aliases, description)`` rows for the ``/help`` output."""
    return [(", ".join("/" + n for n in spec.names), spec.description) for spec in SLASH_COMMANDS]
