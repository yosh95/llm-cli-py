"""Merge the ``-s/--source`` arguments and the positional prompt into one prompt.

``-s`` is a pure text channel: nothing is read from disk and nothing is fetched
over the network. This module also owns the single text-assembly rule for the
whole CLI (``-s`` values one per line, trailing prompt words on one line), so
"the argument is the prompt, verbatim" is testable without an argv or a
terminal.
"""

from __future__ import annotations


def build_prompt(sources: list[str], prompt_words: list[str]) -> str:
    """Return the one prompt text built from ``-s`` values and positional words.

    ``-s a -s b hello world`` becomes ``"a\nb\nhello world"``: each explicit
    source stays on its own line, while the trailing words are one sentence.
    Surrounding whitespace is dropped so an empty result means "no prompt".
    """
    return "\n".join([*sources, " ".join(prompt_words)]).strip()
