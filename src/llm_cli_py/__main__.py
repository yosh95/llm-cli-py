"""Entry point for ``python -m llm_cli_py``.

The same ``main`` as the ``llm-cli-py`` console script (``[project.scripts]``),
reached without a generated launcher: on Windows the script is a small unsigned
``.exe`` that ``pip`` writes into ``Scripts/``, whereas ``python -m`` starts from
the interpreter's own executable and runs the package as plain source. ``prog``
is fixed in ``build_parser``, so ``--help``/``--version`` read the same either
way.

Guarded, so importing this module never starts a session.
"""

from __future__ import annotations

from .main import main

if __name__ == "__main__":
    main()
