# ──────────────────────────────────────────────
# llm-cli-py  Makefile
# ──────────────────────────────────────────────
#
# Everything goes through the standard library `venv` + `pip` inside .venv,
# so no external package manager is required.
#
#   - python:  PYTHON selects the interpreter used to create .venv
#              (override with `make install PYTHON=/usr/bin/python3.13`).
#   - ruff / pytest: both are installed into .venv by `make install-dev`,
#                    so run that once before `make check` / `make test`.
#
# Note: pip updates a package only when it reinstalls it, so `make
# install-dev` reinstalls the project itself every time; that is what
# refreshes the editable install under .venv/lib/.../llm_cli_py.

PYTHON      ?= python3
VENV        := .venv
VENV_PYTHON := $(VENV)/bin/python
RUFF        := $(VENV)/bin/ruff

# Absolute path of this project. Used by install-global so the tool keeps
# working after `make clean` (which removes .venv) and from any cwd.
PROJECT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

.PHONY: help format check test install install-dev install-all install-global uninstall-global clean clean-all check-all venv

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv:  ## Create/update .venv with the stdlib venv module
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

format:  ## Run ruff format (auto-format code)
	$(RUFF) format

check:   ## Run ruff check (linter)
	$(RUFF) check

test:   ## Run pytest
	$(VENV_PYTHON) -m pytest -v

install: venv  ## Install the CLI into .venv (does NOT survive `make clean`)
	$(VENV_PYTHON) -m pip install -e "$(PROJECT_DIR)"

install-dev: venv  ## Install the CLI + dev tools (pytest, ruff) into .venv
	$(VENV_PYTHON) -m pip install -e "$(PROJECT_DIR)[dev]"

install-all: install-dev  ## Alias of install-dev (everything is in the dev extra)

install-global:  ## Install the CLI with pipx (own venv, survives `make clean`)
	@echo "Installing editable CLI from $(PROJECT_DIR) with pipx ..."
	pipx install --force -e "$(PROJECT_DIR)"
	@echo
	@echo "Done. The command 'llm-cli-py' is now independent of $(PROJECT_DIR)/.venv,"
	@echo "so 'make clean' no longer breaks it (edits in src/ still take effect immediately)."
	@echo "If it is not on PATH yet, run 'pipx ensurepath' and restart your shell."

uninstall-global:  ## Remove the CLI installed by install-global (pipx)
	pipx uninstall llm-cli-py

check-all: format check test  ## Run all checks: format → lint → test

clean:  ## Remove intermediate artifacts (keeps .venv, so the CLI stays usable)
	@echo "Removing __pycache__ directories..."
	find . -type d -name __pycache__ -not -path './$(VENV)/*' -exec rm -rf {} +
	@echo "Removing tool caches..."
	rm -rf .pytest_cache .ruff_cache
	@echo "Removing build artifacts..."
	rm -rf dist/ build/
	@echo "Removing egg-info..."
	rm -rf src/*.egg-info/
	@echo "Kept $(VENV) (use 'make clean-all' to remove it)."
	@echo "Done."

clean-all: clean  ## Remove everything including .venv (needs `make install` again)
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Done. Re-create it with 'make install' / 'make install-dev'."
