# ──────────────────────────────────────────────
# llm-cli-py  Makefile
# ──────────────────────────────────────────────
#
# Portable across Termux and normal Linux (Debian/Ubuntu):
#   - ruff:   use system ruff if available (Termux: ruff has no prebuilt
#             wheel and source builds fail without Rust); otherwise fall
#             back to `uv run ruff` (works on Debian/Ubuntu via wheels).
#   - pytest: skip uv re-sync when pytest is already in .venv
#             (Termux: re-sync would try to build ruff and fail);
#             otherwise use `uv run pytest` (installs dev group).

ifeq ($(shell command -v ruff >/dev/null 2>&1 && echo 1 || echo 0),1)
  RUFF := ruff
  SYNC_EXTRA := --no-install-package ruff
else
  RUFF := uv run ruff
  SYNC_EXTRA :=
endif

ifeq ($(shell test -x .venv/bin/pytest && echo 1 || echo 0),1)
  PYTEST := uv run --no-sync pytest
else
  PYTEST := uv run pytest
endif

.PHONY: help format check test clean check-all

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \\
		| sort \\
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

format:  ## Run ruff format (auto-format code)
	$(RUFF) format

check:   ## Run ruff check (linter)
	$(RUFF) check

test:   ## Run pytest
	$(PYTEST) -v

install: ## Run uv sync --no-dev (CLI only)
	uv sync --no-dev

install-dev: ## Run uv sync (CLI + dev tools; skips ruff if system ruff exists)
	uv sync $(SYNC_EXTRA)

install-all: ## Run uv sync (everything; skips ruff if system ruff exists)
	uv sync $(SYNC_EXTRA)

check-all: format check test  ## Run all checks: format → lint → test

clean:  ## Remove all intermediate artifacts (caches, builds, egg-info, venv)
	@echo "Removing __pycache__ directories..."
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	@echo "Removing tool caches..."
	rm -rf .pytest_cache .ruff_cache
	@echo "Removing build artifacts..."
	rm -rf dist/ build/
	@echo "Removing egg-info..."
	rm -rf src/*.egg-info/
	@echo "Removing virtual environment..."
	rm -rf .venv
	@echo "Done."
