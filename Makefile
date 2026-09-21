# ──────────────────────────────────────────────
# llm-cli-py  Makefile
# ──────────────────────────────────────────────

PYTHON      ?= python3

PROJECT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

.PHONY: help format check test install install-dev clean check-all

.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

format:  ## Run ruff format (auto-format code)
	ruff format

check:   ## Run ruff check (linter)
	ruff check

test:   ## Run pytest
	$(PYTHON) -m pytest -v

install:
	$(PYTHON) -m pip install -e "$(PROJECT_DIR)"

install-dev:
	$(PYTHON) -m pip install -e "$(PROJECT_DIR)[dev]"

check-all: format check test

clean:  ## Remove intermediate artifacts
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

