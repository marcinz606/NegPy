# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
PYTEST = $(PYTHON) -m pytest
MYPY = $(PYTHON) -m mypy
FLAKE8 = $(PYTHON) -m flake8
RUFF = $(PYTHON) -m ruff

# Default target
.PHONY: all
all: lint type test

# Style checks (flake8)
.PHONY: lint
lint:
	@echo "Running style checks (flake8)..."
	@$(FLAKE8) .

# Type checks (mypy)
.PHONY: type
type:
	@echo "Running type checks (mypy)..."
	@$(MYPY) .

# Unit tests (pytest)
.PHONY: test
test:
	@echo "Running unit tests (pytest)..."
	@$(PYTEST) tests/

# Auto-format and fix (ruff)
.PHONY: format
format:
	@echo "Running ruff format and fix..."
	@$(RUFF) format .
	@$(RUFF) check --fix .

# Clean up caches
.PHONY: clean
clean:
	@echo "Cleaning up caches..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
