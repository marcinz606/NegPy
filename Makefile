# Variables
VENV = .venv

ifeq ($(OS),Windows_NT)
	PYTHON = $(VENV)\Scripts\python.exe
else
	PYTHON = $(VENV)/bin/python
endif

PYTEST = $(PYTHON) -m pytest
MYPY = $(PYTHON) -m mypy
RUFF = $(PYTHON) -m ruff

# Default target
.PHONY: all
all: lint type test

# Lint checks (ruff)
.PHONY: lint
lint:
	@echo "Running lint checks (ruff)..."
	@$(RUFF) check .

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

# Run the application locally
.PHONY: run
run:
	@echo "Starting NegPy Desktop..."
	@$(PYTHON) desktop.py

# Build the application
.PHONY: build
build:
	@echo "Building NegPy..."
	@$(PYTHON) build.py

# Clean up caches and build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf build
	rm -rf dist
	find . -type d -name "__pycache__" -exec rm -rf {} +