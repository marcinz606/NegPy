# Contributing to NegPy

Thank you for your interest in contributing to **NegPy**!

## 🛠️ Development Setup

NegPy requires **Python 3.13+**.

### 1. Python Environment
We use a virtual environment named `.venv` (used by the Makefile).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Locally
You can run the desktop application directly from source:

```bash
make run
```

## 🏗️ Project Structure

The codebase follows a modular architecture:

- `src/domain/`: Core data models, types, and interfaces.
- `src/features/`: Image processing logic implementations (Exposure, Geometry, Lab, etc.).
- `src/infrastructure/`: Low-level system implementations (GPU resources, file loaders).
- `src/kernel/`: Core system services (Logging, Config, caching).
- `src/services/`: High-level orchestration (Rendering engine, Export service).
- `src/desktop/`: PyQt6 UI implementation (View, Controller, Workers).
- `tests/`: Unit and integration tests.

## 📐 Coding Standards

**Always run `make format` before committing.**

### 1. Style & Formatting
- **Ruff**: Used for both linting and formatting.
- **Type Hints**: Required for all new function definitions (`mypy` is enforced). Using `cast` to get around it is frowned upon.
- **Docstrings**: Use clear, concise docstrings for classes and public methods.
- **Style**: Use double quotes for strings, snake_case for variables and functions, and PascalCase for classes.

### 2. Testing
We use `pytest`. New features should include unit tests in the `tests/` directory.

```bash
make test
```

### 3. Workflow (The Makefile)
The `Makefile` is the central source of truth for developer commands:
- `make lint`: Run Flake8 checks.
- `make type`: Run Mypy type checks.
- `make test`: Run all unit tests.
- `make format`: Auto-format code with Ruff.
- `make all`: Run lint, type, and test in sequence.
- `make clean`: Removes cache from tests.


## 📦 Building and Packaging

To build the standalone application for your current OS:

```bash
make dist
```
This will trigger the Python backend build via PyInstaller and then package the Electron app.
