# Contributing to NegPy

Thank you for your interest in contributing to **NegPy**!

## 🛠️ Development Setup

NegPy requires **Python 3.13+** and **Node.js** (for desktop builds).

### 1. Python Environment
We use a virtual environment named `.venv` (dot is important due to Makefile).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Desktop Environment (Electron)
If you are working on the desktop wrapper:

```bash
npm install
```

### 3. Docker (Optional)
For quicker testing (frontend accessible via `http://localhost:8501`):

```bash
make run-app
```

## 🏗️ Project Structure

The codebase is organized into layers:

- `src/domain/`: Core data models and interfaces.
- `src/features/`: Implementation of specific image processing logic (Inversion, Lab, etc.).
- `src/infrastructure/`: Low-level system implementations (I/O, Loaders).
- `src/kernel/`: Core system services (Logging, Config, Numba caching).
- `src/services/`: Higher-level orchestration logic (Rendering engine, Export service).
- `src/ui/`: Streamlit components and layouts.
- `desktop/`: Electron main process and PyInstaller build scripts.

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
