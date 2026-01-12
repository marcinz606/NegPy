# Contributing to NegPy

Thank you for your interest in contributing to **NegPy**! This project is an open-source tool dedicated to high-quality film negative processing.

## 📜 Code of Conduct

By participating in this project, you agree to abide by the terms of the **GPL-3.0 License** and maintain a professional, respectful environment.

## 🛠️ Development Setup

NegPy requires **Python 3.13+** and **Node.js** (for desktop builds).

### 1. Python Environment
We use a virtual environment named `.venv`.

```bash
# Create and activate venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Desktop Environment (Electron)
If you are working on the desktop wrapper:

```bash
npm install
```

### 3. Docker (Optional)
For a quick setup without local dependencies:

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

We maintain high code quality through automated checks. **Always run `make format` before committing.**

### 1. Style & Formatting
- **Ruff**: Used for both linting and formatting.
- **Type Hints**: Required for all new function definitions (`mypy` is enforced).
- **Docstrings**: Use clear, concise docstrings for classes and public methods.

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

## 🚀 How to Contribute

1. **Check Issues**: Look for existing issues or open a new one to discuss your idea.
2. **Fork & Branch**: Create a feature branch from `main`.
3. **Implement**: Write your code, following existing patterns.
4. **Verify**: Ensure `make all` passes successfully.
5. **PR**: Open a Pull Request with a clear description of your changes.

## 📦 Building and Packaging

To build the standalone application for your current OS:

```bash
make dist
```
This will trigger the Python backend build via PyInstaller and then package the Electron app.
