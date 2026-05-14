# Contributing to NegPy

Thank you for your interest in contributing to **NegPy**!

## 🛠️ Development Setup

NegPy requires **Python 3.13+**. We use **uv** for environment and dependency management.

### 1. Prerequisites
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

**Scanner support (optional):** NegPy's direct scanner integration uses SANE (libsane).

- **Linux** (Debian/Ubuntu):
  ```bash
  sudo apt install libsane-dev
  ```
  After installation, add your user to the `scanner` group and reload udev:
  ```bash
  sudo usermod -a -G scanner $USER
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
  Log out and back in for group changes to take effect. Verify with:
  ```bash
  scanimage -L
  ```

- **macOS**:
  ```bash
  brew install sane-backends
  ```
  Verify with `scanimage -L`. If no scanner is found:
  - Quit Image Capture.app and Preview.app (they may hold a USB claim via ICA).
  - On Apple Silicon, ensure brew is installed at `/opt/homebrew` (arm64).
  - First connection may trigger a macOS USB permission prompt.

- **Windows**: Scanner support is not available on Windows (SANE is Linux/macOS only). Use VueScan / SilverFast / vendor tools to scan, then load the resulting TIFF/DNG via the Files tab.

### 2. Python Environment
The `Makefile` handles synchronization via `uv`. Run this to set up your environment:

```bash
make install
```


### 3. Running Locally

```bash
make run
```

## 🏗️ Project Structure

The codebase follows a modular architecture:

- `negpy/domain/`: Core data models, types, and interfaces.
- `negpy/features/`: Image processing logic implementations (Exposure, Geometry, Lab, etc.).
- `negpy/infrastructure/`: Low-level system implementations (GPU resources, file loaders).
- `negpy/kernel/`: Core system services (Logging, Config, caching).
- `negpy/services/`: High-level orchestration (Rendering engine, Export service).
- `negpy/desktop/`: PyQt6 UI implementation (View, Controller, Workers).
- `tests/`: Unit and integration tests.

## 📐 Coding Standards

**Always run `make format` before committing.**

### 1. Style & Formatting
- **Ruff**: Used for both linting and formatting.
- **Type Hints**: Required for all new function definitions (`ty` is enforced). Using `cast` to get around it is frowned upon.
- **Docstrings**: Use clear, concise docstrings for classes and public methods.
- **Style**: Use double quotes for strings, snake_case for variables and functions, and PascalCase for classes.

### 2. Testing
We use `pytest`. New features should include unit tests in the `tests/` directory.

```bash
make test
```

### 3. Workflow (The Makefile)
The `Makefile` is the central source of truth for developer commands and executes everything via `uv run`:
- `make install`: Set up environment and sync dependencies.
- `make lint`: Run Ruff checks.
- `make type`: Run `ty` type checks.
- `make test`: Run all unit tests.
- `make format`: Auto-format code with Ruff.
- `make all`: Run lint, type, and test in sequence.
- `make clean`: Removes cache and build artifacts.


## 📦 Building and Packaging

To build the standalone application for your current OS:

```bash
make build
```
This will trigger the Python backend build via PyInstaller.
