# 🎞️ DarkroomPy

**DarkroomPy** is an open-source RAW film negative processor built with Python. It provides a non-destructive, professional workflow for converting film scans into high-quality images using physically accurate algorithms that mimic a traditional darkroom.

---

## ✨ Key Features

### 🚀 High-Performance Architecture
- **Disk-Backed Asset Management**: Handles hundreds of 100MB+ RAW files with near-zero idle RAM usage by spooling data to disk and loading only the active frame.
- **Content-Based Fingerprinting**: Uses SHA-256 content hashing to tie edits to the image data itself. Your settings persist even if you rename or move your files.
- **Parallel Processing**: Leverages multi-core background execution for thumbnail generation and high-speed batch exports.

### 🎨 Color & Tonality
- **"True Darkroom" Pipeline**: Follows the physical path of light: *Filtered Light → Negative → Scanner Gain → Inversion → Positive*.
- **Specialized RAW Support**: Seamless, automatic loading of standard RAWs (`.ARW`, `.DNG`, etc.) and specialized headerless formats (e.g., **Pakon scanner** planar `.raw` files).
- **Intelligent Tonality Recovery**: Sensitometric Shadow Toe and Highlight Shoulder controls to prevent digital clipping and preserve analog-like transitions.
- **Dichroic Filtration**: Physically accurate CMY filtration model for neutralizing film base masks without color drift.

### 🔦 Local Adjustments (Dodge & Burn)
- **Layered Adjustments**: Add multiple independent layers for lightening (dodging) or darkening (burning) specific areas.
- **Linear Exposure Math**: Adjustments are applied in linear color space for realistic highlight and shadow transitions.
- **Rubylith Visualization**: Real-time red mask overlay shows exactly where you are painting.
- **Configurable Brushes**: Adjustable size, strength (EV), and feathering for every layer.

### 🛠️ Retouching & Geometry
- **Adaptive Dust Removal**: High-performance algorithm that heals specs while respecting film grain.
- **Manual Healing & Scratch Removal**: Content-aware repair tools with synthetic grain matching for perfect blending.
- **Multi-Format Autocrop**: Automatic edge detection for 3:2, 4:3, 5:4, 6:7, 1:1, and 65:24 (XPan).

---

## 🏗️ Technical Foundation

The project is built on a robust, professional engineering stack:
- **Language**: Python 3.13
- **Type Safety**: 100% compliant with strict **Mypy** type checking.
- **Code Quality**: Enforced by **Ruff** and **Flake8** for consistent, professional formatting.
- **Testing**: Built-in unit test suite via **Pytest** ensuring mathematical integrity.
- **Logging**: Centralized, module-level logging for reliable debugging and monitoring.

---

## 📖 Documentation

For deeper dives into the project, please refer to the following documents:
- 🚀 **[Beginner's Setup Guide](docs/BEGINEER.md)**: A step-by-step guide to getting the app running via Docker without needing programming knowledge.
- 🔬 **[Core Processing Logic](docs/LOGIC.md)**: Detailed technical explanation of the "True Darkroom" mathematical model and algorithms.
- 📓 **[Sensitometric Analysis](docs/READ_THIS.md)**: An in-depth analysis of emulsion response, paper grading, and the scientific foundations of the project.

---

## 🚀 Getting Started

### Using Docker (Recommended)
```bash
docker compose up --build
```
Access the app at `http://localhost:8501`.

### Automation (Makefile)
The project includes a `Makefile` for developers:
- `make test`: Run all unit tests.
- `make type`: Execute Mypy type checks.
- `make lint`: Run style checks.
- `make format`: Automatically fix formatting and style issues.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
