# 🎞️ DarkroomPy

**DarkroomPy** is a professional, open-source RAW film negative processor built with Python. It provides a non-destructive workflow for converting film scans into high-quality images using physically accurate algorithms that mimic traditional darkroom chemistry and physics.

---

## ✨ Key Features

### 🚀 Enterprise-Grade Architecture
- **Vertical Slice Modularization**: A clean, modular design where each feature (Exposure, Geometry, Retouch, etc.) is an independent, testable unit.
- **MVC / MVVM Pattern**: Strict separation of concerns between core logic (Services), coordination (Controllers), and rendering (Views).
- **Zero-Copy Asset Management**: High-performance local workflows. Files are referenced from their original location without duplication, eliminating I/O overhead.
- **Hot Folder Mode**: Real-time asset discovery. Automatically monitors watched directories and imports new RAW files as they appear.
- **Persistent Asset Caching**: Hash-keyed thumbnail cache ensures instant UI responsiveness across sessions.
- **Smart Global Persistence**: Remembers your entire environment (Export settings, Lab parameters, UI state) across application restarts and new file imports.

### ⚙️ Modular Processing Pipeline
DarkroomPy implements an injectable 7-stage physical simulation pipeline:

1.  **Geometry**: High-precision rotation and multi-aspect ratio autocrop (3:2, 4:3, 6:7, XPan).
2.  **Normalization**: Simulates scanner gain to maximize SNR and measures log-density bounds.
3.  **Photometric Engine**:
    *   **H&D Curve Inversion**: Uses a logistic sigmoid characteristic curve to invert the negative.
    *   **Sensitometric Solver**: Automatically determines optimal exposure using *Range-Based Anchoring* (Zone V placement) and *Auto-Grade*.
    *   **Dichroic Filtration**: Professional subtractive CMY color correction model.
4.  **Local Retouching**:
    *   **Auto/Manual Dust Removal**: Adaptive healing algorithm that repairs defects while preserving grain.
    *   **Dodge & Burn**: Linear-space local exposure adjustments with real-time "Rubylith" masks.
5.  **PhotoLab Simulation**:
    *   **Spectral Crosstalk**: Mathematically "un-mixes" film dye impurities (Color Separation).
    *   **Hypertone**: Fuji Frontier-style contrast limited adaptive equalization (CLAHE).
    *   **Chroma Denoise**: Targeted L*a*b* shadow noise filtering.
6.  **Toning & Substrate**:
    *   **Paper Simulation**: Physical modeling of paper tint and D-max boost (Warm Fiber, Cool Glossy).
    *   **Chemical Toning**: Simulates Selenium (shadow cooling) and Sepia (highlight warming).
7.  **Output Mapping**: Final non-destructive crop and resolution-agnostic rendering.

---

## 🚀 Getting Started

### Standalone Desktop App (Recommended)
The easiest way to use DarkroomPy is to download the standalone installer. The desktop app leverages **Native OS Integration**, providing direct filesystem access and native file picker dialogs.

1.  Go to the **[Latest Releases](https://github.com/USER/darkroom-py/releases)**.
2.  Download the installer for your platform:
    *   **Windows**: `.exe` (NSIS Installer)
    *   **macOS**: `.dmg` (Disk Image)
    *   **Linux**: `.AppImage` or `.deb`

### Development Environment
For developers, DarkroomPy provides a robust Docker-based environment and a comprehensive automation suite.

#### Docker
`make run-app`

#### Automation (Makefile)
- `make all`: Run the full verification suite (Lint -> Typecheck -> Test).
- `make test`: Run unit tests.
- `make type`: Execute strict Mypy type checks.
- `make format`: Automatically format and fix style issues using Ruff.

---

## 📂 Data & Persistence
DarkroomPy maintains a professional data structure in your **Documents/DarkroomPy** folder:
- **`edits.db`**: SQLite database storing all non-destructive per-file adjustments.
- **`settings.db`**: Global persistence for environment preferences (Export DPI, Color Spaces, etc.).
- **`cache/thumbnails/`**: Persistent PNG previews keyed by file content hash.
- **`export/`**: Default high-resolution rendering output directory.
- **`presets/`**: JSON-based look definitions.

---

## ⚖️ License
Distributed under the GNU General Public License v3 (GPL-3). See `LICENSE` for more information.