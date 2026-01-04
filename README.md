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

## 🚀 Getting Started

### Standalone Desktop App (Recommended)
The easiest way to use DarkroomPy is to download the standalone installer for your operating system.

1.  Go to the **[Latest Releases](https://github.com/USER/darkroom-py/releases)**.
2.  Download the installer for your platform:
    *   **Windows**: `.exe` (NSIS Installer)
    *   **macOS**: `.dmg` (Disk Image)
    *   **Linux**: `.AppImage` or `.deb`

#### Installation & Launch
- **Windows**: Double-click the `.exe` and follow the installer. Launch from your Start Menu.
- **macOS**: Double-click the `.dmg`, drag to **Applications**. Right-click and select **Open** the first time to bypass security warnings.
- **Linux**: Right-click the `.AppImage`, go to **Properties** -> **Permissions**, check **"Allow executing file as program"**, then double-click to run.

### 🔄 How to Update
When a new version comes out, simply download the new installer and run it. Your settings and edits will remain safe in your documents folder.

---

## 📂 Where are my files?
DarkroomPy creates a folder in your **Documents** called `DarkroomPy`:
- **`edits.db`**: Stores all your per-file edits.
- **`settings.db`**: Remembers your settings (export, borders, etc.).
- **`export/`**: Default location for finished photos.
- **`icc/`**: Imported .ICC profiles.
- **`presets/`**: Your saved looks.
- **`cache/`**: Temporary storage for files being edited (cleared on close).

---

## 🛠️ Development & Advanced Usage

### Docker
I provide a helper script to automatically configure and start the container:
`make run-app`

### Automation (Makefile)
The project includes a `Makefile` for developers:
- `make test`: Run all unit tests.
- `make type`: Execute Mypy type checks.
- `make lint`: Run style checks.
- `make format`: Automatically fix formatting and style issues.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
