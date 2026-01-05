# 🎞️ DarkroomPy

**DarkroomPy** is an open-source RAW film negative processor built with Python. It provides a non-destructive, professional workflow for converting film scans into high-quality images using physically accurate algorithms that mimic a traditional darkroom.

---

## ✨ Key Features

### 🚀 High-Performance Architecture
- **Disk-Backed Asset Management**: Handles hundreds of 100MB+ RAW files with near-zero idle RAM usage by spooling data to disk and loading only the active frame.
- **Content-Based Fingerprinting**: Uses SHA-256 content hashing to tie edits to the image data itself. Your settings persist even if you rename or move your files.
- **Parallel Processing**: Leverages multi-core background execution for thumbnail generation and high-speed batch exports.

### ⚙️ Processing Pipeline
DarkroomPy implements a modular 6-stage physical simulation pipeline:

1.  **Normalization**: Simulates scanner gain to maximize signal-to-noise ratio and measures the negative's log-density bounds.
2.  **Photometric Engine**:
    *   **H&D Curve Inversion**: Uses a logistic sigmoid characteristic curve to invert the negative.
    *   **Sensitometric Solver**: Automatically determines optimal exposure using *Range-Based Anchoring* (Zone V placement) and *Auto-Grade* (matching negative range to paper range).
    *   **Dichroic Filtration**: Subtractive CMY color correction model.
3.  **Local Retouching**:
    *   **Dust Removal**: Adaptive healing algorithm that repairs defects while preserving grain structure.
    *   **Dodge & Burn**: Linear-space local exposure adjustments with real-time masks.
4.  **PhotoLab Simulation**:
    *   **Spectral Crosstalk**: Simulates film dye impurities and scanner channel overlap (Color Separation).
    *   **Hypertone**: Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement.
    *   **Chroma Denoise**: Targeted L*a*b* space filtering for shadow noise.
5.  **Toning & Substrate**:
    *   **Paper Simulation**: Physical modeling of paper tint and D-max boost (e.g., Warm Fiber, Cool Glossy).
    *   **Chemical Toning**: Simulates Selenium (shadow cooling/deepening) and Sepia (highlight bleaching/warming) reactivity.
6.  **Geometry**: High-precision rotation and multi-aspect ratio autocrop (3:2, 4:3, 6:7, XPan).

### 🎨 Color & Tonality
- **"True Darkroom" Workflow**: Follows the physical path of light: *Filtered Light → Negative → Scanner Gain → Inversion → Positive*.
- **Auto-Filtration**: "Base Neutralization" algorithm aligns film base (D-min) to remove orange masks without neutralizing scene colors.
- **Dynamic Range Control**: Visualizes the measured dynamic range of your negative vs. the target paper range.

### 🛠️ Retouching & Tools
- **Layered Adjustments**: Add multiple independent layers for lightening (dodging) or darkening (burning) specific areas.
- **Rubylith Visualization**: Real-time red mask overlay shows exactly where you are painting.
- **Manual Healing**: Content-aware scratch removal with synthetic grain matching.

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
Distributed under the GNU General Public License v3 (GPL-3). See `LICENSE` for more information.
