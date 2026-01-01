# 🎞️ DarkroomPy

**DarkroomPy** is a open source RAW film negative processor built with Python. It provides a non-destructive workflow for converting film scans into high-quality images with specialized tools for retouching, color grading, and local adjustments.

---

## ✨ Key Features

### 🎨 Color & Tonality
- **Intelligent Mask Neutralization:** Automatically detects and removes the orange film base mask.
- **Exposure Control:** Linear exposure adjustment anchored at middle gray via unified Grade control.
- **Split Toning:** Independent color grading for highlights and shadows in the negative domain.
- **Color Separation:** Custom algorithm to enhance color depth in shadows and midtones without shifting luminance.
- **Automatic Shadow Desaturation:** Prevents oversaturated shadows when lifting them, ensuring natural-looking dark tones.
- **Integrated Histogram:** RGB + Luminance histogram integrated into the sidebar for real-time tonal feedback.

### 🛠️ Retouching & Geometry
- **Automatic Dust Removal:** Adaptive algorithm that identifies and heals dust specs while preserving film grain.
- **Manual Healing (Inpainting):** Content-aware manual repair tool for larger spots and scratches using the Telea algorithm with resolution-aware scaling.
- **Scratch Mode:** Remove long vertical or horizontal scratches by defining start and end points.
- **Grain Matching:** Manually healed areas receive modulated synthetic grain to blend perfectly with the surrounding film texture.
- **Multi-Format Autocrop:** Automatically detects film edges and crops to popular aspect ratios including 3:2, 4:3, 5:4, 6:7, 1:1, and 65:24 (XPan).

### 🔦 Local Adjustments (Dodge & Burn)
- **Layered Adjustments:** Add multiple independent layers for lightening (dodging) or darkening (burning) specific areas.
- **Linear Exposure Math:** Adjustments are applied in linear color space for realistic, photographic highlight and shadow transitions.
- **Rubylith Visualization:** Real-time red mask overlay shows exactly where you are painting.
- **Configurable Brushes:** Adjustable size, strength (EV), and feathering for every layer.

### 💾 Workflow & Export
- **Export Color Management:** Support for sRGB (default), Adobe RGB and Greyscale output with ICC profile embedding.
- **Automatic Persistence:** Every adjustment is instantly saved to an internal SQLite database and restored automatically when you reload a file.
- **Named Presets:** Save and load global "looks" across different projects.
- **Batch Export:** High-performance parallel processing for exporting multiple positives to JPEG or TIFF.
- **Enhanced Contact Sheet:** Vertical thumbnail gallery with file names and selection status for rapid navigation.

---

## 🏗️ Architecture

The project follows a modular, professional package structure:

```text
darkroom-py/
├── app.py                 # Application entry point
├── src/
│   ├── backend/           # Core image processing logic
│   │   ├── image_logic/   # Specialized algorithms (color, retouch, local, post)
│   │   ├── config.py      # Centralized settings & constants
│   │   ├── db.py          # SQLite persistence layer
│   │   ├── processor.py   # High-level pipeline orchestration
│   │   └── utils.py       # Shared mathematical helpers
│   └── frontend/          # Streamlit UI implementation
│       ├── components/    # Modular UI elements (contact sheet, image view, etc.)
│       │   └── sidebar/   # Specialized sidebar sections (export, color, retouch)
│       ├── main.py        # UI orchestrator
│       ├── state.py       # Session state & settings lifecycle
│       └── css.py         # Custom application styling
├── icc/                   # ICC Color Profiles
└── user/
       └── edits.db        # SQLite database that persists user edits.
```

---

## 🚀 Getting Started

### Using Docker (Recommended)
The easiest way to run DarkroomPy is using Docker Compose, which handles all dependencies and persistence automatically.

```bash
docker compose up --build
```
Access the app at `http://localhost:8501`.

### Manual Installation
If you prefer to run it locally, ensure you have Python 3.10+ installed.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App:**
   ```bash
   streamlit run app.py
   ```

---

## 📖 Usage Tips
- **Consistent Pipeline:** The application uses an identical processing pipeline for both the 1500px preview and final high-resolution export, ensuring visual consistency.
- **Subtractive Workflow:** Most adjustments (Exposure, Dodge/Burn, WB) happen in a physically accurate way, mimicking a real darkroom process.
- **Soft Proofing:** Load custom ICC profiles in the sidebar to simulate how your edits will look on specific paper stocks or display devices.
- **Navigation:** Use the contact sheet or next/previous buttons to quickly switch between frames in a roll.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
