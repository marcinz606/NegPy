# 🎞️ DarkroomPy

**DarkroomPy** is a professional-grade RAW film negative processor built with Python and Streamlit. It provides a non-destructive workflow for converting film scans into high-quality images with specialized tools for retouching, color grading, local adjustments, and now—AI-powered style matching.

---

## ✨ Key Features

### 🤖 AI Style Assistant (New!)
- **"Magic Fix":** Train a personalized AI model on your editing style.
- **Learn from Edits:** Automatically collects data from your finished exports to understand your preferences for exposure, contrast, and color balance.
- **Auto-Predict:** Apply your unique "look" to new, raw images with a single click using a Random Forest regression model.
- **Deduplicated Training:** Smartly filters training data to learn only from your latest, most refined edits for each photo.

### 🛠️ Professional Retouching
- **Automatic Dust Removal:** Adaptive algorithm that identifies and heals dust specs while preserving film grain.
- **Manual Healing (Inpainting):** Content-aware manual repair tool for larger spots and scratches using the Telea algorithm.
- **Scratch Mode:** Effortlessly remove long vertical or horizontal scratches by defining start and end points.
- **Grain Matching:** Manually healed areas receive modulated synthetic grain to blend perfectly with the surrounding film texture.

### 🔦 Local Adjustments (Dodge & Burn)
- **Layered Adjustments:** Add multiple independent layers for lightening (dodging) or darkening (burning) specific areas.
- **Linear Exposure Math:** Adjustments are applied in linear color space for realistic, photographic highlight and shadow transitions.
- **Rubylith Visualization:** Real-time red mask overlay shows exactly where you are painting.
- **Configurable Brushes:** Adjustable size, strength (EV), and feathering for every layer.

### 🎨 Color & Tonality
- **Intelligent Mask Neutralization:** Automatically detects and removes the orange film base mask.
- **Split Grading:** Independent contrast/gamma controls for highlights and shadows.
- **Exposure Control:** Linear exposure adjustment anchored at middle gray.
- **Black/White Points:** Precise control over the print's clipping points.
- **Selective Color:** Fine-tune Hue, Saturation, and Luminance for specific color ranges.
- **Color Separation:** Custom algorithm to enhance color depth without shifting luminance.

### 💾 Workflow & Persistence
- **Automatic Persistence:** Every adjustment is instantly saved to an internal SQLite database and restored automatically when you reload a file.
- **Named Presets:** Save and load global "looks" across different projects.
- **Batch Export:** High-performance parallel processing for exporting multiple rolls to JPEG or TIFF.
- **Settings Clipboard:** Quickly copy and paste global color adjustments between similar frames.

---

## 🏗️ Architecture

The project follows a modular, professional package structure:

```text
darkroompy/
├── app.py                 # Application entry point
├── src/
│   ├── backend/           # Core image processing logic
│   │   ├── ai/            # AI features extraction, training, and inference
│   │   ├── image_logic/   # Specialized algorithms (color, retouch, local)
│   │   ├── config.py      # Centralized settings & constants
│   │   ├── db.py          # SQLite persistence layer
│   │   ├── processor.py   # High-level pipeline orchestration
│   │   └── utils.py       # Shared mathematical helpers
│   └── frontend/          # Streamlit UI implementation
│       ├── components/    # Modular UI elements (sidebar, viewport, local_ui, ai)
│       ├── main.py        # UI orchestrator
│       └── state.py       # Session state & settings lifecycle
├── data/                  # Collected training vectors (local storage)
├── models/                # Trained AI models
└── presets/               # User-defined global adjustment files
```

---

## 🚀 Getting Started

### Using Docker (Recommended)
The easiest way to run DarkroomPy is using Docker Compose, which handles all dependencies (including machine learning libraries) and persistence automatically.

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
   *Note: This now includes `scikit-learn` for AI features.*

2. **Run the App:**
   ```bash
   streamlit run app.py
   ```

---

## 📖 Usage Tips
- **Training the AI:** Enable "Collect Training Data" in the AI tab (on by default). Export your finished images to save their "style vectors". Once you have a few dozen samples, go to the AI tab and click "Train Model".
- **1:1 Preview:** The app displays images at a fixed 1600px resolution to ensure that the grain you see in the preview matches the logic applied during final export.
- **Subtractive Workflow:** Most adjustments (Exposure, Dodge/Burn, WB) happen in a physically accurate way, mimicking a real darkroom process.
- **Keyboard Shortcuts:** Use standard browser scrolling to navigate the large interaction window during fine retouching.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.