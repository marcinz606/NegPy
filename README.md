# 🎞️ DarkroomPy

**DarkroomPy** is an open-source tool for processing RAW film negatives. I built it because I wanted a way to convert my film scans using algorithms that actually act like a physical darkroom, rather than just another primarly digital photo editor. I throw in some lab-scanner-like features because who wouldn't want to have Fuji Frontier at home?

Also I'm Linux user and linux seriously lacks good options for this kind of workflow.


---

## ✨ Features

### 🛠️ Under the Hood
I've tried to keep the code clean and modular so it's easy to extend:
- **Modular Design**: Each tool (Exposure, Retouch, Geometry, Toning) is its own isolated module.
- **Caching & Persistence**: Thumbnails and settings are cached locally, so it feels snappy even with large libraries.
- **Auto-Save**: All your edits (Exposure, Crop, etc.) are saved automatically to a local SQLite database. We don't touch your raws, we just keep track of all the settings that need to be applied to produce final "print".

---

### 🧪 The Processing Pipeline
Most important part, the image goes through a 7-stage simulation:

[📖 Read about the math behind the pipeline](docs/PIPELINE.md)

1.  **Geometry**: Auto-rotates and auto-crops to standard ratios (3:2, 6:7, etc.) by detecting the film borders.
2.  **Normalization**: Strips away the film base (D-min) to get a clean signal.
3.  **Photometric Engine**:
    *   **Inversion**: Uses a sigmoid curve that mimics H&D film characteristic curves.
    *   **Auto-Exposure**: Tries to find a good starting point (Zone V) automatically.
    *   **Color**: Subtractive CMY filtration, just like a color enlarger head.
4.  **Retouching**:
    *   **Dust Removal**: Automatic median-based healing or manual "spotting" with grain matching.
    *   **Dodge & Burn**: classic local exposure tools with soft masking.
5.  **Lab Tools**:
    *   **Crosstalk/Color Separation**: Fixes color purity by un-mixing dye overlap.
    *   **Hypertone**: A local contrast boost (similar to Fuji Frontier scanners).
    *   **Chroma Denoise**: Targets color noise in the shadows without killing grain.
6.  **Toning**:
    *   **Paper**: Simulates different paper bases (Warm, Cool, Glossy).
    *   **Chemistry**: Simulates Selenium or Sepia toning for archival looks (for B&W mode).
7.  **Output**: Exports your final print.

---

## 🚀 Getting Started

### Download the App
If you just want to use it, grab the installer for your OS from the **[Releases Page](https://github.com/USER/darkroom-py/releases)**.

### 🛡️ Installation & Security
Because DarkroomPy is a hobby, open-source project, the installers are not "digitally signed" by Apple or Microsoft (they want you to pay them for that). You will see a security warning the first time you run it.

#### **macOS (Gatekeeper)**
When you open the app, you may see a message saying it is "corrupted" or from an "unidentified developer."
1.  Drag **DarkroomPy** to your `/Applications` folder.
2.  **Right-Click** (or Ctrl+Click) the app icon and select **Open**.
3.  When the dialog appears, click **Open** again.
4.  *Alternatively*, run this in your Terminal: `xattr -cr /Applications/DarkroomPy.app`

#### **Windows (SmartScreen)**
Windows might show a blue "Windows protected your PC" window.
1.  Click **More info**.
2.  Click **Run anyway**.

---

### For Developers
If you want to contribute or poke around the code, I use Docker to keep the environment consistent.

#### Run with Docker
`make run-app`

#### Build electron app locally
`make dist`

#### Run Tests & Checks
There's a Makefile to help with quality control:
- `make all`: Runs everything (Lint, Typecheck, Tests).
- `make format`: Auto-formats code with Ruff.

---

## 📂 Where's my data?
DarkroomPy keeps everything in your **Documents/DarkroomPy** folder:
- **`edits.db`**: Your edits.
- **`settings.db`**: App preferences.
- **`cache/`**: Thumbnails (safe to delete).
- **`export/`**: Where your finished JPEGs go by default.

---

## ⚖️ License
This project is free software under the **GPL-3 License**. Feel free to use it, study it, and share it. If you use it, also keep it open.

## Support
If you like the project and want to support it, consider buying me a coffee or a roll of film to have material for testing. [Ko-Fi](https://ko-fi.com/marcinzawalski)