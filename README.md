# 🎞️ DarkroomPy

**DarkroomPy** is an open-source tool for processing RAW film negatives. I built it because I wanted something made specifically for film scans but going beyond being simple converter. I try to simulate film & paper behavior while also throwing in some lab-scanner-like features because who wouldn't want to have Fuji Frontier at home?

---

## ✨ Basic Features

- **File support**: Supports raw formats that you would expect, tiff but also weird Planar RAW from Kodak Pakon scanner.
- **Hot folder**: Optionally watch folder for new files and load them automatically.
- **Non-destructive**: It doesn't touch your raws, we just keep track of all the settings that need to be applied to produce final "print".
- **Copy-paste**: Copy-paste settings between images.
- **Presets**: Save your favorite settings as presets. Presets are saved in JSON format so they can be easily shared.
- **Caching**: Thumbnails for film strip view are processed once are cached locally, so it feels snappy even with large libraries.
- **Persistence**: All your edits are tied to hashes calcualated based on file contents (so won't be lost when you rename/move files) and stored in local SQLite database, moving them between computers is as simple as copying the database file.
- **Optimization & Multiprocessing**: To speed up the processing, we compile functions to low level machine code on startup and employ multiprocessing for batch exports process multiple files in parallel.
- **Print preparation**: Export module is tailored towards getting your scans printed. We export with certain print size & DPI in mind, have very convinitent way to add border (while keeping target size) and also soft-proofing module to preview image with applied .icc profile.

---

### 🧪 The Processing Pipeline
Most important part, the image goes through a 7-stage simulation:

[📖 Read about the math behind the pipeline](docs/PIPELINE.md)

1.  **Geometry**: Auto-rotates and auto-crops to standard ratios (3:2, 6:7, etc.) by detecting the film borders.
    you can also choose to keep the full frame in final output.
2.  **Normalization**: Strips away the film base (D-min) to get a clean signal.
3.  **Photometric Engine**:
    *   **Inversion**: Uses a sigmoid curve that mimics H&D film characteristic curves.
    *   **Auto-Exposure**: Tries to find a good starting point (Zone V) automatically.
    *   **Color**: Subtractive CMY filtration, just like a color enlarger head.
4.  **Retouching**:
    *   **Dust Removal**: Automatic median-based healing or manual "spotting" with grain matching.
    *   **Dodge & Burn**: Classic local exposure tools with soft luminance masking.
5.  **Lab Tools**:
    *   **Color Separation**: Fixes color purity by un-mixing dye overlap.
    *   **Hypertone**: A local contrast boost (similar to Fuji Frontier scanners).
    *   **Luma sharpening**: Sharpens the image without affecting grain and color.
6.  **Toning**:
    *   **Paper**: Simulates different paper bases (Warm, Cool, Glossy).
    *   **Chemistry**: Simulates Selenium or Sepia toning for archival looks (for B&W mode).
7.  **Output**: Exports your final print.

---

## Roadmap
I have some more ideas for next features & improvements:

[ROADMAP.md](docs/ROADMAP.md)

## 🚀 Getting Started

### Download the App
Grab the app for your OS from the **[Releases Page](https://github.com/marcinz606/darkroom-py/releases)**.

### 🛡️ Installation & Security
Because DarkroomPy is a hobby, open-source project, the installers are not "digitally signed" by Apple or Microsoft (they want you to pay them $ for that). You will see a security warning the first time you run it.

#### **Linux**

I supply a .AppImage file for Linux, it should work out of the box. Here is [quick guide](https://docs.appimage.org/introduction/quickstart.html) if you never used AppImage before

#### **macOS**
When you open the app, you may see a message saying it is "corrupted" or from an "unidentified developer."
1.  Drag **DarkroomPy** to your `/Applications` folder.
2.  **Right-Click** (or Ctrl+Click) the app icon and select **Open**.
3.  When the dialog appears, click **Open** again.
4.  *Alternatively*, run this in your Terminal: `xattr -cr /Applications/DarkroomPy.app`

#### **Windows**
Windows might show a "Windows protected your PC" window.
1.  Click **More info**.
2.  Click **Run anyway**.
3.  Because proper startup process was blocked, you might get white screen on first run. Just close and restart the app (if it minimizes to tray right click and quit).

---

## 📂 Where's my data?
DarkroomPy keeps everything in your **Documents/DarkroomPy** folder:
- **`edits.db`**: Your edits.
- **`settings.db`**: Global settings like export size, image preview size etc.
- **`cache/`**: Thumbnails (safe to delete).
- **`export/`**: Where your finished positives go by default.

---

### For Developers
If you want to contribute or poke around the code, I use Docker to make building and running the app easy and consistent.

#### Run with Docker
`make run-app`

#### Build electron app locally
`make dist`

#### Run Tests & Checks
There's a Makefile to help with quality control:
- `make all`: Runs everything (Lint, Typecheck, Tests).
- `make format`: Auto-formats code with Ruff.


## ⚖️ License
This project is free software under the **GPL-3 License**. Feel free to use it, study it, and share it. If you use it, also keep it open.

## Support
If you like the project and want to support it, consider buying me a coffee or a roll of film to have material for testing. [Ko-Fi](https://ko-fi.com/marcinzawalski). Also watch the repository for updates and leave a star :) 