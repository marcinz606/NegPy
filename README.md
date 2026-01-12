<div align="center">
  <img src="media/icons/icon.svg" width="96" height="96" alt="NegPy Logo"><h1>NegPy</h1>
</div>


**NegPy** is an open-source tool for processing RAW film negatives. I built it because I wanted something made specifically for film scans but going beyond being simple converter. I try to simulate film & paper behavior while also throwing in some lab-scanner-like features because who wouldn't want to have Fuji Frontier at home?

---

## ✨ Basic Features

- **Math-based Inversion**: No camera profiles (DCP) or "film base color picking" required. It uses per-channel sensitometric normalization to automatically neutralize the orange mask.
- **Physical Modeling**: It doesn't just linearly "invert colors". It simulates the physics of a darkroom print using a Logistic Sigmoid function to model the **H&D Characteristic Curve** of film & photographic paper.
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

[📖 Read about the math behind the pipeline and the basic workflow of NegPy](docs/PIPELINE.md)


---

## 🚀 Getting Started

### Download the App
Grab the app for your OS from the **[Releases Page](https://github.com/marcinz606/NegPy/releases)**.


#### **🐧 Linux**

I supply a .AppImage file for Linux, it should work out of the box. Here is [quick guide](https://docs.appimage.org/introduction/quickstart.html) if you never used AppImage before

I will also add it to Arch User Repository (AUR) as soon as I get around to it.

### 🛡️ Installation & Security
Because NegPy is a hobby, open-source project, the installers are not "digitally signed" by Apple or Microsoft (they want you to pay them ransom for that). You will see a security warning the first time you run it.

#### **🍎 MacOS**
When you open the app, you may see a message saying it is "corrupted" or from an "unidentified developer."
1.  Doubleclick on downloaded `.zip` file to extract it, you should se `.dng` file.
1.  Doubleclick `.dng` **NegPy** and drag it to your `/Applications` folder.
2.  **Right-Click** (or Ctrl+Click) the app icon and select **Open**.
3.  *Alternatively*, run this in your Terminal: `xattr -cr /Applications/NegPy.app`
4.  When the dialog appears, click **Open** again.
5.  After that you should be able to just start the app normally.

#### **🪟 Windows**
Windows might show a "Windows protected your PC" window.
1.  Click **More info**.
2.  Click **Run anyway**.
3.  Because proper startup process was blocked, you might get white screen on first run. Just close and restart the app (if it minimizes to tray right click and quit).


#### **⚠️ Important**
App is compiling functions to machine code on first startup, so it might take a while to start up on slower CPUs.
Also, app closes to tray by default so if you want to kill it, right click tray icon and quit.


## Tips

* You can scale the UI using `ctrl +` and `ctrl -` shortcuts. (`cmd +` and `cmd -` on MacOS)
* Your edits to current file are saved on export or on switching to different file. If you close the app without exporting or switching to different sile edits might be lost.


## 📂 Where's my data?
NegPy keeps everything in your **Documents/NegPy** folder:
- **`edits.db`**: Your edits.
- **`settings.db`**: Global settings like export size, image preview size etc.
- **`cache/`**: Thumbnails (safe to delete).
- **`export/`**: Where your finished positives go by default.
- **`icc/`**: Where your loaded ICC profiles go.

---

## Roadmap
I have some more ideas for next features & improvements:

[ROADMAP.md](docs/ROADMAP.md)

---

### For Developers
If you want to contribute or poke around the code, I use Docker to make building and running the app easy and consistent.

#### Run with Docker
`make run-app`

#### Build electron app locally
`make dist`

#### Run Tests & Checks
There's a Makefile to help with quality control:
- `make all`: Runs everything (Lint, Typecheck, Unit Tests).
- `make format`: Auto-formats code with Ruff.


## ⚖️ License
This project is free software under the copyleft **[GPL-3 License](LICENSE)**. Feel free to use it, study it, and share it. If you use it, also keep it open.

## Support
If you like the project and want to support it, consider buying me a coffee or a roll of film to have material for testing.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/marcinzawalski)