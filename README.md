<div align="center">
  <img src="media/icons/icon.svg" width="96" height="96" alt="NegPy Logo"><h1>NegPy</h1>
</div>

**NegPy** is a tool for processing film negatives. I built it because I wanted something made specifically for film scans that goes beyond a simple inversion tool. It simulates how film and photographic paper work but also throws in some lab-scanner conveniences.

It is built with **Python**, running natively on Linux, macOS, and Windows.

---

![alt text](docs/media/0913_scr.png)
---

## ✨ Features

*   **No Camera Profiles**: It doesn't use camera profiles or ask you to color-pick the border. It uses math to neutralize the orange mask based on channel sensitometry.
*   **Positive/Slide Support**: A dedicated **E-6 mode** for processing slide film with optional normalization to save expired or faded film.
*   **Film Physics**: It models the **H&D Characteristic Curve** of photographic material using a Logistic Sigmoid function instead of doing simple linear inversion.
*   **Batch Normalization**: Perform bounds analysis for all loaded files and apply averaged settings to all.
*   **GPU Acceleration**: Real-time processing and export rendering using Vulkan/Metal.
*   **Dust Removal**: Automatic and manual healing tools with grain synthesis to keep scans clean without looking plastic.
*   **File Support**: Supports standard RAWs/TIFFs, and specialized formats like Kodak Pakon scanner raw files.
*   **Non-destructive**: original files are never touched; edits are stored as recipes.
*   **Keyboard Shortcuts**: [see here](docs/KEYBOARD.md)
*   **Database**: All edits live in a local SQLite database, keyed by file hash. You can move or rename files without losing your work.
*   **Print Ready**: Export module designed for printing, featuring border controls, ICC soft-proofing, and [dynamic filename templating](docs/TEMPLATING.md).

---

### 🧪 How it works

[📖 Read about the math and the pipeline here](docs/PIPELINE.md)

---

## 🚀 Getting Started

### Download
Grab the latest release for your OS from the **[Releases Page](https://github.com/marcinz606/NegPy/releases)**.

#### **🐧 Linux**
I provide an `.AppImage`. Make it executable using `chmod +x` and It should just work. But across all different distros you never know :)

(I'll add it to the AUR eventually).

#### **🛡️ Unsigned Software Warning**
Since this is a free hobby project, I don't pay Apple or Microsoft ransom for their developer certificates. You'll get a scary warning the first time you run it.

**🍎 MacOS**:
1.  Double click `.dmg` file & drag the app to `/Applications`.
2.  Open Terminal and run: `xattr -cr /Applications/NegPy.app` (this gets rid of the warning).
3.  Launch it.

**🪟 Windows**:
1. Run the installer (ignore the warnings)
2. Start the app and click through the warnings.

---

## 📂 Data Location
Everything lives in your `Documents/NegPy` folder:
*   `edits.db`: Your edits.
*   `settings.db`: Global settings like last used export settings or preview size.
*   `cache/`: Thumbnails (safe to delete).
*   `export/`: Default export location.
*   `icc/`: Drop your paper/printer profiles here.

---

## Roadmap
Things I want to add later: [ROADMAP.md](docs/ROADMAP.md)

## Changelog:

[CHANGELOG.md](docs/CHANGELOG.md)

---

### For Developers

Check [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## ⚖️ License
Copyleft under **[GPL-3](LICENSE)**.

## Support
If you like this tool, maybe buy me a roll of film so I have more test data :)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/marcinzawalski)
