<div align="center">
  <img src="media/icons/icon.svg" width="96" height="96" alt="NegPy Logo"><h1>NegPy</h1>
</div>

**NegPy** is a tool for processing film negatives. I built it because I wanted something made specifically for film scans that goes beyond simple inversion tool. It simulates how film & photographic paper works but also throws in some lab-scanner conveniences because who wouldn't want a Fuji Frontier at home?

Also it runs on Linux, macOS and Windows.

---

![alt text](docs/media/090_scr.png)
---

## ✨ Features

*   **No Camera Profiles**: It doesn't use camera profiles or ask you colorpick the border. It uses math to neutralize the orange mask based on channel sensitometry.
*   **Film Physics**: It models the **H&D Characteristic Curve** of film and darkroom paper using a Logistic Sigmoid function instead of doing simple linear inversion.
*   **File Support**: Supports standard RAWs/TIFFs, but also the weird raw files from Kodak Pakon scanners.
*   **Non-destructive**: Your original files are never touched. Edits are just "recipe" for final print.
*   **Database**: All edits are stored in a local SQLite database, keyed by file hash. You can rename your files and your edits won't disappear.
*   **Caching**: Thumbnails are cached so browsing large folders feels snappy.
*   **Print Ready**: The export module is designed for printing (because you should be printing your photos), with easy border controls and soft-proofing.

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
1. Run the installer
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

---

### For Developers

Check [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## ⚖️ License
Copyleft under **[GPL-3](LICENSE)**.

## Support
If you like this tool, maybe buy me a roll of film so I have more test data :)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/marcinzawalski)
