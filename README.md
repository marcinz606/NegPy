<div align="center">
  <img src="media/icons/icon.svg" width="96" height="96" alt="NegPy Logo"><h1>NegPy</h1>
</div>

**NegPy** is a tool for processing film negatives. I built it because I wanted something made specifically for film scans that goes beyond a simple inversion tool. It simulates how film and photographic paper work but also throws in some lab-scanner conveniences.

It is built with **Python**, running natively on Linux, macOS, and Windows.

---

![alt text](docs/media/0350.png)

---

## User Guide
**[Click here to read the USER_GUIDE.md](docs/USER_GUIDE.md)** — A complete walkthrough of the NegPy workflow, features, and controls.

---

## Features

**Conversion & Film Science**
*   **No Camera Profiles**: No camera profiles, no border colour-picking. Math neutralizes the orange mask from channel sensitometry.
*   **Film Physics**: Models the **H&D Characteristic Curve** in density space — an asymmetric toe-linear-shoulder response with independent softplus toe/shoulder knees and ISO-R paper grades — instead of a linear inversion.
*   **Smart Auto Conversion**: Per-frame **Auto Density** and **Auto Grade** meter each negative for sensible brightness/contrast — usable out of the box, easy to fine-tune.
*   **Darkroom Paper Profiles**: Per-paper curve shaping (tone, per-channel gamma, base tint) mapped from Ilford/Kodak/Foma/Fuji datasheets, selectable per roll.
*   **Positive/Slide Support**: Dedicated **E-6 mode** with optional normalization to save expired or faded film.

**Capture & Input**
*   **Camera Scanning**: Capture negatives with a tethered camera straight into NegPy — a single RAW, or automated red/green/blue narrowband triplets driven by an RGB [Scanlight](https://github.com/jackw01/scanlight) that feed the RGB Scan merge. macOS/Linux, optional dependency. [Camera Scanning guide](docs/CAMERA_SCANNING.md)
*   **Scanner Support**: Direct control of SANE-compatible scanners. [supported devices](http://www.sane-project.org/sane-supported-devices.html)
*   **RGB Scan (Trichromatic Capture)**: Merge three narrowband red/green/blue exposures of one negative into a single low-noise colour scan, with automatic sub-pixel alignment to kill fringing.
*   **Flat-Field Correction**: Correct illumination falloff / vignetting from your light source or scanner via a reference scan of the bare light. Named profiles, toggle per image.
*   **File Support**: Standard RAWs/TIFFs plus specialized formats like Kodak Pakon scanner raw files.

**Editing**
*   **Dodge & Burn**: Darkroom-style local lighten/darken with freehand polygon masks — each with its own EV strength and feather. GPU-accelerated with bit-for-bit CPU parity.
*   **Dust Removal**: Automatic and manual healing with grain synthesis — clean scans that don't look plastic.
*   **Batch Normalization**: Bounds analysis across all loaded files, averaged and applied to the roll.
*   **GPU Acceleration**: Real-time processing and export rendering via Vulkan/Metal.

**Colour & Output**
*   **Colour Management**: Full ICC workflow — auto monitor-profile detection (Linux/macOS/Windows), soft proofing including paper/printer profiles, per-image input/output profiles.
*   **Print Ready**: Export built for printing — border controls, ICC soft-proofing, [dynamic filename templating](docs/TEMPLATING.md), **export presets** (save + one-click), and **contact sheets**. Formats: JPEG, TIFF, PNG, WebP, JPEG XL, DNG.
*   **Flat / Digital-Intermediate Export**: Flat, neutral, wide-gamut **16-bit TIFF** (or linear **DNG**) master for Lightroom/Darktable/Photoshop, mapping camera RAWs to ProPhoto via the camera's own matrix.

**Workflow & Data**
*   **Non-destructive**: Original files never touched; edits stored as recipes.
*   **Database**: Edits in a local SQLite db keyed by file hash — move or rename files without losing work.
*   **Persistent Undo/Redo & History**: Up to 100 edits per file. **History panel** lists every step — jump to any state, branch, or export an earlier version. Survives restarts.
*   **Metadata & Gear Library**: Archival metadata for the original analog capture — manage a library of cameras, lenses, and film stocks, apply gear presets per frame, and write real camera/lens/ISO EXIF (plus XMP scan tags) into exports so Lightroom shows your film gear. [see the guide](docs/USER_GUIDE.md#11-metadata-panel)
*   **Keyboard Shortcuts**: [see here](docs/KEYBOARD.md)

---

### How it works

[Read about the math and the pipeline here](docs/PIPELINE.md)

---

## Getting Started

### Download
Grab the latest release for your OS from the **[Releases Page](https://github.com/marcinz606/NegPy/releases)**.

#### Linux
I provide an `.AppImage`. Make it executable using `chmod +x` and It should just work.

**Scanner support** requires SANE to be installed on your system:
```
sudo apt install libsane        # Debian/Ubuntu
sudo pacman -S sane             # Arch
```
Or your distro's equivalent. The app launches fine without so you can ignore that if you don't plan to use a scanner.

**Camera scanning support** (optional) uses `python-gphoto2` for tethered capture, and may need the system `libgphoto2` installed:
```
sudo pacman -S libgphoto2        # Arch
```
Or look up your distro's equivalent package.


#### Unsigned Software Warning
Since this is a free hobby project, I don't pay Apple or Microsoft ransom for their developer certificates. You'll get a scary warning the first time you run it.

**macOS**:
1.  Double click `.dmg` file & drag the app to `/Applications`.
2.  Open Terminal and run: `xattr -cr /Applications/NegPy.app` (this gets rid of the warning).
3.  Launch it.

**Scanner support** requires SANE via [Homebrew](https://brew.sh/):
```
brew install sane-backends
```
The app launches fine without so you can ignore that if you don't plan to use a scanner.

**Camera scanning support** (optional) uses `python-gphoto2`, and may need `libgphoto2` from [Homebrew](https://brew.sh/):
```
brew install libgphoto2
```

**Windows**:
1. Run the installer (ignore the warnings)
2. Start the app and click through the warnings.

Scanner and camera scanning are **not available on Windows**. Both ride on Unix-first free-software libraries — SANE for scanners, libgphoto2 for cameras — that just don't build there. It's not really their fault: the open world spent decades writing generic, vendor-neutral drivers for hundreds of devices, while Windows stuck with closed per-vendor blobs and never grew an equivalent. So the free stack NegPy leans on has nowhere to stand on Windows. Everything else works fine.

Good news: you can install Linux on pretty much any Windows machine. 🐧

---

You can also clone the repo and build it yourself, instruction here: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Data Location
Everything lives in your `Documents/NegPy` folder:
*   `edits.db`: Your edits.
*   `settings.db`: Global settings like last used export settings or preview size.
*   `cache/`: Thumbnails (safe to delete).
*   `export/`: Default export location.
*   `icc/`: Drop your paper/printer profiles here.
*   `override.toml`: Startup overrides — see [Troubleshooting / override.toml](#troubleshooting) below.

---

## Troubleshooting

If NegPy crashes on startup or has rendering issues, edit `Documents/NegPy/override.toml`. It is created automatically on first run with sensible defaults for your OS.

```toml
[rendering]
# Options: "auto", "vulkan" (Linux/Win), "dx12" (Win), "metal" (macOS), "cpu"
backend = "vulkan"

[display]
# Qt scene-graph backend. Options: "auto", "vulkan", "d3d12", "metal", "opengl", "software"
qt_rhi_backend = "auto"

# Window system plugin (Linux only). Options: "auto", "xcb", "wayland"
qt_platform = "auto"

[performance]
# Cap GPU texture size in pixels — useful on low-VRAM cards. "auto" = no limit.
max_texture_size = "auto"

# Force HQ preview on/off. Uncomment to override saved preference.
# force_hq_preview = false

# Preview cache size — keeps recently-viewed photos in memory for instant navigation.
# Lower these on low-RAM machines. Uncomment to override defaults (~1.2 GB / 8 photos).
# preview_cache_max_bytes = 1200000000
# preview_cache_max_entries = 8

[logging]
# "debug", "info", "warning", "error"
level = "info"
```

Setting `backend = "cpu"` disables GPU acceleration entirely — useful if the GPU backend crashes on your hardware.

---

## Roadmap
Things I want to add later: [ROADMAP.md](docs/ROADMAP.md)

## Changelog:

[CHANGELOG.md](docs/CHANGELOG.md)

---

### For Developers

Check [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
Copyleft under **[GPL-3](LICENSE)**.

## Support
If you like this tool, maybe buy me a roll of film so I have more test data :)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/marcinzawalski)
