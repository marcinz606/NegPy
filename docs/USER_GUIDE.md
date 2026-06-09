# NegPy User Guide

## 1. Core Workflow

NegPy follows a non-destructive pipeline:
1.  **Import**: Add files to your session.
2.  **Process**: Choose your film mode and perform roll-wide normalization.
3.  **Geometry**: Crop and fine-tune rotation.
4.  **Exposure**: Fine-tune the density, grade, and characteristic curve (Sigmoid).
5.  **Lab**: Apply local contrast (CLAHE), sharpening, and color enhancements.
6.  **Toning**: Apply split tones, sepia/selenium, or a paper profile.
7.  **Retouch**: Remove dust automatically or by hand.
8.  **Finish**: Add vignette and border.
9.  **Export**: Save your results as high-quality JPEG or TIFF.

The sidebar is organized as collapsible panels in roughly that order, plus a top-level Header (GPU toggle), a Files browser, Presets, an ICC panel, and a Metadata editor in the session tabs.

---

## 2. Header

A thin strip at the top of the sidebar with the NegPy logo, version, and a single switch:

*   **GPU Acceleration**: Toggles the WebGPU rendering path. Default on when a compatible GPU is detected; the checkbox is disabled (with a tooltip explaining why) on hardware without GPU support. Turn off to force the CPU pipeline if you suspect a driver issue or want deterministic numerical results.

---

## 3. Process Panel
The foundation of your edit — film type, exposure analysis, and roll-wide baselines.

*   **Mode**: Selects `C41`, `B&W`, or `E-6`. Changes the negative-to-positive math and invalidates the cache so the pipeline re-runs from scratch.
*   **Lock Bounds**: Freezes the analyzed normalization bounds for this image. Cropping or moving sliders that would normally re-analyze the frame stop doing so, which is useful once you've dialed in good bounds and don't want them recomputed.
*   **Analysis Buffer** (0.0–0.50): Insets the analysis window from the frame edge so film rebate, sprocket holes, and scanner borders don't skew the black/white-point detection. Raise it on scans with wide borders; lower it for tightly-cropped frames.
*   **D-Range Clip** (0–100): Statistical percentile clip on the histogram before NegPy locates the black/white points. Higher values discard more outlier pixels, which helps with dense, fogged, or specular negatives where a few stray bright/dark pixels would otherwise pull the bounds toward the extremes.
*   **White Point** (-0.25–0.25): Manual offset applied on top of the auto-detected white point. Positive values brighten; negative values pull highlights back down. Centered (0) means "use auto exactly."
*   **Black Point** (-0.25–0.25): Manual offset for the black point. Positive values lift the blacks; negative values deepen them.

### AUTO

*   **Normalize** (E-6 only): Auto-stretches the positive's histogram to fill the dynamic range. Useful for faded or expired slides; ignored in C41/B&W modes (where normalization runs on the negative density model instead).

### BATCH

*   **Batch Analysis**: Scans every loaded file and computes a "Roll Average" baseline — average per-channel density and color balance with outliers discarded. Run this once after importing a roll.
*   **Use Roll Average**: Toggles between per-image local normalization and the roll-wide baseline computed by Batch Analysis. Local gives each frame its own auto-exposure; roll-wide produces a consistent look across the roll.

### ROLL

*   **Roll dropdown**: Lists rolls previously saved to the database. Pick one to apply its normalization baseline to the current session.
*   **Load**: Applies the selected roll's bounds and balance to the current workspace.
*   **Save**: Prompts for a name and stores the current Batch Analysis result as a reusable roll (useful when you shoot the same film stock repeatedly).
*   **Delete**: Removes the selected roll from the database. Confirmation expected.

---

## 4. Geometry Panel
Crop and rotation.

*   **Aspect Ratio**: Choose a target ratio for cropping: `Free`, `3:2`, `4:3`, `5:4`, `6:7`, `1:1`, `65:24`, plus vertical variants (`2:3`, `3:4`, `4:5`, `7:6`, `24:65`).
*   **Detect Ratio**: Analyzes the frame and snaps the dropdown to the closest standard ratio. Handy for unknown formats.

### Crop tools (mutually exclusive toggles)

*   **Manual**: Enter manual-crop mode — drag a rectangle on the canvas.
*   **Move**: Translate the current crop rectangle without resizing it. Useful for nudging composition after auto-crop.
*   **Auto**: Runs automatic frame detection using the current ratio and offset. Best on scans with clean rebate.

### Adjustments

*   **Crop Offset** (-5.0 to 100.0 px): Insets the auto-crop border inward from the detected film edge. Positive values trim more; negative values bleed slightly outside the detected edge (useful when the detection clips a tiny bit too aggressively).
*   **Fine Rot** (-5.0° to 5.0°): Sub-degree rotation correction for tilted scans. Applied after auto-crop, so the cropped frame stays axis-aligned to the corrected image.

---

## 5. Exposure Panel
Shaping the light and color.

### Regional CMY

The three CMY sliders target one of three tonal regions selected by the radio group at the top:

*   **Global**: Affects the whole image (overall white balance).
*   **Shadows**: Color shift biased to low densities.
*   **Highlights**: Color shift biased to high densities.

For the selected region:

*   **Cyan** (-1.0–1.0): Cyan ↔ Red axis. Negative values pull toward cyan; positive toward red.
*   **Magenta** (-1.0–1.0): Magenta ↔ Green axis.
*   **Yellow** (-1.0–1.0): Yellow ↔ Blue axis.

Shortcut hints appear in tooltips (e.g. `E/D`, `R/F`) — see [KEYBOARD.md](KEYBOARD.md).

*   **Pick WB**: Activates the canvas eyedropper. Click any pixel you expect to be neutral grey and NegPy computes the CMY offsets that map it to true grey under the current region.
*   **Linear RAW**: When off (default), the camera's as-shot white balance is applied during RAW decode, giving a balanced starting point. Turn on to decode with neutral multipliers (1,1,1,1) and work from completely raw, untoned data.

### Exposure

*   **Density** (0.0–2.0): Overall darkness of the print — simulates exposure time under the enlarger. Lower values = brighter print, higher = denser.
*   **Grade** (0.0–5.0): Contrast grade, like switching paper grades in an analog darkroom. 0 is very soft (low contrast), 5 is very hard.

### Sigmoid Curve

Two transition zones at either end of the tone curve:

*   **Toe** (-1.0–1.0): Shadow transition into black. Positive values lift shadows for a gentler toe; negative values deepen blacks.
*   **Toe Width** (0.1–5.0): How broadly the toe transition is applied. Larger values spread the effect further into the midtones.
*   **Shoulder** (-1.0–1.0): Highlight transition into white. Positive values compress highlights for a gentle film-like roll-off; negative values extend them and risk clipping.
*   **Shoulder Width** (0.1–5.0): Spread of the shoulder region into the midtones.

---

## 6. Lab Panel
Final polish and detail. Several sliders are hidden in B&W mode (where color manipulation doesn't apply).

### Color (hidden in B&W)

*   **Separation** (1.0–2.0): Amplifies differences between the R, G, B channels. Higher values give richer, more saturated color separation; 1.0 is identity.
*   **Denoise** (0.0–5.0): Chroma denoise in Lab space. Smooths color noise (especially in shadows) while preserving the luminance grain that gives film its character.
*   **Saturation** (0.0–2.0): Linear saturation. 1.0 = unchanged, 0 = greyscale, 2.0 = double saturation. Neutral-center slider.
*   **Vibrance** (0.0–2.0): Smart saturation that boosts muted colors more than already-saturated ones — gentler on skin tones than raw Saturation. Neutral-center.

### Detail

*   **CLAHE** (0.0–1.0): Contrast Limited Adaptive Histogram Equalization. Adds local contrast without blowing global highlights or crushing shadows. Use sparingly — values near 1.0 can look cartoonish.
*   **Sharpening** (0.0–1.0): L-channel unsharp mask. Crisps detail without introducing color halos around edges.

### Effects

*   **Glow** (0.0–1.0): Lens bloom — bright highlights scatter equally across all channels, softening edges and giving a dreamy quality.
*   **Halation** (0.0–1.0): Simulates the red glow caused by light scattering back through the film base. Affects only highlights and is strongly red-dominant, as in real film halation.

---

## 7. Toning Panel
Color treatments applied late in the pipeline.

### Toners (B&W only)

*   **Selenium** (0.0–2.0): Simulates selenium toning. Adds a cool blue-purple cast to shadows that deepens with strength.
*   **Sepia** (0.0–2.0): Simulates sepia toning. Warm brown cast across the full tonal range.

### Split Tone

Two color injections — one in shadows, one in highlights — each with its own hue and strength. Works in any process mode.

*   **Shadow Hue**: Hue wheel for the shadow split-tone color.
*   **Shadow Strength** (0.0–1.0): How strongly the shadow hue is mixed in.
*   **Highlight Hue**: Hue wheel for the highlight split-tone color.
*   **Highlight Strength** (0.0–1.0): How strongly the highlight hue is mixed in.

### Paper

*   **Paper Profile**: Dropdown of bundled paper-substrate profiles (warm-tone, neutral, cool-tone, etc.). Picks a baseline tone the rest of the toning stacks on top of.

---

## 8. Retouch Panel
Cleanup and dust removal.

*   **Threshold** (0.01–1.0): Brightness delta above which a pixel is classified as dust during auto-detection. Lower values catch more (including false positives on real detail); higher values are conservative.
*   **Auto Size** (3–8 px): Maximum radius of auto-detected dust spots. Larger values catch bigger blobs but risk eating fine detail.
*   **Auto Dust**: Toggle that enables/disables automatic dust removal using the above two settings.
*   **Heal Tool**: Toggles the manual healing brush. With it on, click dust spots in the preview to paint them out one at a time.
*   **Brush Size** (2–16 px): Radius of the manual heal brush. Only shown while Heal Tool is active.

### HEALS · N

The section header shows the current count of manual spots. Both buttons are disabled when there are no manual heals.

*   **Undo Last**: Removes the most recent manual healing spot.
*   **Clear All**: Removes every manual spot (auto-detected dust is unaffected).

---

## 9. Finish Panel
Vignette and border — applied at the very end of the pipeline.

### Vignette

*   **Strength** (-1.0–1.0): Negative values darken the corners (classic vignette); positive values lighten them. 0 disables.
*   **Size** (0.0–1.0): Falloff radius. Smaller values keep the effect tight around the corners; larger values spread it well into the frame.

### Border

*   **Width** (0.0–2.5): Border thickness as a fraction of the image dimensions. Zero means no border.
*   **Border Color**: Square color swatch — click to open a color picker and pick any RGB color for the border.

---

## 10. ICC Panel
ICC profile for soft-proofing in the preview and (optionally) embedding in the export.

*   **Profile dropdown**: Lists `None` plus every ICC profile NegPy has discovered (system profiles + bundled profiles). Choose one to soft-proof against.
*   **Direction**:
    *   **Input**: Treat the chosen profile as the source profile (rarely used; helpful when you know the scan's profile but the file lacks an embedded tag).
    *   **Output**: Treat it as the destination profile (default). The preview is rendered as it would look through that profile.
*   **Apply to Export**: When on, the selected ICC profile is also applied to (and/or embedded in) exported files. When off, soft-proofing only affects the on-screen preview.

---

## 11. Metadata Panel
Custom film/process metadata plus per-image overrides of EXIF inherited from the source file.

### Custom

These fields are written into the exported file's metadata (XMP/EXIF as supported by the format).

*   **Film**: Film stock name — e.g. `Portra 400`, `HP5+`.
*   **Format**: Film format dropdown — `35mm`, `120`, `4×5`, `8×10`, or `Other`.
*   **Format (Custom)**: Free-text field that appears only when Format = `Other` (e.g. `6×7`).
*   **Developer**: Developer name and dilution, e.g. `D-76 1+1`.
*   **Push / Pull**: Process push or pull from `Push +3` down to `Pull -3`, with `Normal` in the middle.
*   **Scanning**: Scanner or scanning method, e.g. `Noritsu HS-1800`.
*   **Sync custom metadata to all files in batch export**: When on, all files in a batch export receive these custom metadata values.

### Inherited (from source)

Camera, Lens, and Exposure are populated read-only from the source file's EXIF. Each has a small lock-button on the right — click it to unlock the field and override the inherited value for this specific edit. Clicking again re-locks to the EXIF source.

*   **Camera**: `Make` + `Model` from the source EXIF, or your override.
*   **Lens**: `LensModel` from EXIF, or override.
*   **Exposure**: Formatted exposure string (e.g. `1/125s f/2.0 ISO 400`), or override.

---

## 12. Export Panel
Delivering the final image.

### Format

*   **Format**: `JPEG` (compressed) or `TIFF` (high bit-depth).
*   **Color Space**: `Same as Source`, `sRGB`, `Adobe RGB`, `ProPhoto RGB`, `Wide Gamut RGB`, `ACES`, `P3 D65`, `Rec 2020`, `XYZ`, or `Greyscale` (true B&W output).
*   **Paper Aspect Ratio**: Final paper ratio — `Original` (no resize), or one of the standard ratios for fitting print stock.

### Resolution (mutually exclusive)

*   **Original**: Export at the source RAW's full resolution.
*   **Print**: Print-size mode. Reveals:
    *   **Size** (1–500 cm): Long-edge physical print size in centimetres.
    *   **DPI** (72–4800): Print resolution.
*   **Pixels**: Pixel-dimension mode. Reveals:
    *   **Long edge** (256–32768 px): Longest dimension in pixels; the shorter side is derived from the paper aspect ratio.

### Destination

*   **Filename Pattern**: Jinja2 template for output filenames. Available variables: `original_name`, `colorspace`, `format`, `paper_ratio`, `size`, `dpi`, `target_px`, `border`, `date`. See [TEMPLATING.md](TEMPLATING.md) for examples.
*   **Overwrite existing files**: When on, exports replace files with the same name. When off, NegPy refuses to clobber.
*   **Same folder as source**: When on, exports go next to the source file; the Export Path input is disabled.
*   **Export Path**: Target directory when "Same folder as source" is off.
*   **Browse**: Opens a folder picker for the export path.

### Batch

*   **Export All**: Triggers batch export of every loaded file using the current settings.
*   **Sync export settings**: When on (default), the current Format / Color Space / Size / DPI / Border are applied uniformly to every file in the batch. When off, each file keeps its own per-image export settings.

---

## 13. Files (Session)
The file browser at the top of the session panel.

### File actions

*   **File**: Open one or more image files via a file picker.
*   **Folder**: Load every image in a chosen folder.
*   **Clear**: Unload all files from the current session.

### Hot folder & sync

*   **Hot Folder Mode**: Watches the current folder and auto-loads new files as they appear. Useful when paired with a scanner that drops files into the directory.
*   **Sync Edits**: Applies the current image's edit settings to every selected image (excludes crop and rotation, which are inherently per-image).

### Sorting

*   **Name / Date** (mutually exclusive): Sort the file list by filename or by file date.
*   **↑ Ascending / ↓ Descending** (mutually exclusive): Sort direction.

### Filter

*   **Filter input**: Filter the file list by filename (substring match by default).
*   **Regex toggle (`.*`)**: When on, the filter is interpreted as a regular expression.

### Session tabs

Switches the panel below between modes. These are containers, not edit controls — content for each is documented in its own section above:

*   **Analysis**: Histogram and photometric curve.
*   **Export**: Export panel (section 12).
*   **Metadata**: Metadata editor (section 11).
*   **Scan**: Scanner interface (Linux only; unavailable on Windows).

---

## 14. Presets

*   **Preset dropdown**: Lists every saved preset. Pick one and click **Load** to apply its settings to the current image.
*   **Load**: Applies the selected preset.
*   **Preset Name input**: Name for a new preset.
*   **Save**: Stores the current full WorkspaceConfig as a preset under the typed name.

---

## 15. Startup Override (`override.toml`)

If NegPy crashes on launch or has rendering glitches, you can force specific backend settings without touching code. On first run, NegPy creates `Documents/NegPy/override.toml` with defaults for your OS. Edit it and restart the app.

**Key settings:**

| Setting | Values | Effect |
|---------|--------|--------|
| `rendering.backend` | `"auto"`, `"vulkan"`, `"dx12"`, `"metal"`, `"cpu"` | GPU backend for image processing. `"cpu"` disables GPU entirely. |
| `display.qt_rhi_backend` | `"auto"`, `"vulkan"`, `"d3d12"`, `"metal"`, `"opengl"`, `"software"` | Qt UI rendering backend. |
| `display.qt_platform` | `"auto"`, `"xcb"`, `"wayland"` | Window system plugin (Linux only). |
| `performance.max_texture_size` | `"auto"` or a number, e.g. `4096` | Caps GPU texture size — reduce if you see out-of-memory errors on low-VRAM cards. |
| `performance.force_hq_preview` | `true` / `false` (or absent) | Overrides the saved HQ preview toggle. |
| `performance.preview_cache_max_bytes` | a number, e.g. `1200000000` | Memory budget for the preview cache. Lower it on low-RAM machines (default ~1.2 GB). |
| `performance.preview_cache_max_entries` | a number, e.g. `8` | Max recently-viewed photos kept in memory for instant navigation. |
| `logging.level` | `"debug"`, `"info"`, `"warning"`, `"error"` | Controls log verbosity. Use `"debug"` when reporting issues. |

**Common fixes:**

*   **App crashes immediately on Linux** → try `backend = "cpu"` or `qt_rhi_backend = "opengl"`.
*   **Black/blank preview on Windows** → try `backend = "dx12"` or `qt_rhi_backend = "software"`.
*   **Wayland rendering issues** → set `qt_platform = "xcb"` to force X11.
*   **GPU out-of-memory during export** → set `max_texture_size = 4096`.

---

## Additional Info
*   **Hardware Acceleration**: NegPy uses your GPU for near-instant previews & responsive sliders with exceptions of *Process* section (analysis buffer, white/black point offset, normalize) which use CPU for calculations.
*   **Roll Management**: Save your Batch Analysis as a "Roll" to apply the same look to future sessions with the same film stock.
*   **Database**: All edits live in a local SQLite database, keyed by file hash. You can move or rename files without losing your work.
*   **Edits**: Edits are saved to db on export/file change or when you explicitly save them. If you close the app without saving, your edits/settings will be lost.
*   **Keyboard Shortcuts**: [see here](KEYBOARD.md)
*   **Templating**: [see here](TEMPLATING.md)
*   **Pipeline**: [see here](PIPELINE.md)
