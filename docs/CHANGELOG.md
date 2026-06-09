# Change Log

## 0.24.0

- Added **Before/After**: toggle button (and `\` shortcut) to flash the un-graded auto conversion of the current frame, so you can see what your edits changed at a glance.
- Background pre-loading now respects each file's own Linear RAW setting, so neighbouring photos stay instant even when their settings differ.
- **Faster preview loading**: opening and navigating between photos is now near-instant. Previews use a quick decode for display (full quality is still used on export), recently-viewed images are kept in memory, and an embedded thumbnail shows immediately while the full preview loads. Next/previous files are pre-loaded in the background so flipping through a roll feels instant. @reederphill (#230)

## 0.23.2

- Fix: a manually cropped photo no longer fails to load with "asdict() should be called on dataclass instances" after switching to another photo and returning. The crop rectangle was reloaded as a list instead of a tuple, making the config unhashable and crashing the render cache. (#228)

## 0.23.1

- Fix: adjust thresholds for film type detection to reduce negatives falsely recognized as positives.
- Made **process mode autodetect** toggleable using button next to the process dropdown.


## 0.23.0

- **D-Range Clip** now extends into negative values for outward headroom. The slider previously bottomed out at the true min/max (no clipping); pulling it below 0 pushes the normalization bounds *beyond* the histogram extremes, leaving lifted blacks and unclipped highlights for a gentler-than-default stretch. Positive values behave exactly as before (clipping the tails for more aggressive recovery).
- Added **Process mode autodetect**: new files are now analyzed on load and the process mode (C41 / B&W / E-6) is set automatically — orange-mask detection for C41, channel-correlation for B&W, balanced channels for E-6. Detection only applies to files without saved settings, so it never overrides a mode you set yourself.
- Fix: JPEG export now uses **4:4:4** chroma subsampling instead of libjpeg's default 4:2:0, preserving full color resolution at quality 95. Fine film grain and color detail no longer pick up chroma artifacts on export. (#224)

## 0.22.3

- Fix: exported images now honor the source file's EXIF orientation and match the preview — rotation and crop no longer drift on export for files carrying an orientation tag. (#218)

## 0.22.2

- Fix: tiled GPU export now correctly applies IR dust removal — it was silently skipped when the image was split into tiles during high-resolution export. (#216)
- Fix: tiled GPU export no longer applies vignette independently per tile — vignette is now computed over the full frame so seams don't appear on large exports. (#217)

## 0.22.1

- Fix: **Batch Analysis** now respects each file's crop and orientation when computing the roll-wide baseline. Previously, files with large borders (e.g. 6×6 negatives in a 3:2 scan) skewed the average because analysis ran on the full frame. (#213)
- **Sync Crop**: split the Sync Edits button in two — `Sync Edits` (exposure / lab / toning / process settings, preserves per-file crop) and `Sync Crop` (manual crop + rotation only). Useful when every frame on a roll shares the same scanner mask.
- Added **Analysis Buffer overlay**: while moving the Analysis Buffer slider, the canvas shows a dim border around the excluded region with a dashed accent-colored boundary, so you can see exactly what's being analyzed. Disappears shortly after the last slider movement.
- Pre-batch warning: if none of the selected files have a crop set, Batch Analysis prompts before running and points to either cropping or raising the Analysis Buffer.
- Status bar during Batch Analysis now indicates per-file crop state (`[cropped]` vs `[full frame]`).
- Lowered default **Analysis Buffer** to 0.05 and clamped slider max to 0.25 to match the underlying clamp in analysis.

## 0.22.0

- **Tool-aware cursor**: mouse pointer changes shape to reflect the active tool — pointing hand for WB Pick, crosshair for Manual Crop, open/closed hand for Move Crop, hidden cursor (brush circle) for Heal.
- Added **IR Dust Removal**: uses the infrared channel from IR-capable scanners (Nikon Coolscan, Epson flatbeds with SilverFast iSRD, VueScan 4-channel output) to detect and inpaint dust and scratches with near-zero false positives. Toggle and threshold slider in the Retouch panel — controls are disabled automatically when no IR channel is present in the loaded file.
- IR channel is read from: 4-channel TIFFs with ExtraSamples (VueScan, NegPy's own scanner output), multi-page TIFFs with a grayscale IR page (SilverFast iSRD), and `_IR.tif` sidecar files.
- **Tooltips**: added detailed tooltips to every sidebar control — sliders, buttons, dropdowns, and checkboxes. Controls with keyboard shortcuts show dynamic shortcut chips that update when bindings are customised.
- **Optimizations**: optimizations to preview loading speed. @reederphill

  ## 0.21.0

  - Added initial **Scanner support** on Linux and macOS: new Scan tab — select a SANE-compatible scanner, choose resolution, bit depth, output format, and filename template; scanned files auto-load into the session. This is initial implementation, tested with Plustek 8100 on Arch Linux and latest macOS. As it often is with (old in most cases of film scanners) hardware support i cannot guarantee that it will support your scanner. **IMPORTANT:** check [README.md](../README.md) for information about SANE dependencies. **IMPORTANT**
  - Added **Lock Bounds** button in the Process section — freeze normalization bounds so cropping and re-analysis don't overwrite them; useful for locking in exposure after initial analysis.
  - Added **Copy/Paste Bounds** between files — transfer normalization bounds from one file to another (**CTRL+Shift+C** -> CTRL+V).
  - **Canvas zoom**: pinch-to-zoom gesture support; smooth trackpad scrolling; increased min/max zoom bounds. @reederphill
  - Fix: TIFF export with ICC profile applied produced 8-bit tiffs.
  - Fix: Custom metadata not written correctly in tiffs.
  - Fix: Hot folder mode caused session UI debouncing lag.
  - Fix: **Saturation** slider above 1.0 darkened already-saturated reds/blues. Saturation now scales chroma in CIELAB, preserving perceived lightness (matches Vibrance behavior). (#193)
  - Docs: More detailed [USER_GUIDE.md](USER_GUIDE.md)


## 0.20.0

- Added **filtering** options to film strip - details in [FILTERING.md](FILTERING.md)
- Fix: HQ preview scaling up crop offset setting.

## 0.19.0

- Added **Metadata panel**: new "Metadata" tab in the session panel — set Film stock, Format, Developer, Push/Pull, and Scanner info written as EXIF tags into exported files. Shows read-only inherited EXIF from the source file (camera make/model, lens, exposure settings). Optionally sync custom metadata across all files in a batch export.
- Added **Detect Aspect Ratio** button in the Geometry sidebar (crosshairs icon) — finds the film frame in the image and sets the crop ratio to the closest standard aspect ratio.
- Added **Same folder as source** export option — exports files to their source directory instead of a fixed export path.
- Added **Overwrite existing files** toggle — when disabled, exports get incrementing suffixes (`_2`, `_3`) to avoid overwriting.
- **Sticky flip**: Flip Horizontal/Vertical buttons now show a pressed state when active and their state persists across files and app restarts (applied to new files only — files with saved edits keep their own flip).
- Tutorial overlay now shows keyboard navigation hints at the bottom.
- Fix: mouse input in tutorial (Windows) @alessandrv
- Fix: exports are now written atomically — no partial files left on crash or interrupt.
- Fix: Rotate CCW/CW buttons (and `[` / `]` shortcuts) reversed direction when the image was flipped horizontally or vertically (but not both).
- Performance & stability improvements.

## 0.18.2

- Fix: GPU export crop now recomputed at full resolution instead of scaling up the preview ROI — fixes misaligned crop on export when using crop offset with autocrop or manual crop.
- Fix: applying autocrop no longer silently resets the crop ratio to "Free" — the user's chosen ratio is preserved.

## 0.18.1

- Fix: startup crash on Windows systems with non-UTF-8 locale (e.g. Traditional Chinese cp950) caused by reading the stylesheet and other files without explicit encoding.
- Fix: tutorial popup body text cut off on long steps (e.g. Lab panel) — body now scrolls when content exceeds available height.

## 0.18.0

- Added **Interactive Tutorial**: step-by-step overlay walkable from the toolbar, covering the full pipeline from loading files to export.
- Added **Move Crop** tool: translate an existing manual crop rectangle without resizing it — new button in the Geometry sidebar (disabled until a crop rect is set).
- Added **Target Pixel Size** export mode: alongside the existing Print DPI and Original Resolution options, you can now export to a specific long-edge pixel count. Old `use_original_res` workspace files are automatically migrated to the new `export_resolution_mode` field.
- Improved **Autocrop**: more robust edge detection; autocrop is now off by default and resets when the button is deselected. @alessandrv
- Fix: HQ preview no longer inflates bounds analysis — analysis is always run on the downsampled image, eliminating noise from single dead pixels or sharp dust spots that could throw off normalization. (#162)

## 0.17.2

- Fix: batch normalization and batch export progress bars now properly updates on status bar.
- Fix: export filename templating now preserves `original_name` verbatim — dashes, spaces, and multiple/leading/trailing underscores in the original filename are no longer mangled by the cleanup pass.
- Changed default export filename pattern from `positive_{{ original_name }}` to `{{ original_name }}`.
- Changed default export colorspace to be `Same as source` (detected from input file).
- UX: Status bar - replaced zoom % and resolution (shown elsewhere) with a file position counter (`3 / 12`) for batch workflows.

## 0.17.1

- Fix: new vignette shaders not being bundled with appimage on linux
- Fix: queue render cleanup on worker thread to prevent use-after-free (GPU)

## 0.17.0

- Improved default conversion/normalization process, now out-of-the-box results should be better & more natural.
- Camera white balance is now applied by default — images open with the camera's as-shot WB for a balanced starting point.
- Replaced the "Camera WB" toggle with a "Linear RAW" button. Enable it to decode RAW files with neutral (1,1,1,1) multipliers, bypassing the camera WB.
- Added **Vignette effect**: new Finishing sidebar section with Strength and Size sliders. Applies a radial cosine-falloff vignette post-crop in both CPU and GPU pipelines. Negative strength darkens edges, positive brightens them.
- Moved **border controls** from the Export section to the new Finishing section. Border Width is now a slider. Old workspace files with border settings are automatically migrated.
- Increased range of **Analysis buffer** slider.
- Better error recovery in edge cases (swapchain resizing, invalid crop ratios).
- Config deserialization now warns on unknown keys — old workspace files with renamed settings won't silently lose data.
- More tests for GPU-CPU feature parity.
- Removed some dependencies that were not used.
- Added test coverage reporting to CI.
- UI refinements.

## 0.16.0

- Fix: rendering crash on some DNG and TIFF files at extreme D-Range Clip values (`kth out of bounds` error from `np.percentile` on float32 arrays.
- Fix: NaN/inf pixels in loaded images no longer propagate into normalization bounds.
- Fix: color cast on some files (cr3, raf) files.
- Fix: unsupported RAW files now show a clear error in the status bar instead of crashing.
- Updated rawpy/libraw: fixes loading of Panasonic Lumix .RW2 files from newer camera models.
- Updated many other dependencies.

**Dropped intel mac support due to lack of support for newer rawpy/libraw!**

## 0.15.0

- Add customizable slider shortcuts for all controls @alessandrv
- Fix: preview display issues (Windows) @alessandrv
- Fix: JPEG and TIFF scans now correctly linearized from sRGB before processing — density curves and color response now physically accurate for scanned negatives.
- Fix: GPU pipeline failures now log a full stack trace instead of a single-line message, making hardware acceleration issues diagnosable.
- Fix: GPU readback buffer correctly unmapped on error, preventing render failures after a hardware exception.
- UX: Make UI more subdued, replace intensive reds with subtle greys.

## 0.14.2

- Fix regression: loading monochrome DNG files no longer crashes the app (segfault in display conversion and pixel readout).

## 0.14.1

- Added **`override.toml`**: a config file in `Documents/NegPy/` (created on first run) for forcing GPU backend, Qt rendering, texture size cap, and more — without touching the app. See [Troubleshooting](../README.md#-troubleshooting) for details.
- Fix: **D-Range Clip** slider now uses a logarithmic scale for much finer control at low values.
- Fix: **Autocrop Offset** maximum raised to 100, allowing larger offsets on wide-border scans.
- Fix: 16-bit colour TIFF exports now use a proper 16-bit ICC colour transform.
- Fix: copy-pasted edits not persisting without additional manual change or explicit "Save".
- UX: Added empty-state overlay on the canvas when no image is loaded.
- Performance & stability improvements.

## 0.14.0

- Added **Split Toning**: independent shadow and highlight color tinting with hue (0–360°) and strength controls. Works on both color and B&W scans. Applied in Lab space — luminance is preserved exactly.
- **Per-section reset buttons**: each collapsible sidebar section header now has a reset button to restore that section's defaults without touching other settings.
- **Canvas background color**: three swatch buttons in the toolbar (Black / Dark Grey / Mid Grey) to set the viewport background.
- **Keyboard shortcut overlay**: press `?` to open a modal reference of all shortcuts.
- **Persistent UI preferences**: HQ preview toggle, canvas background color, auto dust removal default, and ICC profile settings (path, direction, apply-to-export) now survive app restarts.
- **Slider improvements**: sliders now show a tick mark at the default position; numeric units (px, °) displayed on relevant sliders; drag the slider label horizontally to scrub its value (hold Shift for fine control, Ctrl for coarse).
- **Cursor pixel readout**: hovering over the canvas shows the pixel's RGB and Lab values in the status bar.
- **Right click menu**: added as quicker way to access some tools/functions when right-clickign on image preview.
- Improved normalization: simplified per-channel floor analysis for more consistent results across different film stocks.
- Increased CMY white balance range for stronger correction capability.
- Toning: Selenium & Sepia sliders now hidden in color mode; Paper Profile section moved below toning controls with a clear label.
- Lab: color-only sliders (Separation, Saturation, Vibrance, Denoise) now hidden in B&W mode instead of greyed out.
- Exposure: Global / Shadows / Highlights region selector replaced with three toggle buttons; CMY label colors shift subtly per region for visual context.
- Collapsible sidebar sections now show a red dot indicator when they contain non-default values.
- Added info tooltips to many sliders.
- Many small UI refinements.
- GPU acceleration failure now shown as a status message instead of silently falling back to CPU.
- Stability: fixed GPU staging buffer leak on device loss/OOM; fixed dangling thread on app quit; thumbnail failures now surface as status bar errors.

## 0.13.2

- Fix regression: Changing camera wb, crop ratio, crop offset, manual crop, or resetting crop now forces normalization bounds re-analysis.
- Fix: Process mode change now correctly clears both floor and ceiling bounds (previously only ceiling was cleared).
- Fix: Corrupted RAW files (partial data errors) no longer crash the loader — app falls back gracefully.
- Updated dependencies: PyQt6 6.11.0, wgpu 0.31.0. (might help crashes with newer nvidia drivers)

## 0.13.1

- Fix: Pick WB not updating magenta & yellow sliders

## 0.13.0

- Added *D-Range Clip* slider for controlling the bounds of normalization process.
- Added *Sort Order* settings for contact sheet UI section (sort by name or date, ascending/descending).
- Added *HQ* toggle for viewport preview (renders preview in full resolution).
- Improvements to histogram & photometric curve rendering.
- Overall performance & stability improvements.
- Make "Pick WB" tool more predictable by averaging 8x8 area instead of sampling simple pixel.
- Fix: Linux appimage crashing on some distros.
- Fix: Border added via export menu being picked up by histogram calculation.
- Fix: mouse wheel accidentally changing sliders and dropdowns when scrolling the sidebar.
- Fix: Crash when using the Clear button to remove all files from the session.

## 0.12.0

- Added functionality to reset slider on double click (slider iteself, not only label).
- Added ability to "unzoom" for easier cropping.

## 0.11.0

- Improved normalization/autoexposure.
  - More dynamic range.
  - More neutral defaults.
  - Improved batch analysis (more aggressive outlier detection).
- Streamlined controls
  - Combined shadows+toe & highlights+shoulder sliders
- Added glow & halation effects sliders to Lab section.


## 0.10.1

- Optimized database writes to prevent stuttering during active slider movement.
- Fixed manual crop tool not being restricted to image border.
- Apply/sync export settings to all loaded files by default when using "Export All" button.

## 0.10.0

- Added **Zoom & Pan** for preview:
    - Useful for cleaning dust :)
    - Mouse wheel to zoom (up to 400%).
    - left-click (or middle click when using tools like spot healing brush) drag to pan.
    - Discrete zoom slider in the toolbar.
- Added **Persistent Undo/Redo**:
    - Standard shortcuts (Ctrl+Z / Ctrl+Y).
    - Stores up to 100 steps per file in local SQLite database.
    - History survives app restarts and file switching.
    - Track number of edits on image overlay (lower left corner)
      - Also track number of heal spots in retouch toolbar section.
- Packaged some additional requirements in Linux appimage for easier (I hope) running on debian-derived distros.
- **Fixed(?) UI rendering issues on Windows**

NOTE: due to some backend changes in storing the edits you might get weird colors on your previously edited photos. Reset should get rid of that. Nuclear option is deleting `edits.db` and `settings.db` from NegPy folder in your Documents.


## 0.9.16

- Stability improvements when using Numba-compiled functions.
- Parallelized batch normalization.

## 0.9.15

- Stability improvements when loading files and generating thumbnails (specfically for tiff & pakon files).
- Fix: white & black point offset sliders working in wrong direction when in E6 mode.
- More UI refinements.


## 0.9.14

- Fix: export folder not being correctly set on some configs on first run.
- Fix: Camera WB setting not forcing bounds re-analysis which lead on color cast stacking instead of color cast removal.
- Added [USER_GUIDE.md](USER_GUIDE.md)


## 0.9.13

- Added **Shadow Color Cast Removal**: aggressively target and neutralize color casts in the deepest shadows.
- Added **Regional Color Timing**: independent CMY adjustment for Global, Shadows, and Highlights tonal regions.
- Added **Vibrance Slider**: selectively enhance muted colors while protecting already vibrant tones.
- Added **Chroma Denoise Slider**: reduce digital color noise in LAB space while preserving natural film grain in the L-channel.
- Added **White & Black Point Offsets**: manual sliders to adjust normalization boundaries for precise highlight recovery or shadow recovery on top of auto exposure.
- Added classic Shadows & Highlights slider using dynamic Gaussian-weighted offsets.
- Reordered LAB processing pipeline for maximum signal integrity.
- Many **UI refinements**
- Added popup to ensure that export folder is properly set.

## 0.9.12

- Added macOS Intel build

## 0.9.11

- Fix color casts on exported files when heavy white balance correction is applied 

## 0.9.10

- Initial release of "E-6" mode for positives/slides
    - Optional "Normalize" step that tries to save expired slides
- Fix regression from 0.9.9 that caused colorcasts on some files.

## 0.9.9

- Added button to sync edits to selected files
    - Multiselect files in film strip using ctrl/cmd + click or drag with shift + click
- Fix "tiling" on exports that sometimes appeared on high-res exports when using CLAHE
- Fix image alignment issues when using fine rotation + manual crop

## 0.9.8

- Fix white image/thumbnail on file change when not using batch analysis.

## 0.9.7

- Improve batch normalization by discarding outliers before calculating of averages.
- Bugfixes:
    - Another small fix to Pick WB after recent changes

## 0.9.6

- UI Improvements:
    - Moved process/analysis to separate section.
    - Added options to "save rolls" 
    - Added simple switch between roll average and individual analysis.
- Bugfixes:
    - Fixed regression in Pick WB tool behaviour introduced in 0.9.5

## 0.9.5

- Features:
    - Added "Batch Normalization" button that performs bounds analysis for all loaded files and applies averaged settings to all. 
    - Added button to sync apply export settings for all files.
    - Added support for JPEG scans/files.
- Bugfixes:
    - Improved folder loading & thumbnail generation error handling & stability.
    - Fix fine rotation when manual crop is applied (credit: https://github.com/rodg)
    - Fix occasional wrong autoexposure calculation when file is rotated 90 degrees

## 0.9.4

- Brand new, native desktop UI (pyqt6) instead of electron packaged streamlit app
    - better performance.
    - more responsive.
    - more stable.
    - instant preview when moving sliders.
    - double click on slider label to reset to defaults.
    - native manual crop tool.
    - native file picker.
    - thumbnail re-rendering on inversion.
- Implemented `Analysis Buffer` to ensure that analysis is not thrown off by film border or lightsource outside of it.
- Added `Camera WB` button to use vendor-specific white balance corrections (helps green/nuclear color casts on some files)
- GPU acceleration (Vulkan/Metal)
- [keyboard](KEYBOARD.md) shortcuts
- Bugfixes: improved handling of some raw files that previously resulted in heavy colorcasts and compresssion artifacts.

## 0.9.3

- Added white balance color picker for fine-tuning white balance (click neutral grey)
- Added manual crop options (click top left and bottom right corners to set it)
- Added basic saturation slider
- Added more border options
- Added original resolution export option
- Added Input/Output .icc profile support
- Added input icc profile for narrowband RGB (should mitigate common oversaturation issues)
- Added horizontal & vertical flip options
- UI redesign: main actions moved under the preview, film strip moved to the right.
- Add new version check on startup (Displays tooltip near the logo if new version is available)

## 0.9.2

- Make export consistent with preview (same demosaic + log bounds analysis)

## 0.9.1

- Explicit support for more raw extensions for file picker.
