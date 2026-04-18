# Change Log

## 0.14.0

- Added **Split Toning**: independent shadow and highlight color tinting with hue (0–360°) and strength controls. Works on both color and B&W scans. Applied in Lab space — luminance is preserved exactly.
- Improved normalization: simplified per-channel floor analysis for more consistent results across different film stocks.
- Increased CMY white balance range for stronger correction capability.
- Toning: Selenium & Sepia sliders now hidden in color mode; Paper Profile section moved below toning controls with a clear label.
- Exposure: Global / Shadows / Highlights region selector replaced with three toggle buttons; CMY label colors shift subtly per region for visual context.
- Collapsible sidebar sections now show a red dot indicator when they contain non-default values.
- **Per-section reset buttons**: each collapsible sidebar section header now has a reset button to restore that section's defaults without touching other settings.
- Lab: color-only sliders (Separation, Saturation, Vibrance, Denoise) now hidden in B&W mode instead of greyed out.
- Added info tooltips to sliders
- - **Canvas background color**: three swatch buttons in the toolbar (Black / Dark Grey / Mid Grey) to set the viewport background.
- **Keyboard shortcut overlay**: press `?` to open a modal reference of all shortcuts.
- GPU acceleration failure now shown as a status message instead of silently falling back to CPU.
- Stability: fixed GPU staging buffer leak on device loss/OOM; fixed dangling thread on app quit; thumbnail failures now surface as status bar errors.
- **Persistent UI preferences**: HQ preview toggle, canvas background color, auto dust removal default, and ICC profile settings (path, direction, apply-to-export) now survive app restarts.
- **Slider improvements**: sliders now show a tick mark at the default position; numeric units (px, °) displayed on relevant sliders; drag the slider label horizontally to scrub its value (hold Shift for fine control, Ctrl for coarse).
- **Cursor pixel readout**: hovering over the canvas shows the pixel's RGB and Lab values in the status bar.

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
- Added [USER_GUIDE.md](docs/USER_GUIDE.md)


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
- [keyboard](docs/KEYBOARD.md) shortcuts
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
