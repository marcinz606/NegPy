# NegPy User Guide

## 1. Core Workflow

NegPy follows a non-destructive pipeline:
1.  **Import**: Add files to your session.
2.  **Process**: Choose your film mode and perform roll-wide normalization.
3.  **Exposure**: Fine-tune the density, grade, and characteristic curve (Sigmoid).
4.  **Lab**: Apply local contrast (CLAHE), sharpening, and color enhancements.
5.  **Export**: Save your results as high-quality JPEG or TIFF.

---

## 2. Process Panel
The foundation of your edit.

*   **Process Mode**: Select between `C41 Negative`, `B&W Negative`, and `E6 (Positive)`.
*   **Analysis Buffer**: Adjusts the safety margin for auto-exposure. Increase if your scans have a lot of space around the actual image.
*   **White/Black Point Offset**: Manually shift the auto-normalization boundaries for more or less contrast.
*   **Normalize (E6 only)**: Automatically stretches the histogram for positive film. Useful for faded/expired slides.
*   **Batch Analysis**: Analyzes all loaded files to find a consistent "Roll Average" baseline. Calculates average density and color balance for the entire roll (after discarding outliers).
*   **Use Roll Average**: Toggles between local (per-image) and roll-wide exposure normalization.

---

## 3. Exposure Panel
Shaping the light and color.

*   **Regional CMY (Cyan, Magenta, Yellow)**: 
    *   **Global**: Adjusts the overall white balance.
    *   **Shadows/Highlights**: Applies targeted color shifts to specific regions of the density curve.
*   **Pick WB**: Select a neutral area in the image to automatically calculate white balance shifts.
*   **Camera WB**: Uses the white balance metadata from your RAW file as a starting point. REQUIRED for some cameras, turn on if your images come with heavy yellow/blue/green color cast out of the box.
*   **Density**: Controls the overall brightness, simulating exposure time in an analog darkroom. Lower values = brighter.
*   **Grade**: Controls the contrast, simulating different paper grades.
*   **Sigmoid Curve (Toe/Shoulder)**:

    *   **Toe**: Controls how shadows transition to black. Positive values add density to the shadow region.
    *   **Shoulder**: Controls how highlights transition to white. Positive values compress the highlights for a gentler roll-off.
    *   **Width**: Controls how broadly each transition region is applied.

---

## 4. Lab Panel
Final polish and detail.

*   **Separation**: Enhances color distinction by applying a separation matrix.
*   **Chroma Denoise**: Selectively reduces color noise in shadow areas using a Gaussian LAB pass.
*   **Saturation**: Basic saturation boost or reduction.
*   **Vibrance**: Smart saturation that targets muted colors more than vibrant ones.
*   **CLAHE**: (Contrast Limited Adaptive Histogram Equalization) provides local contrast enhancement without over-blowing highlights.
*   **Sharpening**: L-channel Unsharp Masking for crisp details without introducing color halos.
*   **Glow**: Simulates lens bloom. Bright highlights scatter light equally across all channels, softening edges and giving a dreamy quality.
*   **Halation**: Simulates the red glow caused by light scattering back through the film base. Affects only highlights and is strongly red-dominant, as in real film halation.

---

## 5. Retouch Panel
Cleanup and dust removal.

*   **Auto Dust**: Automatically detects and removes small particles based on a density threshold.
*   **Heal Tool**: Manual dust removal. Toggle the tool, then click on dust spots in the preview.
*   **Brush Size**: Controls the radius of the manual healing tool.
*   **Undo Last / Clear All**: Manage your manual retouching spots.

---

## 6. Export Panel
Delivering the final image.

*   **Format**: Choose between compressed `JPEG` or high-bit-depth `TIFF`.
*   **Color Space**: Standard `sRGB`, `Adobe RGB`, `Greyscale` (for true B&W) and some others.
*   **Resolution**: Export at `Original` RAW resolution or resize to a specific print size (cm) and DPI.
*   **Border**: Add a procedural border with custom width and color.
*   **Batch Export**: Process and save all loaded files using current or individual settings.

---

## Additional Info
*   **Hardware Acceleration**: NegPy uses your GPU for near-instant previews & responsive sliders with exceptions of *Process* section (analysis buffer, white/black point offset, normalize) which use CPU for calculations.
*   **Roll Management**: Save your Batch Analysis as a "Roll" to apply the same look to future sessions with the same film stock.
*   **Database**: All edits live in a local SQLite database, keyed by file hash. You can move or rename files without losing your work.
*   **Edits**: Edits are saved to db on export/file change or when you explicitly save them. If you close the app without saving, your edits/settings will be lost.
*   **Keyboard Shortcuts**: [see here](KEYBOARD.md)
*   **Templating**: [see here](TEMPLATING.md)
*   **Pipeline**: [see here](PIPELINE.md)
