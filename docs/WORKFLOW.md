# User Guide: The DarkroomPy Workflow

This guide walks you through the typical workflow of processing a RAW film scan in DarkroomPy, from importing your negatives to exporting a finished digital print.

---

## 1. Importing Files

DarkroomPy supports a variety of RAW formats (`.dng`, `.nef`, `.arw`, etc.) as well as TIFF files.

### Manual Import
Use the **Files** section in the sidebar to pick individual files or a whole folder. DarkroomPy calculates a unique hash for each file, so your edits remain linked even if you move or rename the files later.

### Hot Folder Mode
If you are scanning in real-time, enable **Hot Folder Mode**. Select the directory where your scanner saves files, and DarkroomPy will automatically load new images as they appear in the film strip.

---

## 2. The Processing Pipeline

DarkroomPy uses a non-destructive pipeline. Your original RAW files are never touched; instead, a set of instructions is applied in a specific order to generate the preview and the final export.

### Step A: Geometry & Composition
Before adjusting colors, fix the orientation and framing.
- **Rotation**: Rotate the image in 90° increments.
- **Auto-Crop**: DarkroomPy attempts to detect the film borders automatically. If the detection fails (common with thin negatives), use the **Manual Assist** tool to click on the film base area.
- **Aspect Ratio**: Choose standard ratios like 3:2, 6:7, or 4:5.
- **Keep Full Frame**: Toggle this if you want to see the film borders and sprockets in your final print.

**Cropping is crucial for image analisis in next steps**. Having film borders or even worse, lightsource outside of negative visible in the working image will throw off the auto-exposure and auto-color algorithms. Make sure that the image is cropped to the film borders.

You can select **Keep Full Frame** if you want to see the film borders and sprockets in your final print/export.

### Step B: Photometric Exposure (The "Inversion")
This is the heart of the app. It simulates the response of photographic paper using an H&D characteristic curve.
- **Density**: Adjust the overall brightness/exposure.
- **Contrast**: Change the "Paper Grade." High values produce a punchy look; lower values preserve more shadow and highlight detail.
- **CMY Filtration**: Simulate a color enlarger head. Use these sliders to fine-tune color balance (e.g., add Yellow to reduce Blue).
- **Toe & Shoulder**: Fine-tune the "roll-off" of deep shadows and bright highlights to prevent digital clipping.

### Step C: Retouching
- **Dust Removal**: 
    - **Auto**: Detects and heals small dust specs automatically. You can play with detection threshold to fine-tune the results. Setting it too low might accidentally remove fine detail in the highlights misinterpreting it as dust.
    - **Manual Spotting**: Click on the image to remove larger distractions. DarkroomPy synthesizes matching grain to hide the repair.
- **Dodge & Burn**: Add local adjustment layers. Use the brush to paint exposure changes on specific areas. You can use **Luminance Masking** to ensure your "Burn" only affects the highlights.

### Step D: Photo Lab Tools
Simulate the enhancements of high-end lab scanners (like the Fuji Frontier).
- **Color Separation**: Improves color purity by reducing dye "cross-talk."
- **Hypertone**: Boosts micro-contrast to make details pop without changing the overall exposure.
- **Sharpening**: Apply luma-based sharpening that ignores film grain.

### Step E: Toning & Paper Base
- **Paper Tone**: Choose between Neutral, Warm, or Cool paper bases.
- **Chemical Toning**: For B&W images, simulate archival processes like **Selenium** and **Sepia** toning.

---

## 3. Managing Your Session

- **Film Strip**: The bottom bar shows all loaded files. Thumbnails are cached for speed.
- **Copy/Paste Settings**: Right-click or use the buttons to copy settings from one frame and apply them to another (or a whole selection).
- **Presets**: Save your favorite "look" as a JSON preset to reuse later.

---

## 4. Exporting the Print

The Export module is designed to prepare your file for physical printing or high-quality sharing.
- **Print Size & DPI**: Instead of pixel dimensions, specify the physical size (e.g., 30cm @ 300 DPI). DarkroomPy handles the math.
- **Soft Proofing**: Preview how your image will look when printed by loading your lab's `.icc` profile.
- **Borders**: Add a clean white (or custom) border. The border is added *inside* the target print size, ensuring it fits your paper perfectly.
- **Format**: Export to high-quality JPEG or 16-bit TIFF.

---

## Pro Tip: Caching
DarkroomPy caches the results of each pipeline stage. If you only change the **Toning** settings, the engine doesn't need to re-process the **Geometry** or **Exposure** steps, making adjustments feel snappy.
