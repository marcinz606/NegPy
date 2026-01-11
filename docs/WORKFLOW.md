# User Guide: The DarkroomPy Workflow

This guide walks you through the typical workflow of processing a RAW film scan in DarkroomPy, from importing your negatives to exporting a finished digital print.

---

## 1. Importing Files

DarkroomPy supports a variety of RAW formats (`.dng`, `.nef`, `.arw`, etc.) as well as TIFF files.

### Manual Import
Use the **Files** section in the sidebar to pick individual files or a whole folder. DarkroomPy calculates a unique hash for each file, so your edits remain linked even if you move or rename the files later.

### Hot Folder Mode
If you are scanning in real-time, enable **Hot Folder Mode**. Select the directory where your scanner saves files, and DarkroomPy will automatically load new images into the film strip.

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
This is the heart of the app. It simulates the response of film & photographic paper using an H&D characteristic curve.
- **CMY Filtration**: Simulate a color enlarger head. Use these sliders to fine-tune color balance to your liking (e.g., add Yellow to reduce Blue).
- **Density**: Adjust the overall brightness/exposure.
- **Contrast**: Change the "Paper Grade." High values produce a punchy look; lower values preserve more shadow and highlight detail.
- **Toe & Shoulder**: Fine-tune the "roll-off" of deep shadows and bright highlights to prevent digital clipping.

### Step C: Retouching
- **Dust Removal**: 
    - **Auto**: Detects and heals small dust specs automatically. You can play with detection threshold to fine-tune the results. Setting it too low might accidentally remove fine detail in the highlights misinterpreting it as dust.
    - **Manual Spotting**: Click on the image to remove larger distractions. DarkroomPy synthesizes matching grain to hide the repair.
- **Dodge & Burn**: Add local adjustment layers. Use the brush to paint exposure changes on specific areas. You can use **Luminance Masking** to ensure your "Burn" only affects the highlights.

### Step D: Photo Lab Tools
Simulate the enhancements of high-end lab scanners (like the Fuji Frontier). Play with color separation, CLAHE and sharpening to get the result you want.

### Step E: Toning & Paper Base
Optionally simulate different paper types and chemical toning (Selenium and Sepia, available in B&W mode only).

---

## 3. Managing Your Session
- **Copy/Paste Settings**: Use the buttons on side panelto copy settings from one frame and apply them to another. You can also easily reset the settings to default.
- **Presets**: Save your favorite "look" as a JSON preset to reuse later. Files are saved in `Documents/DarkroomPy/presets`. As simple json that you can edit "offline".

---

## 4. Exporting the Print

The Export module is designed to prepare your file for physical printing or high-quality sharing.
- **Print Size & DPI**: Specify the physical size of long edge (e.g., 30cm @ 300 DPI). DarkroomPy handles the math.
- **Soft Proofing**: Preview how your image will look when printed by loading your lab's `.icc` profile (Put it in `Documents/DarkroomPy/icc`)
- **Borders**: Add a clean white (or custom) border. The border is added *inside* the target print size, ensuring it fits your paper perfectly.
- **Format**: Export to high-quality JPEG or 16-bit TIFF in sRGB, AdobeRGB or Greyscale color space.
- **File Name**: You can template output file names using Jinja2 templating (see the tooltip for more info)
