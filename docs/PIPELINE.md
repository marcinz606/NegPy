# How the Pipeline Works

DarkroomPy is a **Hybrid Analog/Digital** engine. Most editors treat your photo as a grid of pixels (RGB values). We treat it as a **physical negative** and the software as a simulation of light passing through it onto paper.

Here's a breakdown of what happens to your image, step-by-step. I've split each section into two parts:
1.  **The Darkroom Analogy**: What we're trying to simulate.
2.  **The Geeky Stuff**: How the code actually does it.

---

## 1. Geometry (Straighten & Compose)
**Code**: `src.features.geometry`

### 🎞️ In the Darkroom
You put your negative into the enlarger carrier. It's rarely perfectly straight, so you rotate the easel until the horizon looks good. Then you adjust the easel blades to crop out the messy film borders.

### 🤓 Under the Hood
*   **Rotation**: We spin the image array (90° steps) and then fine-tune it with affine transformations (`cv2.warpAffine`). We use bilinear interpolation so we don't lose sharpness.
*   **Autocrop**: The code scans the image to find where the "film base" (the clear edge) ends and the actual picture begins. It detects the density jump.
*   **Deferred Cropping**: We figure out the crop *now*, but we don't actually cut the pixels until the very *end*. This keeps the full image available for things like dust detection or blur algorithms that hate image edges.

---

## 2. Normalization (The Digital Negative)
**Code**: `src.features.exposure.normalization`

### 🎞️ In the Darkroom
Every roll of film has a "base density"—the thickness of the plastic backing plus the chemical fog ($D_{min}$). To get a good print, you need to ignore that base and focus on the actual image range.

### 🤓 Under the Hood
*   **Logarithmic Conversion**: Digital cameras see light linearly. Film sees it logarithmically. We convert the raw data to **Optical Density** using $D = \log_{10}(T)$.
*   **Finding Bounds**: We calculate the "floor" (1st percentile) and "ceiling" (99.5th percentile) of the image data.
*   **Normalization**: We stretch that range to `0.0 - 1.0`. This effectively strips away the film base so the rest of the pipeline works on pure image data.

---

## 3. Photometric Exposure (Making the Print)
**Code**: `src.features.exposure`

### 🎞️ In the Darkroom
The main event. You turn on the enlarger light.
*   **Filters**: You dial in some Cyan, Magenta, or Yellow on the color head to fix weird casts.
*   **Time**: The longer the light is on, the darker the print gets (Exposure/Density).
*   **Grade**: You pick a paper contrast. Grade 0 is flat and soft; Grade 5 is punchy and hard.
*   **Toe & Shoulder**: Photographic paper doesn't react linearly. Highlights roll off gently (Toe), and shadows crunch down (Shoulder).

### 🤓 Under the Hood
We use a **JIT-compiled Fused Kernel** here because it needs to be fast.
*   **The Math**: We model the film/paper response using a **Logistic Sigmoid Function**:
    $$D_{out} = \frac{L}{1 + e^{-k(x - x_0)}}$$
    *   $k$ is your Contrast Grade.
    *   $x_0$ is your Exposure Pivot (Zone V).
*   **Filtration**: We subtract C/M/Y offsets from the density values *before* the curve, just like physical filters block light.
*   **Curve Shaping**: We damp the sigmoid at the ends with polynomial functions to mimic the chemical saturation limits of real silver halides.

### 🧪 Science Break: The H&D Curve
In 1890, Ferdinand Hurter and Vero Charles Driffield published a paper that changed photography forever. They established the relationship between **Log Exposure** and **Optical Density**, creating the "Characteristic Curve" (or H&D Curve).

*   **Toe**: The curved bottom part. This is the "inertia" of the emulsion—it takes a certain amount of light to even start waking up the silver halides. This creates that soft, roll-off look in the highlights of a negative print.
*   **Linear Region (Gamma)**: The straight middle part. The slope of this line ($\gamma$) defines the contrast. A steep slope means high contrast (Hard Grade); a shallow slope means low contrast (Soft Grade).
*   **Shoulder**: The curved top part. Eventually, you run out of silver to react. The density flattens out, compressing the shadows so they don't clip harshly to black.

We don't just use a simple S-curve; we mathematically construct this exact behavior using a parameterized logistic function. When you move the "Grade" slider, you are literally changing the $\gamma$ (gamma) slope of this simulated emulsion.

---

## 4. Retouching (Dodge, Burn & Spot)
**Code**: `src.features.retouch`

### 🎞️ In the Darkroom
*   **Spotting**: You take a tiny brush and dye to fill in dust spots on the final print.
*   **Dodging**: You use your hands or a card to block light from shadow areas so they don't go pitch black.
*   **Burning**: You make a hole with your hands to blast extra light onto highlights to darken them.

### 🤓 Under the Hood
*   **Healing (Dust)**:
    *   **Auto**: We look for high-frequency spikes (dust is sharp and small) and swap them with median values from the neighbors.
    *   **Manual**: We use **Telea Inpainting**. To stop the healed spot looking too smooth/blurred, we synthesize fake grain and blend it in.
*   **Local Adjustments**:
    *   **Masks**: We generate masks from your brush strokes.
    *   **Luminosity Masks**: We can restrict the mask to specific luminance (e.g., "only affect the bright stuff").
    *   **Application**: Mathematically, dodging/burning is just applying a local multiplier to the light intensity *after* the exposure curve but *before* the final output.

---

## 5. Photo Lab Physics (Chemistry & Optics)
**Code**: `src.features.lab`

### 🎞️ In the Darkroom
*   **Acutance**: You might use a specific developer (like Rodinal) to make edges sharper.
*   **Cross-Talk**: Film dyes aren't perfect. The Magenta layer often absorbs a bit of Green light it shouldn't.
*   **Grain**: The physical texture of the silver crystals.

### 🤓 Under the Hood
*   **Spectral Crosstalk**: We use a $3 \times 3$ matrix multiplication to "un-mix" the colors, effectively correcting for the impure spectral response of the film dyes. Lab scanners like Frontier and Noritsu have it "baked in".
*   **CLAHE**: Contrast Limited Adaptive Histogram Equalization applied in LAB space. It boosts local micro-contrast (texture) without messing up the overall brightness. Fuji Frontier calls this "Hypertone".
*   **Unsharp Mask (USM)**: Standard sharpening technique. We subtract a blurred version of the Lightness channel from the original to make edges pop without affecting the colors.

---

## 6. Toning & Paper (The Final Look)
**Code**: `src.features.toning`

### 🎞️ In the Darkroom
*   **Paper Base**: Paper isn't pure white. "Warm Fiber" is creamy; "Cool RC" is bluish.
*   **Selenium**: A chemical bath that turns silver into silver selenide. It makes blacks deeper ($D_{max}$ boost) and adds a purple-red tint.
*   **Sepia**: Turns silver into silver sulfide for that classic warm brown look.

### 🤓 Under the Hood
*   **Paper Simulation**: We tint the highlights (paper base) and apply a gamma curve to the shadows to mimic the reflectivity of different paper surfaces.
*   **Chemical Toning**: We use a **Fused Kernel** that calculates pixel luminance and interpolates the color towards a Selenium or Sepia target vector. Since toners affect silver, the effect is strongest in the dense areas (shadows) and weak in the highlights.

---

## 7. Final Crop
**Code**: `src.features.geometry.processor.CropProcessor`

### 🎞️ In the Darkroom
Trimming the white borders off the dry print with a paper cutter.

### 🤓 Under the Hood
Remember that crop we calculated in Step 1? We finally apply it here. By waiting until the very end, we ensured that all our median filters and blur kernels had plenty of border data to work with, preventing weird edge artifacts.
