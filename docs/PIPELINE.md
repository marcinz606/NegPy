# How the Pipeline Works

Here's a breakdown of what happens to your image, step-by-step. We apply all the transformations in order, so the later steps can use the results of the previous ones.

## 1. Geometry (Straighten & Compose)
**Code**: `src.features.geometry`

* **Rotation**: We spin the image array (90° steps) and then fine-tune it with affine transformations (`cv2.warpAffine`). We use bilinear interpolation so we don't lose sharpness during the fine rotation.
* **Autocrop**: The code scans the image to find where the "film base" (the clear edge) ends and the actual picture begins by detecting the density jump. Some scans are tricky (not scanned flat, light source showing outside the film mask, light leaks, etc.), so as a fallback, there is a button to manually point to where the film border is.
* **Deferred Cropping**: We figure out the crop *now*, but we don't actually cut the pixels until the very *end*. This keeps the full image available for things like dust detection or blur algorithms that hate image edges. It is also crucial for the next normalization step, which can be easily thrown off by the film base or, even worse, the light source appearing outside of the negative.

---

## 2. Normalization (The Digital Negative)
**Code**: `src.features.exposure.normalization`

* **Logarithmic Conversion**: Digital cameras see light linearly; film sees it logarithmically. We convert the raw data to **Optical Density** using the standard formula $D = -\log_{10}(T)$.
* **Finding Bounds**: We calculate the "floor" (0.5th percentile) and "ceiling" (99.5th percentile) of the image data.
* **Normalization**: We stretch that range to `0.0 - 1.0`. This effectively strips away the film base so the rest of the pipeline works on pure image data.

---

## 3. Photometric Exposure (Making the Print)
**Code**: `src.features.exposure`

* **Concept**: This is the key part of the pipeline. We don't use a simple linear inversion. We model the response based on [H&D Curve](https://www.shutterbug.com/content/darkroombrwhat-do-those-h-and-d-curves-really-mean), simulating how the film emulsion and darkroom paper respond to light, effectively making a **positive print**.
* **Filtration**: We subtract C/M/Y offsets from the density values *before* the curve, effectively simulating physical filters in an enlarger head blocking specific wavelengths of light.
* **The Math**: We model the paper response using a **Logistic Sigmoid Function**:
    $$D_{out} = \frac{L}{1 + e^{-k(x - x_0)}}$$
    * $k$ is your Contrast Grade.
    * $x_0$ is your Exposure Pivot (Zone V).
* **The Toe (Shadows)**: The curved bottom part of the graph (low input, low output). This controls the deep shadows. By gently curving out of black, we ensure rich shadows without "crushing" texture immediately to zero.
* **Linear Region (Gamma)**: The straight middle part. The slope of this line ($\gamma$) defines the contrast. A steep slope means high contrast (Hard Grade); a shallow slope means low contrast (Soft Grade).
* **The Shoulder (Highlights)**: The curved top part of the graph (high input, high output). This rolls off the bright values gently. Instead of clipping harsh white (255), the highlights compress smoothly, mimicking the way photographic paper gradually loses its ability to reflect light.

The sliders in the Exposure & Tonality UI allow you to control the parameters of this curve (aka "your print"). You can observe the effect on the plotted curve, histogram, and the preview image itself.

---

## 4. Retouching (Dodge, Burn & Spot)
**Code**: `src.features.retouch`

* **Healing (Dust)**:
    * **Auto**: We look for high-frequency spikes (dust is sharp and small) and swap them with median values from the neighbors.
    * **Manual**: We use **Telea Inpainting**. To prevent the healed spot from looking too smooth or blurred, we synthesize fake grain and blend it back in to match the surrounding texture.
* **Local Adjustments**:
    * **Masks**: We generate masks from your brush strokes.
    * **Luminosity Masks**: We can restrict the mask to specific luminance ranges (e.g., "only affect the bright stuff").
    * **Application**: Mathematically, dodging/burning is applying a local multiplier to the light intensity *after* the exposure curve but *before* the final output.

---

## 5. Lab Scanner
**Code**: `src.features.lab`

The less "darkroom-y" part of the pipeline. We try to replicate the automated enhancements that lab scanners like the Frontier or Noritsu perform.

* **Color Separation**: We use a $3 \times 3$ **Color Correction Matrix (CCM)** to "un-mix" the colors. This corrects for the impure spectral response of the film dyes (cross-talk). Lab scanners have these matrices "baked in" for specific film stocks.
* **CLAHE**: Contrast Limited Adaptive Histogram Equalization applied in LAB space. It boosts local micro-contrast without distorting the global brightness. This is functionally similar to the "Hypertone" feature found in Fuji Frontier scanners.
* **Unsharp Mask (USM)**: A standard sharpening technique. We subtract a blurred version of the Lightness channel from the original to make edges pop without creating color artifacts.

---

## 6. Toning & Paper
**Code**: `src.features.toning`

* **Paper Simulation**: We tint the highlights (to simulate paper base color) and apply a gamma curve to the shadows to mimic the dynamic range and reflectivity of different paper surfaces (glossy vs. matte).
* **Chemical Toning**: We use a function that calculates pixel luminance and interpolates the color towards a Selenium or Sepia target vector. Since chemical toners react with silver, the effect is strongest in the dense areas (shadows) and weakest in the highlights (where there is less silver).

---

## 7. Final Crop
**Code**: `src.features.geometry`

Calculated and determined in Step 1, but applied last. By waiting until the very end, we ensure that all our median filters and blur kernels have plenty of border data to work with, preventing weird edge artifacts.

During export, we can also easily apply a border (white or any other color) without affecting the total desired print size.