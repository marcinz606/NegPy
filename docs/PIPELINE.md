# How the Pipeline Works

Here's a breakdown of what happens to your image, step-by-step. We apply all the transformations in order, so the later steps can use the results of the previous ones.

## 1. Geometry (Straighten & Compose)
**Code**: `src.features.geometry`

*   **Rotation**: We spin the image array (90° steps) and then fine-tune it with affine transformations (`cv2.warpAffine`). We use bilinear interpolation so we don't lose sharpness.
*   **Autocrop**: The code scans the image to find where the "film base" (the clear edge) ends and the actual picture begins. It detects the density jump. Some scans are tricky (not scanned flat, light source showing outside the film mask, light leaks, etc.) so as little help for the algorithm there is a button to to manually point where the film border is.
*   **Deferred Cropping**: We figure out the crop *now*, but we don't actually cut the pixels until the very *end*. This keeps the full image available for things like dust detection or blur algorithms that hate image edges. It is also crucial for next normalization step that can be easily thrown off by film base or even worse, light source outside of the negative.

---

## 2. Normalization (The Digital Negative)
**Code**: `src.features.exposure.normalization`

*   **Logarithmic Conversion**: Digital cameras see light linearly. Film sees it logarithmically. We convert the raw data to **Optical Density** using $D = \log_{10}(T)$.
*   **Finding Bounds**: We calculate the "floor" (0.5th percentile) and "ceiling" (99.5th percentile) of the image data.
*   **Normalization**: We stretch that range to `0.0 - 1.0`. This effectively strips away the film base so the rest of the pipeline works on pure image data.
---

## 3. Photometric Exposure (Making the Print)
**Code**: `src.features.exposure`

* Key part of the pipeline. We don't do simple linear inversion or basic s-curve. We base it on the [The H&D Curve](https://www.shutterbug.com/content/darkroombrwhat-do-those-h-and-d-curves-really-mean) (highly recommended reading).
*   **Filtration**: We subtract C/M/Y offsets from the density values *before* the curve, just like physical filters in enlarger head block light.
*   **The Math**: We model the film/paper response using a **Logistic Sigmoid Function**:
    $$D_{out} = \frac{L}{1 + e^{-k(x - x_0)}}$$
    *   $k$ is your Contrast Grade.
    *   $x_0$ is your Exposure Pivot (Zone V).
*   **Toe**: The curved bottom part. This is the "inertia" of the emulsion—it takes a certain amount of light to even start waking up the silver halides. This creates that soft, roll-off look in the highlights of a negative print.
*   **Linear Region (Gamma)**: The straight middle part. The slope of this line ($\gamma$) defines the contrast. A steep slope means high contrast (Hard Grade); a shallow slope means low contrast (Soft Grade).
*   **Shoulder**: The curved top part. Eventually, you run out of silver to react. The density flattens out, compressing the shadows so they don't clip harshly to black.

Sliders in Exposure & Tonality UI section allow you to control parameters of our our sigmoid curve. Aka "your print". You can observe the effect on ploted curve, histogram and on preview image itself.

## 4. Retouching (Dodge, Burn & Spot)
**Code**: `src.features.retouch`

*   **Healing (Dust)**:
    *   **Auto**: We look for high-frequency spikes (dust is sharp and small) and swap them with median values from the neighbors.
    *   **Manual**: We use **Telea Inpainting**. To stop the healed spot looking too smooth/blurred, we synthesize fake grain and blend it in.
*   **Local Adjustments**:
    *   **Masks**: We generate masks from your brush strokes.
    *   **Luminosity Masks**: We can restrict the mask to specific luminance (e.g., "only affect the bright stuff").
    *   **Application**: Mathematically, dodging/burning is just applying a local multiplier to the light intensity *after* the exposure curve but *before* the final output.

---

## 5. Lab Scanner
**Code**: `src.features.lab`

Less darkroom-y part of the pipeline. We try to replicate things that lab scanners like Frontier or Noritsu do well.

*   **Spectral Crosstalk**: We use a $3 \times 3$ matrix multiplication to "un-mix" the colors, effectively correcting for the impure spectral response of the film dyes. Lab scanners like Frontier and Noritsu have it "baked in".
*   **CLAHE**: Contrast Limited Adaptive Histogram Equalization applied in LAB space. It boosts local micro-contrast (texture) without messing up the overall brightness. Fuji Frontier calls this "Hypertone".
*   **Unsharp Mask (USM)**: Standard sharpening technique. We subtract a blurred version of the Lightness channel from the original to make edges pop without affecting the colors.

---

## 6. Toning & Paper 
**Code**: `src.features.toning`

*   **Paper Simulation**: We tint the highlights (paper base) and apply a gamma curve to the shadows to mimic the reflectivity of different paper surfaces.
*   **Chemical Toning**: We use a function that calculates pixel luminance and interpolates the color towards a Selenium or Sepia target vector. Since toners affect silver, the effect is strongest in the dense areas (shadows) and weak in the highlights.

---

## 7. Final Crop
**Code**: `src.features.geometry.processor.CropProcessor`

Calculated and detemined in 1st step but we apply it last. By waiting until the very end, we ensured that all our median filters and blur kernels had plenty of border data to work with, preventing weird edge artifacts.

During export we can also easily apply nice border (white or any other color you like) without affecting total desired print size. 