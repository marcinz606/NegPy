# How the Pipeline Works

Here's a breakdown of what happens to your image, step-by-step. We apply all the transformations in order, so the later steps can use the results of the previous ones.

## 1. Geometry (Straighten & Compose)
**Code**: `src.features.geometry`

* **Rotation**: We spin the image array (90° steps) and then if needed fine-tune it with affine transformations (`cv2.warpAffine`). We use bilinear interpolation so we don't lose sharpness during the fine rotation.
* **Autocrop**: The code scans the image to find where the "film base" (the clear edge) ends and the actual picture begins by detecting the density jump. Some scans are tricky (not scanned flat, light source showing outside the film mask, light leaks, etc.), so as a fallback, there is a button to manually point to where the film border is.

**Cropping is crucial for the next step as normalization is easily thrown off by the film base or, even worse, the light source appearing outside of the negative.** So make sure that you crop the image to the actual picture Using autocrop + crop offset. If you want to keep your film border in your final export you can just select `Keep Borders` checkbox.


---

## 2. Normalization
**Code**: `src.features.exposure.normalization`

* **Logarithmic Conversion**: Digital cameras see light linearly; film sees it logarithmically. We convert the raw data to **Optical Density** using the standard formula $D = -\log_{10}(T)$.
* **Finding Bounds**: We calculate the "floor" (0.5th percentile) and "ceiling" (99.5th percentile) of the image data.
* **Normalization**: We stretch that range to `0.0 - 1.0`. This effectively strips away the film base so the rest of the pipeline works on pure image data.

---

## 3. Photometric Exposure
**Code**: `src.features.exposure`

* **Concept**: This is the key part of the pipeline. We don't use a simple linear inversion. We model the response based on [H&D Curve](https://www.shutterbug.com/content/darkroombrwhat-do-those-h-and-d-curves-really-mean), simulating how the film emulsion and darkroom paper respond to light, effectively making a **positive print**.
* **Filtration**: We subtract C/M/Y offsets from the density values *before* the curve
* **The Math**: We model the paper response using a **Logistic Sigmoid Function**:
    $$D_{out} = \frac{L}{1 + e^{-k(x - x_0)}}$$
    * $k$ is your Contrast Grade.
    * $x_0$ is your Exposure Pivot (Zone V).
* **The Toe (Shadows)**: The curved bottom part of the graph (low input, low output). This controls the deep shadows. By gently curving out of black, we ensure rich shadows without "crushing" texture immediately to zero.
* **Linear Region (Gamma)**: The straight middle part. The slope of this line ($\gamma$) defines the contrast. A steep slope means high contrast (Hard Grade); a shallow slope means low contrast (Soft Grade).
* **The Shoulder (Highlights)**: The curved top part of the graph (high input, high output). This rolls off the bright values gently. Instead of clipping harsh white (255), the highlights compress smoothly, mimicking the way photographic paper gradually loses its ability to reflect light.

The sliders in the Exposure & Tonality UI allow you to control the parameters of this curve (aka "your print"). You can observe the effect on the plotted curve, histogram, and the preview image itself.

***There is no one right way on how the print should look like, automatic exposure is just a neutral starting point that you can adjust to your liking.***

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

* **Color Separation**: Color separation is achieved by applying a row-normalized mixing matrix $M$ to the RGB density vectors $\mathbf{v}$: $\mathbf{v}_{out} = M \mathbf{v}_{in}$. The matrix $M$ is linearly interpolated between the identity matrix $I$ and the generalized color negative calibration matrix $C$ based on strength $\beta$, ensuring neutral axis linearity:

$$
M = \text{normalize}((1 - \beta)I + \beta C)
$$
* **CLAHE**:  
Local micro-contrast is enhanced by applying Contrast Limited Adaptive Histogram Equalization (CLAHE) strictly to the luminance component $L$ in CIELAB space. The final output is a linear blend of the original lightness and the equalized signal $L_{eq}$, controlled by the strength parameter $\alpha$:

    $$
    L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CLAHE}(L, \text{clip}=2.5\alpha)
    $$

  This is functionally similar to the "Hypertone" feature found in Fuji Frontier scanners.
* **Luma Sharpening**: The algorithm implements Luminance-Preserving Unsharp Masking in the CIELAB color space. It isolates the $L$ channel and applies a high-pass filter: $L' = L + \lambda(L - G_\sigma * L)$, subject to a noise threshold $|\Delta| > 2.0$. The chrominance channels $a$ and $b$ remain mathematically invariant to prevent saturation artifacts..

---

## 6. Toning & Paper
**Code**: `src.features.toning`

* **Paper Simulation**: The physical characteristics of the photographic paper are simulated by modifying the spectral reflectance and dynamic range of the image. Given an input channel value $v_{in}$, a per-channel tint factor $t_c$, and a density boost factor $\gamma$ (D-max):

$$
v_{out} = \text{clip}\left( (v_{in} \cdot t_c)^\gamma, 0, 1 \right)
$$

* **Tint ($t_c$):** Multiplicative factor simulating the base color of the paper stock (e.g., Warm Fiber vs. Cool Glossy).
* **Density Boost ($\gamma$):** A power-law transformation ($\gamma \neq 1.0$) that deepens blacks and expands effective density, simulating the specific D-max capability of the paper.

* **Chemical Toning**: The effect is applied selectively based on pixel luminance $Y$ (calculated via Rec. 709 coefficients). Reactivity masks are generated to target specific tonal ranges:

    * **Selenium (Shadows):** Targets high-density regions using a quadratic inversion of luminance.
    $$M_{sel} = S_{sel} \cdot (1 - Y)^2$$
    * **Sepia (Midtones):** Targets mid-tones using a Gaussian distribution centered at $Y=0.6$ with variance $\sigma^2 = 0.08$.
    $$M_{sep} = S_{sep} \cdot \exp\left(-\frac{(Y - 0.6)^2}{0.08}\right)$$

    The toning is applied sequentially by linearly interpolating between the original pixel value $P$ and the toned value ($P \cdot C_{tone}$):

    $$
    P' = (1 - M) \cdot P + M \cdot (P \cdot C_{tone})
    $$

    Where $C_{sel} \approx [0.85, 0.75, 0.85]$ (Purple shift) and $C_{sep} \approx [1.10, 0.99, 0.83]$ (Warm shift).

---

## 7. Final Crop
**Code**: `src.features.geometry`

Calculated and determined in Step 1, but applied last. By waiting until the very end, we ensure that all our median filters and blur kernels have plenty of border data to work with, preventing weird edge artifacts.

During export, we can also easily apply a border (white or any other color) without affecting the total desired print size.