# How the Pipeline Works

Here's a breakdown of what happens to your image, step-by-step. We apply all the transformations in order, so the later steps can use the results of the previous ones.

## 1. Geometry (Straighten & Compose)
**Code**: `src.features.geometry`

* **Rotation**: We spin the image array (90° steps) and then if needed fine-tune it with affine transformations (`cv2.warpAffine`). We use bilinear interpolation so we don't lose sharpness during the fine rotation.
* **Autocrop**: The code scans the image to find where the "film base" (the clear edge) ends and the actual picture begins by detecting the density jump. Some scans are tricky (not scanned flat, light source showing outside the film mask, light leaks, etc.), so as a fallback, there is a button to manually point to where the film border is.

**Cropping is crucial for the next step as normalization is easily thrown off by the film base or, even worse, the light source appearing outside of the negative.** So make sure that you crop the image to the actual picture using autocrop & crop offset. If you want to keep your film border in your final export you can just select `Keep Borders` checkbox.


---

## 2. Scan Normalization
**Code**: `src.features.exposure.normalization`

* **Physical Model**: We treat the input image not as a digital photo, but as a **radiometric measurement of the negative**. The sensor data represents light passing through the film density.
* **Logarithmic Inversion**: Since film density is logarithmic ($D \propto \log E$), and the scanner sensor is linear, we must mathematically invert the signal to recover the latent image:
$$E_{log} = \log_{10}(I_{scan})$$
* **D-Min / D-Max Calibration**: We analyze the scan to find the physical boundaries of the film strip:
  * **Floor ($P_{0.5}$):** The film base + fog (unexposed celluloid).
  * **Ceiling ($P_{99.5}$):** The densest highlight on the negative.
* **Normalization**: We stretch these bounds to $[0, 1]$. This effectively subtracts the orange film base and standardizes the density range, ensuring the next stage processes only the captured image data.

---

## 3. Photometric Exposure (The "Print")
**Code**: `src.features.exposure`

* **Virtual Darkroom**: This stage simulates the **optical printing process**. We are effectively shining light through the normalized "digital negative" onto virtual photographic paper to create a Positive.
* **Color Timing**: Before the print is made, subtractive filtration (CMY) is applied to the digital negative to correct color casts, mimicking the usage of a dichroic color enlarger head.
* **The H&D Model**: We model the paper's response using the **Hurter–Driffield Characteristic Curve**. For a given negative density $x$, the resulting print density $D$ is calculated via a **Logistic Sigmoid**:
$$D_{print} = \frac{D_{max}}{1 + e^{-k \cdot (x - x_0)}}$$
    * $D_{max}$: The deepest black the paper can achieve.
    * $k$: The paper contrast grade (slope).
    * $x_0$: The exposure time (Mid-gray point).
* **Slope Modulation (Toe & Shoulder)**: To capture the "analog look," the slope $k$ is dynamically modulated at the extremes to shape the positive image:
    * **Toe (Shadows)**: Controls the bottom of the curve (deep blacks). A softer toe ensures rich shadows without crushing texture immediately to zero.
    * **Shoulder (Highlights)**: Controls the top of the curve (bright whites). This rolls off highlights smoothly, simulating the chemical saturation limit of the paper emulsion.
* **Final Visualization**: The calculated print density is converted to reflected light (Transmittance) and gamma-corrected for display/digital printing:
$$I_{out} = (10^{-D_{print}})^{1/\gamma}$$

The sliders in the Exposure & Tonality UI allow you to control the parameters of this curve (aka "your print"). You can observe the effect on the plotted curve, histogram, and the preview image itself.

***There is no one right way on how the print should look like, automatic exposure is just a neutral starting point that you can adjust to your liking.***

---

## 4. Retouching (Dodge, Burn & Spot)
**Code**: `src.features.retouch`

* **Restoration (Dust & Scratch)**:
    * **Auto-Detection**: We perform statistical analysis of the local texture. The algorithm calculates the local standard deviation ($\sigma$) to distinguish between actual image detail and defects. Dust is identified only where the pixel deviation exceeds a threshold modulated by local flatness:
$$|I - \text{median}(I)| > T \cdot f(\sigma)$$
    * **Grain Re-synthesis**: When manually healing spots (using **Telea Inpainting**), simple blurring creates "plastic" artifacts. We solve this by injecting synthetic grain back into the healed area. The noise intensity is modulated by luminance ($L \cdot (1-L)$) to mimic the physics of film grain visibility (strongest in midtones).
* **Local Adjustments (Dodge & Burn)**:
    * **Geometry & Range**: Adjustments are defined by vector strokes converted to masks. These can be constrained by **Luminosity Masking** (e.g., "Burn only the highlights") using a soft-ramp function defined by range $[low, high]$ and softness $S$.
    * **Photometric Math**: we apply a true **Exposure Value (EV) Offset**. The pixel intensity is multiplied exponentially akin to increasing/decreasing exposure time in the darkroom:
$$I_{out} = I_{in} \cdot 2^{(\text{strength} \cdot \text{mask})}$$
      * A `strength` of +1.0 doubles the light (adds 1 Stop).
      * A `strength` of -1.0 halves the light (subtracts 1 Stop).

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

$$ L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CLAHE}(L, \text{clip}=2.5\alpha) $$

*  This is functionally similar to the "Hypertone" feature found in Fuji Frontier scanners.
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