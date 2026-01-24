# The Pipeline

Here is what actually happens to your image. We apply these steps in order, passing the buffer from one stage to the next.

## 1. Geometry (Straighten & Crop)
**Code**: `src.features.geometry`

*   **Rotation**: We spin the image array (90° steps) and fine-tune with affine transformations. We use bilinear interpolation so it stays sharp.
*   **Autocrop**: I try to detect where the film ends and the scanner bed begins by looking for the density jump. It's not perfect (light leaks or weird scanning holders can fool it), so there's a manual override.

**Note:** Cropping happens early because the normalization step needs to know what is "image" and what is "border" to calculate the black/white points correctly.

---

## 2. Scan Normalization
**Code**: `src.features.exposure.normalization`

*   **Physical Model**: We treat the file not as a photo, but as a **radiometric measurement**. The pixel values represent how much light passed through the negative.
*   **Inversion**: Film density is logarithmic ($D \propto \log E$), but scanners and camera sensors are linear. So we invert it to get back to the log space:

    $$E_{log} = \log_{10}(I_{scan})$$
    *   $I_{scan}$: Raw linear input from scanner/camera using 16-bit precision.
*   **Bounds**: We find the floor (film base + fog) and the ceiling (densest highlight) We set those at 0.5th percentile and 99.5th percentile.
*   **Stretch**: We normalize these bounds to $[0, 1]$. This effectively subtracts the orange mask and gives us a clean signal to work with.

---

## 3. The Print (Exposure)
**Code**: `src.features.exposure`

*   **Virtual Darkroom**: This step simulates shining light through the negative onto paper.
*   **Color Timing**: We apply subtractive filtration (CMY) to the digital negative. This is exactly like using a dichroic head on an enlarger to remove color casts.
*   **The H&D Curve**: Paper doesn't respond linearly. We model its response using a **Logistic Sigmoid**:

    $$D_{print} = \frac{D_{max}}{1 + e^{-k \cdot (x - x_0)}}$$
    *   $D_{max}$: Deepest black the paper can do.
    *   $k$: Contrast grade.
    *   $x_0$: Exposure time.
    *   $x$: Input logarithmic exposure.
*   **Toe & Shoulder**: We tweak the curve at the ends.
    *   **Toe**: Controls how fast shadows go to pure black.
    *   **Shoulder**: Controls how highlights roll off.
*   **Output**: Finally, we convert that print density back to light (Transmittance) for your screen:

    $$I_{out} = (10^{-D_{print}})^{1/\gamma}$$
    *   $I_{out}$: Final display intensity.
    *   $\gamma$: Display gamma correction (2.2).

The defaults should be somewhat neutral, but you can (and should) use the sliders to match the curve shape (your "print") to your liking.

---

## 4. Retouching
**Code**: `src.features.retouch`

This stage removes physical artifacts like dust, hairs, and scratches from the negative. We use two complementary approaches:

*   **Automatic Dust Removal**:
    A resolution-invariant impulse detector and patching engine.
    
    1.  **Statistical Gating**: Uses dual-radius analysis. A local window ($3\times$ scaled) identifies luminance spikes, while a wide window ($4\times$ scaled) provides texture context. A cubic variance penalty ($w\_std^3$) aggressively raises detection thresholds in high-frequency regions (foliage, rocks) to minimize false positives.
    2.  **Peak Integrity**: Validates candidates via a strict 3x3 Local Maximum check and a $Z > 3.0$ sigma outlier gate. A strong-signal bypass ensures saturation-limited artifacts (hairs/scratches) are captured even if they form plateaus.
    3.  **Annular Sampling (SPS)**: Background data is reconstructed via Stochastic Perimeter Sampling. Samples are fetched from a ring strictly exterior to the artifact footprint, ensuring zero contamination from the dust luminance itself.
    4.  **Soft Patching**: Healed regions are integrated using distance-weighted alpha blending with cubic falloff and procedural grain injection to match local noise characteristics.

*   **Manual Healing (Stochastic Boundary Sampling - SBS)**:
    When you use the Heal tool, we fill the brush area using information from its own perimeter.
    
    1.  **Perimeter Characterization**: The tool identifies the cleanest background luminance at the edge of the brush circle. This sets a "Perimeter-Safe" floor to prevent dark artifacts in bright areas like skies.
    2.  **Stochastic Sampling**: For every pixel inside the brush, we sample the immediate boundary with small angular jitter:
        $$I_{patch} = \frac{1}{3} \sum_{j=1}^{3} \text{min3x3}(P_{\theta + \Delta \theta_j})$$
        *   $P_{\theta + \Delta \theta_j}$: Perimeter point at pixel's angle $\theta$ with random jitter $\Delta \theta$.
        *   This reconstructs the natural grain and texture of the surrounding area without using "synthetic" noise.
    3.  **Luminance Keying**: To preserve original details and grain within the brush, we only apply the patch to pixels that are significantly brighter than the reconstructed background:
        $$m_{luma} = \text{smoothstep}(0.04, 0.12, I_{curr} - I_{patch})$$
    4.  **Cumulative Patching**: Patches can be overlaid and stacked. The tool intelligently heals long hairs or scratches by basing each new patch on the current accumulated state.

*   **Resolution Independence**:
    Retouching coordinates and sizes are scaled relative to the full-resolution RAW data, ensuring that edits made on the preview translate perfectly to the high-resolution export.

---

## 5. Lab Scanner Mode
**Code**: `src.features.lab`

This mimics what lab scanners like Frontier or Noritsu do automatically.

*   **Color Separation**: We use a mixing matrix to push colors apart. It mixes between a neutral identity matrix and a "calibration" matrix based on how much pop you want.
  
    $$M = \text{normalize}((1 - \beta)I + \beta C)$$
    *   $I$: Identity matrix (neutral).
    *   $C$: Calibration matrix (vibrant).
    *   $\beta$: Separation strength.
        
    We use hardcoded calibration matrix for now that should be good for most cases but later I plan to add option to set your own presets for different filmstocks/looks.

*   **CLAHE**: Adaptive histogram equalization. It boosts local contrast in the luminance channel.
  
    $$L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CLAHE}(L)$$
    *   $L$: Luminance channel.
    *   $\alpha$: Blending strength.

*   **Sharpening**: We sharpen just the Lightness channel ($L$) in LAB space using Unsharp Masking (USM). We apply a threshold to avoid amplifying noise.
  
    $$L_{diff} = L - \text{GaussianBlur}(L, \sigma)$$
    $$L_{final} = L + L_{diff} \cdot \text{amount} \cdot 2.5 \quad \text{if } |L_{diff}| > 2.0$$
    *   $\sigma$: Blur radius (scale factor).
    *   $2.5$: Hardcoded USM boosting factor.
    *   $2.0$: Noise threshold.

---

## 6. Toning & Paper Simulation
**Code**: `src.features.toning`

*   **Paper Tint**: We multiply the image by a base color (e.g., warm cream for fiber paper) and tweak the D-max (density boost).
  
    $$I_{tinted} = (I_{in} \cdot C_{base})^{\gamma_{boost}}$$
    *   $I_{in}$: Input image.
    *   $C_{base}$: Paper tint RGB color.
    *   $\gamma_{boost}$: D-max density boost.

*   **Chemical Toning**: We simulate toning by blending the original pixel with a tinted version based on luminance ($Y$) masks.
    *   **Selenium**: Targets the shadows (inverse squared luminance).
      
        $$m_{sel} = S_{sel} \cdot (1 - Y)^2$$
        $$I' = I_{tinted} \cdot (1 - m_{sel}) + (I_{tinted} \cdot C_{selenium}) \cdot m_{sel}$$
        *   $Y$: Pixel Luminance.
        *   $S_{sel}$: Selenium strength.
        *   $C_{selenium}$: Selenium target color (purple/reddish).
    *   **Sepia**: Targets the midtones using a Gaussian bell curve centered at $0.6$ luminance.
      
        $$m_{sep} = S_{sep} \cdot \exp\left(-\frac{(Y - 0.6)^2}{0.08}\right)$$
        $$I_{out} = I' \cdot (1 - m_{sep}) + (I' \cdot C_{sepia}) \cdot m_{sep}$$
        *   $S_{sep}$: Sepia strength.
        *   $C_{sepia}$: Sepia target color (brown/gold).
