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
*   **Inversion**: Film density is logarithmic ($D \propto \log E$), but scanners are linear. So we invert it to get back to the latent image:
    $$E_{log} = \log_{10}(I_{scan})$$
*   **Bounds**: We find the floor (film base + fog) and the ceiling (densest highlight).
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
*   **Toe & Shoulder**: We tweak the curve at the ends.
    *   **Toe**: Controls how fast shadows go to pure black.
    *   **Shoulder**: Controls how highlights roll off.
*   **Output**: Finally, we convert that print density back to light (Transmittance) for your screen:
    $$I_{out} = (10^{-D_{print}})^{1/\gamma}$$

This is where the "look" comes from. The defaults are neutral, but you can twist the curve however you want.

---

## 4. Retouching
**Code**: `src.features.retouch`

*   **Dust & Scratches**: We look for sharp spikes in local texture. If a pixel is way different from its neighbors (based on standard deviation), it's probably dust.
    $$|I - \text{median}(I)| > T \cdot f(\sigma)$$
*   **Grain Injection**: When you heal a spot, simple blurring looks fake ("plastic"). So we inject synthetic grain back into the healed area, scaled by the brightness (since grain is most visible in midtones).
*   **Dodge & Burn**: Standard darkroom tools. We multiply the pixel intensity to simulate giving it more or less light.
    $$I_{out} = I_{in} \cdot 2^{(\text{strength} \cdot \text{mask})}$$

---

## 5. Lab Scanner Mode
**Code**: `src.features.lab`

This mimics what a Fuji Frontier or Noritsu scanner does automatically.

*   **Color Separation**: We use a mixing matrix to push colors apart. It mixes between a neutral identity matrix and a "calibration" matrix based on how much pop you want.
    $$M = \text{normalize}((1 - \beta)I + \beta C)$$
*   **CLAHE**: Adaptive histogram equalization. It boosts local contrast in the luminance channel. This is basically the "Hypertone" feature from Frontier scanners.
    $$L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CLAHE}(L)$$
*   **Sharpening**: We sharpen just the Lightness channel in LAB space so we don't introduce color fringing.

---

## 6. Toning & Paper Simulation
**Code**: `src.features.toning`

*   **Paper Tint**: We multiply the image by a base color (e.g., warm cream for fiber paper) and tweak the D-max (density boost).
    $$v_{out} = \text{clip}\left( (v_{in} \cdot t_c)^\gamma, 0, 1 \right)$$
*   **Chemical Toning**: We can simulate Selenium or Sepia toning.
    *   **Selenium**: Targets the shadows ($1-Y^2$).
    *   **Sepia**: Targets the midtones using a Gaussian curve.
    $$P' = (1 - M) \cdot P + M \cdot (P \cdot C_{tone})$$
