# The Pipeline

Here is what actually happens to your image. We apply these steps in order, passing the buffer from one stage to the next.

## 1. Geometry (Straighten & Crop)
**Code**: `negpy.features.geometry`

*   **Rotation**: We spin the image array (90° steps) and fine-tune with affine transformations. We use bilinear interpolation so it stays sharp.
*   **Autocrop**: I try to detect where the film ends and the scanner bed begins by looking for the density jump. It's not perfect (light leaks or weird scanning holders can fool it), so there's a manual override.

**Note:** Cropping happens early because the normalization step needs to know what is "image" and what is "border" to calculate the black/white points correctly. Instead of cropping you can also use the "Analysis buffer" option to exclude outer X% of the image from the analysis. This is useful when you have a border around the film.

---

## 2. Scan Normalization
**Code**: `negpy.features.exposure.normalization`

*   **Physical Model**: We treat the input as a **radiometric measurement**. Pixel values represent linear transmittance captured by the sensor.
*   **Log Conversion**: Film density is logarithmic ($D \propto \log E$). We convert the raw signal to log-space to align with the physics of the film layers:
    $$E_{log} = \log_{10}(I_{raw})$$
*   **Bounding & Polarity**:
    The engine uses statistical percentiles to detect the usable signal range. To maintain a unified pipeline, we always map the target **White Point** to the **Floor** ($0.0$) and the **Black Point** to the **Ceiling** ($1.0$).
    *   **Negative (C-41/B&W)**: Raw low-signal (Film Base) maps to Floor ($0.0$). Raw high-signal (Highlights) maps to Ceiling ($1.0$). Range: 0.5% to 99.5%.
    *   **Positive (E-6)**: Raw high-signal (Highlights) maps to Floor ($0.0$). Raw low-signal (Shadows) maps to Ceiling ($1.0$). Range: 99.9% to 0.01%.
    
    The bounds are customizable via two controls:
    *   **D-Range Clip**: Tunes how aggressively the percentile window is set. **Positive** values symmetrically tighten the window before bounds detection — useful for very dense or fogged negatives where a few outlier pixels would otherwise pull the white or black point to an extreme. **Zero** uses robust extremes (a block-median prefilter rejects dust and speculars, and a small base clip excludes tiny outlier populations). **Negative** values push the bounds *outward* beyond the extremes, leaving lifted blacks and unclipped highlights as headroom.
    *   **White & Black Point Offsets**: Fine-tunes the detected bounds after statistical analysis. Shifting the White Point floor or Black Point ceiling enables precise highlight recovery or shadow crushing without re-running the analysis.
*   **Stretch**: All modes use independent channel bounding. This neutralizes the orange mask in negatives and base tints/fading in reversal film by stretching each channel to the full $[0, 1]$ range. The result is **not clamped**: tones outside the detected bounds are kept and rolled off later by the soft toe/shoulder of the print curve, rather than being truncated here.
*   **Per-frame metering**: Normalization also measures a few statistics used later by the Print stage's automatic helpers — per-channel **shadow references** ($P_{97.5}$, for Cast Removal) and a per-frame **exposure anchor** ($P_{50}$ luminance) and **textural range** ($P_{10}\text{–}P_{90}$, for Auto Density / Auto Grade). See §3.

---

## 3. The Print (Exposure)
**Code**: `negpy.features.exposure`

*   **Virtual Darkroom**: Simulates shining light through the normalized log-signal onto paper.
*   **Color Timing**: Applies subtractive filtration (CMY) in log-space. This mimics a dichroic head on an enlarger. Adjustments can be targeted to **Global**, **Shadows**, or **Highlights** regions; the shadow/highlight offsets are weighted by the toe/shoulder tonal masks (below).
*   **The H&D Curve**: Models paper response using a **Richards curve** (generalized logistic). The plain sigmoid is raised to a power $\nu$, which gives an asymmetric, paper-like toe while keeping a smooth shoulder. The curve lives in **density** space:
    $$D = D_{min} + (D_{asym} - D_{min}) \cdot \sigma(k \cdot x_{adj})^{\nu}$$
    *   $D_{min} = 0.06$: Paper white (the base isn't pure black). Toggle with **Paper White Base** (`paper_dmin`); off uses $D_{min} = 0$.
    *   $D_{asym} = 2.7$: The curve's *virtual* asymptote. The actual deepest black is the physical **D-max** $= 2.3$, enforced by a soft-saturation shoulder so density rolls into the floor instead of clipping.
    *   $\nu = 3.0$: Paper-toe sharpness.
    *   $k$: Per-channel slope (contrast), derived from **Grade**.
    *   $x_{adj}$: Adjusted input log-exposure (after toe, shoulder, and CMY offsets).
*   **Grade (ISO-R)**: Contrast is set as an **ISO range (R) value**, default 115, range 50–180 (R110 ≈ classic paper grade 2; higher R = softer). Edits saved under the old 0–5 paper-grade scale are auto-migrated via $R = 150 - 20 \cdot G$.
*   **Toe & Shoulder**: Two independent levers, both anchored so the rest of the tone scale stays put (slider values are scaled by $0.85$ internally):
    *   **Shoulder** — an integrated-sigmoid term on the *input* axis that modulates highlight local contrast, leaving the pivot tone invariant.
    *   **Toe** — a *density-domain* shadow lever that begins at an onset density of $1.2$, with its tangent removed so highlights remain invariant at any width. Lifts blacks or deepens shadows without touching the upper scale.
*   **Output**: Converts print density back to light (Transmittance), then encodes:
    $$I_{out} = 10^{-D}$$
    *   **Note**: The sRGB transfer (display gamma) is applied as the final encode.

### Automatic helpers

The defaults are tuned to look right straight out of the box; these helpers do per-frame work so you don't have to. All correct **partially** — they nudge toward a good result while preserving the photograph's intent. Turn them off to let the conversion follow the negative honestly (a dense negative prints dense, a flat one prints flat).

*   **Auto Density** (`auto_exposure`, **on**): Meters each frame's median tone and sets a sensible brightness. The exposure anchor is a partial move from an assumed key toward the measured median:
    $$\text{anchor} = 0.46 + 0.25 \cdot (P_{50} - 0.46), \quad \text{clamped to } \pm 0.12$$
    The $0.25$ blend (and $\pm 0.12$ band) means a deliberately low-key or high-key shot keeps its mood instead of being flattened to neutral grey.
*   **Auto Grade** (`auto_normalize_contrast`, **on**): Chooses contrast to suit each scene from the textural density range ($P_{10}\text{–}P_{90}$). Letting $r$ be the ratio of the full bounded range to the textural range, the effective contrast target is:
    $$0.6 \cdot \big(2.0 + 0.4 \cdot (r - 2.0)\big)$$
    The $0.4$ adaptation strength dampens contrast swings gently — a flat scene gets a small lift, a punchy scene stays punchy, nothing is pushed to a harsh extreme.
*   **Cast Removal** (`cast_removal`, **on**): Neutralizes the colour cast a negative leaves in the print, balancing each layer so greys read neutral from deep shadows through highlights — not just at the midtone (the usual cause of shadows/highlights drifting off-colour after a C-41 midtone white balance). Using the per-channel shadow refs ($P_{97.5}$), each non-green channel gets its own slope so its shadow ref lines up with green's, while the pivot (midtone) stays neutral:
    $$k_{ch} = k \cdot \frac{\text{anchor} - r_{green}}{\text{anchor} - (r_{green} - \text{cast}_{ch})}$$
    The per-channel cast is bounded ($\pm 0.125$) so the tilt can't run away.
*   **Contrast Lift / Surround** (`surround`, off): A gentle Bartleson–Breneman dim-surround correction. Prints viewed in a normal room want slightly more midtone contrast than a 1:1 reproduction, so this expands contrast about paper white:
    $$D = D_{min} + 1.10 \cdot (D - D_{min})$$
*   **Flare** (`flare`, off): A darkroom-style veiling-glare floor that lifts the deepest blacks and softens the toe for a more film-like look, while leaving paper white fixed. Applied in linear reflectance:
    $$I_{out} = \frac{I + f \cdot I_{white}}{1 + f}, \quad f = 0.005,\ I_{white} = 10^{-D_{min}}$$

With the helpers off, the conversion **shows you your photography** — exactly how the frame was exposed and developed. The defaults should be neutral, but you can (and should) use the sliders to match the curve shape (your "print") to your liking.

---

## 4. Retouching
**Code**: `negpy.features.retouch`

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
        $$I_{patch} = \frac{1}{3} \sum_{j=1}^{3} \text{min3x3}(P_{θ + Δ θ_j})$$
        *   $P_{θ + Δ θ_j}$: Perimeter point at pixel's angle $θ$ with random jitter $Δ θ$.
        *   This reconstructs the natural grain and texture of the surrounding area without using "synthetic" noise.
    3.  **Luminance Keying**: To preserve original details and grain within the brush, we only apply the patch to pixels that are significantly brighter than the reconstructed background:
        $$m_{luma} = \text{smoothstep}(0.04, 0.12, I_{curr} - I_{patch})$$
    4.  **Cumulative Patching**: Patches can be overlaid and stacked. The tool intelligently heals long hairs or scratches by basing each new patch on the current accumulated state.

*   **Resolution Independence**:
    Retouching coordinates and sizes are scaled relative to the full-resolution RAW data, ensuring that edits made on the preview translate perfectly to the high-resolution export.

---

## 5. Lab Scanner Mode
**Code**: `negpy.features.lab`

This mimics what lab scanners like Frontier or Noritsu do automatically. For maximum signal quality, the steps are applied in the following sequence:

1.  **Chroma Denoise**: Applies a Gaussian filter to the A and B channels in LAB space. This reduces color noise and digital "chroma speckle" while leaving the L-channel (and its film grain) completely untouched.
2.  **Crosstalk**: We use a mixing matrix (in density space) to push colors apart. It blends between a neutral identity matrix and a "calibration" matrix based on how much pop you want.
  
    $$M = \text{normalize}((1 - \beta)I + \beta C)$$
    *   $I$: Identity matrix (neutral).
    *   $C$: Calibration matrix.
    *   $\beta$: Crosstalk strength.

    A **profile dropdown** selects the calibration matrix $C$. The built-in matrix is always available as *Default*; you can also drop your own `.toml` calibration matrices into the `NegPy/crosstalk` folder (seeded with a starter example on first run) and pick one per film stock or scanner. See [`CROSSTALK.md`](CROSSTALK.md) for the file format and how to contribute matrices to the bundled gallery.

3.  **Vibrance**: Selectively boosts the saturation of muted colors using a chroma mask. The mask is strongest at zero chroma and fades to zero for already vibrant colors, preventing over-saturation of sensitive areas like skin tones.
4.  **Global Saturation**: A linear boost applied to all colors via the HSV saturation channel.
5.  **CLAHE**: Adaptive histogram equalization. It boosts local contrast in the luminance channel.
  
    $$L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CLAHE}(L)$$
    *   $L$: Luminance channel.
    *   $\alpha$: Blending strength.

6.  **Sharpening**: We sharpen just the Lightness channel ($L$) in LAB space using Unsharp Masking (USM). We apply a threshold to avoid amplifying noise.

    $$L_{diff} = L - \text{GaussianBlur}(L, \sigma)$$
    $$L_{final} = L + L_{diff} \cdot \text{amount} \cdot 2.5 \quad \text{if } |L_{diff}| > 2.0$$
    *   $\sigma$: Blur radius (scale factor).
    *   $2.5$: Hardcoded USM boosting factor.
    *   $2.0$: Noise threshold.

7.  **Glow**: Simulates lens bloom by blurring highlights and compositing them back using screen blending.

    $$I_{out} = 1 - (1 - I)(1 - B_{glow} \cdot s_{glow})$$
    $$B_{glow} = \text{GaussianBlur}(I \cdot m_{hl})$$

    *   $m_{hl}$: Luminance-based highlight mask, quadratically ramped from 50% to 100%.
    *   Applied equally to all three channels.

8.  **Halation**: Simulates the red scatter caused by light reflecting back through the film base. Uses a larger-radius Gaussian than Glow and a strongly red-biased highlight source.

    $$I_{out} = 1 - (1 - I)(1 - B_{hal} \cdot s_{hal})$$
    $$B_{hal} = \text{GaussianBlur}(I_R \cdot m_{hl} \cdot C_{hal})$$

    *   $I_R$: Red channel used as the scatter source.
    *   $C_{hal}$: Per-channel tint weights $(1.0,\ 0.3,\ 0.05)$ for red-dominant scatter.

---

## 6. Toning
**Code**: `negpy.features.toning`

*   **Chemical Toning** (B&W mode only): We simulate toner by blending the original pixel with a tinted version based on luminance ($Y$, Rec. 709) masks.
    *   **Selenium**: Targets the shadows (inverse squared luminance).
      
        $$m_{sel} = S_{sel} \cdot (1 - Y)^2$$
        $$I' = I \cdot (1 - m_{sel}) + (I \cdot C_{selenium}) \cdot m_{sel}$$
        *   $Y$: Pixel Luminance.
        *   $S_{sel}$: Selenium strength.
        *   $C_{selenium}$: Selenium target color $(0.85,\ 0.75,\ 0.85)$ (cool purple).
    *   **Sepia**: Targets the midtones using a Gaussian bell curve centered at $0.6$ luminance.
      
        $$m_{sep} = S_{sep} \cdot \exp\left(-\frac{(Y - 0.6)^2}{0.08}\right)$$
        $$I_{out} = I' \cdot (1 - m_{sep}) + (I' \cdot C_{sepia}) \cdot m_{sep}$$
        *   $S_{sep}$: Sepia strength.
        *   $C_{sepia}$: Sepia target color $(1.1,\ 0.99,\ 0.825)$ (warm gold).

*   **Chromaticity-Preserving Black Point** (B&W mode only): After chemical toning, the black point is re-seated about the $0.05$ luminance percentile, preserving the toner's hue:
    $$I_{out} = \frac{I - bp}{1 - bp}$$
    *   $bp$: $0.05$-percentile luminance.

*   **Split Toning** (all modes): Additive tint in LAB ($a^*b^*$) space, so luminance — and therefore grain and detail — is preserved. Shadows and highlights are pushed toward independent hue angles. With $L$ the CIELAB lightness ($0$–$100$):
    $$m_{shadow} = \text{clip}(1 - L/50,\ 0,\ 1), \qquad m_{highlight} = \text{clip}((L - 50)/50,\ 0,\ 1)$$
    For each region (using its hue $\theta$, strength $S$, and mask $m$):
    $$a^* \mathrel{+}= \cos\theta \cdot 20 \cdot S \cdot m, \qquad b^* \mathrel{+}= \sin\theta \cdot 20 \cdot S \cdot m$$