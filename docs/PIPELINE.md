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
    
    Bounds are sampled on **two independent axes** (`_sample_log_bounds`): a **luma** pass fixes the floor/ceil mean (centre + span), and a **colour** pass fixes each channel's deviation from that mean. The two are recombined, so colour balance is tunable without compressing the luminance range. Identical channels (mono) give zero deviation at any clip. The controls:
    *   **Luma Range Clip** (`luma_range_clip`): Tunes how aggressively the *luminance* percentile window is set — the black/white-point span (dynamic range). **Positive** values symmetrically tighten the window before bounds detection — useful for very dense or fogged negatives where a few outlier pixels would otherwise pull the white or black point to an extreme. **Zero** uses robust extremes (a block-median prefilter rejects dust and speculars, and a small base clip excludes tiny outlier populations). **Negative** values push the bounds *outward* beyond the extremes, leaving lifted blacks and unclipped highlights as headroom.
    *   **Colour Clip** (`color_range_clip`): The absolute per-tail clip percentile for the per-channel colour deviation (white balance / orange-mask cast), independent of the luma span. A **tighter** (larger) clip gives a more robust, outlier-resistant channel balance; a **gentler** (smaller) clip samples nearer the extremes. The default neutral is `base_color_clip` ($1.0$); the slider spans log-interpolated percentiles either side of it.
    *   **White & Black Point Offsets**: Fine-tunes the detected bounds after statistical analysis. Shifting the White Point floor or Black Point ceiling enables precise highlight recovery or shadow crushing without re-running the analysis. A **[Global / R / G / B]** selector on the Process page scopes the sliders to per-layer trims (`white_point_trim_*` / `black_point_trim_*`) added on top of the global offsets — per-dye-layer film-base (Dmin) and Dmax correction, scanner-style per-channel levels (`per_channel_point_offsets`, single source for CPU/GPU; E6 negates; hidden in B&W).
*   **Stretch**: All modes use independent channel bounding. This neutralizes the orange mask in negatives and base tints/fading in reversal film by stretching each channel to the full $[0, 1]$ range. The result is **not clamped**: tones outside the detected bounds are kept and rolled off later by the soft toe/shoulder of the print curve, rather than being truncated here.
*   **Per-frame metering**: Normalization also measures a few statistics used later by the Print stage's automatic helpers — per-channel **shadow references** ($P_{98}$, for Cast Removal) and a per-frame **exposure anchor** ($P_{50}$ luminance) and **textural range** ($P_{10}\text{–}P_{90}$, for Auto Density / Auto Grade). See §3.
*   **Spectral crosstalk / dye unmix** (`crosstalk_strength` / `crosstalk_matrix`): applies a spectral-crosstalk matrix (`.toml` profiles, see docs/CROSSTALK.md) to the raw **negative** log densities *before* bounds analysis and the stretch. This is the physically correct domain — secondary dye absorptions are linear in negative dye density (Beer–Lambert), and the bundled matrices are derived from negative spectral dye-density curves. The matrix is blended with identity by strength and row-normalized (grays preserved); every meter (bounds, anchor, shadow refs, neutral axis) reads the unmixed film. Batch Analysis applies the same unmix — bounds measured under a different matrix are invalid for the render, so re-run it (and re-check locked bounds) after changing the matrix or strength.
*   **Scan-clip warning**: the fraction of source pixels at/above sensor white ($\ge 0.99$ linear) is reported per channel (`scan_clip_fractions`). In a negative scan the film base and scene shadows sit near sensor white, so clipping there irreversibly collapses distinct densities to $D=0$; the Analysis panel warns above 1%. This is a capture-side problem — no reconstruction is attempted.

---

## 3. The Print (Exposure)
**Code**: `negpy.features.exposure`

*   **Virtual Darkroom**: Simulates shining light through the normalized log-signal onto paper.
*   **Color Timing**: Applies subtractive filtration (CMY) in log-space. This mimics a dichroic head on an enlarger. Adjustments can be targeted to **Global**, **Shadows**, or **Highlights** regions; the shadow/highlight offsets are weighted by a smooth sigmoid about the midtone — $w_{sh} = \sigma(3 \cdot (v - z))$, where $z$ is the midtone zone centre (`anchor_target_density`) — so shadow weight rises with density and highlight weight falls. The Temperature slider, WB picker and temperature roll-lock all operate on the *selected region's* M/Y pair.
*   **The H&D Curve**: Models paper response as an **asymmetric toe-linear-shoulder** curve in **density** space. A straight line of slope $k$ through the exposure pivot is smoothly bounded above by the **toe** (shadows rolling into paper black) and below by the **shoulder** (highlights rolling into paper white). Both bounds are independent **softplus** knees, so each slider shapes only its own end of the scale (film/print convention). With $v = k \cdot (x_{adj} - x_0)$:
    $$v_1 = D_{min} + \frac{\text{softplus}\big(a_{hl} (v - D_{min})\big)}{a_{hl}} \qquad \text{(shoulder → paper white)}$$
    $$D = D_{max} - \frac{\text{softplus}\big(a_{sh} (D_{max} - v_1)\big)}{a_{sh}} \qquad \text{(toe → paper black)}$$
    *   $D_{min} = 0.06$: Paper white (the base isn't pure black). Toggle with **Paper White Base** (`paper_dmin`); off uses $D_{min} = 0$.
    *   $D_{max} = 2.3$: Physical deepest black (paper D-max). There is no separate virtual asymptote — the softplus toe rolls density into $D_{max}$ directly.
    *   $a_{sh}, a_{hl}$: Toe / shoulder knee sharpness, from `toe_sharpness_base` ($4.0$) and `shoulder_sharpness_base` ($3.0$) scaled by `toeshoulder_width_ref`$/$width.
    *   $k$: Per-channel slope (contrast), derived from **Grade**.
    *   $x_{adj}$: Adjusted input log-exposure (after CMY offsets); $x_0$ is the pivot.
*   **Variable-gamma paper S-curve**: Before the bounds, a midtone gamma boost adds an anchor-preserving S-shape — $v \mathrel{+}= \gamma \cdot w \cdot \tanh\big((v - v^{\ast})/w\big)$ (`paper_midtone_gamma` $= 0.15$, `paper_gamma_width` $= 0.6$). Centred on the reference tone $v^{\ast}$ so the anchor is preserved, easing to zero toward toe and shoulder — a real paper's continuously varying gamma. The **Snap** slider (`midtone_gamma`) is a user trim added to the paper's baseline $\gamma$; in R/G/B mode it retargets to per-layer trims (`midtone_gamma_trim_*`) on top of that — midtone crossover, evaluated per channel (`per_channel_midtone_gamma`, single source for CPU/GPU/chart).
*   **Grade (ISO-R)**: Contrast is set as an **ISO range (R) value**, default 115, range 50–180 (R110 ≈ classic paper grade 2; higher R = softer). The straight-line slope is $k = \text{(grade contrast scale)} \cdot \text{range} / (R/100)$ (`grade_contrast_scale` $= 2.9$), clamped to $[2.0, 10.0]$ — the literal H&D gamma (negative density range over paper exposure range). Edits saved under the old 0–5 paper-grade scale are auto-migrated via $R = 150 - 20 \cdot G$.
*   **Per-layer trims (crossover correction)**: each dye layer has its own characteristic curve; the **Global / R / G / B** selector on the Tone page trims one layer relative to the shared curve. CMY filtration can only *shift* a layer's curve in parallel — it cannot fix **crossover** (shadows cast one colour, highlights the complement), which is a per-layer curve-*shape* mismatch. The trims are:
    *   **Grade trim** (`grade_trim_*`, ±30 ISO-R points): folds into the layer's slope exactly like a paper's `channel_gamma` — since $k \propto 1/R$, a trim is the pure ratio $R/(R+\Delta R)$ — and the pivot is re-solved per channel, so the layer rotates about the anchor and midtones stay neutral.
    *   **Toe / Shoulder trims** (`toe_trim_*` / `shoulder_trim_*`, ±1 on top of the global knee): per-layer endpoint casts — one layer's shadow or highlight knee moves, the other layers and the opposite end stay put. Effective per-channel values are clamped to the slider domain (`per_channel_toe_shoulder`, single source for CPU/GPU/chart).
    *   **Snap trim** (`midtone_gamma_trim_*`, ±0.5 on top of the global Snap): per-layer midtone gamma — a cast that lives only in the midtones while endpoints and the anchor stay neutral (midtone crossover).
    *   **Width trims** (`toe_width_trim_*` / `shoulder_width_trim_*`, ±2 on top of the global Widths, effective values clamped to the width domain [0.1, 5]): per-layer knee *sharpness* — how far one layer's roll-off reaches into the tonal scale, complementing the height trims (`per_channel_widths`, single source for CPU/GPU/chart).
*   **Toe & Shoulder**: Two independent levers (slider values scaled by $0.85$ internally), evaluated **per channel** (global value + layer trim). The slider sets roll-off **height**; **sharpness** comes from the width control — itself per channel (global width + layer trim):
    *   **Toe** — shadows. Lifts the paper-black ceiling: $D_{max,eff} = D_{max} - \text{toe} \cdot 0.90$ (`toe_height`). Deliberately larger than `shoulder_height`: density is $\log_{10}$, so a $\Delta D$ near $D_{max}$ is perceptually far smaller than the same $\Delta D$ near $D_{min}$ — 0.90 roughly evens out the two sliders in $L^{\ast}$. Negative toe instead *sharpens* the shadow knee — and, with **Paper Black** off, raises the BPC clip point (see Output below), which is what makes exact black attainable.
    *   **Shoulder** — highlights. Lifts the paper-white floor (compresses/greys highlights): $D_{min,eff} = D_{min} + \text{shoulder} \cdot 0.35$ (`shoulder_height`).
    *   **Grade-coupled baseline**: hard grades (high slope) physically have snappier toes and compressed shoulders, so a slope-proportional amount is added automatically (`toe_grade_strength` $\approx 0.058$ — rescaled with the `toe_height` retune so the baseline $\Delta D$ matches the old $0.15 \cdot 0.35$ — and `shoulder_grade_strength` $= 0.12$, scaled by the normalized slope).
*   **Zone Density (ΔD)**: two achromatic sliders (`shadow_density` ±0.9, `highlight_density` ±0.5) brighten/darken the shadow and highlight zones without reshaping the knees — the slider value is a literal density offset at full zone weight. Unlike the regional CMY (a broad complementary blend that pushes half of each offset into the mids), each slider has its own **mid-sparing** weight centred in the three-quarter/quarter tones: $v \mathrel{+}= \Delta D_{sh} \cdot \sigma\big(k(v - z_{sh})\big) + \Delta D_{hl} \cdot \big(1 - \sigma(k(v - z_{hl}))\big)$ with $z_{sh} = z + 0.75$, $z_{hl} = z - 0.40$, $k = 4$ (`zone_density_*` constants, mirrored as literals in `exposure.wgsl`) — midtones get neither offset. Applied before the softplus bounds, so a shadow burn can never exceed paper black and a highlight bleach never crosses paper white; a highlight burn shows first in the quarter-tones (near paper white the shoulder bound absorbs it, like a real print). Ranges are asymmetric because density is $\log_{10}$ — an equal $\Delta D$ reads far smaller near $D_{max}$ than near $D_{min}$. The chart mirrors the shift (`CharacteristicCurve`).
*   **Output**: Converts print density back to **scene-linear** reflectance (transmittance):
    $$I_{out} = 10^{-D}$$
    *   **Paper Black** (`paper_black`, off): off applies black point compensation, the same idea as ICC relative-colorimetric soft-proofing — a reflection print's D-max ($2.3$) floors reflectance at $10^{-2.3} \approx 0.005$, but the adapted eye reads paper black as black, so the display should too; on preserves the paper's lifted D-max instead. With compensation (the default), per channel, with $t_b = 10^{-D_b}$:
        $$I_{out} = \frac{I - t_b}{1 - t_b}, \quad D_b = D_{max} + \text{toe}_{ch} \cdot 0.90 \text{ (for } \text{toe}_{ch} < 0\text{)},\ D_{max} \text{ otherwise}$$
        clamped at $0$. The curve reaches $D_{max}$ only asymptotically, so a **negative toe raises the clip point** into the shadows — that's what makes exact $0$ reachable ("negative toe deepens blacks", literally); a lifted toe and per-layer shadow casts survive because the reference is the *physical* $D_{max}$, not $D_{max,eff}$. A negative per-layer toe trim (with compensation on) tints the deepest black.
    *   **Note**: The pipeline is **scene-linear internally** — the exposure stage emits linear light and every creative stage (Retouch, Lab, Local, Toning, Finish) operates on it. The working-space OETF (the **Adobe RGB (1998) TRC**, a pure $563/256 \approx 2.199$ power with no linear segment) is applied **only as the final engine step** (the output transform), so it composes correctly with the Adobe RGB ICC profile at the display/export boundary. Retouch's dust *detection* is perceptual, so on the CPU it computes its luma on a display-encoded copy while healing in linear; the GPU keeps a single encoded perceptual region (exposure → clahe/retouch encoded → lab decodes back to linear).

### Automatic helpers

The defaults are tuned to look right straight out of the box; these helpers do per-frame work so you don't have to. All correct **partially** — they nudge toward a good result while preserving the photograph's intent. Turn them off to let the conversion follow the negative honestly (a dense negative prints dense, a flat one prints flat).

*   **Auto Density** (`auto_exposure`, **on**): Meters each frame's median tone and sets a sensible brightness. The exposure anchor is a linear partial pull from the assumed key (`assumed_anchor` $= 0.46$) toward the measured median:
    $$\text{anchor} = \text{assumed} + s \cdot (P_{50} - \text{assumed}), \quad \text{clamped to } \pm b$$
    with $s =$ `anchor_meter_strength` ($0.2$) and $b =$ `anchor_meter_band` ($0.12$). The partial blend (and the band) means a deliberately low-key or high-key shot keeps its mood instead of being flattened to neutral grey. The anchor then prints at `anchor_target_density` ($0.74$), which is what sets overall print brightness.
*   **Auto Grade** (`auto_normalize_contrast`, **on**): Chooses contrast to suit each scene from the textural density range ($P_{10}\text{–}P_{90}$). Letting $r$ be the ratio of the full bounded range to the textural range, the effective contrast target is:
    $$K \cdot \big(n + \sigma \cdot (r - n)\big)$$
    with $K =$ `auto_grade_target` ($0.5$), $n =$ `auto_grade_nominal_ratio` ($2.0$, the ratio of a "normal" negative) and $\sigma =$ `auto_grade_strength` ($0.5$). The adaptation strength dampens contrast swings — at $\sigma = 0$ every frame gets the same fixed grade, at $\sigma = 1$ every frame is normalized to identical contrast; the default sits between, so a flat scene gets a lift and a punchy scene stays punchy without being pushed to a harsh extreme.
*   **Set Targets** (app-global): the five numbers above that set the *aim* — `anchor_target_density`, `anchor_meter_strength`, `anchor_meter_band`, `auto_grade_target`, `auto_grade_strength` — are user-tunable from a dialog beside the two toggles (`TUNABLE_TARGETS` in `features/exposure/models.py`, ranges declared there). They are a **calibration, not per-image state**: `apply_targets()` overlays them onto `EXPOSURE_CONSTANTS` and they persist in the `exposure_targets` global setting, so they apply to every frame including already-edited ones. Because no `WorkspaceConfig` hash sees them, both engines fold a `TARGETS_REVISION` counter into their cache keys — the CPU base stage (which does the metering) and, on the GPU, the analysis cache key plus the exposure-stage diff. Anything added to `TUNABLE_TARGETS` that is read *outside* those stages needs its own invalidation.
*   **Cast Removal** (`cast_removal_strength`, default $0.5$; $0$ turns it off): Neutralizes the colour cast a negative leaves in the print, balancing each layer so greys read neutral from deep shadows through highlights — not just at the midtone (the usual cause of shadows/highlights drifting off-colour after a C-41 midtone white balance). The applied strength is `confidence × slider` (`effective_cast_strength`): how cleanly the frame's near-neutrals read biases the correction, so a scene with few greys is corrected gently and the slider trims on top.

    The primary solve fits each non-green channel to green's **neutral axis** — per-channel references taken at a highlight, midtone and shadow luma band, each over that band's lowest-chroma pixels (`neutral_axis_*` constants). R and B get a **quadratic** through all three green-matched points, so highlights don't extrapolate past neutral; the midtone is pinned exactly via the pivot. Each channel's deviation from green is clamped ($\pm 0.2$, `midtone_cast_max_offset`) and the curvature is bounded to a fraction of the slope (`neutral_axis_curv_max_ratio` $= 0.45$) to keep the per-channel core monotonic on $[0,1]$.

    When no neutral axis is available (too few trustworthy near-neutrals), it falls back to a **two-point tie** on the per-channel shadow refs ($P_{98}$): each non-green channel gets its own slope so its shadow ref lines up with green's, with the luma anchor pinning the midtone:
    $$k_{ch} = k \cdot \frac{\text{anchor} - r_{green}}{\text{anchor} - (r_{green} - \text{cast}_{ch})}$$
    The per-channel cast is bounded ($\pm 0.1$, `cast_removal_max_offset`) so the tilt can't run away.
With the helpers off, the conversion **shows you your photography** — exactly how the frame was exposed and developed. The defaults should be neutral, but you can (and should) use the sliders to match the curve shape (your "print") to your liking.

### Paper profiles
**Code**: `negpy.features.exposure.papers`

A **paper profile** (`paper_profile`, default *Neutral*) overrides the print *character* — the H&D curve shape — without touching contrast or exposure. Each profile sets the paper's $D_{max}$/$D_{min}$, toe/shoulder knee sharpness and height, and midtone gamma; colour papers add a per-channel slope crossover (`channel_gamma`, the dye-layer divergence at the extremes) and a paper-base tint (`base_tint_cmy`). Grade still owns contrast and the Density/toe/shoulder sliders still trim on top — the *Neutral* profile reproduces the defaults exactly.

Profiles are **mode-aware**: C-41 exposes the RA4 colour papers, B&W exposes the tonal-only B&W papers (paper tone is a Toning job, so B&W profiles carry no colour terms), and E-6 gets only *Neutral*. An incompatible stored value collapses to *Neutral* so it can never leak into a render. Bundled papers: **Neutral**; *B&W* — Ilford Multigrade RC, Ilford Multigrade FB Classic, Foma Fomatone, Foma Fomabrom; *RA4* — Kodak Endura Premier, Fujicolor Crystal Archive. Values are loosely mapped from datasheets (mainly $D_{max}$ is grounded; the knee/midtone tweaks are light character touches).

### Flat (log) master — "for editing elsewhere"
**Code**: `negpy.features.exposure.processor.PhotometricProcessor._process_flat` → `apply_flat_curve`

When the render intent is **Flat** (`RenderIntent.FLAT`), the Print stage is replaced by a **true log encoding** for use as a digital intermediate — flat, milky, low-contrast, like S-Log/LogC video before a LUT. It does **not** run the H&D curve at all.

The key point: the normalized signal $\text{val} \in [0,1]$ from §2 is *already* a log measurement of the scene. The print path's $I_{out} = 10^{-D}$ is therefore a log→linear **decode** — it (with the working-space OETF) is exactly what turns the signal back into a normal-contrast positive. The flat master **skips both**, emitting the log signal **directly** as the output value (positive-oriented, $1 - \text{val}$):

$$I_{out} = \text{clip}\big(\text{lift} + \text{gain} \cdot (1 - \text{val}),\ 0,\ 1\big)$$

*   `flat_log_gain` $= 0.65$: contrast (range of output used); $<1$ keeps it flat.
*   `flat_log_lift` $= 0.10$: the output value the scene **shadow** lands on (black lift).
*   Result: scene shadow → $0.10$, mid-grey → $\approx 0.46$, highlight → $0.75$ — headroom above white and below black, fully invertible for downstream grading.

Both are **fixed** (no per-frame metering) so an evenly-exposed roll renders identically; manual white balance still rides as an additive per-channel shift in log space. The engine also **bypasses** the creative stages (Retouch, Lab, Local, Toning, Finish) for a flat intent; only Geometry → Normalization → this log map → Crop run. Export is full-resolution; the colour space follows the export selection (color-managed at encode like the print path), as 16-bit TIFF or Linear DNG. CPU engine is forced (no GPU flat shader) for numerical exactness.

---

## 4. Local Contrast (CLAHE)
**Code**: `negpy.features.lab.logic.apply_clahe` (CPU) / `negpy/features/lab/shaders/clahe_{hist,cdf,apply}.wgsl` (GPU)

Contrast Limited Adaptive Histogram Equalization on the CIELAB $L^{\ast}$ channel (computed from linear working RGB, Adobe RGB/D65). Chroma ($a^{\ast}/b^{\ast}$) is untouched, so boosted local contrast never pumps saturation. The algorithm is **identical on CPU and GPU** (mirrored bin-for-bin; the parity test pins them to ~1e-6):

*   **Fixed $8\times8$ tile grid** over the full frame at every render scale (tile fraction constant → preview predicts export), 256 histogram bins over $L^{\ast} \in [0, 100]$.
*   **Clip limit**: $\max(1, \lfloor \text{strength} \cdot 2.5 \cdot N_{tile} / 256 \rfloor)$ counts; the clipped excess is redistributed evenly across all bins (remainder to the lowest bins), conserving the tile total exactly.
*   **Per-pixel remap**: smoothstep-weighted bilinear blend of the four neighbouring tile CDFs (tile centres at $(\text{pos}/\text{dims}) \cdot 8 - 0.5$, edge-clamped), then
    $$L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CDF}(L) \cdot 100$$
    with $\alpha$ = `clahe_strength`.
*   The GPU keeps the CDF from the preview render and reuses it for tiled full-res export (`clahe_cdf_override`), so export tiles share one seam-free global mapping.

The control lives in the Lab sidebar (`lab.clahe_strength`), but the stage runs **before Retouching** so dust healing operates on the final local-contrast rendition.

---

## 5. Retouching
**Code**: `negpy.features.retouch`

This stage removes physical artifacts like dust, hairs, and scratches from the negative. We use two complementary approaches:

*   **Automatic Dust Removal**:
    A resolution-invariant impulse detector and patching engine.
    
    1.  **Statistical Gating**: Uses dual-radius analysis. A local window ($3\times$ scaled) identifies luminance spikes, while a wide window ($4\times$ scaled) provides texture context. A cubic variance penalty ($w_{std}^3$) aggressively raises detection thresholds in high-frequency regions (foliage, rocks) to minimize false positives.
    2.  **Peak Integrity**: Validates candidates via a strict 3x3 Local Maximum check and a $Z > 3.0$ sigma outlier gate. A strong-signal bypass ensures saturation-limited artifacts (hairs/scratches) are captured even if they form plateaus.
    3.  **Annular Sampling (SPS)**: Background data is reconstructed via Stochastic Perimeter Sampling. Samples are fetched from a ring strictly exterior to the artifact footprint, ensuring zero contamination from the dust luminance itself.
    4.  **Soft Patching**: Healed regions are integrated using distance-weighted alpha blending with cubic falloff and procedural grain injection to match local noise characteristics.

*   **Manual Healing (Stochastic Boundary Sampling - SBS)**:
    When you use the Heal tool, we fill the brush area using information from its own perimeter.
    
    1.  **Perimeter Characterization**: The tool identifies the cleanest background luminance at the edge of the brush circle. This sets a "Perimeter-Safe" floor to prevent dark artifacts in bright areas like skies.
    2.  **Stochastic Sampling**: For every pixel inside the brush, we sample the immediate boundary with small angular jitter:
        $$I_{patch} = \frac{1}{3} \sum_{j=1}^{3} \text{min3x3}(P_{\theta + \Delta\theta_j})$$
        *   $P_{\theta + \Delta\theta_j}$: Perimeter point at pixel's angle $\theta$ with random jitter $\Delta\theta$.
        *   This reconstructs the natural grain and texture of the surrounding area without using "synthetic" noise.
    3.  **Luminance Keying**: To preserve original details and grain within the brush, we only apply the patch to pixels that are significantly brighter than the reconstructed background:
        $$m_{luma} = \text{smoothstep}(0.04, 0.12, I_{curr} - I_{patch})$$
    4.  **Cumulative Patching**: Patches can be overlaid and stacked. The tool intelligently heals long hairs or scratches by basing each new patch on the current accumulated state.

*   **Resolution Independence**:
    Retouching coordinates and sizes are scaled relative to the full-resolution RAW data, ensuring that edits made on the preview translate perfectly to the high-resolution export.

---

## 6. Lab Scanner Mode
**Code**: `negpy.features.lab`

This mimics what lab scanners like Frontier or Noritsu do automatically. For maximum signal quality, the steps are applied in the following sequence:

1.  **Chroma Denoise**: Applies a Gaussian filter to the A and B channels in LAB space. This reduces color noise and digital "chroma speckle" while leaving the L-channel (and its film grain) completely untouched.


2.  **Vibrance**: Selectively boosts the saturation of muted colors using a chroma mask. The mask is strongest at zero chroma and fades to zero for already vibrant colors, preventing over-saturation of sensitive areas like skin tones.
3.  **Global Saturation**: A linear boost applied to all colors via the HSV saturation channel. Before applying, the factor is multiplied by a grade-coupled chroma damping term $(k_{min}/k_g)^{strength}$ ("Dye Mute", default 0.5), where $k_g$ is the green print-curve slope and $k_{min}$ the softest printable slope. Per-channel H&D curves inflate chroma as contrast rises; the damping counters it, mimicking paper dyes' unwanted absorptions. Strength 0 disables. (The default was tuned against the old ProPhoto working gamut — it may run strong now that the working space is Adobe RGB.)
4.  **Sharpening**: We sharpen just the Lightness channel ($L$) in LAB space using Unsharp Masking (USM). We apply a threshold to avoid amplifying noise.

    $$L_{diff} = L - \text{GaussianBlur}(L, \sigma)$$
    $$L_{final} = L + L_{diff} \cdot \text{amount} \cdot 2.5 \quad \text{if } |L_{diff}| > 2.0$$
    *   $\sigma$: Blur radius (scale factor).
    *   $2.5$: Hardcoded USM boosting factor.
    *   $2.0$: Noise threshold.

5.  **Glow**: Simulates lens bloom (a print-side effect) by blurring highlights and adding them back in linear light.

    $$I_{out} = I + B_{glow} \cdot s_{glow}$$
    $$B_{glow} = \text{GaussianBlur}(I \cdot m_{hl})$$

    *   $m_{hl}$: **Display-domain** highlight mask (lens bloom follows perceived print brightness), quadratically ramped from code value 0.5 to 1.0.
    *   Applied equally to all three channels; the sum is clamped at the stage output.

6.  **Halation**: Simulates the red scatter caused by light reflecting back through the film base at capture. Uses a larger-radius Gaussian than Glow and a strongly red-biased highlight source. Because scattered light is *added exposure*, the composite is additive in linear light (not a screen blend), and the mask thresholds **linear reflectance** ($t = 0.65$) so the halation footprint is fixed by scene exposure instead of moving with Grade/Density.

    $$I_{out} = I + B_{hal} \cdot s_{hal}$$
    $$B_{hal} = \text{GaussianBlur}(I_R \cdot m_{lin} \cdot C_{hal})$$

    *   $I_R$: Red channel used as the scatter source.
    *   $m_{lin}$: Linear-light highlight mask, quadratically ramped from reflectance 0.65 to 1.0.
    *   $C_{hal}$: Per-channel tint weights $(1.0,\ 0.3,\ 0.05)$ for red-dominant scatter.

---

## 7. Toning
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

*   **Split Toning** (all modes): Additive tint in LAB ($a^{\ast}b^{\ast}$) space, so luminance — and therefore grain and detail — is preserved. Shadows and highlights are pushed toward independent hue angles. With $L$ the CIELAB lightness ($0$–$100$):
    $$m_{shadow} = \text{clip}(1 - L/50,\ 0,\ 1), \qquad m_{highlight} = \text{clip}((L - 50)/50,\ 0,\ 1)$$
    For each region (using its hue $\theta$, strength $S$, and mask $m$):
    $$a^{\ast} \mathrel{+}= \cos\theta \cdot 20 \cdot S \cdot m, \qquad b^{\ast} \mathrel{+}= \sin\theta \cdot 20 \cdot S \cdot m$$

## 7. Finish
**Code**: `negpy.features.finish`

Post-crop print finishing, in scene-linear before the output transform. Stage order: edge burn → filed carrier (the layout extras below run at compositing time).

*   **Edge Burn (Vignette)**: printer's card work in stops — a true exposure change, $I_{out} = I \cdot 2^{-s \cdot m}$ with $s$ the burn in stops (negative = hold back) and $m$ the cosine falloff mask. **Roundness** morphs the distance metric from radial (lens-like) to rectangular following the print edges (card-like); **Size** sets the falloff midpoint.

*   **Filed Carrier**: full-frame printing with a filed-out negative carrier — the clear rebate prints max black. A frame of the given mm-of-print width is multiplied to zero, its inner edge jittered by fixed-seed roughness profiles (`carrier_profiles()`), so the same "carrier" prints every frame of the roll and the GPU samples the identical table (storage buffer).

*   **Layout extras** (`services/export/print.py` + `layout.wgsl` mirror): **bottom-weighted mat** (window-mat proportions) and **match paper white** (mat colour derived by running paper white through the toning stack).
