# The Pipeline

Here is what actually happens to your image. We apply these steps in order, passing the buffer from one stage to the next.

**Colour handling — no input colorspace.** NegPy works on **linear RGB straight from the raw decode** (`output_color=raw`, `gamma=(1,1)`, unity white balance): the sensor's own channels, never converted through camera primaries into a colorimetric space. The entire pipeline processes this sensor-native data — channel balance is handled in film terms instead (independent per-channel normalization bounds in §2, spectral crosstalk unmix, cast removal in §3). Adobe RGB (1998) is an *assumed boundary profile*, not an input characterisation (`WORKING_COLOR_SPACE` in `infrastructure/display/color_spaces.py`): stages needing a perceptual model (CLAHE, Lab, Toning) compute CIELAB from the linear data using Adobe RGB primaries/D65, and the Adobe RGB TRC is applied as the final engine step (§3 Output). Colorspace primaries are applied only on the way **out** — the preview is colour-managed from the working profile to the display profile, and export converts to the selected target space and embeds its ICC profile.

## 1. Geometry (Straighten & Crop)
**Code**: `negpy.features.geometry`

*   **Rotation**: We spin the image array (90° steps) and fine-tune with affine transformations. We use bilinear interpolation so it stays sharp.
*   **Lens distortion**: a radial $k_1$ coefficient — a rig property mirrored from the active flat-field profile (`flatfield.k1`) — is corrected in the same resample.
*   **Autocrop**: I try to detect where the film ends and the scanner bed begins by looking for the density jump. It's not perfect (light leaks or weird scanning holders can fool it), so there's a manual override.

**Note:** The crop *selection* is resolved here (it becomes the analysis ROI) because the normalization step needs to know what is "image" and what is "border" to calculate the black/white points correctly — but the pixels are only actually cropped **after Toning** (before Finish), so retouching, Lab and the preview overlays all operate on the full frame. Instead of cropping you can also use the "Analysis buffer" option to exclude the outer X% of the image from the analysis, or draw a freehand **analysis region** (`analysis_rect`) that the meters then read exactly. Useful when a border rides around the film.

---

## 2. Scan Normalization
**Code**: `negpy.features.exposure.normalization`

*   **Physical Model**: We treat the input as a **radiometric measurement**. Pixel values represent linear transmittance captured by the sensor.
*   **Source corrections** (linear domain, before the log conversion):
    *   **Flat-field** (`negpy.features.flatfield`): divides out illumination falloff using a blank reference frame — a per-channel gain map $\text{mean}(\text{blur})/\text{blur}$ (computed on a 256 px copy, clamped to $[0.25, 4]$) multiplies the linear source. The reference's identity is folded into the render's source hash.
    *   **Sensor crosstalk unmix** (`sensor_matrix`, `features/process/sensor.py`): for single-shot narrowband camera scans, the camera's CFA passbands overlap the light source's bands, so a pure R/G/B exposure leaks into the other channels — a fixed property of the sensor+light pair, independent of film. Calibrated once from three bare-light exposures (response columns normalized to a unit diagonal so per-capture exposure cancels, then inverted) and applied as a 3×3 unmix of the **linear** capture — before the log/inversion where the film-dye crosstalk below lives. The related **narrowband scan** toggle instead applies the bundled RGBScan *input* profile at the display/export boundary (an explicit Input ICC overrides it).
*   **Log Conversion**: Film density is logarithmic ($D \propto \log E$). We convert the raw signal to log-space to align with the physics of the film layers:
    $$E_{log} = \log_{10}(I_{raw})$$
*   **Bounding & Polarity**:
    The engine uses statistical percentiles to detect the usable signal range. To maintain a unified pipeline, we always map the target **White Point** to the **Floor** ($0.0$) and the **Black Point** to the **Ceiling** ($1.0$).
    *   **Negative (C-41/B&W)**: Raw low-signal (dense Highlights) maps to Floor ($0.0$). Raw high-signal (Film Base / Shadows) maps to Ceiling ($1.0$). Range: 0.01% to 99.99%.
    *   **Positive (E-6)**: Raw high-signal (Highlights) maps to Floor ($0.0$). Raw low-signal (Shadows) maps to Ceiling ($1.0$). Range: 99.99% to 0.01%.
    
    Bounds are sampled on **two independent axes** (`_sample_log_bounds`): a **luma** pass fixes the floor/ceil mean (centre + span), and a **colour** pass fixes each channel's deviation from that mean. The two are recombined, so colour balance is tunable without compressing the luminance range. Identical channels (mono) give zero deviation at any clip. The controls:
    *   **Luma Range Clip** (`luma_range_clip`): Tunes how aggressively the *luminance* percentile window is set — the black/white-point span (dynamic range). **Positive** values symmetrically tighten the window before bounds detection — useful for very dense or fogged negatives where a few outlier pixels would otherwise pull the white or black point to an extreme. **Zero** uses robust extremes (a block-median prefilter rejects dust and speculars, and a small base clip excludes tiny outlier populations). **Negative** values push the bounds *outward* beyond the extremes, leaving lifted blacks and unclipped highlights as headroom.
    *   **Colour Clip** (`color_range_clip`): The sampling depth for the per-channel colour deviation (white balance / orange-mask cast), independent of the luma span. A **tighter** (larger) value samples deeper into the histogram for a more robust, outlier-resistant channel balance; a **gentler** (smaller) value samples nearer the extremes. The default neutral is `base_color_clip` ($1.0$); the slider spans log-interpolated values either side of it. The two ends are sampled differently: the **thin end** (film base) uses per-channel percentiles — density on real film is bounded below by base, so those are physically anchored — while the **dense end** (scene highlights) reads one shared, chroma-gated pixel set (the luma-extreme band's lowest-chroma subset, base-anchored) so coloured highlight content can't masquerade as film cast the way independent per-channel percentiles allow. When the band holds no trustworthy neutrals (and always for E-6), the dense end falls back to the per-channel percentile pass.
    *   **White & Black Point Offsets**: Fine-tunes the detected bounds after statistical analysis. Shifting the White Point floor or Black Point ceiling enables precise highlight recovery or shadow crushing without re-running the analysis. A **[Global / R / G / B]** selector on the Process page scopes the sliders to per-layer trims (`white_point_trim_*` / `black_point_trim_*`) added on top of the global offsets — per-dye-layer film-base (Dmin) and Dmax correction, scanner-style per-channel levels (`per_channel_point_offsets`, single source for CPU/GPU; E6 negates; hidden in B&W).
    *   **Roll baseline & locks**: Batch Analysis measures every frame of a roll; **Use luma / colour average** (`use_luma_average` / `use_colour_average`) swap in the roll-wide baseline independently per axis (luma span, colour cast) so the whole roll normalizes consistently, and **Lock bounds** (`lock_bounds`) freezes the stored floors/ceils against re-analysis.
*   **Stretch**: All modes use independent channel bounding. This neutralizes the orange mask in negatives and base tints/fading in reversal film by stretching each channel to the full $[0, 1]$ range. The result is **not clamped**: tones outside the detected bounds are kept and rolled off later by the soft toe/shoulder of the print curve, rather than being truncated here.
*   **Per-frame metering**: Normalization also measures a few statistics used later by the Print stage's automatic helpers — per-channel **shadow references** ($P_{98}$, for Cast Removal) and a per-frame **exposure anchor** ($P_{50}$ luminance) and **textural range** ($P_{10}\text{–}P_{90}$, for Auto Density / Auto Grade). See §3.
*   **Spectral crosstalk / dye unmix** (`crosstalk_strength` / `crosstalk_matrix`): applies a spectral-crosstalk matrix (`.toml` profiles, see docs/CROSSTALK.md) to the raw **negative** log densities *before* bounds analysis and the stretch. This is the physically correct domain — secondary dye absorptions are linear in negative dye density (Beer–Lambert), and the bundled matrices are derived from negative spectral dye-density curves. The matrix is blended with identity by strength and row-normalized (grays preserved); every meter (bounds, anchor, shadow refs, neutral axis) reads the unmixed film. Batch Analysis applies the same unmix — bounds measured under a different matrix are invalid for the render, so re-run it (and re-check locked bounds) after changing the matrix or strength.
*   **Scan-clip warning**: the fraction of source pixels at/above sensor white ($\ge 0.99$ linear) is reported per channel (`scan_clip_fractions`). In a negative scan the film base and scene shadows sit near sensor white, so clipping there irreversibly collapses distinct densities to $D=0$; the Analysis panel warns above 1%. This is a capture-side problem — no reconstruction is attempted.

---

## 3. The Print (Exposure)
**Code**: `negpy.features.exposure`

*   **Virtual Darkroom**: Simulates shining light through the normalized log-signal onto paper.
*   **B&W (panchromatic)**: in B&W mode the normalized signal collapses to a **single density** (its luminance) *before* the curve — the H&D curve shapes one channel instead of mixing three, like paper under a B&W negative; per-channel colour controls are hidden.
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
*   **Split Grade** (`shadow_grade` / `highlight_grade`, ISO-R points, negative = harder): zone-local contrast, the split-grade print — the curve rotates about the shadow/highlight zone centres, $v \mathrel{+}= \Delta k_{ch} \cdot w \cdot (v - z_{zone})$, using the same mid-sparing sigmoid weights as Zone Density ($z_{sh} = z + 0.75$, $z_{hl} = z - 0.40$, $k = 4$). The ISO-R points fold into a slope ratio exactly like Grade trims (`split_grade_deltas`, single source for CPU/GPU/chart), with per-layer trims (`shadow_grade_trim_*` / `highlight_grade_trim_*`) on top. Runs **before** Zone Density as its own block — sequential blocks stay monotone where shared weights would not.
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
*   **Dodge & Burn** (`negpy.features.local`): polygon masks drawn over the print, each with a strength in **EV stops** (positive = dodge / hold back, negative = burn) and a Gaussian feather ($\sigma$ as a fraction of the short side). The masks rasterize to a per-pixel EV map added to the log-exposure input alongside the CMY offsets — a true print-exposure change that rides the full curve, not a brightness overlay. One stop is $\log_{10}(2)$ scaled by each channel's stretch range (`local_ev_scale`), so a 1-stop dodge holds back exactly one stop of print exposure regardless of the frame's bounds. Vertices are stored in raw-image coordinates and follow geometry (rotation, flips, distortion). The Flat intent skips them.
*   **Output**: Converts print density back to **scene-linear** reflectance (transmittance):
    $$I_{out} = 10^{-D}$$
    *   **Paper Black** (`paper_black`, off): off applies black point compensation, the same idea as ICC relative-colorimetric soft-proofing — a reflection print's D-max ($2.3$) floors reflectance at $10^{-2.3} \approx 0.005$, but the adapted eye reads paper black as black, so the display should too; on preserves the paper's lifted D-max instead. With compensation (the default), per channel, with $t_b = 10^{-D_b}$:
        $$I_{out} = \frac{I - t_b}{1 - t_b}, \quad D_b = D_{max} + \text{toe}_{ch} \cdot 0.90 \text{ (for } \text{toe}_{ch} < 0\text{)},\ D_{max} \text{ otherwise}$$
        clamped at $0$. The curve reaches $D_{max}$ only asymptotically, so a **negative toe raises the clip point** into the shadows — that's what makes exact $0$ reachable ("negative toe deepens blacks", literally); a lifted toe and per-layer shadow casts survive because the reference is the *physical* $D_{max}$, not $D_{max,eff}$. A negative per-layer toe trim (with compensation on) tints the deepest black.
    *   **Note**: The pipeline is **scene-linear internally** — the exposure stage emits linear light and every creative stage (Local Contrast, Retouch, Lab, Toning, Finish) operates on it. The working-space OETF (the **Adobe RGB (1998) TRC**, a pure $563/256 \approx 2.199$ power with no linear segment) is applied **only as the final engine step** (the output transform), so it composes correctly with the Adobe RGB ICC profile at the display/export boundary. Retouch is a perceptual op, so the CPU brackets that stage through the OETF (encode → heal → decode); the GPU keeps a single encoded perceptual region (exposure → clahe/retouch encoded → lab decodes back to linear).

### Automatic helpers

The defaults are tuned to look right straight out of the box; these helpers do per-frame work so you don't have to. All correct **partially** — they nudge toward a good result while preserving the photograph's intent. Turn them off to let the conversion follow the negative honestly (a dense negative prints dense, a flat one prints flat).

*   **Auto Density** (`auto_exposure`, **on**): Meters each frame's median tone and sets a sensible brightness. The exposure anchor is a linear partial pull from the assumed key (`assumed_anchor` $= 0.46$) toward the measured median:
    $$\text{anchor} = \text{assumed} + s \cdot (P_{50} - \text{assumed}), \quad \text{clamped to } \pm b$$
    with $s =$ `anchor_meter_strength` ($0.2$) and $b =$ `anchor_meter_band` ($0.12$). The partial blend (and the band) means a deliberately low-key or high-key shot keeps its mood instead of being flattened to neutral grey. The anchor then prints at `anchor_target_density` ($0.75$), which is what sets overall print brightness.
*   **Auto Grade** (`auto_normalize_contrast`, **on**): Chooses contrast to suit each scene from the textural density range ($P_{10}\text{–}P_{90}$). Letting $r$ be the ratio of the full bounded range to the textural range, the effective contrast target is:
    $$K \cdot \big(n + \sigma \cdot (r - n)\big)$$
    with $K =$ `auto_grade_target` ($0.6$), $n =$ `auto_grade_nominal_ratio` ($2.0$, the ratio of a "normal" negative) and $\sigma =$ `auto_grade_strength` ($0.5$). The adaptation strength dampens contrast swings — at $\sigma = 0$ every frame gets the same fixed grade, at $\sigma = 1$ every frame is normalized to identical contrast; the default sits between, so a flat scene gets a lift and a punchy scene stays punchy without being pushed to a harsh extreme.
*   **Set Targets** (app-global): the five numbers above that set the *aim* — `anchor_target_density`, `anchor_meter_strength`, `anchor_meter_band`, `auto_grade_target`, `auto_grade_strength` — are user-tunable from a dialog beside the two toggles (`TUNABLE_TARGETS` in `features/exposure/models.py`, ranges declared there). They are a **calibration, not per-image state**: `apply_targets()` overlays them onto `EXPOSURE_CONSTANTS` and they persist in the `exposure_targets` global setting, so they apply to every frame including already-edited ones. Because no `WorkspaceConfig` hash sees them, both engines fold a `TARGETS_REVISION` counter into their cache keys — the CPU base stage (which does the metering) and, on the GPU, the analysis cache key plus the exposure-stage diff. Anything added to `TUNABLE_TARGETS` that is read *outside* those stages needs its own invalidation.
*   **Cast Removal** (`cast_removal_strength`, default $0.5$; $0$ turns it off): Neutralizes the colour cast a negative leaves in the print, balancing each layer so greys read neutral from deep shadows through highlights — not just at the midtone (the usual cause of shadows/highlights drifting off-colour after a C-41 midtone white balance). The applied strength is `confidence × slider` (`effective_cast_strength`): how cleanly the frame's near-neutrals read biases the correction, so a scene with few greys is corrected gently and the slider trims on top.

    The primary solve fits each non-green channel to green's **neutral axis** — per-channel references taken at a highlight, midtone and shadow luma band, each over that band's lowest-chroma pixels (`neutral_axis_*` constants). R and B get a **quadratic** through all three green-matched points, so highlights don't extrapolate past neutral; the midtone is pinned exactly via the pivot. Each channel's deviation from green is clamped ($\pm 0.2$, `midtone_cast_max_offset`) and the curvature is bounded to a fraction of the slope (`neutral_axis_curv_max_ratio` $= 0.45$) to keep the per-channel core monotonic on $[0,1]$.

    When no neutral axis is available (too few trustworthy near-neutrals), it falls back to a **two-point tie** on the per-channel shadow refs ($P_{98}$): each non-green channel gets its own slope so its shadow ref lines up with green's, with the luma anchor pinning the midtone:
    $$k_{ch} = k \cdot \frac{\text{anchor} - r_{green}}{\text{anchor} - (r_{green} - \text{cast}_{ch})}$$
    The per-channel cast is bounded ($\pm 0.1$, `cast_removal_max_offset`) so the tilt can't run away.
With the helpers off, the conversion **shows you your photography** — exactly how the frame was exposed and developed. The defaults should be neutral, but you can (and should) use the sliders to match the curve shape (your "print") to your liking.

### Paper profiles
**Code**: `negpy.features.exposure.papers`

A **paper profile** (`paper_profile`, default *Neutral*) overrides the print *character* — the H&D curve shape — without touching contrast or exposure. Each profile sets the paper's $D_{max}$/$D_{min}$, toe/shoulder knee sharpness and height, and midtone gamma; colour papers add a per-channel slope crossover (`channel_gamma`, the dye-layer divergence at the extremes), a paper-base tint (`base_tint_cmy`, an addition to the minimum-density floor that shows in highlights) and a **dye-coupling matrix** (`dye_matrix`, $D_{rgb} = M \cdot D_{dye}$ above base — the dyes' unwanted absorptions, row-normalized at use). Grade still owns contrast and the Density/toe/shoulder sliders still trim on top — the *Neutral* profile reproduces the defaults exactly.

Profiles are **mode-aware**: C-41 exposes the RA4 colour papers, B&W exposes the tonal-only B&W papers (paper tone is a Toning job, so B&W profiles carry no colour terms), and E-6 gets only *Neutral*. An incompatible stored value collapses to *Neutral* so it can never leak into a render. Bundled papers: **Neutral**; *B&W* — Ilford Multigrade RC, Ilford Multigrade FB Classic, Foma Fomatone, Foma Fomabrom; *RA4* — Kodak Endura Premier, Fujicolor Crystal Archive. Values are loosely mapped from datasheets (mainly $D_{max}$ is grounded; the knee/midtone tweaks are light character touches).

### Flat (log) master — "for editing elsewhere"
**Code**: `negpy.features.exposure.processor.PhotometricProcessor._process_flat` → `apply_flat_curve`

When the render intent is **Flat** (`RenderIntent.FLAT`), the Print stage is replaced by a **true log encoding** for use as a digital intermediate — flat, milky, low-contrast, like S-Log/LogC video before a LUT. It does **not** run the H&D curve at all.

The key point: the normalized signal $\text{val} \in [0,1]$ from §2 is *already* a log measurement of the scene. The print path's $I_{out} = 10^{-D}$ is therefore a log→linear **decode** — it (with the working-space OETF) is exactly what turns the signal back into a normal-contrast positive. The flat master **skips both**, emitting the log signal **directly** as the output value (positive-oriented, $1 - \text{val}$):

$$I_{out} = \text{clip}\big(\text{lift} + \text{gain} \cdot (1 - \text{val}),\ 0,\ 1\big)$$

*   `flat_log_gain` $= 0.65$: contrast (range of output used); $<1$ keeps it flat.
*   `flat_log_lift` $= 0.10$: the output value the scene **shadow** lands on (black lift).
*   Result: scene shadow → $0.10$, mid-grey → $\approx 0.46$, highlight → $0.75$ — headroom above white and below black, fully invertible for downstream grading.

Both are **fixed** (no per-frame metering) so an evenly-exposed roll renders identically; manual white balance still rides as an additive per-channel shift in log space. The engine also **bypasses** the creative stages (Local Contrast, Retouch, Lab, Toning, Finish — dodge/burn masks are skipped too) for a flat intent; only Geometry → Normalization → this log map → Crop run. Export is full-resolution; the colour space follows the export selection (color-managed at encode like the print path), as 16-bit TIFF. CPU engine is forced (no GPU flat shader) for numerical exactness.

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

This stage removes physical artifacts like dust, hairs, and scratches from the negative. Defects come from three complementary sources — the IR channel (hardware), statistical detection (software), and your manual strokes — and all repairs run through one shared membrane-clone engine, with a routed inpaint for defects a clone can't cover:

*   **Infrared (IR) Dust Removal** (scans carrying an IR channel — Coolscan, SilverFast iSRD, VueScan DNG):
    Dust and scratches block infrared light while the film's dyes pass it, so the IR plane is a defect map independent of the photograph. This path runs on the **linear source before normalization**, so every meter reads the cleaned film. Algorithm concepts are ported from digital-fauxice (see `NOTICE.md`), a validated recreation of Digital ICE.

    1.  **Normalized ratio**: the IR plane is divided by its own local clean base — $r = IR / \text{blur}(\text{dilate}(IR))$ — reading ~$1.0$ on clean film and dipping under defects, independent of illumination. A crosstalk fit divides out the visible image's IR ghost first.
    2.  **Division tier**: semi-transparent dust *attenuates* rather than blocks, so the image beneath is recovered directly: $RGB / r^{\gamma}$ with per-channel γ fitted per frame. The gain never lifts a pixel past its local clean base (defect-excluded mean $- \sigma$) — the guard that used to be a grain-biased local max and printed dark rings around specks.
    3.  **Score-weighted fill**: the ratio maps to a continuous defect score $s \in [0.02, 1]$ ($1$ = clean; the IR Threshold slider moves the ramp — nothing is ever thresholded, so there is no abort and no mask edge). Opaque cores and hairs are rebuilt as a multiscale score-weighted average over nested supports:
        $$\text{fill} = \frac{\sum I \cdot s \cdot w}{\sum s \cdot w}$$
        Defective neighbours self-exclude, edges continue through defects (the finest support with clean data wins). Written under the **original-floor rule**: dust is dark in negative transmittance, so a repair may only lighten — a dark halo cannot be produced.
    4.  **Routed inpaint**: only defects with an interior the fill can't see across (chebyshev radius ≥ 5 at detection scale) go to structure-following inpaint, composited with an alpha feather. A 2% frame budget bounds this heavy path; the fill itself always runs.

    B&W silver and Kodachrome block IR like dust does; such frames are auto-detected (the IR plane mirrors the image) and skipped.

*   **Automatic Dust Removal** (`dust_remove`, with Threshold / Size):
    A resolution-invariant impulse detector; what it finds is healed by the same membrane engine as the manual tool.

    1.  **Detection proxy**: percentile-normalized source **density** ($-\log_{10}$ of linear luminance, 0.5–99.5% window). Grade-independent — dust reads bright in every process mode, and a defect's step stays proportional to its physical density excess (a print-like tone map would compress it below threshold on wide-range scans).
    2.  **Statistical gating**: dual-window statistics — a local window ($3\times$ size) supplies the spike mean/σ, a wide window ($4\times$ size) a texture penalty (cubic in the wide σ) that raises the threshold in high-frequency regions (foliage, rocks). Candidates must clear the adaptive threshold, a $Z > 3.0$ outlier gate and a strict $3\times3$ local-max check, with a strong-signal bypass so saturation-limited plateaus (hairs, scratches) still register. The stat maps are threshold-independent and cached — dragging the Threshold slider re-runs only the cheap gate.
    3.  **Specks → membrane strokes**: connected components become heal regions for the membrane engine (compact specks a disc, mildly elongated ones a ≤8-point capsule chain). Each gets an auto-picked **clean clone source** nearby, scored on the per-channel density proxy so dusty or wrong-colour sources are rejected.
    4.  **Hairs → routed inpaint**: strongly elongated defects (thinness $= \text{area}/\text{thickness}^2$ — bending-invariant where PCA aspect calls a twisted hair "compact") can't be cloned with a single offset; they're rebuilt by structure-following inpaint (Navier–Stokes), encoded against the local clean range, alpha-feathered over the dilated PSF skirt, and baked into the **linear source** (folded into the render's source identity).

*   **Healing engine (membrane clone)**:
    One engine heals everything — manual strokes and spots, auto-detected specks, IR-routed cores — via **mean-value-coordinates membrane cloning** (the healing-brush construction):
    $$\text{out}(p) = I(p + \text{off}) + \sum_i \hat{w}_i \big( I(b_i) - I(b_i + \text{off}) \big)$$
    The copied source patch (at offset $\text{off}$) carries real grain; the MVC-weighted boundary-difference field is the smooth membrane that matches it into the destination rim — no synthetic noise, no smudging.
    *   **Dust guard**: every clone and boundary sample passes a clean-sample filter, so specks in the source patch or on the boundary are never recloned.
    *   **Destination dust gate**: dust regions heal only pixels *brighter* than the membrane-predicted clean value (smoothstep on the excess) — a brush marks a search area, and clean grain inside it stays untouched. Manual clone regions (gate off) copy unconditionally, with a wider rim feather.
    *   Healing is perceptual work, so the stage brackets the linear buffer through the working OETF (encode → heal → decode; see §3 Output note).

*   **Resolution Independence**:
    Retouching coordinates and sizes are scaled relative to the full-resolution RAW data, ensuring that edits made on the preview translate perfectly to the high-resolution export.

---

## 6. Lab Scanner Mode
**Code**: `negpy.features.lab`

This mimics what lab scanners like Frontier or Noritsu do automatically. For maximum signal quality, the steps are applied in the following sequence:

1.  **Chroma Denoise**: Applies a Gaussian filter to the A and B channels in LAB space. This reduces color noise and digital "chroma speckle" while leaving the L-channel (and its film grain) completely untouched.


2.  **Vibrance**: Selectively boosts the saturation of muted colors via a chroma mask in LAB space. The mask is strongest at zero chroma and fades to zero for already vibrant colors, preventing over-saturation of sensitive areas like skin tones.
3.  **Global Saturation**: A linear chroma scale ($a^{\ast}/b^{\ast}$) in CIELAB — lightness-preserving, unlike HSV S-scaling which darkens already-saturated colours when S clips. Before applying, the factor is multiplied by a grade-coupled chroma damping term $(k_{min}/k_g)^{strength}$ ("Dye Mute", default 0.5), where $k_g$ is the green print-curve slope and $k_{min}$ the softest printable slope. Per-channel H&D curves inflate chroma as contrast rises; the damping counters it, mimicking paper dyes' unwanted absorptions. Strength 0 disables.
4.  **Sharpening**: A **Method** selector picks Unsharp Mask or Deconvolution; both share the Amount/Radius/Masking controls and the same $\text{radius} \cdot \text{scale factor}$ Gaussian taps from `gaussian_kernel_1d` (uploaded to the `sharpen_k` storage buffer, convolved identically on CPU `cv2.sepFilter2D` and the separable WGSL passes), so CPU and GPU match bit-for-bit.

    **Unsharp Mask** — on the Lightness channel ($L$) in LAB space, with halo suppression (`lab_sharpen_h/v.wgsl`):

    $$L_{diff} = L - \text{blur}(L, \sigma), \qquad \sigma = \text{radius} \cdot \text{scale factor}$$
    $$\text{gain} = \text{amount} \cdot 2.5 \cdot \text{smoothstep}(1.5, 2.0, |L_{diff}|) \cdot m$$
    $$L_{final} = \text{clamp}\big(L + L_{diff}\cdot\text{gain},\; L_{min}-2,\; L_{max}+1\big)$$
    *   **Radius** (px): blur $\sigma$, scaled to the render size so preview and export match.
    *   **Masking** ($m$): edge mask from the boxed $|\nabla L|$, $\text{smoothstep}(0.5t, t, |\nabla L|)$ with $t = 10\cdot\text{masking}$ — protects flat areas (sky, skin, grain); off at 0.
    *   Smoothstep noise gate over $[1.5, 2.0]$ replaces a hard threshold. Overshoot clamp to the local $3\times3$ range ($L_{min}, L_{max}$) kills halos, tighter above (+1) than below (−2) because $L^{\ast}$-domain USM exaggerates light halos.

    **Deconvolution** — Richardson-Lucy on linear luminance $Y$ (Gaussian PSF), reversing the scanner's optical blur (`rl_*.wgsl`). Runs on $Y$, not $L^{\ast}$: optical blur is physical, so the model must live in linear light.

    $$\hat{o}_{n+1} = \hat{o}_n \cdot \left(K \otimes \frac{o}{\max(K \otimes \hat{o}_n,\ \epsilon)}\right), \qquad \hat{o}_0 = o = Y$$

    Iterations are fixed by radius, $\text{clamp}(\text{round}(10\cdot\text{radius}), 5, 20)$ — a function of the *user* radius, never the scaled $\sigma$, so preview and export run identical counts (no per-pixel early stop, no damping; the edge mask alone governs grain, matching RawTherapee). The result is applied as an RGB ratio (chroma-preserving), gated by the same $L^{\ast}$ edge mask $m$:

    $$\mathrm{RGB}_{out} = \mathrm{clamp}\left(\mathrm{RGB} \cdot \max\left(1 + \left(\frac{\hat{o}_N}{\max(o,\epsilon)} - 1\right) \cdot \mathrm{amount} \cdot m,\ 0\right),\ 0,\ 1\right)$$

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

*   **Chemical Toning** (B&W mode only): six bath simulations — **Selenium**, **Sepia**, **Gold**, **Iron Blue**, **Copper**, **Vanadium Green** — modelled as a **silver ledger** in density space (`TONING_CONSTANTS`). The pixel's original mean density $D_0$ is its metallic-silver reservoir; each toner converts a fraction of it, and converted silver is locked away from later baths (Rudman/Ilford — the archival selenium-then-sepia split; "no silver left" exhaustion).
    *   **Susceptibility**: each toner's conversion $c_i$ is a pure function of $D_0$ (a grain property — sequence only decides who claims silver first):
        *   *silver-proportional, shadows first*: Selenium $c = S \cdot (D_0/D_{ref})^{p}$; Iron Blue and Copper likewise, with a low $D_{ref}$ ($0.9$) so the colour reaches the mids instead of hiding in the deep shadows.
        *   *bleach-limited, highlights first*: Sepia, Gold, Vanadium $c = S \cdot (1 - D_0/D_{ref})^{p}$ — the thinnest silver converts first (split-sepia character comes from the exponent).
        Strength $> 1$ reads as a longer bath; conversion caps at 1 (all remaining silver toned).
    *   **The ledger**: the untoned fraction $a$ starts at 1; in bath order (selenium → sepia → gold → blue → copper → vanadium) each toner claims $f_i = a \cdot c_i$, $a \mathrel{-}= f_i$. **Gold is the one lock-out exception**: it also plates the *sulfide* (sepia) fraction — the classic gold-over-sepia orange-red shift, with compounded covering power.
    *   **Covering power**: the final per-channel density is the mix
        $$D_{ch} = D_0 \cdot \Big(a + \sum_i f_i \cdot g_{i,ch}\Big), \qquad I_{out} = 10^{-D_{ch}}$$
        where each gain triplet $g_i$ carries the deposit's colour and density change: selenium all $\ge 1$ (the Dmax boost it's used for, eggplant shadow hue), sepia's sulfide lower covering power (lifts and warms), gold slight intensification (cool blue-black), Prussian blue net $> 1$ with G at exactly $1.00$ (what lets the classic sepia+blue *green* split emerge from the mix), copper net $< 1$ (the ferricyanide bleaches while it tones — brick red), vanadium R/B absorbed with slight density loss (green print, black blacks).

*   **Split Toning** (all modes): Additive tint in LAB ($a^{\ast}b^{\ast}$) space, so luminance — and therefore grain and detail — is preserved. Shadows and highlights are pushed toward independent hue angles. With $L$ the CIELAB lightness ($0$–$100$):
    $$m_{shadow} = \text{clip}(1 - L/50,\ 0,\ 1), \qquad m_{highlight} = \text{clip}((L - 50)/50,\ 0,\ 1)$$
    For each region (using its hue $\theta$, strength $S$, and mask $m$):
    $$a^{\ast} \mathrel{+}= \cos\theta \cdot 20 \cdot S \cdot m, \qquad b^{\ast} \mathrel{+}= \sin\theta \cdot 20 \cdot S \cdot m$$

---

## 8. Finish
**Code**: `negpy.features.finish`

Post-crop print finishing, in scene-linear before the output transform. Stage order: edge burn → filed carrier (the layout extras below run at compositing time).

*   **Edge Burn (Vignette)**: printer's card work in stops — a true exposure change, $I_{out} = I \cdot 2^{-s \cdot m}$ with $s$ the burn in stops (negative = hold back) and $m$ the cosine falloff mask. **Roundness** morphs the distance metric from radial (lens-like) to rectangular following the print edges (card-like); **Size** sets the falloff midpoint.

*   **Filed Carrier**: full-frame printing with a filed-out negative carrier — the clear rebate prints max black between two boundaries of different character, framed by a margin of unexposed paper ($0.7 \cdot w$) in the mat colour (`PrintService.effective_paper_linear`, scene-linear since this stage precedes the output OETF while the mat does not). The picture side is the camera's film gate: machine-cut, so it only wobbles (`carrier_profiles()` rows 0-3, no slider) and prints soft. The paper side is the filed aperture: **Roughness** swings it off rows 4-7 — file-mark-scale bites over a slow uneven wander, not per-sample noise, which reads as a machine-cut comb — and prints nearly crisp. **Corners** rounds it by an arc measured from the aperture corner, not the print edge; both boundaries take the same arc so the band keeps its width around it. Profiles are fixed-seed, so one "carrier" prints every frame of the roll and the GPU samples the identical table (storage buffer).

    A **2-D fBm** then displaces both boundaries' distance fields (`carrier_noise`). A 1-D profile is a height field — one depth per position — so it cannot overhang, shed a fleck, or tuft both ways at one spot; the 2-D term supplies all three and blotches the flare. Hand-rolled hash value noise rather than a library, because WGSL must reproduce it bit for bit: the hash is u32 wrap-around only. Cell size scales with $w$, so features hold their print size at any resolution.

    **Flare** is the reflection off the bared metal of the bevel: a weight peaking on the filed edge, reaching $0.35 \cdot w$ into the rebate and $0.7 \times$ that back onto the paper, applied as a lerp toward a tint rather than an addition — so it glows on the black and stains the white. The stray light never passes the orange mask, so it is coloured (hue drifting slowly off the gate profile) and neutral in B&W. Edges compose in order 0-3 on both paths; unlike the rebate's products, the lerp is order-dependent.

*   **Layout extras** (`services/export/print.py` + `layout.wgsl` mirror): **bottom-weighted mat** (window-mat proportions) and **match paper white** (mat colour derived by running paper white through the toning stack).
