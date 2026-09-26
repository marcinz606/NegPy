# The Pipeline

These steps run in order. Each stage passes its buffer to the next.

**Color handling: no input colorspace.** NegPy works on **linear RGB straight from the raw decode** (`output_color=raw`, `gamma=(1,1)`, unity white balance): the sensor's own channels, never converted through camera primaries. Channel balance is handled in film terms: per-channel normalization bounds (§2), crosstalk unmix and cast removal (§3). Adobe RGB (1998) is an assumed boundary profile, not an input characterization (`WORKING_COLOR_SPACE` in `infrastructure/display/color_spaces.py`): CLAHE, Lab and Toning compute CIELAB from linear data with Adobe RGB primaries and D65, and the Adobe RGB TRC is the final engine step. Primaries apply only on the way **out**: the preview is color-managed from the working profile to the display profile, and export converts to the target space and embeds its ICC profile. Every render, preview, detection and thumbnail decode passes `adjust_maximum_thr=0.0`, which pins the scale to the camera's calibrated linearity limit (`user_sat`, or the format's generic white level when the body has no calibration table) instead of LibRaw's per-frame maximum. A roll then decodes on one scale, and the preview matches export. Linear Output's decode keeps the roll-consistency pin but not `user_sat`, because it hands off the least-processed data it can.

## 1. Geometry (Straighten & Crop)
**Code**: `negpy.features.geometry`

*   **Rotation**: 90° steps, then affine fine rotation with bilinear interpolation.
*   **Lens distortion**: a radial $k_1$ coefficient (`geometry.distortion_k1`, ±0.10), corrected in the same resample.
*   **Tilt and Swing** (`geometry.converge_v` / `converge_h`, ±15%): perspective correction named for the enlarger movements. Tilt straightens converging verticals, swing converging horizontals: the plane-to-plane projectivity of a tilted easel (Hartley & Zisserman §2.3). It runs last in the forward chain, after $k_1$, because a projectivity cannot be fitted to a barrel-distorted frame. The unit is percent of the frame, not degrees, because convergence is $(H/2)\sin\tau / D$ and no magnification or focal length is modeled. Output keeps the canvas size and replicates the wedge, as fine rotation does.

    `keystone_matrix_normalized` is the single definition of the quad; the pixel-space matrix and the GPU's inverse derive from it. Every geometry reader applies it: the uv grid, `map_coords_to_geometry`, autocrop's replay and detection key, and the GPU's analysis replay. Off-frame card-edge handles need `CoordinateMapping` to fit the grid projectively.
*   **Crop by Default** (`geometry.crop_to_valid`): the largest axis-aligned, frame-centered rect that fine rotation and keystone leave void-free, used as the ROI when no manual or detected crop is set. Distortion is excluded, because it scales to fill. The valid region is a convex quadrilateral, so `compute_geometry_crop_rect` ternary-searches the half-width, with a nested binary search for the half-height.
*   **Autocrop**: finds where the film ends and the scanner bed begins from the density jump. Light leaks and odd holders can fool it, so a manual override exists.

    Detection runs once per edit, in `ImageProcessor` ahead of either engine, and the rect is stored on the edit (`geometry.crop_rect`; `crop_from_auto` marks its origin). Both engines only slice that rect, because a preview and a full-res export can stop on different edges. The rect keeps its detection key (`autocrop_detection_key`: orientation, ratio, mode, rebate trim); a key change re-detects on the next render. Crop Offset is not in the key, because it is re-applied every render.

    The surround of a detected box must read as bed (uniform and near-clipping, not merely bright) or as a dark holder; otherwise detection falls back to the full frame. The box then has opaque edges trimmed, up to 20% a side, against a near-black level no negative reaches. A run that uses the whole allowance trims nothing, so a slide's black shadows keep their edges. A box under a quarter of the frame is not film, and detection falls back to the full frame; in a roll, the frame is placed from the roll or the roll declines.

    The crop is then squared to the Ratio setting. On `Free` (default) the ratio comes from the detected frame, snapped to the nearest real film format, so a 6x6 stays square. The snap is checked against the sensor dimensions only where the detected box is the longer of the two.
*   **Rebate Trim**: in *Image only* mode the film box is refined inward to the exposed image, by whichever of three routes can read the frame.
    *   A **uniformity probe**, when bed, rebate and image tiers separate cleanly. It classifies the box against the rebate/image midpoint and takes the longest run of occupied rows, then columns. A run shorter than half its side is refused, because no rebate is that thick.
    *   A **per-side edge walk** (`measure_film_edges`), for camera scans, where mask and base are too thin for the ring (its probe is 4% of the box). Each line is walked alone and depths are pooled per side, because a slight tilt mixes mask, base and picture on one row. The walk recognizes mask (near-black) and bed (near-clipping) by level, and a rebate brighter or a holder edge darker than the picture on the same line. Where it ends on a relative test, the edge is the strongest gradient in the run and must be a real step by magnitude, so a tapering subject is not read as a border. Sides are independent; a cap and a support test bound the risk. The walk runs twice: the first pass reaches 3% of the side and is accepted as is; the second reaches 10% (wide holder margins, sprocket rails, a neighboring frame) and is adopted only where it goes deeper and passes four tests. It stopped inside its allowance, its per-line depths fit a straight line, its outermost pixels hold one level along the side, and its closing step carries most of the drop from that level to the picture.
    *   A **per-side ring measurement** (`measure_film_border`), gated on an opposite pair because a bright sky at the frame edge reads like film base. It reads a wide rebate below bed level.

    Rebate Trim scales the refinement: 0 stops at the film edge, 1.0 lands on the detected image edge, above 1.0 cuts into the picture. The same factor scales the roll-pooled inset in the roll auto crop (Crop › Roll), where walk and ring both feed the roll median and each side is pooled on its own sample count. A roll too short for any side to reach that count trims by the ring alone, gated on an opposite pair.

    In a roll, a portrait frame is resolved from its own single-frame crop, because the roll is landscape. A frame whose film box was not found is placed from the roll, its left and right edges taken from a vertical-edge profile. When no frame in the roll found a box (film overfills the sensor), the roll falls back to the threshold box of single-frame Auto Crop, trimmed by the border walk, only where that box is near the whole frame. Deskew is not pooled: the angle comes from the film-box contour, refined by a line fit on the top edge where the two agree, and a frame keeps its own tilt against the roll.

**Note:** the crop *selection* is resolved here and becomes the analysis ROI, so normalization can tell image from border. The pixels are cropped only **after Toning**, before Finish, so Lab and the preview overlays see the full frame. The "Analysis buffer" option excludes the outer X% from analysis, and a freehand **analysis region** (`analysis_rect`) sets exactly what the meters read.

---

## 2. Scan Normalization
**Code**: `negpy.features.exposure.normalization`

*   **Physical model**: the input is a **radiometric measurement**: pixel values are linear transmittance.
*   **Source corrections** (linear domain, before the log conversion):
    *   **Flat-field** (`negpy.features.flatfield`): divides out illumination falloff with a blank reference frame. A per-channel gain map $\text{mean}(\text{blur})/\text{blur}$, computed on a 256 px copy and clamped to $[0.25, 4]$, multiplies the linear source. The gain is **baked once** into a profile (an `.npz` in `APP_CONFIG.flatfield_dir`, keyed by an opaque id), so moving or deleting the reference file is harmless. The edit stores only the profile id. The render resolves the gain through a provider (`set_gain_provider`, wired to `services/assets/flatfield.py`) and caches it; the profile id and a gain content token fold into the source hash.
    *   **Sensor crosstalk unmix** (`sensor_matrix`, `features/process/sensor.py`): for single-shot narrowband camera scans, CFA passbands overlap the light's bands, so each channel leaks into the others. It is a property of the sensor and light, not the film. It is calibrated from three bare-light exposures (columns normalized to a unit diagonal), inverted, and applied as a 3×3 unmix of the **linear** capture, ahead of the log. `unmix_block_reason` refuses it on a transparency (E-6 is not scanned narrowband) and on a camera-WB decode (`linear_raw` off), where a diagonal gain does not commute with the unmix. The **narrowband scan** toggle instead applies the bundled RGBScan *input* profile at the display and export boundary; an explicit Input ICC overrides it.
    *   **HDR merge** (`features/hdr`): combines a bracket of one frame into a single linear source at decode, next to the triplet merge. It has no shader and no GPU parity surface.

        The **reference** is the longest exposure with a negligible clipped fraction (`choose_reference`), and the merge is computed in its units, so the result stays in $[0, 1]$ (`features/process/logic.py` clips at pipeline entry). The render exposure (`output_scale`, clamped to $\le 1$) comes from `hdr_anchor`, a frame the user nominates, because the longest unclipped capture sits above the metered exposure. With no anchor it is the bracket's median ratio in log space. The final clip runs after this scale, so an anchor below the reference keeps radiance above the reference's white. `anchor_choices` offers only frames the bracket contains.

        The merge is cached **unscaled** and `apply_render_exposure` scales it on the way out, so an exposure change needs no new decode. The merge key (`hdr_merge_token`) excludes the exposure; the source identity (`hdr_token`) includes it. The cached merge holds values **above 1.0**.

        A fresh merge is seeded with a Shadows Density lift (`seed_shadow_density`, $-0.30$ per stop of `shadow_reach_stops`, capped at $-0.70$). It is an edit on the composite, so the slider shows it and zero restores the faithful render. It is bounded so the merged render stays quieter than the single metered frame. A merge recovers shadow precision, not range.

        Exposure ratios are solved from the images, because several loaders are scanner formats with no exposure tags. `pair_ratio` is the median of per-sample ratios over samples above the noise floor in the shorter frame and unsaturated in the longer, chained outward from the reference between **exposure-adjacent** frames. `exposure_order_key` orders the bracket by clipped fraction, then level, because `level` pins at $1.0$ for every frame clipping more than 1%. Samples combine by **inverse variance** (weight $r^2$), so the merge is no worse than the best single frame. The weight rolls off to zero between $0.90$ and $0.995$ per channel. Output is float32, because the recovered detail sits below the reference's quantization step.

        The merge runs **before** flat-field and the sensor unmix, because a gain map applied first would move the saturation point the weights key on. Every frame decodes on the **reference frame's** white balance, because per-frame as-shot multipliers (`use_camera_wb`) make `pair_ratio` solve wrong ratios, which print as contour rings around a blown highlight. The solve (`HdrWorker.run`) decodes one frame at a time with `hdr` cleared and is pinned separately: it passes the first frame's multipliers to the rest, since the reference is not yet known.
    *   **Composite assembly** (`features/rgbscan`, `features/stitch`): a frame may be built from several files before any of the above. A **Trichrome triplet** takes each output channel from the exposure lit by that band, green and blue registered to red by phase correlation. Its channels never mixed in the sensor, so the crosstalk unmix is skipped. Every exposure decodes on a neutral white balance. A **stitch composite** decodes each part separately, warps the parts into a shared canvas by a registration fixed at stitch time, matches per-channel gain across the overlap, then cuts a minimum-error seam through it and cross-fades over a narrow band at that cut. The IR defect mask follows the same cut, so it describes the part each pixel actually came from. Averaging the whole overlap instead draws twice any detail the registration could not line up, which film that is not flat between shots always leaves behind. The two nest: every part carries its own triplet, and flat-field and unmix apply **per part, never to the composite canvas**.
*   **Log conversion**: film density is logarithmic ($D \propto \log E$), so the raw signal goes to log space:
    $$E_{log} = \log_{10}(I_{raw})$$
*   **Bounding and polarity**:
    Statistical percentiles detect the usable signal range. The target **White Point** always maps to the **Floor** ($0.0$) and the **Black Point** to the **Ceiling** ($1.0$).
    *   **Negative (Color Negative / B&W Negative)**: raw low signal (dense highlights) maps to Floor ($0.0$). Raw high signal (film base, shadows) maps to Ceiling ($1.0$). Range: 0.01% to 99.99%.
    *   **Transparency** takes none of this metering: its window is fixed and the camera matrix is applied instead. See *Transparency transfer* in §3.

    Bounds are sampled on **two independent axes** (`_sample_log_bounds`). A **luma** pass fixes the floor and ceiling mean (center plus span), and a **color** pass fixes each channel's deviation from it, so color balance is tunable without compressing the luminance range. Mono gives zero deviation at any clip.
    *   **Luma Range Clip** (`luma_range_clip`): the *luminance* percentile window (dynamic range). **Positive** values tighten it symmetrically, for dense or fogged negatives with outlier pixels. **Zero** uses robust extremes: a block-median prefilter rejects dust and speculars, and a small base clip excludes tiny outlier populations. **Negative** values push the bounds *outward*, leaving lifted blacks and unclipped highlights as headroom.
    *   **Color Clip** (`color_range_clip`): the sampling depth for per-channel color deviation (white balance, orange-mask cast). A **tighter** (larger) value samples deeper for a more outlier-resistant balance; a **gentler** (smaller) value samples nearer the extremes. The neutral default is `base_color_clip` ($1.0$); the slider spans log-interpolated values either side.

        The **thin end** (film base) uses per-channel percentiles, anchored because density is bounded below by base. The **dense end** (scene highlights) reads one shared, chroma-gated pixel set (the lowest-chroma subset of the luma-extreme band, base-anchored), so colored highlights cannot pass as film cast. With no trustworthy neutrals, and always in Transparency, it falls back to per-channel percentiles.
    *   **White and Black Point offsets**: fine-tune the detected bounds without re-running the analysis. They are roll defaults on the Roll tab's Metering card, as Dmin and Dmax belong to the stock. A **[Global / R / G / B]** selector scopes the sliders to per-layer trims (`white_point_trim_*` / `black_point_trim_*`) added on top of the global offsets, for per-dye-layer Dmin and Dmax correction (`per_channel_point_offsets`, single source for CPU and GPU; Transparency negates; hidden in B&W Negative).
    *   **Roll baseline and locks**: Roll Analysis measures every frame with its own crop, freehand analysis region and crosstalk. A frame whose luma-free color offsets (per channel and bound) lie further than `POOL_OUTLIER_DISTANCE` log density from the median is an outlier: shot under a different light, or scanned with a different per-channel balance. The baseline pools the other frames: luma is the median of their luma-weighted floors and ceilings, color the mean of their offsets. An outlier keeps its own bounds (both averages off), on this run and on every later apply of the same baseline, because its color offset also shifts luma through G. **Use average: Luma / Color** (`use_luma_average` / `use_color_average`, the Roll Analysis card's roll defaults) apply the baseline independently per axis. **Lock bounds** (`lock_bounds`) exempts a frame from Roll Analysis, on the first run and every re-analysis. A **scene** is a hand-picked group of frames with its own baseline: Scene Analysis pools only its members and writes to them alone. Roll Analysis and a picked roll baseline skip scene members. The analysis also pools each inlier frame's Cast Removal **neutral axis** (raw log, each frame's own unmix and region) by a confidence-weighted median per band and channel (`pool_neutral_axis`); the highlight band pools only when at least half the frames have one. **Use average: Cast** (`use_cast_average`, Color Negative only) renders Cast Removal with that pooled axis (`locked_neutral_axis`), on both engines: the R/B-vs-G curve is the pool's (film and process), the midtone level moves toward the frame's own by `weight × confidence` (`blend_neutral_axis`). The weight is the pool's, $1 - \sigma^2_{noise}/\sigma^2_{offset}$: the spread of the frames' midtone offsets from the pooled curve against the meter noise, estimated as half the spread of their shadow-minus-midtone offsets, since the curve's shape is the same for every frame. Frames under one light give a weight near zero and render alike; a pool mixing lights keeps each frame's own level. Below three frames the weight is zero.
*   **Stretch**: every mode stretches each channel independently to $[0, 1]$, which neutralizes the orange mask in negatives and base tints and fading in reversal film. The result is not clamped: tones outside the bounds are rolled off later by the print curve's toe and shoulder.
*   **Per-frame metering**: normalization also measures inputs for the Print stage helpers: per-channel **shadow references** ($P_{98}$, for Cast Removal), an **exposure anchor** (for Auto Density) and a **textural range** ($P_{10}$ to $P_{90}$ luminance, for Auto Grade). The anchor and textural meters read only **textured cells** of the block-median grid: sectors of 2×2 blocks of `activity_block` cells vote when their block means span more than `activity_gate_density` ($0.05$ log D; Boyack & Juenger, US 5,724,456). When fewer than `activity_min_fraction` of sectors pass, every cell votes. See §3.
*   **Channel unmix / crosstalk** (`crosstalk_strength` / `crosstalk_matrix`): a 3×3 unmix (`.toml` profiles, see docs/CROSSTALK.md) of the raw **negative** log densities *before* bounds analysis and the stretch, where secondary dye absorptions are linear (Beer-Lambert). The light spectrum and CFA passbands are the same linear operator in log density, so a profile belongs to a *scanning setup*, not a film stock. The sensor matrix (linear, before the log) and Hue Trim in §3 are not interchangeable with it. The matrix is blended with identity by strength and row-normalized, so grays are preserved, and every meter (bounds, anchor, shadow refs, neutral axis) reads the unmixed film, including Roll Analysis. After changing the matrix or the strength, re-run Roll Analysis and re-check locked bounds.
*   **Scan-clip warning**: the per-channel fraction of source pixels at or above sensor white ($\ge 0.99$ linear) is reported (`scan_clip_fractions`). In a negative, film base and shadows sit near sensor white, so clipping collapses them to $D=0$; the Analysis panel warns above 1%. The engine attempts no reconstruction.

---

## 3. The Print (Exposure)
**Code**: `negpy.features.exposure`

*   **Virtual darkroom**: simulates light through the normalized log signal onto paper.
*   **B&W Negative (panchromatic)**: the normalized signal collapses to a **single density** (its luminance) *before* the curve, so the H&D curve shapes one channel. Per-channel color controls are hidden.
*   **Color timing**: subtractive CMY filtration in log space, like an enlarger's dichroic head. Targets **Global**, **Shadows** or **Highlights**. Shadow and highlight offsets are weighted by a sigmoid about the midtone, $w_{sh} = \sigma(3 \cdot (v - z))$ with $z$ the midtone zone center (`anchor_target_density`). The Temperature slider, the WB picker and the temperature roll-lock act on the *selected region's* M/Y pair.
*   **The H&D curve**: an **asymmetric toe-linear-shoulder** curve in **density** space. A line of slope $k$ through the exposure pivot is bounded above by the **toe** (into paper black) and below by the **shoulder** (into paper white). Both bounds are independent **softplus** knees, so each slider shapes only its own end. With $v = k \cdot (x_{adj} - x_0)$:
    $$v_1 = D_{min} + \frac{\text{softplus}\big(a_{hl} (v - D_{min})\big)}{a_{hl}} \qquad \text{(shoulder → paper white)}$$
    $$D = D_{max} - \frac{\text{softplus}\big(a_{sh} (D_{max} - v_1)\big)}{a_{sh}} \qquad \text{(toe → paper black)}$$
    *   $D_{min} = 0.06$: paper white. **Paper White Base** (`paper_dmin`) toggles it; off uses $D_{min} = 0$.
    *   $D_{max} = 2.3$: paper D-max. The softplus toe rolls density into $D_{max}$ directly, with no separate virtual asymptote.
    *   $a_{sh}, a_{hl}$: knee sharpness, from `toe_sharpness_base` ($6.0$) and `shoulder_sharpness_base` ($3.0$) scaled by `toeshoulder_width_ref` / width.
    *   $k$: per-channel slope, from **Grade**.
    *   $x_{adj}$: input log-exposure after CMY offsets. $x_0$ is the pivot.
*   **Preflash** (`preflash`, 0 to 1, a fraction of the threshold exposure): a uniform exposure added to the image exposure, applied to the straight line before any shaping, $v \leftarrow v_{th} + \gamma \log_{10}\big(10^{(v - v_{th})/\gamma} + f\big)$. $f$ is the flash, $\gamma = 2.9 / (R/100)$ the paper's density per log exposure at the frame's ISO-R grade, and $v_{th}$ the value that prints `preflash_threshold_density` ($0.04$, the ISO R start) above $D_{min}$ (`preflash_params`). The flash is a paper exposure, so the scan's log range does not change it. It follows dodge/burn, so a mask never dodges the flash; bare paper fogs neutral and stays at or under the threshold. Highlight hold measures its tone after the flash.
*   **Variable-gamma paper S-curve**: before the bounds, $v \mathrel{+}= \gamma \cdot w \cdot \tanh\big((v - v^{\ast})/w\big)$ (`paper_midtone_gamma` $= 0.05$, `paper_gamma_width` $= 0.6$). Centered on the reference tone $v^{\ast}$ (the anchor holds), it eases to zero toward toe and shoulder. **Snap** (`midtone_gamma`) is a user trim on the paper's baseline $\gamma$. In R/G/B mode it adds per-layer trims (`midtone_gamma_trim_*`), which is midtone crossover, evaluated per channel (`per_channel_midtone_gamma`, single source for CPU, GPU and chart).
*   **Grade (ISO-R)**: contrast as an **ISO range (R) value**, default 115, range 50 to 180. R110 is about paper grade 2; higher R is softer. $k = \text{(grade contrast scale)} \cdot \text{range} / (R/100)$ (`grade_contrast_scale` $= 2.9$), clamped to $[2.0, 10.0]$ Edits on the old 0 to 5 paper-grade scale migrate with $R = 150 - 20 \cdot G$.
*   **Split Grade** (`shadow_grade` / `highlight_grade`, ISO-R points, negative = harder): zone-local contrast. The curve rotates about the shadow and highlight zone centers, $v \mathrel{+}= \Delta k_{ch} \cdot w \cdot (v - z_{zone})$, with the same mid-sparing sigmoid weights as Zone Density ($z_{sh} = z + 0.75$, $z_{hl} = z - 0.40$, $k = 4$). The points fold into a slope ratio like Grade trims (`split_grade_deltas`, single source for CPU, GPU and chart), with per-layer trims (`shadow_grade_trim_*` / `highlight_grade_trim_*`). It runs **before** Zone Density as its own block, because sequential blocks stay monotone and shared weights do not.
*   **Per-layer trims (crossover correction)**: the **Global / R / G / B** selector on the Tone page trims one dye layer against the shared curve. CMY only *shifts* a layer's curve; **crossover** (shadows one color, highlights the complement) needs a shape change:
    *   **Grade trim** (`grade_trim_*`, ±30 ISO-R points): folds into the slope like a paper's `channel_gamma`. As $k \propto 1/R$, the trim is the ratio $R/(R+\Delta R)$, and the pivot is re-solved per channel, so the layer rotates about the anchor and midtones stay neutral.
    *   **Toe / Shoulder trims** (`toe_trim_*` / `shoulder_trim_*`, ±1 on the global knee): per-layer endpoint casts. Effective values are clamped to the slider domain (`per_channel_toe_shoulder`, single source for CPU, GPU and chart).
    *   **Snap trim** (`midtone_gamma_trim_*`, ±0.5 on the global Snap): a cast only in the midtones; endpoints and anchor stay neutral.
    *   **Width trims** (`toe_width_trim_*` / `shoulder_width_trim_*`, ±2 on the global widths, effective values clamped to [0.1, 5]): per-layer knee *sharpness*, how far one layer's roll-off reaches (`per_channel_widths`, single source for CPU, GPU and chart).
*   **Toe and Shoulder**: independent, slider values scaled by $0.85$ internally, evaluated **per channel** (global value plus layer trim). The slider sets roll-off **height**; sharpness comes from the per-channel width control:
    *   **Toe** (shadows): lifts the paper-black ceiling, $D_{max,eff} = D_{max} - \text{toe} \cdot 0.90$ (`toe_height`). The 0.90 (above `shoulder_height`) roughly equalizes the two sliders in $L^{\ast}$. Negative toe *sharpens* the shadow knee and, with **Paper Black** off, raises the BPC clip point (see Output), which makes exact black reachable.
    *   **Shoulder** (highlights): lifts the paper-white floor, compressing and graying highlights, $D_{min,eff} = D_{min} + \text{shoulder} \cdot 0.35$ (`shoulder_height`).
    *   **Grade-coupled baseline**: hard grades get snappier toes and compressed shoulders automatically, scaled by the normalized slope (`toe_grade_strength` $\approx 0.058$, a baseline $\Delta D$ of $0.15 \cdot 0.35$; `shoulder_grade_strength` $= 0.12$).
*   **Zone Density (ΔD)**: two achromatic sliders (`shadow_density` ±0.9, `highlight_density` ±0.5), a literal density offset at full zone weight, that move the shadow and highlight zones without reshaping the knees. Unlike regional CMY, each has a **mid-sparing** weight centered in the three-quarter and quarter tones: $v \mathrel{+}= \Delta D_{sh} \cdot \sigma\big(k(v - z_{sh})\big) + \Delta D_{hl} \cdot \big(1 - \sigma(k(v - z_{hl}))\big)$ with $z_{sh} = z + 0.75$, $z_{hl} = z - 0.40$, $k = 4$ (`zone_density_*` constants, mirrored as literals in `exposure.wgsl`). It runs before the softplus bounds, so a shadow burn never passes paper black and a highlight bleach never crosses paper white. The chart mirrors the shift (`CharacteristicCurve`).
*   **Dodge & Burn** (`negpy.features.local`): masks with a print exposure in **stops** (`LocalMask.stops`, ±2, default 0; positive burns, negative dodges, signed like `vignette_stops`) and a Gaussian feather ($\sigma$ as a fraction of the short side). They rasterize to a per-pixel stop map added to the log-exposure input with the CMY offsets. One stop is $\log_{10}(2)$ scaled by each channel's stretch range (`local_ev_scale`). Vertices are in raw-image coordinates and follow geometry (rotation, flips, distortion). The Flat intent skips them.
*   **Mask shapes** (`LocalMask.shape`): vertices are the universal store and `shape` says how to read them; every control point goes through `map_coords_to_geometry` the same way.
    *   *Polygon*: N points, closed and Catmull-Rom smoothed (`smooth_polyline`).
    *   *Oval*: 3 points $(c, p_1, p_2)$. Outline $c + u\cos\theta + v\sin\theta$ over 64 samples, $u = p_1 - c$, $v = p_2 - c$. The axes need not be perpendicular. Points map to pixels before the outline is built, which handles the raw-image aspect ratio.
    *   *Gradient* (the card edge): 2 points $(a, b)$, alpha $= 1 - \text{smoothstep}(t)$ with $t = \big((p-a)\cdot d\big)/|d|^2$ clamped to $[0,1]$, $d = b - a$. Full exposure at and behind $a$, none at and past $b$. Feather does not apply; handle spacing sets the softness.
    *   *Invert* (`LocalMask.invert`): $\alpha \rightarrow 1 - \alpha$ after the feather.
    *   *Enabled* (`LocalMask.enabled`, default on): a disabled mask is skipped in `compute_local_maps` and adds nothing to either plane. It stays editable, with no canvas tint, printing-notes badge or recipe line.

    One function (`local/logic.rasterise`) serves the render, the canvas tint and the printing-notes map. The GPU uses the same CPU-rasterised map (`compute_local_maps` → the dodge/burn texture), so shapes have no parity surface.
*   **Local Grade** (`LocalMask.grade`, ISO-R points off the frame's Grade, negative = harder): a burn through a different filter. `compute_local_maps` rasterizes a second plane (plane 0 EV, plane 1 summed $\Delta R$), turned into a per-pixel slope multiplier $R/(R+\Delta R)$ clamped to the ISO-R ladder (`local_grade_factor_map`, single source for the CPU kernel and the GPU's uploaded map). It multiplies the straight-line slope only, $v = k \cdot g \cdot (x_{adj} - x_0) + c \cdot x_{adj}^2$, rotating **about the channel pivot**, the same in all three channels; the cast-removal curvature $c$ stays global. On the GPU it rides the dodge/burn texture's green channel. Metrics and the zone ruler describe the frame-wide grade.
*   **Tone limit** (`LocalMask.key` *Highlights* / *Shadows*, `key_zone` and `key_softness` in print zones): the lith mask in register (Kodak O-3), so a burn stops at the subject's own edge. A limited mask's weight is $\alpha \cdot s(t)$, $t = (L - e_0)/(e_1 - e_0)$ clamped, $s = t^2(3-2t)$ (`tone_key_weight`), where $L$ is the luma of the **unburned** normalized log, so no mask can move the pixels it selects. The edges invert the achromatic print curve at $Z \mp S/2$ (`key_edges`, the inverse of `predicted_zone`) from the render's own metrics. The weight depends on each pixel, so the first four limited masks (`MAX_KEYED_MASKS`) keep their own shape planes after planes 0–1 (on the GPU, one texture) and a fifth prints unlimited. Limited grades add to plane 1's $\Delta R$ before the one clamp to the ISO-R ladder; the GPU carries that $\Delta R$ in the dodge/burn texture's blue channel.
*   **Contrast Mask** (`ExposureConfig.contrast_mask`, ±0.5, default 0): the darkroom unsharp mask, a blurred low-gamma film mask sandwiched with the negative: $D' = D - g\,\text{blur}(D) + \text{const}$, a linear high-boost in log density (Ctein, *Post Exposure*; Bond, *Unsharp Masking*; Adams, *The Print*).

    $g$ is the signed mask gamma. Positive (a blurred positive) scales the low-frequency range by $(1-g)$; negative (same polarity as the negative) by $(1+g)$. Detail above the blur scale passes at unity either way, so unlike a harder Grade it leaves micro-contrast alone, and it still works when a flat negative sits on the 2.0 slope floor.

    It is not a stage. `contrast_mask_plane` builds a zero-mean plane that becomes stops of print exposure read with the dodge/burn map, equal in every channel (no per-layer trim). Both engines call the same helper on the same pre-geometry array, with $\sigma$ a fraction of the analysis grid, so preview, export and both engines match. The slider is a scalar on the plane (`contrast_mask_scale`) and rebuilds nothing: the CPU caches the plane at render size (`expand_mask_plane`); the GPU keeps it on the analysis grid in its own texture and upscales in the shader.

    **Mask Spacer** (`ExposureConfig.mask_spacer`, 2 to 6%, default 4) is the blur $\sigma$ as a per-cent of the analysis grid: a frequency cut-off, not a strength. A thin spacer reaches further down and compresses harder; the floor keeps it from collapsing into a plain $(1-g)$ Grade. It rebuilds the plane, so it keys both engines' plane caches, including the single plane a tiled export shares across tiles.

    The plane covers the printed frame only (a blurred rebate would print as a vignette), placed back at the crop and edge-replicated outside so the crop tool's full-frame preview has no seam. Hidden on the transparency transfer path, which takes no dodge/burn map. Instruments read the unmasked negative.
*   **Output**: print density back to **scene-linear** reflectance (transmittance):
    $$I_{out} = 10^{-D}$$
    *   **Paper Black** (`paper_black`, off): off applies black point compensation, as in ICC relative-colorimetric soft-proofing, so the display shows paper black as black. On keeps the paper's lifted D-max ($10^{-2.3} \approx 0.005$). With compensation each channel is $I_{out} = (I - t_b) / (1 - t_b)$, clamped at $0$, $t_b = 10^{-D_b}$, where $D_b$ is the physical $D_{max}$, or $D_{max} + \text{toe}_{ch} \cdot 0.90$ when that layer's toe is negative. A **negative toe raises the clip point**, which makes exact $0$ reachable. The reference is the *physical* $D_{max}$, not $D_{max,eff}$, so a lifted toe and per-layer shadow casts survive. A negative per-layer toe trim (with compensation on) tints the deepest black.
    *   **Scene-linear internally**: the exposure stage emits linear light and every creative stage (Local Contrast, Retouch, Lab, Toning, Finish) works on it. The working-space OETF (the **Adobe RGB (1998) TRC**, a pure $563/256 \approx 2.199$ power, no linear segment) is the final engine step, matching the Adobe RGB ICC profile at the display/export boundary. Retouch is perceptual: the CPU brackets it through the OETF (encode → heal → decode); the GPU keeps one encoded region (exposure → clahe/retouch encoded → lab decodes back to linear).
    *   **The soft proof is a color table, not a stage.** `soft_proof_preview` runs on a $65^3$ identity grid inside `soft_proof_lut`, uploaded as the 3D texture the canvas shader samples; the CPU display path uses `apply_lut_f32`. Rendering intent, black point compensation, paper white, ink black and the gamut warning cost one cached rebuild on change, with no shader and no parity surface. `ProofCondition` holds them as one hashable value that keys the table and the render memo. Paper white is the absolute intent; ink black drops black point compensation; the gamut warning sets unprintable grid nodes to flat gray.
    *   **Printability**: both engines bin encoded content into a joint $32^3$ RGB histogram (`color_hist.wgsl` and `analysis.color_histogram`, mirrored constant `COLOR_HIST_BINS`), since gamut is a property of the triple. The gamut mask is built on the CPU from the proofed output profile (`ImageProcessor.gamut_lut`, cached per profile pair) by round-tripping the identity grid source → output → source, and the Analysis panel combines the two. The tolerance sits above the transform's 8-bit noise, so the read-out under-reports at the boundary.
    *   **Input ICC overrides primaries only, never the tone curve.** The rendered buffer's TRC is always the working-space OETF. For a **matrix/TRC** profile (`infrastructure/display/icc_profile.py`) the boundary transform takes only its primaries (Bradford-adapted to D65, through its `chad` tag when present) as a plain matrix; the declared TRC is inert (`sRGB-*-g10.icc` and `sRGB-*-srgbtrc.icc` render **identically**). A **LUT** (A2B0/B2A0) profile runs through the full CMS transform, so its input curves apply; the bundled narrowband `RGBScan.icc` puts its compensation there, authored against this boundary encoding.

### Automatic helpers

The helpers correct each frame partially. With them off, a dense negative prints dense and a flat one prints flat.

*   **Auto Density** (`auto_exposure`, **on**): meters the textured content. The metered tone $m$ is the average of the trimmed window's mean and midpoint ($P_5$ to $P_{95}$ of the textured cells; Boyack & Juenger, US 5,724,456), so a skewed histogram is placed by its detail-bearing span. The anchor is a partial pull from the assumed key (`assumed_anchor` $= 0.46$):
    $$\text{anchor} = \text{assumed} + s \cdot (m - \text{assumed}), \quad \text{clamped to } \pm b$$
    with $s =$ `anchor_meter_strength` ($0.2$) and $b =$ `anchor_meter_band` ($0.12$), so a low-key or high-key shot keeps its mood. The anchor prints at `anchor_target_density` ($0.75$), which sets overall print brightness.
*   **Auto Grade** (`auto_normalize_contrast`, **on**): picks the grade from the textural density range $t$ ($P_{10}$ to $P_{90}$ of the textured cells, log D), a partial ISO R match (Alkofer, US 4,731,671). With $r$ the floor-to-ceil range, the effective contrast range is:
    $$K \cdot r \cdot \min\!\Big((1 - s) + s \cdot \frac{n}{t},\; c \cdot \frac{n}{t}\Big)$$
    with $K =$ `auto_grade_target` ($0.85$), $n =$ `auto_grade_nominal_range` ($0.9$, a normal negative's textural range), $s =$ `auto_grade_strength` ($0.4$) and $c =$ `auto_grade_max_overfill` ($1.2$). At $s = 0$ every frame prints on one paper; at $s = 1$ every textural range prints to the same span. The cap $c$ keeps a very wide negative from clipping at both ends.

    **Shadow reach** (Gindele, US 7,113,649): the darkest textured tone ($P_{99}$ of the textured cells, `shadow_reach_percentile`) must print at least `shadow_reach_density` ($1.9$, straight-line) with the anchor at its target. If the grade falls short, the slope is raised to the line through both points, never lowered. It reads the same normalized luma as Auto Density.

    **Highlight hold**, the soft-exposure half of a split-grade print (Agfa, US 4,104,069 / US 3,839,036): the brightest textured tone ($P_2$, `highlight_hold_percentile`) must print at least `highlight_hold_density` ($0.10$; $0$ turns it off). If it would print brighter, an automatic highlight burn is added to the Zone Density highlight term, solved against that term's weight at the tone so it lands exactly and stays under the shoulder, capped at `highlight_hold_max` ($0.5$). It never lifts. Highlights Grade is not the actuator, because it pivots at the highlight zone center, where these tones sit.
*   **Set Targets** (app-global): a dialog beside the two toggles tunes the seven aim values (`anchor_target_density`, `anchor_meter_strength`, `anchor_meter_band`, `auto_grade_target`, `auto_grade_strength`, `shadow_reach_density`, `highlight_hold_density`; `TUNABLE_TARGETS` in `features/exposure/models.py` declares ranges). They are a **calibration, not per-image state**: `apply_targets()` overlays them onto `EXPOSURE_CONSTANTS`, persisted in the `exposure_targets` global setting. No `WorkspaceConfig` hash sees them, so both engines fold a `TARGETS_REVISION` counter into their cache keys: the CPU base stage (the metering), and on the GPU the analysis cache key plus the exposure-stage diff. A `TUNABLE_TARGETS` entry read *outside* those stages needs its own invalidation.
*   **Cast Removal** (`cast_removal_strength`, default $1.0$ on a negative, $0$ on a slide; $0$ is off): balances each layer against the frame's own grays so neutrals stay neutral from shadows to highlights, not only at the midtone. Applied strength is `confidence × slider` (`effective_cast_strength`), so a frame with few clean near-neutrals is corrected gently.

    It runs on both **color** processes and is hidden for **B&W Negative**. On a color negative it removes the **orange mask** (start $1.0$); on a transparency a cast can be the photograph (start $0$), and it corrects uneven E-6 dye fade. `cast_removal_for_mode` (`features/process/models.py`) is the one place that swap lives: every route into a mode passes through it, and it rewrites only the other mode's default, so a user-chosen strength survives a mode switch. It is a roll default on the Calibration card (`cast_removal` in `ROLL_DEFAULT_FIELDS`); a Film Mode pushed to the roll carries the roll's strength through the same rule. `services/assets/migrations/cast_removal.py` sweeps legacy slides once at startup (a row rewrite guarded by a done flag, because a load cannot tell a legacy row from a fresh save).

    The **neutral axis** meter reads the space the curve consumes: the unmixed film log on the print curve; the working-space log after the camera matrix, against the fixed transfer window, on the transparency curve (`processor.py`, `needs_axis` in `gpu_engine.py`). With **Use average: Cast** on, the roll or scene axis replaces this meter (see *Roll baseline and locks*, §2).

    The primary solve fits each non-green channel to green's **neutral axis**, from references at a highlight, midtone and shadow luma band, each over the band's lowest-chroma pixels (`neutral_axis_*` constants). R and B get a **quadratic** through the three points, with the midtone pinned through the pivot. Deviation from green is clamped ($\pm 0.2$, `midtone_cast_max_offset`) and curvature is bounded (`neutral_axis_curv_max_ratio` $= 0.45$ of the slope) to keep each channel monotonic on $[0,1]$. Outside that range the quadratic is held at its vertex (`quadratic_core`), so a value far past the frame's bounds, such as the Filed Carrier's rebate ramp, never folds back.

    With too few trustworthy near-neutrals, Color Negative falls back to a **two-point tie** on the per-channel shadow refs ($P_{98}$, calibrated for a negative), with the luma anchor pinning the midtone:
    $$k_{ch} = k \cdot \frac{\text{anchor} - r_{green}}{\text{anchor} - (r_{green} - \text{cast}_{ch})}$$
    The cast is bounded ($\pm 0.1$, `cast_removal_max_offset`). A slide with no neutral axis gets no correction.

    The **transparency curve** has no per-channel slope, so it uses a per-channel **affine on density** (`neutral_axis_affine`): a gain and offset that land each channel's midtone and shadow refs on green's, applied before every other control on that curve. Green, a missing axis, zero strength and a degenerate fit are the identity, so the default slide render is a pass-through. The same $\pm 0.2$ clamp applies, the gain is bounded (`cast_affine_gain_limit` $= 2.0$ and its reciprocal), and there is no curvature term.

### Paper profiles
**Code**: `negpy.features.exposure.papers`

A **paper profile** (`paper_profile`, default *Neutral*) sets the H&D curve shape without touching contrast or exposure: $D_{max}/D_{min}$, toe and shoulder sharpness and height, and midtone gamma. Color papers add a per-channel slope crossover (`channel_gamma`), a paper-base tint (`base_tint_cmy`, added to the minimum-density floor, visible in highlights) and a **dye-coupling matrix** (`dye_matrix`, $D_{rgb} = M \cdot D_{dye}$ above base, row-normalized at use). Grade owns contrast, the Density, toe and shoulder sliders trim on top, and *Neutral* reproduces the defaults exactly.

Profiles are **mode-aware**. Color Negative offers the RA4 papers. B&W Negative offers the B&W papers, which carry no RA4 color terms (paper tone is a Toning job) but carry `lith_path`, which the Lith stage reads. Transparency gets only *Neutral*. An incompatible stored value collapses to *Neutral*. Bundled: **Neutral**; *B&W*: Ilford Multigrade RC, Ilford Multigrade FB Classic, Foma Fomatone, Foma Fomabrom; *RA4*: Kodak Endura Premier, Fujicolor Crystal Archive. Values are loosely mapped from datasheets; mainly $D_{max}$ is grounded.

### Dye Separation
**Code**: `negpy.features.exposure.papers.resolve_saturation_matrix` / `compose_density_matrices`

Saturation in density, in the same matrix slot as a paper's `dye_matrix` (compare §6's CIELAB Global Saturation). For density above paper base $e = D - D_{min}$, $M(k) = k \cdot I + (1-k) \cdot J$, where $J$ has every entry $1/3$: $k=1$ is identity, $k=0$ collapses the row to the achromatic mean, $k>1$ boosts separation. $M_{total} = M(k) \cdot M_{dye}$: saturation outermost, so the paper's crosstalk does not reabsorb it. $k$ (`dye_separation`, default 1.0, slider 0.5 to 1.5) takes per-channel trims (`dye_separation_trim_red/green/blue`) like Grade, Toe and Shoulder. Each row sums to 1 whatever its $k$, so neutrals stay flat. $k = 1$ on every channel returns `None` from `resolve_saturation_matrix`, so the default path stays byte-exact.

$M(k)$ is the isotropic case of the standard row-normalized 3×3 masking operator. Real negatives land around $k = 0.8$ to $1.4$.

### Separation Damping
**Code**: `negpy.features.exposure.logic.separation_damping_gain`

**Separation Damping** (`separation_damping`, 0 to 1, default 0) makes $k$ depend on the pixel's existing color. With $c$ the spread of $e$ across channels (`_rms_chroma`'s hue-symmetric measure) and $\bar e$ its mean:

$$h(c) = \frac{c_0 - c}{c_0 + c}, \qquad k_{eff,ch} = k_{ch}^{\,(1-\tau)\,+\,\tau\,h(c)}, \qquad e'_{ch} = \bar e + k_{eff,ch}\,(e_{ch} - \bar e)$$

At $\tau = 0$ this is the matrix above, byte-exact. At $\tau = 1$ the gain is $k$ on a near-neutral, 1 at the reference spread $c_0$ (`separation_damping_ref_spread` $= 0.35$, also a WGSL literal) and $1/k$ far above it: muted and vivid colors move in **opposite** directions. Per-pixel $k$ cannot fold into a matrix, so with damping live the `dye_mix` slot carries only the paper's coupling and the separation runs after it in the kernel.

It is inert at $k = 1$, so the UI disables it there. It stays monotone in $c$ for $k < e^2$, which covers the clamped $[0, 3]$ domain of $k$ and $k_{eff}$.

$c_0$ and the exponent are a designed control, not a measurement. Lateral dye and inhibitor spread (about 20 to 200 µm) is not modeled; it belongs with sharpening.

### Hue Trim (light-source hue rotation)
**Code**: `negpy.features.process.hue.apply_hue_trim`, mirrored in `exposure.wgsl`

An unusual scanning light rotates hues ($|\Delta H|$ is flat across CIELAB chroma), so the correction is a rotation:

$$\begin{pmatrix} a' \\ b' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix}$$

with $L^{\ast}$ untouched. $\theta$ is `process.hue_trim` in degrees (0 = off, ±30 range). A rotation fixes the origin, so neutrals do not move and it is orthogonal to §2's per-channel color clip, which places the gray axis.

It runs on the **scene-linear print**, after the H&D curve and before the working OETF, inside the exposure stage (a drag does not re-run normalization), and it stays active under `RenderIntent.FLAT`, since a light's hue error is a capture defect. The GPU mirror inlines `rgb_to_lab` and `lab_to_rgb`, copied verbatim from `lab.wgsl` (WGSL has no includes), and rotates `transmittance` before `oetf_encode`. $\theta$ rides the `hue` vec4's x lane in `ExposureUniforms`; its yzw lanes carry the preflash fraction, threshold value and paper gamma.

### Flat (log) master, "for editing elsewhere"
**Code**: `negpy.features.exposure.processor.PhotometricProcessor._process_flat` → `apply_flat_curve`

With render intent **Flat** (`RenderIntent.FLAT`), a **true log encoding** replaces the Print stage: flat, low-contrast, like S-Log or LogC before a LUT. The H&D curve does not run.

The normalized signal $\text{val} \in [0,1]$ from §2 is already log. The flat master skips the print path's $10^{-D}$ decode and emits it positive-oriented:

$$I_{out} = \text{clip}\big(\text{lift} + \text{gain} \cdot (1 - \text{val}),\ 0,\ 1\big)$$

*   `flat_log_gain` $= 0.65$: contrast. $<1$ keeps it flat.
*   `flat_log_lift` $= 0.10$: where the scene **shadow** lands (black lift).
*   Result: shadow → $0.10$, mid-gray → $\approx 0.46$, highlight → $0.75$, with headroom at both ends.

Both are fixed (no metering). Manual white balance still applies as a per-channel log shift. The engine bypasses the creative stages (Local Contrast, Lab, Toning, Finish, dodge/burn masks) and bakes no defect repair: only Geometry → Normalization → this log map → Crop run. Export is full-resolution, color-managed at encode into the selected color space, always 16-bit whatever the export Bit Depth says: TIFF at the chosen compression, or lossless JPEG XL if JXL is selected and the color space is JXL-taggable. The CPU engine is forced (there is no GPU flat shader).

### Transparency transfer, "the slide as captured"
**Code**: `negpy.features.transparency` (+ `shaders/transfer.wgsl`), `negpy.features.process.capture_color`

For **Transparency** a **transfer curve** replaces the Print stage. It is exactly the identity at default settings, so the render is the capture. **Positive** (a Slide-only setting) takes the same curve. `process.path.render_path` makes the choice (`TRANSFER`, `POSITIVE`, or `PRINT` for every negative), and the engine routes the base and exposure stages to `features/transparency` on it; the negative's processors hold no slide branch. A Flat master of a slide takes the same base, then the flat log map.

**Normalization** (`TransparencyBaseProcessor`) does no metering to shape the window, but on a Positive frame it meters for Auto Density/Auto Grade, as the paper path does:

*   **Neutral decode.** This path always decodes without the as-shot white balance, whatever Linear RAW says (`effective_linear_raw`), because the camera matrix folds the multipliers back in. Decode, CPU and GPU matrix and every cache token ask that one helper, or the render is tinted by the raw green-to-red ratio. The same helper makes the loader read a TIFF as literal linear data whatever its profile. **Positive** is exempt (`TiffLoader.load`'s `positive_source` parameter): its embedded profile decides the decode (untagged falls back to sRGB).
*   **Highlight reconstruction** (`highlight_reconstruction` on `ProcessConfig`, `effective_highlight_reconstruction`) sets libraw's `highlight_mode`: `0` Clip (default), `2` Blend, or Reconstruct `3`-`9`; the sidebar exposes Off/Blend/Reconstruct (level `5`). It applies to either Transparency render and is forced to `0` off Transparency, under Narrowband and on a merged bracket, whose clip-detection weighting (`features/hdr/logic.py`) needs pixels near the sensor ceiling. `WorkspaceConfig.__post_init__` holds that invariant once `hdr` names the bracket; `HdrWorker.run`'s solve zeroes it per frame, since it runs before the merge exists. On this path, active reconstruction bakes the real white balance into the decode and skips the matrix fold (`highlight_reconstruction_bakes_wb`, `should_fold_camera_wb`), because libraw finds clipping from its own per-channel gains. A non-Clip mode normalizes libraw's gains against the widest multiplier, which darkens the decode; `highlight_reconstruction_bright_gain` passes that ratio back as libraw's `bright` scale wherever a real white balance (baked or native) meets a non-Clip mode.
*   **Camera matrix.** RAW is decoded `output_color=raw` (camera primaries). The decoder's XYZ→camera matrix (libraw `rgb_xyz_matrix`, on `PipelineContext.cam_xyz`) becomes the *working→camera* matrix, is **row-normalized there**, then inverted (dcraw's `cam_xyz_coeff` order). Normalizing after inversion gives a magenta cast that grows with saturation. It is validated against libraw's own cam→sRGB and applied in **linear**, before the log. A source without a matrix (scanner TIFF, JPEG) passes through. The WB fold asks `should_fold_camera_wb`, which refuses a narrowband capture (read through `narrowband_profile_active`, so a flag left on from a negative is inert on a slide). An explicit Input ICC swaps `cam_xyz` for `wb_only_cam_xyz` (`AppController._effective_cam_xyz`), so only the white-balance fold remains: `_XYZ_TO_WORKING` through the same forward/normalize/invert steps gives exactly identity, which the soft-proof and export ICC bypass also assume.
*   **Fixed bounds.** The stretch is a constant window: floor at the decoder's white level ($D = 0$), ceiling `TRANSFER_DENSITY_RANGE` $= 3.0$ decades below, the same for every frame, so a bracketed set does not converge. White and Black Point (`per_channel_point_offsets`, shared with the measured path) shift the window additively and are the identity at zero. The dye-crosstalk unmix is skipped (it models negative film).

**Print** (`TransferProcessor` → `apply_transfer_curve`) inverts the window and deviates only by what the user moved. With $n$ the normalized signal and $R$ the density range:

$$D = R \cdot n - \Delta_{exp} + R \cdot \text{cmy} \qquad D \leftarrow p + (D - p) \cdot c$$
$$D \mathrel{+}= \text{cmy}_{sh} \cdot w_{sh} + \text{cmy}_{hl} \cdot (1 - w_{sh}) \qquad w_{sh} = \sigma\big(k_{wb}(D - z_{wb})\big)$$
$$D \leftarrow D - t \cdot \text{softplus}(D - k_t,\ w_t) + s \cdot \text{softplus}(k_s - D,\ w_s) \qquad I_{out} = \text{ODT}\big(2^{b} \cdot 10^{-D}\big)$$

At neutral settings $\Delta_{exp} = 0$, $c = 1$, $t = s = 0$, so $D = R \cdot n$ and $10^{-Rn}$ inverts the normalization. Every term vanishes at its neutral value, so this holds exactly in float32. The toe compresses *above* its knee and the shoulder *below* its own; `softplus` is monotonic for $t, s \in [-1, 1]$.

**Display rendering.** $\text{ODT}$ is applied last: a fixed baseline exposure $b$ = `transfer_baseline_ev` = 0.7 EV (darktable's raw default), then Narkowicz's closed-form fit to the ACES RRT plus sRGB ODT. A raw capture needs it, because the decode anchors to the **sensor white level** (`no_auto_bright`, `adjust_maximum_thr=0`) and a frame exposed below clipping arrives dark. Both terms are fixed, never metered.

**Positive** (`ProcessConfig.positive_source`) skips both. `apply_transfer_curve` returns the scene stage's output clipped to display range, and the Print sliders still act. The GPU packs `baseline_gain` as `1.0` and carries the skip in the `zone_taper.y` lane.

The controls are the Print sliders, each neutral at its default: **Print Density** → $\Delta_{exp}$ (stops, `transfer_density_stops` $= 2.0$ per unit, higher is darker); **ISO-R Grade** → $c$ (`transfer_grade_ref` $/$ grade, so the shipped $100$ is unity); **Toe** / **Shoulder** → $t, s$ with knees $k_t = 1.6$ and $k_s = 0.35$ density; their **Width** sliders → $w_t, w_s$ scaled about `transfer_width_ref` $= 2.5$; white balance → per-channel density offsets through `filtration_offsets`; its **Shadows / Highlights** split → the Regional CMY term (`wb_split_geometry`); **Shadows / Highlights Density** → the Zone Density term. The last two apply after contrast and before the knees. R/G/B trims go through `per_channel_toe_shoulder` and `per_channel_widths`, as on the print curve.

**Auto Density/Auto Grade** (`transfer_auto_terms`) meter as on a negative, restated on this curve. A slide is a deliberate exposure, so `auto_meter_for_mode` starts them off on Slide, rewriting only the setting still at the mode being left (as `cast_removal_for_mode` does); a live merge holds both off (`WorkspaceConfig.__post_init__`). Auto Density places the anchor at `transfer_contrast_pivot`, the mid-gray density Grade rotates about (`transfer_assumed_anchor` is that reference's fraction of the fixed window). Auto Grade uses `effective_grade_range`/`grade_to_slope` unchanged. Shadow Reach then raises contrast (never lowers) so the textured dark tail reaches `shadow_reach_density`, and Highlight Hold adds a burn through the Zone Density kernel. `shadow_reach_density`/`highlight_hold_density` map onto this curve by tonal position.

Zone Density's geometry comes from the print path's (`zone_geometry()`) by **tonal position**, because the scales differ (print $d_{min}$ 0.06 to $d_{max}$ 2.3, here 0 to `TRANSFER_DENSITY_RANGE`): the shadow center 1.50 re-centers to 1.93, with sharpness scaled likewise. `wb_split_geometry()` derives the Regional CMY center $z_{wb}$ and sharpness $k_{wb}$ the same way, from `anchor_target_density` and the print's fixed sharpness of 3.0.

The shadow lift is **tapered to nothing at the bottom of the window** (`_black_taper`, smoothstep over `ZONE_BLACK_TAPER` = 1.0 density), because there is no paper black to stop it lifting the black point. Split Grade is not mirrored: the transfer curve has no per-zone slope.

> `transfer_grade_ref` and `transfer_width_ref` **mirror** `DEFAULT_WORKSPACE_CONFIG`. If they drift apart, the default render stops being the identity. `test_transparency_transfer.py` asserts they agree.

The paper model cannot be the identity (`d_max` floors the blacks, `anchor_target_density` places mid-gray, the midtone snap and paper-white reference stay live at neutral), so the paper-specific controls (paper profile, Paper White/Black, split grade) and the normalization tuning are hidden in this mode. Downstream stages run as usual.

Dye Separation and Separation Damping stay, since neither needs a paper. Each channel's $k$ (`dye_separation` + `dye_separation_trim_red/green/blue`) applies to density directly, $D'_{ch} = \bar D + k_{eff,ch}(D_{ch} - \bar D)$, §3's $M(k)$ as one scalar per channel. $\bar D$ is the mean of the channels, each softly capped (softplus) at the higher of 2.5 D and the pixel's lightest channel plus 1.5 D. The cap is set on the capture density, where the window clamp always sits at 3 D, and each capped channel then takes its own curve, so Print Density, Grade, WB and the knees cannot move a clamped channel back under the cap. Nothing bounds a channel here the way the paper curve does, and a channel far denser than the rest near the dense end of the window (the red of an out-of-gamut blue sky) is mostly noise, which the plain mean would spread onto the other two. Grays and ordinary colors never reach the cap, so for them $\bar D$ is the plain mean. The cap only rises with its inputs, so $\bar D$ never falls as a channel darkens and a smooth gradient stays smooth. Separation Damping applies §3's $k_{eff}(c)$ law to each channel's $k$ from the shared chroma $c$, measured on the capped channels.

**Capture toggles.** `linear_raw`, `narrowband_scan` and `sensor_matrix` are sticky global settings, so each is *inert* where it does not apply and its control is grayed with the reason, not hidden.

Linear RAW decodes with `user_wb=[1,1,1,1]`; `camera_to_working_matrix` folds the as-shot multipliers back in, normalized to green so exposure does not move, which makes the render independent of the decode with no shader change (the folded matrix rides the existing `cam*` rows). Linear RAW is inert on this path only; with Positive Source on, it decides the decode again.

`narrowband_profile_active` refuses Narrowband for E-6 on every path: `RGBScan.icc` characterizes *negative* dyes, and no input profile can recover a slide from three isolated wavelengths. `effective_input_icc` suppresses the implicit profile, `proof_active` follows, and an explicit user Input ICC still wins. `unmix_block_reason` refuses the sensor matrix on the same grounds.

The GPU mirror is a separate shader, not a branch in `exposure.wgsl`. Like that shader it emits **display-encoded** values (`oetf_encode` at the end of the pass); the CPU stays linear to the end of the engine.

### Linear Output
**Code**: `negpy.services.export.linear_output`

With render intent **Linear** the darkroom pipeline is bypassed. The source decodes to its native linear buffer, lossless geometry (EXIF orientation plus user rotation and flip) is applied, and a 16-bit file (whatever the export Bit Depth says) is written: TIFF (default, at the Export panel's TIFF compression) or standalone lossless JPEG XL, chosen in the Format combo. An Effort slider (1 to 9, default 7) trades JPEG XL encoder speed for compression. No normalization, exposure, color management, flatfield or sensor correction runs. The IR sidecar, when present and not consumed by ICE, uses the same Format; it is single-channel grayscale, and on the JXL gray path `transfer=LINEAR` is its only tag.

**Only the TIFF path is untagged.** A JPEG XL codestream always carries a `ColorEncoding`, even with `photometric`, `primaries` and `transfer` unset (true of `imagecodecs.jpegxl_encode()` and `cjxl`). `_write_jxl()` pins `transfer=LINEAR` but leaves `primaries` at the sRGB default, which is wrong for camera or scanner-native primaries, so a spec-compliant JXL viewer applies a wrong sRGB→display transform. `imagecodecs` cannot pass real chromaticities (`PRIMARIES.CUSTOM`) or an ICC profile (`jpegxl_encode()` has no `iccprofile=`, though libjxl's `JxlEncoderSetICCProfile` has one). NegPy re-imports it correctly, because `JxlLoader` reads no color metadata (`imagecodecs.jpegxl_decode()` surfaces none).

*   **Pakon RAW**: uint16 scanner data is scaled by an expansion factor. F135 (14-bit sensor, confirmed) defaults to 4× (`PAKON_EXPANSION`); F335 (16-bit, detected by file size) defaults to 1×, off. The 2k Square and Panoram specs are assumed 14-bit like the F135 (not verified; override if needed). 4× puts a typical F135 negative peak around 50 to 55% of range (some external Pakon tools use 2×). The Expansion combo overrides it, and the factor is recorded in the TIFF's ImageDescription tag.
*   **LinearRaw DNG**: 4-channel VueScan (RGB+IR) and 3-channel SilverFast HDRi files are read through tifffile, bypassing rawpy. IR, when present, is written as a separate grayscale file with an `_ir` suffix in the main output's Format. Expansion defaults to off; 2× and 4× are available.

    A 3-channel LinearRaw DNG that libraw fails to unpack falls back to the same tifffile decode, replaying the `LinearizationTable`, `BlackLevel`/`WhiteLevel` and `DefaultCropOrigin`/`DefaultCropSize` tags. This covers DNG 1.7 JPEG-XL from DxO PhotoLab/PureRAW or Lightroom Enhance ([rawpy#207](https://github.com/letmaik/rawpy/issues/207)). No camera-to-XYZ matrix is computed; only the as-shot white balance (`AsShotNeutral`) applies, under the **Linear RAW** toggle. The fallback is failure-driven, so a future libraw or rawpy that unpacks these files is used automatically. Linear Output's 3-channel path uses the same decode (`_peek_linear_dng_rgb`), so tagged files export at the correct exposure and SilverFast HDRi files are unchanged. A default crop that trims the RGB plane while a same-file HDRi IR page stays full size raises an error.
*   **Camera RAW**: demosaiced by rawpy (AHD on a CFA sensor, or the Demosaic panel's choice) with `user_wb=[1,1,1,1]` (unity), `output_color=raw` (sensor-native), `gamma=(1,1)` (linear), `no_auto_bright=True`. The resolved algorithm (the actual choice on an X-Trans CFA; see `resolve_demosaic` in `loaders/helpers.py`) is recorded in ImageDescription; a de-mosaiced source records none. Green-normalized as-shot WB multipliers go into XMP as `RAW-WB: R G B` inside `dc:description` (the `RAW-WB` convention of external linear-workflow tools), the source filename into `crs:RawFileName`. No expansion (sensors use the full bit depth).
*   **Coolscan NEF** (`negpy.infrastructure.loaders.nef_loader`): TIFF-structured, with the full-res RGB image in a SubIFD chain (tag 0x014A). The loader picks the largest RGB SubIFD by pixel count and rejects files with CFA/Bayer SubIFDs (camera NEFs). Scanner NEFs are processed images whose content depends on the Nikon Scan settings (curves, gain, DigitalICE and so on), so linear output needs the right settings at scan time. The loader assumes nothing about linearity or color space. No IR channel; extra channels are dropped. No expansion.
*   **Flextight FFF** (`negpy.infrastructure.loaders.fff_loader`): Imacon/Hasselblad files in two variants. Uncompressed 16-bit RGB (FlexColor export) is a big-endian TIFF with the image in a top-level IFD, selected by pixel count because SubfileType is unreliable. SGI LogLuv (raw `.3fr`/`.fff`, compression tags 34676/34677) decodes through `negpy.infrastructure.loaders.logluv`, LogLuv32/24 → CIE XYZ → linear sRGB, ported from [flexcolor-tool](https://github.com/rohanpandula/flexcolor-tool) (MIT).

    LogLuv values exceed 1.0, so the decoder applies per-channel percentile normalization (0.2th/99.8th) after XYZ → linear RGB, as flexcolor-tool does, giving gain-normalized linear transmittance with no gamma. FlexColor metadata comes from tags 50457 (Apple plist: film stock, film type, gamma, DPI, scan date) and 46279 (firmware version, scanner serial). No IR; extra channels dropped. No expansion.
*   **Noritsu RAW** (`negpy.infrastructure.loaders.noritsu_loader`): headerless BGR 16-bit little-endian dumps. Dimensions come from file size against a table of known Noritsu scan sizes (exact match, known width with novel height, novel-width fallback). BGR is swapped to RGB. 12-bit data in 16-bit range; default expansion 16× (`NORITSU_EXPANSION`).
*   **TIFF** (standalone `_decode_tiff`, independent of `TiffLoader` and its sRGB linearization): a 4th channel is IR only when ExtraSamples is 0 (UNSPECIFIED) or missing; 1 and 2 (associated and unassociated alpha) are dropped. Sidecar IR (`_ir.tif`, with optional `_ir_valid` mask) and IR in secondary TIFF pages (SilverFast iSRD) are detected. **Input gamma** declares the source encoding (linear, 1.8, 2.2 or sRGB) for linearization before export. Expansion is available, off by default.
*   **Trichrome triplets**: all three narrowband exposures decode and merge through `merge_rgb_triplet()` into one TIFF. The red exposure is primary (and supplies WB and device metadata); green and blue paths come from the frame's `RgbScanConfig`. No sensor correction.
*   **Stitch composites**: each part is decoded and corrected (flatfield plus sensor correction for single-shot parts, flatfield only for triplet parts), then assembled through `stitch_composite()` with gain compensation, a minimum-error seam and a narrow cross-fade at the cut. Parts can be Trichrome triplets, so a 4-part stitch of triplets turns 12 RAW files into 1 TIFF.

**ICE dust removal** (visible when an IR channel is available, off by default): IR-based dust and scratch correction on the linear buffer before writing.

**Optional corrections** (camera RAW only), all off by default: *Apply white balance* multiplies by the green-normalized as-shot WB gains; *Apply flatfield* applies the configured flatfield gain; *Apply sensor correction* applies the crosstalk unmixing matrix. Stitch composites always get flatfield and sensor correction per part, whatever the toggles, to avoid seams.

`wb_bake_block_reason()` makes *Apply white balance* inert for a Trichrome triplet or a Single-Shot Narrowband capture ("trichrome"/"narrowband", "" otherwise), because as-shot gains correct a broadband scene. Single-Shot Narrowband is detected from the Narrowband toggle or a calibrated sensor matrix. The sidebar checkbox grays on the same reason.

**Output metadata.** The file holds raw pixels plus Make, Model and DateTime from the source; source ICC profiles, EXIF color space tags and XMP color metadata are never copied. For **TIFF**, the description records source format, expansion, white balance and applied corrections (including whether ICE ran) and ends with "no color management"; as-shot WB also goes into XMP. For Flextight FFF, Make includes film stock and type from the plist, and Model includes the scanner serial.

**JPEG XL** carries the same record: `imagecodecs.jpegxl_encode()` has no metadata parameter, so `_write_jxl()` writes the description, Make, Model, DateTime and XMP into the container's Exif and `xml ` boxes after the encode. The IR sidecar is untagged in both formats.

### Peek Negative
**Code**: `AppController.toggle_negative_peek` / `_paint_negative_peek`

The canvas equivalent of Linear Output, not a pipeline stage. The decoded source (`AppState.preview_raw`, pre-bake: before flat-field, sensor unmix and every defect repair) runs through `GeometryProcessor` and `CropProcessor`, so it keeps rotation, flip, straighten, keystone and crop, then `working_oetf_encode` for display. Nothing else runs, and `content_rect` is cleared. The state is transient, exclusive with the flat peek and the before/after split, and any render with no config override drops it.

The peek applies `camera_to_working_matrix` before the encode (a source with no matrix passes through), folding in the as-shot multipliers whenever the decode skipped them (`effective_linear_raw`), including narrowband, unlike `should_fold_camera_wb`. `lightbox_level` then applies one scalar gain that puts a high percentile of the frame at display white, measured before the crop and applied after; a frame with no signal returns None and paints unlifted. The frame is *not* marked `splash`; it takes the normal working-to-display conversion with `proof` set False.

### Peek Embedded Preview
**Code**: `AppController.toggle_embedded_peek` / `_paint_embedded_peek`

The camera's own JPEG (`PreviewManager.try_splash_preview`, the buffer the load splash paints) through `GeometryProcessor` and `CropProcessor`, nothing else. Its context uses the preview's own shape, not `original_res`. It is marked `splash`, so the display transform takes it as sRGB. It is read on the first peek and held in `AppState.preview_embedded` until the frame changes, with the active asset's half-frame slice applied. No metric is derived from it.

---

## 4. Local Contrast (CLAHE)
**Code**: `negpy.features.lab.logic.apply_clahe` (CPU) / `negpy/features/lab/shaders/clahe_{hist,cdf,apply}.wgsl` (GPU)

Contrast Limited Adaptive Histogram Equalization on CIELAB $L^{\ast}$, computed from linear working RGB (Adobe RGB, D65). Chroma ($a^{\ast}/b^{\ast}$) is untouched, so saturation does not rise. CPU and GPU mirror each other bin for bin; the parity test pins them to about 1e-6.

*   **Fixed $8\times8$ tile grid** over the full frame at every render scale, so the preview predicts the export. 256 bins over $L^{\ast} \in [0, 100]$.
*   **Clip limit**: $\max(1, \lfloor \text{strength} \cdot 2.5 \cdot N_{tile} / 256 \rfloor)$ counts. The excess is spread evenly across all bins, remainder to the lowest, so the tile total is conserved.
*   **Per-pixel remap**: a smoothstep-weighted bilinear blend of the four neighboring tile CDFs (tile centers at $(\text{pos}/\text{dims}) \cdot 8 - 0.5$, edge-clamped), then
    $$L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CDF}(L) \cdot 100$$
    with $\alpha$ = `clahe_strength`.
*   A tiled full-res export hands the CDF of its downsampled meter render to every tile (`clahe_cdf_override`), so the tiles share one seam-free mapping.

The control is in the Lab sidebar (`lab.clahe_strength`). Defect repair (§5) runs earlier, on the linear source.

---

## 5. Retouching
**Code**: `negpy.features.retouch`

Removes dust, hairs and scratches. Three sources find defects: the IR channel, statistical detection and manual strokes. Each gives the same continuous **defect score** ($1$ = clean film, floor $0.02$ = certain defect), repaired by one shared fill (step 3 below), plus a routed inpaint for defects too wide for the fill.

This is not a pipeline stage. Every repair is baked into the **linear source before geometry and normalization**, like flat-field, so the meters read cleaned film and CPU and GPU agree without either implementing it.

*   **Infrared (IR) dust removal** (scans with an IR channel: Coolscan, SilverFast iSRD, VueScan DNG):
    Dust blocks infrared while the dyes pass it, so the IR plane is a defect map. Concepts are ported from digital-fauxice (see `NOTICE.md`), a recreation of Digital ICE.

    The IR plane is downsampled for detection and **follows the buffer** being repaired: at most 1.5× under it, capped at 3600 px. Detecting far below repair resolution inflates the mask and the fill supports, and a defect on a tonal edge then prints a dark blotch. Footprint constants below are pixels at a 1600 px reference and scale with the plane; the score's 3×3 erode does not. Only *large* regions under the dead floor count as non-film, since an opaque hair blocks IR as fully as the holder.

    1.  **Normalized ratio**: $r = IR / \text{blur}(\text{dilate}(IR))$, about $1.0$ on clean film, lower under defects. A crosstalk fit first divides out the image's IR ghost. The clean-film floor is then rescaled onto the identity point and the dip scale onto a reference σ per frame, stretching only, so a scanner at or above the reference renders bit-identically.
    2.  **Division tier**: semi-transparent dust attenuates, so the image is recovered as $RGB / r^{\gamma}$, per-channel γ fitted per frame. The gain never lifts a pixel past its local clean base (defect-excluded mean $- \sigma$); a grain-biased local max prints dark rings.
    3.  **Score-weighted fill**: the ratio maps to $s \in [0.02, 1]$ (the IR Threshold slider moves the ramp). Nothing is thresholded, so no mask edge. Cores and hairs are rebuilt as a multiscale score-weighted average over nested supports, $\text{fill} = \left(\sum I \cdot s \cdot w\right) / \left(\sum s \cdot w\right)$; the finest support with clean data wins, so edges continue through defects. Three rungs sit at the detection plane's footprint and one coarse rung at the reference footprint, which alone reaches across a wide defect.

        **Original-floor rule**: dust is dark in negative transmittance, so a repair may only lighten (no dark halo). The rule lets this path measure each support's clean fraction on the raw score and correct an all-defect support downstream. It compares the *low-frequency* deficit, since per pixel it keeps grain peaks and clips troughs.
    4.  **Grain transplant**: both repair paths paste back detail high-passed from the nearest pixel the score calls clean, mirrored *through* the donor so a row across a hair does not streak into bands. Deep in a wide defect the paste flattens and the multiscale blend carries the interior.
    5.  **Routed inpaint**: defects with chebyshev radius ≥ 5 at detection scale go to structure-following inpaint, alpha-feathered, with the same grain transplant. A 2% frame budget bounds it; the fill always runs.

    B&W silver and Kodachrome block IR like dust. Such frames are auto-detected (the IR plane mirrors the image) and skipped.

*   **ICE at scan time** (Scan tab, nkscan backend only):
    The nkscan driver runs its own openICE port during the scan. The repair is baked into the file, not an edit. Use it for speed on a batch; use the other paths when the repair must stay editable.

*   **IR removal, the OpenICE method** (`ir_method = "openice"`, `negpy/features/retouch/openice.py`):
    A port of openICE (see `NOTICE.md`), a reverse-engineering of Nikon Scan's Digital ICE verified byte-exact against the original. It replaces steps 1 to 4 above, shares no code with them, keeps the routed inpaint, and runs at the same point.

    All work is in log density, $D(v) = \frac{M}{16\ln 2}\ln(vM + 1)$ with $M = 65535$ (Beer-Lambert: a neutral defect is a fixed subtraction, the red dye's IR absorption one scalar).

    1.  **Calibration**, once per frame from a roughly 281 px box downsample. The dye→IR crosstalk $c$ is a least-squares slope of IR deviation on red deviation over the four 4×4 quadrants of every all-clear-film 8×8 tile, gated to $|\delta^{IR}/\delta^{R}| \le 0.2$ and weighted $(\delta^R)^2(\sum IR)^2$. Two IR²-weighted reference levels give $IR_\text{ref} = (IR_\text{raw} - c\,R_\text{ref})/(1-c)$.
    2.  **IR gate**: $g = (d_{IR} - c\,d_R)/(1-c) - \theta$, at $IR_\text{ref}$ on clear film, dropping by the light a defect stole.
    3.  **Clean-confidence weight**: $w = \text{clamp}\!\left(1 - (IR_\text{ref} + b - g)/\Delta,\ 0.02,\ 1\right)$ on the horizontal 3-tap minimum of the gate. ICE's fixed $b$ and $\Delta$ do not transfer between scanners, so both come per frame from the gate's median and MAD σ over sampled whole rows (after crosstalk subtraction and the min, at working resolution). The IR Threshold slider biases $\Delta$ between 2σ and 11σ.
    4.  **Two normalized-convolution pyramids** over four scales: 9×9 octagon (69 cells), 5×5 octagon (21), 3×3 binomial tent (16), pixel. $C_\ell = \sum k_i w_i$; gate pyramid $P_\ell$ and color pyramid $L_\ell$ are $\left(\sum k_i w_i x_i\right)/C_\ell$.
    5.  **Reconstruction**: $\text{acc} = L_0 + \gamma_{ch}(IR_\text{ref} - P_0)$, then per scale $\text{detail}^{(\ell)} = (L_\ell - L_{\ell-1}) \cdot 1.25$ through a dead zone $[\beta_\text{lo}, \beta_\text{hi}]$ (min and max of $P_\ell - P_{\ell-1}$ over a 5-point cross). Detail within the IR's own contrast is dust and is dropped; the rest passes, less the threshold, scaled by $\min(2C_1, 1)$, $C_2$, $w^2$. This keeps the fine structure the fill averages away.
    6.  **Give up rather than invent**: a pixel is left alone when $w \ge 1$, when any channel reconstructs non-positive, or when one of four 9-sample probes on the 9×9 perimeter is entirely below the dust floor (ICE's 6.5% of clear film, carried as a transmittance fraction). The floor sits near-opaque, so giving up is a last resort.

        Output is $\max(L_3, \text{acc} + \text{dither})$ ("only fill, never darken"), written at full strength wherever $w < 1$ with no confidence ramp, as ICE does. Unlifted pixels are written back verbatim, since $D^{-1}(D(v))$ is not bit-exact in float32.
    7.  **Grain**: ICE's zero-mean density dither, $\text{dither} = \text{env}(x)\,(u - \tfrac{1}{2})\,\alpha_{ch}x$, $u$ uniform, $\alpha = (0.015, 0.015, 0.025)$, parabolic envelope over $[D(\lfloor 0.01M \rfloor), D(\lfloor 0.99M \rfloor)]$ peaking at 1 mid-band; none unless $x + \text{dither}$ stays in the band. The draw is a hash of the absolute pixel coordinate, not ICE's frame-global LCG, so a banded pass matches a whole-frame one.
    8.  **Routing**: unreconstructable defects go to the structure-following inpaint. Components over 0.2% of the frame are dropped first, since some scanners report the rebate as one huge defect that would blow the 2% budget.

    Byte-exactness is out of scope (80-bit x87 intermediates, frame-global grain order). Density through calibration, gate, all four confidences and all four gate-pyramid levels match the C reference to float32 noise, and the give-up maps agree.

*   **Automatic dust removal** (`dust_remove`, with Threshold and Size):
    A statistical detector on the visible scan, feeding the same fill.

    1.  **Detection plane**: the source min-pooled to the IR detection resolution (`ir_detect_target`), by the same erode-then-average, because an area average dilutes a two-pixel speck below the grain. The preview keeps its own min-pooled plane from load. A segmented RGBI preview filters its IR channel before each segment is reduced, including defects on a segment boundary.
    2.  **Detection proxy**: percentile-normalized source **density** ($-\log_{10}$ linear luminance, 0.5 to 99.5% window), grade-independent in every process mode.
    3.  **Excess and σ**: background = max of a median over $3\times$ Size and a morphological opening by Size, so specks cannot lift their own background and wider features stay. The excess is averaged $3\times3$ and divided by its local MAD σ over $4\times$ Size, both on 8-bit quantized planes. Windows scale with the plane, so Size is a film footprint.
    4.  **Gate**: Threshold maps to a seed bar in σ (3 loosest, 9 default, 48 tightest; geometric above the default). Seeds grow through connected pixels to a lower bar within $2\times$ Size, so a hair becomes one component. The wide-window σ raises the bar for compact marks (a thin dense image line looks like a hair); hair-thin components are exempt. Stat maps are cached, so dragging Threshold re-runs only the gate.
    5.  **Specks → score**: at the floor on the defect, ramping to clean across a small pad, then the shared fill. Where IR Removal already repaired a pixel, the optical mark is released.
    6.  **Hairs → routed inpaint**: strongly elongated defects (thinness $= \text{area}/\text{thickness}^2$, bending-invariant unlike PCA aspect) go to Navier-Stokes inpaint, encoded against the local clean range and alpha-feathered over the dilated PSF skirt.
    7.  **Exclusions**: right-painted strokes (source-normalized points, width at the heal reference scale) release the optical detections they cover: score back to clean, hair mask cleared. A mark crossing the rim keeps its repair outside the brush. The rim feathers like a manual heal, and the band is the whole path, smoothed and round-capped. This runs on the detection plane after the gate and before the fill, and leaves the IR and manual routes alone. An all-clean score skips the bake. Strokes fold into the source token.

*   **Manual heals (Heal and Scratch tools)**:
    A painted stroke is a **search area, not a stamp**: damaged pixels inside it are measured, and clean grain comes back byte-identical.

    The measure is a **two-sided** $|z|$: density high-passed against a local mean (window $3\times$ the brush radius), averaged $3\times3$, over its MAD σ in the stroke's neighborhood. Two-sided because dust is bright in density and a **scratch dark**. **Hysteresis**: $|z| \ge 8$ seeds, connected pixels to $|z| \ge 2$ join. A stroke that never reaches 8 seeds against its own maximum; below $|z| = 5$ it is a no-op. The kept region is padded like the detector's.

    The fill runs with the **original-floor rule off**, since a scratch reads brighter and must be free to darken. Instead it measures each support's clean fraction *above* the score floor, with supports at the film footprint, not buffer pixels. Too-wide cores route to the inpaint. Sparse defects (strokes and detected specks) are repaired each in a padded crop, with the support ladder still from the whole frame, so results match a full-buffer repair.

*   **Transport scratches (the line tool)**:
    A nearly straight scratch a few px wide along the film, too faint per pixel for a brush; it shows only when integrated along the line.

    A click gives a start point. The frame is band-passed *across* the scratch and normalized by a **local** noise scale. The line's **slope** is fitted, pulled toward the click. **Extent and width are measured**: the repair covers the stretch where the ridge holds, and the band grows outward by the same hysteresis as the brush, so a stored line stays valid at any resolution. The mask takes the same skirt pad and fill, original-floor rule off. **Line Sensitivity** sets the ridge bar for both and re-measures placed lines. There is no auto-detection, because a full-length ridge is as likely a horizon, and telling them apart needs a cross-frame pass.

*   **Resolution independence**:
    Coordinates and sizes are relative to the full-resolution RAW. A half frame reports the cropped and split full-resolution dimensions even when its preview uses a half-size RAW decode. Strokes are in raw-frame coordinates and the repair runs before geometry, so rotations, flips and distortion correction need no mapping.

---

## 6. Lab Scanner Mode
**Code**: `negpy.features.lab`

Mimics lab scanners such as Frontier or Noritsu. Steps, in order:

1.  **Chroma Denoise**: a Gaussian filter on the A and B channels in LAB. L, and its grain, stays untouched.

2.  **Global Saturation**: a lightness-preserving chroma scale ($a^{\ast}/b^{\ast}$) in CIELAB, post-decode and independent of the print curve. Paper-dependent color is §3's job (Dye Separation).

    Below 1.0 it is a flat scale. Above 1.0 it is **gamut-aware** (`gamut_aware_chroma_scale` in `kernel/image/logic.py`, mirrored in `lab.wgsl`), because a per-channel RGB clamp would shift hue. `_in_gamut_lab` checks whether the boosted (L,a,b) decodes to linear working RGB in $[0,1]$. In-gamut pixels get the flat scale; the rest get a softplus-style knee toward their own headroom (the print curve's toe and shoulder shape), the max in-gamut scale found by bisection to under 0.1% error in 10 iterations.

3.  **Skin Protection** (`skin_chroma_rein` in `kernel/image/logic.py`, mirrored in `lab.wgsl`, `lab.skin_protection`, default 0.5): a soft chroma ceiling inside a skin mask, after saturation. It also runs at Chroma 1.0.

    The mask (`_skin_weight`) multiplies a Gaussian on hue at $52°\pm20°$, a chroma window at full weight to $C^{\ast}=35$ and zero by $60$, and a lightness rolloff below $L^{\ast}=15$ and above $95$. Pixels under $C^{\ast}=2$ get zero. The chroma window (skin locus $C^{\ast}$ 12 to 40) separates skin from sunset, terracotta and brick; skin above $C^{\ast}\approx 50$ keeps partial weight, and wood, tan leather and sand count as skin.

    With $w$ the mask weight and $s$ the slider:

    $$C_{\text{ceil}} = \frac{22}{s}, \qquad C_0 = 0.6\,C_{\text{ceil}}, \qquad C_{\text{knee}} = C_0 + (C_{\text{ceil}}-C_0)\left(1 - e^{-(C-C_0)/(C_{\text{ceil}}-C_0)}\right)$$

    for $C > C_0$, identity below, $C_{\text{out}} = C + w\,(C_{\text{knee}} - C)$. $a^{\ast}$ and $b^{\ast}$ scale by $C_{\text{out}}/C$, so hue and $L^{\ast}$ are kept. The reciprocal ceiling fades out continuously as $s \to 0$. Chroma is only reduced, so `saturation = 0.0` still reaches gray and the gamut knee's fit holds.

4.  **Sharpening**: **Method** picks Unsharp Mask or Deconvolution. Both share Amount, Radius and Masking and the $\text{radius}$ Gaussian taps from `gaussian_kernel_1d` (`sharpen_k` storage buffer), convolved identically by `cv2.sepFilter2D` and the separable WGSL passes, so CPU and GPU match bit for bit.

    **Unsharp Mask** on $L$ with halo suppression (`lab_sharpen_h/v.wgsl`):

    $$L_{diff} = L - \text{blur}(L, \sigma), \qquad \sigma = \text{radius}$$
    $$\text{gain} = \text{amount} \cdot 2.5 \cdot \text{smoothstep}(0.25, 0.33, |L_{diff}|) \cdot m \cdot g(L)$$
    $$L_{final} = \text{clamp}\big(L + L_{diff}\cdot\text{gain},\; L_{min}-2,\; L_{max}+1\big)$$
    *   **Radius** (px): blur $\sigma$ in output pixels, so judge sharpening at 1:1, not fit-to-window.
    *   **Masking** ($m$): $\text{smoothstep}(0.5t, t, |\nabla L|)$ on the boxed gradient, $t = 10\cdot\text{masking}$. Protects flat areas; off at 0.
    *   **Shadow gain** ($g(L)$, fixed): $\tfrac{1}{3} + \tfrac{2}{3}\,\text{smoothstep}(0, 35, L)$, since the thinnest negative has the most grain (Gallagher & Gindele, US 7,228,004). Both methods apply it; Deconvolution reads $L^{\ast}$ of the observed $Y$.
    *   The noise gate over $[0.25, 0.33]$ is sized for $|L_{diff}|$ at a 1 px radius (tops out near 1.0). The clamp to the local $3\times3$ range is tighter above (+1) than below (−2), because $L^{\ast}$ USM exaggerates light halos.

    **Deconvolution**: Richardson-Lucy on linear luminance $Y$ with a Gaussian PSF, reversing the scanner's optical blur (`rl_*.wgsl`).

    $$\hat{o}_{n+1} = \hat{o}_n \cdot \left(K \otimes \frac{o}{\max(K \otimes \hat{o}_n,\ \epsilon)}\right), \qquad \hat{o}_0 = o = Y$$

    Iterations are $\text{clamp}(\text{round}(10\cdot\text{radius}), 5, 20)$, so preview and export match. No early stop, no damping; the edge mask alone governs grain, as in RawTherapee. Applied as a chroma-preserving RGB ratio:

    $$\mathrm{RGB}_{out} = \mathrm{clamp}\left(\mathrm{RGB} \cdot \max\left(1 + \left(\frac{\hat{o}_N}{\max(o,\epsilon)} - 1\right) \cdot \mathrm{amount} \cdot m,\ 0\right),\ 0,\ 1\right)$$

5.  **Glow**: lens bloom, a print-side effect: blurred highlights added back in linear light.

    $$I_{out} = I + B_{glow} \cdot s_{glow}$$
    $$B_{glow} = \text{GaussianBlur}(I \cdot m_{hl})$$

    *   $m_{hl}$: **display-domain** highlight mask, linear ramp from code value 0.5 to 1.0.
    *   Equal on all three channels; the sum is clamped at the stage output.

6.  **Halation**: red scatter from light reflecting back through the film base. A larger Gaussian than Glow, additive in linear light, with the mask on **linear reflectance** ($t = 0.65$) so it follows scene exposure, not Grade and Density.

    $$I_{out} = I + B_{hal} \cdot s_{hal}$$
    $$B_{hal} = \text{GaussianBlur}(I_R \cdot m_{lin} \cdot C_{hal})$$

    *   $I_R$: red channel, the scatter source.
    *   $m_{lin}$: linear ramp from reflectance 0.65 to 1.0.
    *   $C_{hal}$: tint weights $(1.0,\ 0.3,\ 0.05)$.

---

## 7. Alternative Processes
**Code**: `negpy.features.lith`, `negpy.features.cyanotype`; config `negpy.features.altprocess`

One optional non-enlarging process (B&W Negative mode only), between Lab and Toning so the toners act on it. Lith and cyanotype are mutually exclusive (one `alt_process` enum), default neither; then both engines skip the stage.

### 7.1 Lith

Two phases (Rudman's "New Rules", Moersch Lessons 1-6): a long low-gamma highlight branch fixed by exposure, then infectious development that drives the shadows to Dmax almost vertically, with no separation past it (Moersch's "lith-band"). On the print density $D_0$, with the paper's $D_{max}$:

$$D_0' = D_0 + 0.301 E, \qquad D_h = f_{max}\left(1 - e^{-f_{rate} (D_0 + 0.301 E v) / f_{max}}\right)$$
$$D = D_h + (D_{max} - D_h) \cdot \sigma\!\left(\frac{D_0' - K}{w}\right)$$

*   **Exposure** $E$ (stops) shifts the image up the exposure axis and brings the knee forward. The highlight branch takes only a fraction $v$ (`foot_veil`), so paper white does not fog by a full quarter-stop of density at the practitioner-standard +2.
*   **Snatch Point** sets the knee density $K = D_{max}(\text{knee}_{\text{lo}} - \text{knee}_{\text{span}}\cdot\text{snatch})$, a development-time proxy. Later snatch widens the black band.
*   **Abruptness** sets the knee width $w$ (Moersch: "an almost abrupt blackening sets in"). A hydroquinone-rich, low-sulphite bath (Solution A end) reaches $w \approx 0.03$.
*   **Color** comes from the paper, with no strength slider: its $(a^{\ast}, b^{\ast})$ path on $u = D/D_{max}$, anchors at $u = 0.10/0.35/0.65/1.00$: peach, ochre, **olive**, neutral. Keep the olive knot: warmtone lith goes green between warm highlights and cold blacks. $L^{\ast}$ comes from the density, so the RGB→Lab transform is skipped. Hue tracks silver particle size, which tracks density (Kong & Shore, *J. Imaging Sci. Technol.* 51(3), 2007). The path is `PaperProfile.lith_path`, for the paper chosen in the Exposure panel.

Not modelled: semiquinone and bromide diffusion halos, pepper fog, snowballs and other faults.

### 7.2 Cyanotype

An iron process: ferric salt, contact print under UV, wash. No development, so the print depends on the light through the negative and the sensitizer's density range, which is the contrast control (Ware: about 1.0 to 1.2 for Herschel, about 2.4 for his own; Simple Cyanotype variants at 1.8, 2.3 and 2.7). Midtones compress within that range, so the mid gamma is below one.

On $D_0$, with exposure $E$ in stops and exposure scale $S$ in log D:

$$t = \mathrm{clamp}\!\left(\frac{D_0 + 0.301 E}{S},\ 0,\ 1\right), \qquad v = 2t - 1$$
$$u_0 = (1-m)\,t + \tfrac{m}{2}\left(1 + v\,|v|\right)$$
$$u_b = u_0\left(1 - B\left(1 - \beta u_0\right)\right), \qquad u = u_b + T\left(r\,u_0 - u_b\right)$$
$$D = D_{max}\,(1 + T g)\,u$$

*   **Exposure** $E$: more UV drives more of the scale into blue.
*   **Exposure Scale** $S$: the negative density range printed, so the contrast. A short scale clips both ends of a normal negative.
*   Midtone compression is the reverse-S term, $m = 0.45$, written $v|v|$ so both engines evaluate it identically, and blended with a straight line so the center slope is $1-m$ (a zero slope would posterize).
*   **Bleach** $B$ (washing soda) strips pigment highlights-first, leaving the deepest shadow at $\beta = 0.15$ of its density at full strength. **Tannin** $T$ re-develops the bleached iron as iron tannate, slightly past the blue ($r = 1.05$), with more covering power ($g = 0.15$).
*   **Color**: Prussian blue absorbs mostly red (peak near 700nm), so it goes blue, never black. As in lith, an $(a^{\ast}, b^{\ast})$ path on $u$, anchors at $u = 0.00/0.15/0.55/1.00$: rag white, **green highlight stain** (residual yellow sensitizer, per Ware), mid blue, Prussian blue; $L^{\ast}$ from density. Tannin mixes toward an iron-tannate direction scaled by $u$, so a partial bleach splits blue-brown. **Sensitizer** picks $D_{max}$ and path: Classic (Herschel) 0.95 and greener, New (Ware) 1.40 and deeper (Ware's red-channel Dmax 0.55-1.05 for classic sensitisers, about 1.5 for a good modern print).
*   No silver, so the six chemical toners are skipped. Split toning still applies.

Not modelled: **solarisation** (the reversal dries back to blue, so the finished print does not show it), bronzing, paper texture.

---

## 8. Toning
**Code**: `negpy.features.toning`

*   **Chemical Toning** (B&W Negative mode only): six baths (**Selenium**, **Sepia**, **Gold**, **Iron Blue**, **Copper**, **Vanadium Green**) modelled as a **silver ledger** in density space (`TONING_CONSTANTS`). The mean density $D_0$ is the silver reservoir; each toner converts a fraction, locked away from later baths (Rudman/Ilford: selenium-then-sepia split, "no silver left" exhaustion).
    *   **Susceptibility**: $c_i$ is a pure function of $D_0$; sequence only decides who claims silver first.
        *   *Silver-proportional, shadows first*: Selenium $c = S \cdot (D_0/D_{ref})^{p}$. Iron Blue and Copper likewise, with $D_{ref} = 0.9$ so color reaches the mids.
        *   *Bleach-limited, highlights first*: Sepia, Gold, Vanadium $c = S \cdot (1 - D_0/D_{ref})^{p}$; the exponent sets the split-sepia character.

        Strength $> 1$ is a longer bath; conversion caps at 1.
    *   **The ledger**: untoned fraction $a$ starts at 1; in bath order (selenium → sepia → gold → blue → copper → vanadium) each claims $f_i = a \cdot c_i$, $a \mathrel{-}= f_i$. **Gold** also plates the sulfide (sepia) fraction: the gold-over-sepia orange-red shift.
    *   **Covering power**: $D_{ch} = D_{0,ch} \cdot \left(a + \sum_i f_i \cdot g_{i,ch}\right)$, $I_{out} = 10^{-D_{ch}}$. Each channel uses its own input density, so existing color (a lith print) survives; on a gray print $D_{0,ch} = D_0$.

        Gain triplets $g_i$: selenium all $\ge 1$ (Dmax boost, eggplant shadows), sepia lower covering power (lifts, warms), gold slight intensification (cool blue-black), Prussian blue net $> 1$ with G at exactly $1.00$ (so sepia-plus-blue splits green), copper net $< 1$ (bleaches while it tones: brick red), vanadium R/B absorbed with slight density loss (green print, black blacks).

*   **On a lith print** (`LITH_TONING_CONSTANTS`, when the Lith stage is on): fine lith silver moves further in the same bath. Only two baths change, and only they are exposed in the UI:
    *   **Selenium**: $D_{ref} = 1.2$, exponent $1.0$, so it reaches the mids; a green-led gain turns the green-black shadow magenta (Moersch, 1+5 for 20 s), mean well above 1 for Dmax lift.
    *   **Gold**: flat $c = S$ ("gold toner attacks all densities evenly"), gain blue-violet.
    *   Sepia, iron blue, copper and vanadium are unchanged and inert or redundant on lith, so the sidebar disables them.

*   **Split Toning** (all modes): an additive tint in LAB ($a^{\ast}b^{\ast}$), so luminance, grain and detail are kept. With $L$ the CIELAB lightness ($0$ to $100$):
    $$m_{shadow} = \text{clip}(1 - L/50,\ 0,\ 1), \qquad m_{highlight} = \text{clip}((L - 50)/50,\ 0,\ 1)$$
    For each region, with hue $\theta$, strength $S$ and mask $m$:
    $$a^{\ast} \mathrel{+}= \cos\theta \cdot 20 \cdot S \cdot m, \qquad b^{\ast} \mathrel{+}= \sin\theta \cdot 20 \cdot S \cdot m$$

---

## 9. Finish
**Code**: `negpy.features.finish`

Post-crop print finishing in scene-linear, before the output transform. Order: edge burn → filed carrier; layout extras run at compositing time.

*   **Edge Burn (Vignette)**: $I_{out} = I \cdot 2^{-s \cdot m}$, $s$ the burn in stops (negative = hold back), $m$ a cosine falloff mask. **Roundness** morphs from radial (lens-like) to rectangular (card-like); **Size** sets the falloff midpoint.

*   **Filed Carrier**: full-frame printing with a filed-out carrier. The clear rebate prints between two boundaries, framed by an unexposed margin ($0.7 \cdot w$) in the mat color (`PrintService.effective_paper_linear`, scene-linear since this stage precedes the output OETF and the mat does not).

    The rebate is printed, not painted. `rebate_tone` pushes a neutral density ramp off the film base (the normalization's thin bound plus `REBATE_BASE_MARGIN` $= 0.15$ log units) through the frame's exposure kernel, saturation and toning into a 64-entry table over the exposure fraction $t$ that reaches the paper, sampled at $t = u^3$. The table is divided by its $t = 0$ entry and multiplies the mat color, so bare paper meets the mat exactly. The paper layers' different contrasts and the filtration put a hue into the toe, and with Paper Black on the rebate stops at the paper's D-max. Alternative processes, slides and flat masters use plain light ($1 - t$). Both engines read the one table (GPU storage buffer), so the shader has no print model.

    The picture side is the camera gate: machine-cut, it only wobbles (`carrier_profiles()` rows 0-3, no slider), prints soft and has a small fixed corner radius. The paper side is the filed aperture: **Roughness** swings it off rows 4-7, straight file strokes with raised-cosine nicks weighted toward the corners, printing nearly crisp. **Corners** rounds only the aperture. The aperture is evaluated in its own frame, offset from the picture by `CARRIER_OFFSET_X/Y` of $w$, so the rebate is wider at the top and left. The aperture also gates the picture (weight $A_{in} A_{out}$). Profiles and offset are fixed-seed, so one carrier prints the whole roll.

    A fine **2-D fBm** (`carrier_noise`) displaces both boundaries' distance fields, adding the fibrous hairline and the flecks a 1-D profile cannot. It is hand-rolled hash value noise (u32 wrap-around only) so WGSL reproduces it bit for bit. Cell size scales with $w$.

    **Flare** is the bevel's reflection: exposure added to $t$, peaking on each filed edge over $0.25 \cdot w$ and gated by the other three edges, so it stops at the aperture's corners. It reads through the same table, so it stains the paper in the toe color, neutral in B&W.

*   **Layout extras** (`services/export/print.py` plus the `layout.wgsl` mirror): **bottom-weighted mat** (window-mat proportions) and **match paper white** (mat color from paper white run through the toning stack).

*   **Preview paper size**: an interactive render sizes the paper from the preview long edge (`preview_render_size`) instead of the export DPI, and resamples the content to fit, never above its own resolution, because the canvas quotes zoom against the pipeline's buffer and an upscale would make 1:1 read closer than one scan pixel per device pixel. The display shader magnifies instead.

