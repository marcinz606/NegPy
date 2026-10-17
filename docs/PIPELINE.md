# The Pipeline

These steps run in order. Each stage passes its buffer to the next. A helper named in this document as the "single source" for a per-channel value serves the CPU engine, the GPU engine and the curve chart alike.

**Color handling: no input colorspace.** NegPy works on **linear RGB straight from the raw decode** (`output_color=raw`, `gamma=(1,1)`, unity white balance): the sensor's own channels, never converted through camera primaries. Adobe RGB (1998) is an assumed boundary profile, not an input characterization (`WORKING_COLOR_SPACE` in `infrastructure/display/color_spaces.py`): CLAHE, Lab and Toning compute CIELAB from linear data with Adobe RGB primaries and D65, and the Adobe RGB TRC is the final engine step. Primaries apply only on the way **out**: the preview is color-managed from the working profile to the display profile, and export converts to the target space and embeds its ICC profile. Every render, preview, detection and thumbnail decode passes `adjust_maximum_thr=0.0`, which pins the scale to the camera's calibrated linearity limit (`user_sat`, or the format's generic white level when the body has no calibration table), not LibRaw's per-frame maximum, so a roll decodes on one scale and the preview matches export. Linear Output's decode keeps the pin but not `user_sat`.

## 1. Geometry (Straighten & Crop)
**Code**: `negpy.features.geometry`

*   **Rotation**: 90° steps, then affine fine rotation with bilinear interpolation.
*   **Lens distortion**: a radial $k_1$ coefficient (`geometry.distortion_k1`, ±0.10), corrected in the same resample.
*   **Tilt and Swing** (`geometry.converge_v` / `converge_h`, ±15%): the plane-to-plane projectivity of a tilted easel (Hartley & Zisserman §2.3). Tilt straightens converging verticals, swing converging horizontals. It runs last in the forward chain, after $k_1$, because a projectivity cannot fit a barrel-distorted frame. The unit is percent of the frame: convergence is $(H/2)\sin\tau / D$, and no magnification or focal length is modeled. Output keeps the canvas size and replicates the wedge, as fine rotation does. A quarter-turn swaps the two between axes; a flip negates the one on the flipped axis (`rotate_geometry_and_analysis`, `toggle_flip`).

    `keystone_matrix_normalized` is the single definition of the quad; the pixel-space matrix and the GPU's inverse derive from it. Every geometry reader applies it: the uv grid, `map_coords_to_geometry`, autocrop's replay and detection key, and the GPU's analysis replay. Off-frame card-edge handles need `CoordinateMapping` to fit the grid projectively.
*   **Crop by Default** (`geometry.crop_to_valid`): the largest axis-aligned, frame-centered rect that fine rotation and keystone leave void-free, used as the ROI when no manual or detected crop is set. Distortion is excluded, because it scales to fill. `compute_geometry_crop_rect` ternary-searches the half-width of the convex valid quadrilateral, with a nested binary search for the half-height.
*   **Auto Skew** (`geometry.skew`): sets fine rotation, tilt and swing from the frame's own straight edges, never the picture. A length-weighted mode of long segments turns the frame near square. The film detector's boxes (or, on a tight scan, the border ring) place the film and gate edges. Each side is fitted as a line, and Gauss-Newton levels every line through the app's own warp. A side's line is searched inside its box, and outside it only within the detector's snap window. A film box on the canvas border loses its sides on that axis. On a slide, the picture box found from the canvas stands in for the film box. A bowed edge is leveled by its chord. A correction applies only with box-led lines (or a border ring with film base outside it), because a picture line can fit perfectly, and only with an opposite pair: left/right for tilt, top/bottom for swing. A keystone is kept only between 0.3% and 5%, and only where it explains the edges clearly better than rotation alone. Auto-crop all frames takes each frame's rotation from the same fit.
*   **Autocrop**: finds where the film ends and the scanner bed begins from the density jump. A manual override covers light leaks and odd holders.

    Detection runs once per edit, in `ImageProcessor` ahead of either engine, and stores the rect on the edit (`geometry.crop_rect`; `crop_from_auto` marks its origin). Both engines only slice that rect, because a preview and a full-res export can stop on different edges. The rect keeps its detection key (`autocrop_detection_key`: orientation, ratio, mode, rebate trim), and a key change re-detects on the next render. Crop Offset is not in the key, because every render re-applies it. **Square Before Crop** (`autocrop_square`) runs the Auto Skew fit (`skew.squared_geometry`) once per arm, on a frame without its own skew, and detects the crop on the squared geometry. The key holds Square only when it is on. Auto-crop all frames takes the fit's Tilt and Swing per frame and leaves rotation to its pooled deskew.

    The surround of a detected box must read as bed (uniform and near-clipping) or as a dark holder; otherwise detection falls back to the full frame. Opaque edges are then trimmed off the box, up to 20% a side, against a near-black level no negative reaches. A run that uses the whole allowance trims nothing, so a slide's black shadows keep their edges. A box under a quarter of the frame is not film: detection falls back to the full frame, and in a roll the frame is placed from the roll or the roll declines.

    The crop is then squared to the Ratio setting. On `Free` (default) the ratio comes from the detected frame, snapped to the nearest real film format. The snap is checked against the sensor dimensions only where the detected box is the longer of the two.
*   **Rebate Trim**: in *Image only* mode the film box is refined inward to the exposed image, by whichever of three routes can read the frame.
    *   A **uniformity probe**, when bed, rebate and image tiers separate cleanly. It classifies the box against the rebate/image midpoint and takes the longest run of occupied rows, then columns. It refuses a run shorter than half its side.
    *   A **per-side edge walk** (`measure_film_edges`), for camera scans, where mask and base are too thin for the ring (its probe is 4% of the box). Each line is walked alone and depths are pooled per side, because a slight tilt mixes mask, base and picture on one row. The walk recognizes mask (near-black) and bed (near-clipping) by level, and a rebate brighter or a holder edge darker than the picture on the same line. Where it ends on a relative test, the edge is the strongest gradient in the run and must be a real step by magnitude. Sides are independent; a cap and a support test bound the risk. The first pass reaches 3% of the side and is accepted as is. The second reaches 10% and is adopted only where it goes deeper and passes four tests: it stopped inside its allowance, its per-line depths fit a straight line, its outermost pixels hold one level along the side, and its closing step carries most of the drop from that level to the picture.
    *   A **per-side ring measurement** (`measure_film_border`), gated on an opposite pair because a bright sky at the frame edge reads like film base. It reads a wide rebate below bed level.

    Rebate Trim scales the refinement: 0 stops at the film edge, 1.0 lands on the detected image edge, above 1.0 cuts into the picture. The same factor scales the roll-pooled inset in Auto-crop all frames, where walk and ring both feed the roll median and each side pools on its own sample count. A roll too short for any side to reach that count trims by the ring alone, gated on an opposite pair.

    In a roll, a portrait frame resolves from its own single-frame crop, because the roll is landscape. A frame with no film box is placed from the roll, its left and right edges taken from a vertical-edge profile. When no frame in the roll found a box (film overfills the sensor), the roll falls back to the threshold box of single-frame Auto Crop, trimmed by the border walk, only where that box is near the whole frame. Deskew is not pooled: the angle comes from the film-box contour, refined by a line fit on the top edge where the two agree.

**Note:** the crop *selection* resolves here and becomes the analysis ROI. The pixels are cropped only **after Toning**, before Finish, so Lab and the preview overlays see the full frame. The "Analysis buffer" option excludes the outer X% from analysis, and a freehand **analysis region** (`analysis_rect`) sets exactly what the meters read.

A frame with no crop meters a **meter rect** (`process.meter_rect`) in place of the whole frame: the rect Auto Crop would keep, but only when its detector finds the film inside an opaque holder or on bare light (`resolve_meter_rect`). An opaque holder is denser than any film, so in the meters it takes the print's white point. A frame the film fills keeps whole-frame metering. `ImageProcessor` resolves the rect once ahead of both engines, and the controller freezes it with its key (`meter_detection_key`: the detection key plus the source's aspect). A new rect drops per-frame bounds measured without it. Roll Analysis reads the same rect.

---

## 2. Scan Normalization
**Code**: `negpy.features.exposure.normalization`

*   **Physical model**: pixel values are linear transmittance.
*   **Source corrections** (linear domain, before the log conversion):
    *   **Flat-field** (`negpy.features.flatfield`): divides out illumination falloff with a blank reference frame. A per-channel gain map $\text{mean}(\text{blur})/\text{blur}$, computed on a 256 px copy and clamped to $[0.25, 4]$, multiplies the linear source. The gain is stored in **sensor layout**: the bake undoes the reference's EXIF orientation, and each frame's decode orientation turns the gain first, so a frame tagged differently from its flat gets the gain where the light fell. The Gaussian blur has $\sigma$ = 1/64 of the long side, small enough to follow the light source's own pattern. The blur and the mean cover only the lit area: a normalized convolution masks out pixels below 0.35 of the 95th-percentile luminance, so a dark border or its soft shadow edge does not overcorrect the frame next to it. Pixels too far from the lit area take the nearest valid value. A profile also stores a 512 px copy of the reference (float16) for its self-check: Check Flat Field shows that copy times the gain, each channel over its lit-area median, ±10% to black..white, or 1/gain over ±15% without the copy. The gain is **baked once** into a profile (an `.npz` in `APP_CONFIG.flatfield_dir`, keyed by an opaque id) and kept until the profile is rebuilt from its reference. The edit stores only the profile id. The render resolves the gain through a provider (`set_gain_provider`, wired to `services/assets/flatfield.py`) and caches it; the profile id and a gain content token fold into the source hash.
    *   **Embedded lens correction** (`negpy.features.lens`): the scanning camera's own distortion and lateral CA profile (DNG `WarpRectilinear`, Sony ARW, and a radial distortion polynomial for Panasonic RW2, with no lateral CA, after darktable's reader), applied at decode after flat-field and before the unmix, so neither engine needs a shader. It is skipped for composites and RGB+IR sources. A correction that reads past the frame edge is scaled about its center to fill (`fill_scale` ≤ 1); a CA-only correction is not scaled.
    *   **Sensor crosstalk unmix** (`sensor_matrix`, `features/process/sensor.py`): in single-shot narrowband camera scans the CFA passbands overlap the light's bands, so each channel leaks into the others. It is calibrated from three bare-light exposures (columns normalized to a unit diagonal), inverted, and applied as a 3×3 unmix of the **linear** capture, ahead of the log. Where the film passes almost none of a band's light, the subtraction leaves a value below its grain and calibration error, which the log turns into maximum density. `sensor_unmix` picks the handling (default `two_scale`):
        *   `linear` clips at zero.
        *   `density` applies $C = \mathrm{diag}(1/(M b))\,M\,\mathrm{diag}(b)$, the unmix linearized at the film base $b$ (rows sum to 1), to $\ln x - \ln b$. Each channel is held at $10^{-3} b_c$ or above.
        *   `two_scale` is the clipped linear unmix where its own-channel gain $M_{cc}\bar x_c / (M\bar x)_c$ stays below 1.15 $C_{cc}$. Above that it blends by a smoothstep (full at 1.5 $C_{cc}$, one weight per pixel) toward $\mathrm{SF}(M\bar x)\,\exp\!\big(C(\ln x - \ln \bar x)\big)$. $\bar x$ is the capture blurred by $\sigma = $ long edge / 1200. $\mathrm{SF}$ is the smooth maximum $\tfrac12\big(u + f + \sqrt{(u - f)^2 + f^2}\big)$ of each unmixed channel, with $f = 0.15\,M_{cc}\bar x_c$. A channel whose detail $\ln(x_c/\bar x_c)$ sits $\ln 64$ below both others' is a dead photosite and takes their mean.

        $b$ is the median color of the frame-interior cells thinnest in all three channels together. A channel whose 99.5th percentile sits at its maximum is clipped, and both methods fall back to `linear`. A stitch is unmixed once, assembled, and a half-frame after it is sliced, so preview and export read the same base. `unmix_block_reason` refuses the unmix on a transparency and on a camera-WB decode (`linear_raw` off), where a diagonal gain does not commute with the unmix.
    *   **HDR merge** (`features/hdr`): combines a bracket of one frame into a single linear source at decode, next to the triplet merge. It has no shader and no GPU parity surface.

        The **reference** is the longest exposure with a negligible clipped fraction (`choose_reference`), and the merge is computed in its units, so the result stays in $[0, 1]$ (`features/process/logic.py` clips at pipeline entry). The render exposure (`output_scale`, clamped to $\le 1$) comes from `hdr_anchor`, a frame the user nominates; with no anchor it is the bracket's median ratio in log space. The final clip runs after this scale, so an anchor below the reference keeps radiance above the reference's white.

        The merge is cached **unscaled**, with values **above 1.0**, and `apply_render_exposure` scales it on the way out, so an exposure change needs no new decode. The merge key (`hdr_merge_token`) excludes the exposure; the source identity (`hdr_token`) includes it.

        A fresh merge is seeded with a Shadows Density lift (`seed_shadow_density`, $-0.30$ per stop of `shadow_reach_stops`, capped at $-0.70$). It is an edit on the composite, so zero restores the faithful render.

        Exposure ratios are solved from the images (several scanner formats have no exposure tags). `pair_ratio` is the median of per-sample ratios over samples above the noise floor in the shorter frame and unsaturated in the longer, chained outward from the reference between **exposure-adjacent** frames. `exposure_order_key` orders the bracket by clipped fraction, then level, because `level` pins at $1.0$ for every frame clipping more than 1%. Samples combine by **inverse variance** (weight $r^2$). The weight rolls off to zero between $0.90$ and $0.995$ per channel. Output is float32, because the recovered detail sits below the reference's quantization step.

        The merge runs **before** flat-field and the sensor unmix, because a gain map applied first would move the saturation point the weights key on. Every frame decodes on the **reference frame's** white balance; per-frame as-shot multipliers (`use_camera_wb`) make `pair_ratio` solve wrong ratios. The solve (`HdrWorker.run`) decodes one frame at a time with `hdr` cleared and passes the first frame's multipliers to the rest, since the reference is not yet known.
    *   **Composite assembly** (`features/rgbscan`, `features/stitch`): a frame may be built from several files before any of the above. A **Trichrome triplet** takes each output channel from the exposure lit by that band, with green and blue registered to red by phase correlation. The crosstalk unmix is skipped, and every exposure decodes on a neutral white balance. A **stitch composite** decodes each part separately, warps the parts into a shared canvas by a registration fixed at stitch time, and matches per-channel gain across the overlap. It then cuts a minimum-error seam through the overlap and cross-fades over a narrow band at that cut; averaging the whole overlap draws twice any detail the registration could not line up. The IR defect mask follows the same cut. Every part carries its own triplet, and flat-field and unmix apply **per part, never to the composite canvas**.
*   **Log conversion**: the raw signal goes to log space:
    $$E_{log} = \log_{10}(I_{raw})$$
*   **Bounding and polarity**:
    Statistical percentiles detect the usable signal range. The target **White Point** always maps to the **Floor** ($0.0$) and the **Black Point** to the **Ceiling** ($1.0$).
    *   **Negative (Color Negative / B&W Negative)**: raw low signal (dense highlights) maps to Floor ($0.0$). Raw high signal (film base, shadows) maps to Ceiling ($1.0$). Range: 0.01% to 99.99%.
    *   **Transparency** takes none of this metering. See *Transparency transfer* in §3.

    Bounds are sampled on **two independent axes** (`_sample_log_bounds`). A **color** pass reads each channel's floor and ceiling reference, and a **luma** pass sets how far the window reaches past them, so color balance is tunable without compressing the luminance range. The reach is one shared fraction of each channel's own span $s_c$ between its references: $f_c = r_c^f + (h_f + k)\,s_c$ with $h_f = \sum_c (l_c^f - r_c^f) / \sum_c s_c$, and the ceiling alike. A film whose channels differ in density range, a faded negative for example, keeps a gray scale neutral, and a negative Luma Range Clip widens every channel by the same fraction. $k$ is the base's mean-minus-median channel offset over the mean span. Spans of mixed sign take one log reach for all channels. Mono gives zero deviation at any clip.
    *   **Luma Range Clip** (`luma_range_clip`): the *luminance* percentile window. **Positive** values tighten it symmetrically. **Zero** uses outlier-resistant extremes: a block-median prefilter rejects dust and speculars, and a small base clip excludes tiny outlier populations. **Negative** values push the bounds *outward*, leaving lifted blacks and unclipped highlights as headroom.
    *   **Color Clip** (`color_range_clip`): the sampling depth for per-channel color deviation. A larger value samples deeper for a more outlier-resistant balance; a smaller value samples nearer the extremes. The neutral default is `base_color_clip` ($1.0$); the slider spans log-interpolated values either side.

        The **thin end** (film base) uses per-channel percentiles. The **dense end** (scene highlights) reads one shared, chroma-gated pixel set (the lowest-chroma subset of the luma-extreme band, base-anchored), so colored highlights cannot pass as film cast. With no trustworthy neutrals, and always in Transparency, it falls back to per-channel percentiles.
    *   **White and Black Point offsets**: shift the detected bounds without re-running the analysis. Per-layer trims (`white_point_trim_*` / `black_point_trim_*`) add on top of the global offsets (`per_channel_point_offsets`, single source; Transparency negates).
    *   **Roll baseline and locks**: Roll Analysis measures every frame with its own crop, freehand analysis region and crosstalk. A frame whose luma-free color offsets (per channel and bound) lie further than `POOL_OUTLIER_DISTANCE` log density from the median is an outlier. The baseline pools the other frames: luma is the median of their luma-weighted floors and ceilings, color the mean of their offsets. An outlier keeps its own bounds (both averages off), on this run and on every later apply of the same baseline, because its color offset also shifts luma through G.

        `use_luma_average` / `use_color_average` apply the baseline independently per axis. `lock_bounds` exempts a frame from Roll Analysis. A **scene** pools only its members and writes to them alone; Roll Analysis and a picked roll baseline skip scene members.

        A frame saves its measured bounds with the edit and reuses them. At startup, bounds saved under an older `BOUNDS_VERSION` are dropped, except under `lock_bounds`, so each frame meters again. A sidecar records the version and drops older bounds when read. A roll or scene baseline records it too and is flagged, not dropped, until its analysis runs again.

        The analysis also pools each inlier frame's Cast Removal **neutral axis** (raw log, each frame's own unmix and region) by a confidence-weighted median per band and channel (`pool_neutral_axis`). The highlight band pools only when at least half the frames have one. `use_cast_average` (Color Negative only) renders Cast Removal with that pooled axis (`locked_neutral_axis`) on both engines. The R/B-vs-G curve is the pool's; the midtone level moves toward the frame's own by `weight × confidence` (`blend_neutral_axis`). The weight is $1 - \sigma^2_{noise}/\sigma^2_{offset}$: the spread of the frames' midtone offsets from the pooled curve against the meter noise, estimated as half the spread of their shadow-minus-midtone offsets. Below three frames the weight is zero.
*   **Stretch**: every mode stretches each channel independently to $[0, 1]$, which neutralizes the orange mask and base tints. The result is not clamped: the print curve's toe and shoulder roll off tones outside the bounds.
*   **Per-frame metering**: normalization also measures an **exposure anchor** (for Auto Density) and a **textural range** ($P_{10}$ to $P_{90}$ luminance, for Auto Grade). The anchor and textural meters read only **textured cells** of the block-median grid. Sectors of 2×2 blocks of `activity_block` cells vote when their block means span more than `activity_gate_density` ($0.05$ log D; Boyack & Juenger, US 5,724,456). When fewer than `activity_min_fraction` of sectors pass, every cell votes.
*   **Channel unmix / crosstalk** (`crosstalk_strength` / `crosstalk_matrix`): a 3×3 unmix (`.toml` profiles, see docs/CROSSTALK.md) of the raw **negative** log densities *before* bounds analysis and the stretch, where secondary dye absorptions are linear. The light spectrum and CFA passbands are the same linear operator in log density, so a profile belongs to a *scanning setup*, not a film stock. The matrix is blended with identity by strength and row-normalized, so grays are kept. Every meter (bounds, anchor, shadow refs, neutral axis) reads the unmixed film, Roll Analysis included, so a matrix or strength change needs a Roll Analysis re-run.
*   **Density above base**: `density_above_base` is (Dmin, Dmed, Dmax) above the clear base, luminance-weighted: the thin bound and dense bound the frame is normalized with (before the trims; the roll's under Roll Analysis) and the metered median, each minus the base, so Dmax − Dmin is the Negative row's range. The base is the roll's picked one (`rolls.roll_film_base`, raw log space, carried into the render as `film_base`, `base_source` `picked`), else the rebate reader's (`read`); either is unmixed like the meters. The reader: the crop's film-box detector (`film_and_picture`, at `AUTOCROP_DETECT_RES`) on the raw source before geometry, then the flat 8-px blocks (std over mean under 0.03, unclipped) in the film box's outer 15% band, grouped by brightness with a 0.04-decade gap; the largest group of at least five blocks is the base, read once per source by a preview. None without a film box, without such a group, with a base darker than the picture, or on the transfer path. A flat picture region in the band larger than the base is taken for it. The Negative row's hover shows it.
*   **Scan-clip warning**: `scan_clip_fractions` reports the per-channel fraction of source pixels at or above sensor white ($\ge 0.99$ linear). In a negative, clipping collapses film base and shadows to $D=0$; the Analysis panel warns above 1%. The engine attempts no reconstruction.

---

## 3. The Print (Exposure)
**Code**: `negpy.features.exposure`

*   **B&W Negative (panchromatic)**: the normalized signal collapses to a **single density** (its luminance) *before* the curve, so the H&D curve shapes one channel.
*   **Color timing**: subtractive CMY filtration in log space. Targets **Global**, **Shadows** or **Highlights**. Shadow and highlight offsets are weighted by a sigmoid about the midtone, $w_{sh} = \sigma(3 \cdot (v - z))$, with $z$ the midtone zone center (`anchor_target_density`). The Temperature slider, the WB picker and the temperature roll-lock act on the *selected region's* M/Y pair.
*   **The H&D curve**: an **asymmetric toe-linear-shoulder** curve in **density** space. A line of slope $k$ through the exposure pivot is bounded above by the **toe** (into paper black) and below by the **shoulder** (into paper white). Both bounds are independent **softplus** knees. With $v = k \cdot (x_{adj} - x_0)$:
    $$v_1 = D_{min} + \frac{\text{softplus}\big(a_{hl} (v - D_{min})\big)}{a_{hl}} \qquad \text{(shoulder → paper white)}$$
    $$D = D_{max} - \frac{\text{softplus}\big(a_{sh} (D_{max} - v_1)\big)}{a_{sh}} \qquad \text{(toe → paper black)}$$
    *   $D_{min} = 0.06$: paper white. **Paper White Base** (`paper_dmin`) toggles it; off uses $D_{min} = 0$.
    *   $D_{max} = 2.3$: paper D-max, reached directly by the softplus toe.
    *   $a_{sh}, a_{hl}$: knee sharpness, from `toe_sharpness_base` ($6.0$) and `shoulder_sharpness_base` ($3.0$), scaled by `toeshoulder_width_ref` / width.
    *   $k$: per-channel slope, from **Grade**.
    *   $x_{adj}$: input log-exposure after CMY offsets. $x_0$ is the pivot.
*   **Preflash** (`preflash`, 0 to 1, a fraction of the threshold exposure): a uniform exposure added on the straight line, before any shaping: $v \leftarrow v_{th} + \gamma \log_{10}\big(10^{(v - v_{th})/\gamma} + f\big)$. $f$ is the flash, $\gamma = 2.9 / (R/100)$ the paper's density per log exposure at the frame's ISO-R grade, $v_{th}$ the value that prints `preflash_threshold_density` ($0.04$, the ISO R start) above $D_{min}$ (`preflash_params`). The flash is a paper exposure, so the scan's log range does not change it. It follows dodge/burn (the frame's flash and a mask's). Bare paper fogs neutral, at or under the threshold. Highlight hold measures its tone after the flash.
*   **Variable-gamma paper S-curve**: before the bounds, $v \mathrel{+}= \gamma \cdot w \cdot \tanh\big((v - v^{\ast})/w\big)$ (`paper_midtone_gamma` $= 0.05$, `paper_gamma_width` $= 0.6$), centered on the reference tone $v^{\ast}$. **Snap** (`midtone_gamma`) is a user trim on the paper's baseline $\gamma$, with per-layer trims (`midtone_gamma_trim_*`) evaluated per channel (`per_channel_midtone_gamma`, single source).
*   **Grade (ISO-R)**: contrast as an **ISO range (R) value**, default 115, range 50 to 180. R110 is about paper grade 2; higher R is softer. $k = \text{(grade contrast scale)} \cdot \text{range} / (R/100)$ (`grade_contrast_scale` $= 2.9$), clamped to $[2.0, 10.0]$. Edits on the legacy 0 to 5 paper-grade scale migrate with $R = 150 - 20 \cdot G$.
*   **Split Grade** (`shadow_grade` / `highlight_grade`, ISO-R points, negative = harder): the curve rotates about the shadow and highlight zone centers, $v \mathrel{+}= \Delta k_{ch} \cdot w \cdot (v - z_{zone})$, with the Zone Density weights ($z_{sh} = z + 0.75$, $z_{hl} = z - 0.40$, $k = 4$). The points fold into a slope ratio like Grade trims (`split_grade_deltas`, single source), with per-layer trims (`shadow_grade_trim_*` / `highlight_grade_trim_*`). It runs **before** Zone Density as its own block, because sequential blocks stay monotone and shared weights do not.
*   **Per-layer trims (crossover correction)**: trim one dye layer against the shared curve. CMY only *shifts* a layer's curve; **crossover** needs a shape change:
    *   **Grade trim** (`grade_trim_*`, ±30 ISO-R points): folds into the slope like a paper's `channel_gamma`. As $k \propto 1/R$, the trim is the ratio $R/(R+\Delta R)$. The pivot is re-solved per channel, so the layer rotates about the anchor and midtones stay neutral.
    *   **Toe / Shoulder trims** (`toe_trim_*` / `shoulder_trim_*`, ±1 on the global knee), effective values clamped to the slider domain (`per_channel_toe_shoulder`).
    *   **Snap trim** (`midtone_gamma_trim_*`, ±0.5 on the global Snap): a cast only in the midtones.
    *   **Width trims** (`toe_width_trim_*` / `shoulder_width_trim_*`, ±2 on the global widths, effective values clamped to [0.1, 5]): per-layer knee *sharpness* (`per_channel_widths`).
*   **Toe and Shoulder**: independent, slider values scaled by $0.85$ internally, evaluated **per channel** (global value plus layer trim). The slider sets roll-off **height**; the width control sets sharpness:
    *   **Toe** (shadows): lifts the paper-black ceiling, $D_{max,eff} = D_{max} - \text{toe} \cdot 0.90$ (`toe_height`). Negative toe *sharpens* the shadow knee and, with **Paper Black** off, raises the BPC clip point (see Output).
    *   **Shoulder** (highlights): lifts the paper-white floor, $D_{min,eff} = D_{min} + \text{shoulder} \cdot 0.35$ (`shoulder_height`).
    *   **Grade-coupled baseline**: hard grades get snappier toes and compressed shoulders, scaled by the normalized slope (`toe_grade_strength` $\approx 0.058$, a baseline $\Delta D$ of $0.15 \cdot 0.35$; `shoulder_grade_strength` $= 0.12$).
*   **Zone Density (ΔD)**: two achromatic density offsets (`shadow_density` ±1.0, `highlight_density` ±1.0) at full zone weight, with a **mid-sparing** weight centered in the three-quarter and quarter tones: $v \mathrel{+}= \Delta D_{sh} \cdot \sigma\big(k(v - z_{sh})\big) + \Delta D_{hl} \cdot \big(1 - \sigma(k(v - z_{hl}))\big)$, with $z_{sh} = z + 0.75$, $z_{hl} = z - 0.40$, $k = 4$ (`zone_density_*` constants, mirrored as literals in `exposure.wgsl`). It runs before the softplus bounds, so a shadow burn never passes paper black and a highlight bleach never crosses paper white. The chart mirrors the shift (`CharacteristicCurve`).
    *   **Hold Black** (`hold_black`, off): on a lift ($\Delta D_{sh} < 0$), the shadow weight is multiplied by a smoothstep that falls from 1 at $D_{max,eff} - \tau$ to 0 at $D_{max,eff}$ (`zone_black_hold_taper` $\tau = 0.6$, per channel). Tones at or past paper black get no lift, so the black point stays. The taper only falls where the lift is negative, so it steepens the curve and keeps it monotone. A burn is unchanged.
*   **Dodge & Burn** (`negpy.features.local`): masks with a print exposure in **stops** (`LocalMask.stops`, ±2, default 0; positive burns, signed like `vignette_stops`) and a Gaussian feather ($\sigma$ as a fraction of the short side). They rasterize to a per-pixel stop map added to the log-exposure input with the CMY offsets. One stop is $\log_{10}(2)$ scaled by each channel's stretch range (`local_ev_scale`). Vertices are in raw-image coordinates and follow geometry. The Flat intent skips them.
*   **Mask shapes** (`LocalMask.shape`): `shape` says how to read the stored vertices. Every control point goes through `map_coords_to_geometry`.
    *   *Polygon*: N points, closed and Catmull-Rom smoothed (`smooth_polyline`).
    *   *Oval*: 3 points $(c, p_1, p_2)$. Outline $c + u\cos\theta + v\sin\theta$ over 64 samples, $u = p_1 - c$, $v = p_2 - c$; the axes need not be perpendicular. Points map to pixels before the outline is built.
    *   *Gradient*: 2 points $(a, b)$, alpha $= 1 - \text{smoothstep}(t)$ with $t = \big((p-a)\cdot d\big)/|d|^2$ clamped to $[0,1]$, $d = b - a$. Feather does not apply.
    *   *Invert* (`LocalMask.invert`): $\alpha \rightarrow 1 - \alpha$ after the feather.
    *   *Enabled* (`LocalMask.enabled`, default on): `compute_local_maps` skips a disabled mask.

    One function (`local/logic.rasterise`) serves the render, the canvas tint and the printing-notes map. The GPU uses the same CPU-rasterized map (`compute_local_maps` → the dodge/burn texture), so shapes have no parity surface.
*   **Local Grade** (`LocalMask.grade`, ISO-R points off the frame's Grade, negative = harder): `compute_local_maps` rasterizes a second plane (plane 0 EV, plane 1 summed $\Delta R$) into a per-pixel slope multiplier $R/(R+\Delta R)$, clamped to the ISO-R ladder (`local_grade_factor_map`; `exposure.wgsl` mirrors it). It multiplies the straight-line slope only, $v = k \cdot g \cdot (x_{adj} - x_0) + c \cdot x_{adj}^2$, rotating **about the channel pivot**, the same in all three channels; the cast-removal curvature $c$ stays global. On the GPU the $\Delta R$ plane rides the dodge/burn texture's blue channel and the shader applies the frame's grade, so a Grade change uploads no map.
*   **Local Flash** (`LocalMask.flash`, 0 to 1): the mask's own preflash. `compute_local_maps` pre-sums it into plane 2 as $\sum f_i \alpha_i$, and the preflash formula runs with the frame's $f$ plus that sum, with the frame's $v_{th}$ and $\gamma$. On the GPU the plane rides the dodge/burn texture's green channel. A tone-limited mask carries no flash.
*   **Tone limit** (`LocalMask.key` *Highlights* / *Shadows*, `key_zone` and `key_softness` in print zones; Kodak O-3): a limited mask's weight is $\alpha \cdot s(t)$, $t = (L - e_0)/(e_1 - e_0)$ clamped, $s = t^2(3-2t)$ (`tone_key_weight`). $L$ is the luma of the **unburned** normalized log, so no mask can move the pixels it selects. The edges invert the achromatic print curve at $Z \mp S/2$ (`key_edges`, the inverse of `predicted_zone`) from the render's own metrics. The first four limited masks (`MAX_KEYED_MASKS`) keep their own shape planes after planes 0–1 (on the GPU, one texture); a fifth prints unlimited. Limited grades add to plane 1's $\Delta R$ before the one clamp to the ISO-R ladder.
*   **Contrast Mask** (`ExposureConfig.contrast_mask`, ±0.5, default 0): a blurred low-gamma film mask sandwiched with the negative, $D' = D - g\,\text{blur}(D) + \text{const}$, a linear high-boost in log density (Ctein, *Post Exposure*; Bond, *Unsharp Masking*; Adams, *The Print*).

    $g$ is the signed mask gamma. Positive scales the low-frequency range by $(1-g)$; negative by $(1+g)$. Detail above the blur scale passes at unity, and the mask still works when a flat negative sits on the 2.0 slope floor.

    It is not a stage. `contrast_mask_plane` builds a zero-mean plane that becomes stops of print exposure, read with the dodge/burn map, equal in every channel. Both engines call the same helper on the same pre-geometry array, with $\sigma$ a fraction of the analysis grid. The slider is a scalar on the plane (`contrast_mask_scale`) and rebuilds nothing: the CPU caches the plane at render size (`expand_mask_plane`); the GPU keeps it on the analysis grid in its own texture and upscales in the shader.

    **Mask Spacer** (`ExposureConfig.mask_spacer`, 2 to 6%, default 4) is the blur $\sigma$ as a percent of the analysis grid. It rebuilds the plane, so it keys both engines' plane caches, including the single plane a tiled export shares across tiles.

    The plane covers the printed frame only (a blurred rebate would print as a vignette), placed back at the crop and edge-replicated outside so the crop tool's full-frame preview has no seam. The transfer path takes no dodge/burn map. Instruments read the unmasked negative.
*   **Diffusion** (`ExposureConfig.diffusion`, 0 to 1, default 0): a diffuser at the enlarger lens with a fixed spread: $\sigma$ is `DIFFUSION_SPREAD`, 2% of the printed frame's short side. The paper sees $I' = (1-d)\,I + d\,\text{blur}_\sigma(I)$, a mix of light, not of density. Per channel, $v \leftarrow \log_{10}\big((1-d)\,10^{vr} + d\,10^{pr}\big)/r$, with $r$ the channel's log range (`channel_density_ranges`), $p$ the blurred frame in the same normalization, and $d$ half the slider value (`diffusion_mix`). `diffusion_grid` blurs the unmixed light on the analysis grid (`analysis_grid`, the Contrast Mask's own). The normalization is a per-channel gain on that light, so `diffusion_plane` only applies the print's final bounds to the grid. It runs on the normalized value before filtration and dodge/burn, which commute with it. Both engines sample the one grid plane bilinearly with clamped taps (`plane_taps`, `exposure.wgsl`); a B&W print mixes a luma plane. On the GPU it rides the Contrast Mask texture's other three channels. The crop and the geometry rebuild the grid; a White/Black Point change only re-normalizes it. Print curve only: off on the transfer path, in the Flat master and in the carrier strip.
*   **Output**: print density back to **scene-linear** reflectance (transmittance):
    $$I_{out} = 10^{-D}$$
    *   **Paper Black** (`paper_black`, off): off applies black point compensation, as in ICC relative-colorimetric soft-proofing. On keeps the paper's lifted D-max ($10^{-2.3} \approx 0.005$). With compensation each channel is $I_{out} = (I - t_b) / (1 - t_b)$, clamped at $0$, $t_b = 10^{-D_b}$, where $D_b$ is the physical $D_{max}$, or $D_{max} + \text{toe}_{ch} \cdot 0.90$ when that layer's toe is negative. A **negative toe raises the clip point**, which makes exact $0$ reachable. The reference is the *physical* $D_{max}$, not $D_{max,eff}$, so a lifted toe and per-layer shadow casts survive.
    *   **Scene-linear internally**: the exposure stage emits linear light, and every creative stage (Local Contrast, Retouch, Lab, Toning, Finish) works on it. The working-space OETF (the **Adobe RGB (1998) TRC**, a pure $563/256 \approx 2.199$ power, no linear segment) is the final engine step. Retouch is perceptual: the CPU brackets it through the OETF (encode → heal → decode); the GPU keeps one encoded region (exposure → clahe/retouch encoded → lab decodes back to linear).
    *   **The soft proof is a color table, not a stage.** `soft_proof_preview` runs on a $65^3$ identity grid inside `soft_proof_lut`, uploaded as the 3D texture the canvas shader samples; the CPU display path uses `apply_lut_f32`. Rendering intent, black point compensation, paper white, ink black and the gamut warning cost one cached rebuild on change, with no shader and no parity surface. `ProofCondition` holds them as one hashable value that keys the table and the render memo. Paper white is the absolute intent; ink black drops black point compensation; the gamut warning sets unprintable grid nodes to flat gray.
    *   **Printability**: both engines bin encoded content into a joint $32^3$ RGB histogram (`color_hist.wgsl` and `analysis.color_histogram`, mirrored constant `COLOR_HIST_BINS`). The CPU builds the gamut mask from the proofed output profile (`ImageProcessor.gamut_lut`, cached per profile pair) by round-tripping the identity grid source → output → source. The tolerance sits above the transform's 8-bit noise, so the read-out under-reports at the boundary.
    *   **Waveform**: both engines bin encoded luma into 128 tone bins for each of 256 image columns (`waveform.wgsl` and `analysis.waveform`, mirrored constants `WAVEFORM_BINS` and `WAVEFORM_COLS`), over the content the RGBL histogram reads. The same pass sums each cell's encoded R, G and B in fixed point (`analysis.waveform_color`, mirrored `WAVEFORM_COLOR_SCALE`) for the cell's mean color, and bins each encoded channel on its own for the RGB parade (`analysis.waveform_parade`). The density histogram pass also bins each channel's scan density per crop column for the scan Parade (`density_hist.wgsl` and `analysis.density_parade`, on `SCAN_DENSITY_RANGE`), inverting the stretch with the final bounds, and counts each column's linear pixels at `scan_clip_level` before the unmix (`analysis.scan_full_scale_columns`). The film base line is `normalization.base_scan_density`. A peek renders nothing, so it reads the last print render.
    *   **Vectorscope**: both engines bin the clamped linear content's CIELAB a\*b\* into 180 hue × 64 chroma bins, chroma on a log axis with a knee at 4 (`vectorscope.wgsl` and `analysis.vectorscope_histogram`, mirrored `SCOPE_*` constants). The last chroma bin holds everything past chroma 60.
    *   **Input ICC overrides primaries only, never the tone curve.** The rendered buffer's TRC is always the working-space OETF. For a **matrix/TRC** profile (`infrastructure/display/icc_profile.py`) the boundary transform takes only its primaries (Bradford-adapted to D65, through its `chad` tag when present) as a plain matrix; the declared TRC is inert, so `sRGB-*-g10.icc` and `sRGB-*-srgbtrc.icc` render **identically**. A **LUT** (A2B0/B2A0) profile runs through the full CMS transform, so its input curves apply.

### Automatic helpers

*   **Auto Density** (`auto_exposure`, **on**): the metered tone $m$ is the average of the trimmed window's mean and midpoint ($P_5$ to $P_{95}$ of the textured cells; Boyack & Juenger, US 5,724,456). The anchor is a partial pull from the assumed key (`assumed_anchor` $= 0.46$):
    $$\text{anchor} = \text{assumed} + s \cdot (m - \text{assumed}), \quad \text{clamped to } \pm b$$
    with $s =$ `anchor_meter_strength` ($0.2$) and $b =$ `anchor_meter_band` ($0.12$). The anchor prints at `anchor_target_density` ($0.75$).
*   **Auto Grade** (`auto_normalize_contrast`, **on**): picks the grade from the textural density range $t$ ($P_{10}$ to $P_{90}$ of the textured cells, log D), a partial ISO R match (Alkofer, US 4,731,671). With $r$ the floor-to-ceil range, the effective contrast range is:
    $$K \cdot r \cdot \min\!\Big((1 - s) + s \cdot \frac{n}{t},\; c \cdot \frac{n}{t}\Big)$$
    with $K =$ `auto_grade_target` ($0.85$), $n =$ `auto_grade_nominal_range` ($0.9$), $s =$ `auto_grade_strength` ($0.4$) and $c =$ `auto_grade_max_overfill` ($1.2$).

    **Shadow reach** (Gindele, US 7,113,649): the darkest textured tone ($P_{99}$, `shadow_reach_percentile`) must print at least `shadow_reach_density` ($1.9$, straight-line) with the anchor at its target. If the grade falls short, the slope is raised to the line through both points, never lowered.

    **Highlight hold** (Agfa, US 4,104,069 / US 3,839,036): the brightest textured tone ($P_2$, `highlight_hold_percentile`) must print at least `highlight_hold_density` ($0.10$; $0$ turns it off). If it would print brighter, an automatic highlight burn joins the Zone Density highlight term, solved against that term's weight at the tone, capped at `highlight_hold_max` ($0.5$). It never lifts. Highlights Grade is not the actuator, because it pivots at the highlight zone center, where these tones sit.

    **Shadow hold**: a partial lift of Shadow Reach's tail ($P_{99}$). With $v$ the tail's straight-line value and $v_r$ the value that prints `shadow_reach_density`, the Zone Density shadow term gets $-\min\big(h \cdot (v - v_r - o),\; m\big)$ when $v > v_r + o$, with $h =$ `shadow_hold_strength` ($0.40$; $0$ turns it off), $o =$ `shadow_hold_overshoot` ($0.15$) and $m =$ `shadow_hold_max` ($0.3$). It never burns.
*   **On the sliders** (`features/exposure/auto_sliders.py`, transfer twin in `features/transparency/logic.py`): the stored values are offsets on the meter. The Tone panel shows their manual equivalent and converts an edit back: Density shifts by $(\text{anchor} - \text{assumed})/$`density_multiplier`, Grade is the ISO R that prints the auto slope on the frame's own range (`slope_to_grade`), and Shadows/Highlights Density add the holds. The meters come from the frame's last plain render (`desktop/auto_sliders.py`).
*   **Set Targets** (app-global): tunes eight aim values (`anchor_target_density`, `anchor_meter_strength`, `anchor_meter_band`, `auto_grade_target`, `auto_grade_strength`, `shadow_reach_density`, `shadow_hold_strength`, `highlight_hold_density`; `TUNABLE_TARGETS` in `features/exposure/models.py` declares ranges). `apply_targets()` overlays them onto `EXPOSURE_CONSTANTS`, persisted in the `exposure_targets` global setting. No `WorkspaceConfig` hash sees them, so both engines fold a `TARGETS_REVISION` counter into their cache keys: the CPU base stage, and on the GPU the analysis cache key plus the exposure-stage diff. A `TUNABLE_TARGETS` entry read *outside* those stages needs its own invalidation.
*   **Cast Removal** (`cast_removal_strength`, default $1.0$ on a negative, $0$ on a slide; $0$ is off): balances each layer against the frame's own grays, from shadows to highlights. Applied strength is `confidence × slider` (`effective_cast_strength`).

    It runs on both **color** processes. `cast_removal_for_mode` (`features/process/models.py`) is the one place the per-mode default swap lives: every route into a mode passes through it, and it rewrites only the other mode's default, so a user-chosen strength survives a mode switch. It is a roll default (`cast_removal` in `ROLL_DEFAULT_FIELDS`). `services/assets/migrations/cast_removal.py` sweeps legacy slides once at startup (a row rewrite guarded by a done flag, because a load cannot tell a legacy row from a fresh save).

    The **neutral axis** meter reads the space the curve consumes: the unmixed film log on the print curve, the working-space log after the camera matrix on the transparency curve, against the fixed transfer window (`processor.py`, `needs_axis` in `gpu_engine.py`). With `use_cast_average` on, the roll or scene axis replaces this meter (§2).

    The primary solve fits each non-green channel to green's **neutral axis**, from references at a highlight, midtone and shadow luma band, each over the band's lowest-chroma pixels (`neutral_axis_*` constants). On the print curve, foliage, sky and water never enter a set: a cell whose offset from the stretched neutral is above `neutral_axis_foliage_chroma` ($0.03$) at a hue from green to yellow ($-25°$ to $80°$ of $\operatorname{atan2}(-s, t)$, where $t$ and $s$ are the mean and half-difference of the R−G and B−G offsets), or above `neutral_axis_sky_chroma` ($0.06$) at a hue from azure to blue ($-150°$ to $-60°$), is dropped first. In a frame with no grays the lowest-chroma pixels are desaturated leaves, or the pale sky near the horizon, and neutralizing them prints magenta or yellow. The transparency curve meters against a fixed window, whose neutral is not a gray, so it keeps every cell. A reference is green's median plus the set's median R−G and B−G offsets. The confidence falls as the interdecile width of those offsets passes `neutral_axis_width_deadzone`. R and B get a **quadratic** through the three points, with the midtone pinned through the pivot. Deviation from green is clamped ($\pm 0.2$, `midtone_cast_max_offset`), and curvature is bounded (`neutral_axis_curv_max_ratio` $= 0.45$ of the slope) to keep each channel monotonic on $[0,1]$. Outside that range the quadratic is held at its vertex (`quadratic_core`), so a value far past the frame's bounds, such as the Filed Carrier's rebate ramp, never folds back.

    A frame with no neutral axis (too few trustworthy near-neutrals) gets no correction.

    The **transparency curve** has no per-channel slope, so it uses a per-channel **affine on density** (`neutral_axis_affine`): a gain and offset that land each channel's midtone and shadow refs on green's, applied before every other control on that curve. Green, a missing axis, zero strength and a degenerate fit are the identity. The same $\pm 0.2$ clamp applies, the gain is bounded (`cast_affine_gain_limit` $= 2.0$ and its reciprocal), and there is no curvature term.

### Print stocks
**Code**: `negpy.features.exposure.papers`

A **print stock** (`paper_profile`, default *Digital*) sets the H&D curve shape without touching contrast or exposure: $D_{max}/D_{min}$, toe and shoulder sharpness and height, and midtone gamma. Color stocks add a per-channel slope crossover (`channel_gamma`), a paper-base tint (`base_tint_cmy`, added to the minimum-density floor) and their **dyes** (`print_color`). *Digital* and the B&W papers have no dyes (an identity `DyeMatrix`), so *Digital* reproduces the defaults exactly. A color stock's dyes are a 3D LUT (`DyeLut`, `negpy/features/exposure/print_luts/<key>.npz`, $33^3$): it maps the dye densities above base $e = D - D_{min}$ to the print's color in linear Adobe RGB, media-relative, with tetrahedral interpolation, so $T = 10^{-D_{min}} \cdot \mathrm{LUT}(e)$ in place of $10^{-D}$. The LUT follows how each dye changes hue and saturation with its amount and how real dyes mix, which a matrix cannot. On its gray axis the LUT keeps the curve's luminance, $Y(\mathrm{LUT}(e, e, e)) = 10^{-e}$, so the tone tools stay true and grays drift only in chroma, as the stock's dyes do. Black point compensation then maps the luminance of the stock's black to 0 with one scalar for all channels, so a tinted black keeps its tint. `scripts/build_print_luts.py` bakes the LUTs. Endura Premier and Ultra Endura (E-4070), Portra and Supra Endura (E-4021), 2383 (H-1-2383) and 2393 (H-1-2393t) come from the datasheets' spectral dye densities, in CIE 1931 2° colorimetry under D50 for paper and a Kinoton 75P xenon projector for print film, Bradford-adapted to D65. The E-4021 and 2393 curves are spektrafilm's traces (CC BY-SA 4.0, `NOTICE.md`). The three Crystal Archive stocks come from Fujifilm's Frontier-PD printer profiles: the Frontier exposes each layer with its own laser, so the profile's gray axis gives one curve per dye, and the LUT reads the measured print colors through them. *Digital (Pakon Minilab)* keeps Digital's curve; its LUT is the Pakon profile `tests/fixtures/RGBScan.icc` applied to a Digital print, exact at the defaults.

Profiles are **mode-aware**. Color Negative offers the RA4 papers. B&W Negative offers the B&W papers, which carry no RA4 color terms but carry `lith_path`, which the Lith stage reads. Transparency gets only *Digital*. An incompatible stored value collapses to *Digital*. Bundled: **Digital (Adobe RGB)**, **Digital (Pakon Minilab)** (Color Negative); *B&W*: Ilford Multigrade RC, Ilford Multigrade FB Classic, Foma Fomatone, Foma Fomabrom, Kodak Polymax II RC, Kodak Polycontrast IV RC; *RA4*: Kodak Endura Premier, Fujicolor Crystal Archive, Fujicolor Crystal Archive HD, Kodak Portra Endura, Kodak Supra Endura, Kodak Ultra Endura; *print film*: Kodak Vision 2383, Kodak Vision Premier 2393. The Kodak Polymax, Polycontrast, Portra, Supra and Ultra profiles are fitted to their datasheet curves (`scripts/fit_paper_profile.py`) at the slope the Grade gives each curve's ISO R. Crystal Archive, whose sheets publish no curve, takes its knees from the same fit to koraks.nl's measured grey wedge and its midtone from Endura Premier; its $D_{min}$, base tint and $D_{max}$ come from the Frontier-PD profile, as do those of Crystal Archive HD, which keeps its curve. 2383 and 2393 are fitted at the Grade of their steepest gamma, with their layers aligned at mid density and their projection $D_{max}$ (4 to 5.5) scaled into the paper range, because black-point compensation maps the profile's $D_{max}$ to display black. The others are loosely mapped from datasheets; mainly $D_{max}$ is grounded.

### Dye Separation
**Code**: `negpy.features.exposure.papers.resolve_saturation_matrix` / `compose_density_matrices`

Saturation in density, in the same matrix slot as a matrix stock's `DyeMatrix`. For density above paper base $e = D - D_{min}$, $M(k) = k \cdot I + (1-k) \cdot J$, where $J$ has every entry $1/3$: $k=1$ is identity, $k=0$ collapses the row to the achromatic mean, $k>1$ boosts separation. $M_{total} = M(k) \cdot M_{dye}$: saturation is outermost, so a matrix's crosstalk does not reabsorb it. On a LUT stock the slot holds $M(k)$ alone, before the LUT: separation changes the dye amounts, as a more saturated negative would, so the same $k$ gives more chroma on a stock with cleaner dyes. $k$ (`dye_separation`, default 1.0, slider 0.5 to 1.5) takes per-channel trims (`dye_separation_trim_red/green/blue`). Each row sums to 1, so neutrals stay flat. $k = 1$ on every channel returns `None` from `resolve_saturation_matrix`, so the default path stays byte-exact. Real negatives land around $k = 0.8$ to $1.4$.

### Separation Damping
**Code**: `negpy.features.exposure.logic.separation_damping_gain`

**Separation Damping** (`separation_damping`, 0 to 1, default 0) makes $k$ depend on the pixel's existing color. With $c$ the spread of $e$ across channels (`_rms_chroma`'s hue-symmetric measure) and $\bar e$ its mean:

$$h(c) = \frac{c_0 - c}{c_0 + c}, \qquad k_{eff,ch} = k_{ch}^{\,(1-\tau)\,+\,\tau\,h(c)}, \qquad e'_{ch} = \bar e + k_{eff,ch}\,(e_{ch} - \bar e)$$

At $\tau = 0$ this is the matrix above, byte-exact. At $\tau = 1$ the gain is $k$ on a near-neutral, 1 at the reference spread $c_0$ (`separation_damping_ref_spread` $= 0.35$, also a WGSL literal) and $1/k$ far above it. Per-pixel $k$ cannot fold into a matrix, so with damping live the `dye_mix` slot carries only the paper's coupling, and the separation runs after it in the kernel.

It is inert at $k = 1$. It stays monotone in $c$ for $k < e^2$, which covers the clamped $[0, 3]$ domain of $k$ and $k_{eff}$. Lateral dye and inhibitor spread is not modeled.

### Hue Trim (light-source hue rotation)
**Code**: `negpy.features.process.hue.apply_hue_trim`, mirrored in `exposure.wgsl`

A rotation of CIELAB hue:

$$\begin{pmatrix} a' \\ b' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix}$$

with $L^{\ast}$ untouched. $\theta$ is `process.hue_trim` in degrees (0 = off, ±30 range). Neutrals do not move.

It runs on the **scene-linear print**, after the H&D curve and before the working OETF, inside the exposure stage (a drag does not re-run normalization). It stays active under `RenderIntent.FLAT`. The GPU mirror inlines `rgb_to_lab` and `lab_to_rgb`, copied verbatim from `lab.wgsl` (WGSL has no includes), and rotates `transmittance` before `oetf_encode`. $\theta$ rides the `hue` vec4's x lane in `ExposureUniforms`; its yzw lanes carry the preflash fraction, threshold value and paper gamma.

### Flat (log) master, "for editing elsewhere"
**Code**: `negpy.features.exposure.processor.PhotometricProcessor._process_flat` → `apply_flat_curve`

With render intent **Flat** (`RenderIntent.FLAT`), a **true log encoding** replaces the Print stage. The H&D curve does not run. The normalized signal $\text{val} \in [0,1]$ from §2 is already log; the flat master skips the $10^{-D}$ decode and emits it positive-oriented:

$$I_{out} = \text{clip}\big(\text{lift} + \text{gain} \cdot (1 - \text{val}),\ 0,\ 1\big)$$

*   `flat_log_gain` $= 0.65$: contrast.
*   `flat_log_lift` $= 0.10$: where the scene **shadow** lands.
*   Result: shadow → $0.10$, mid-gray → $\approx 0.46$, highlight → $0.75$.

Both are fixed. Manual white balance still applies as a per-channel log shift. Only Geometry → Normalization → this log map → Crop run: no creative stages, dodge/burn masks or defect repair. Light leak reduction still bakes in: a leak is a fault of the film, not a print decision. Export is full-resolution, color-managed at encode into the selected color space, and always 16-bit: TIFF at the chosen compression, or lossless JPEG XL if JXL is selected and the color space is JXL-taggable. The CPU engine is forced (there is no GPU flat shader).

### Transparency transfer, "the slide as captured"
**Code**: `negpy.features.transparency` (+ `shaders/transfer.wgsl`), `negpy.features.process.capture_color`

For **Transparency** a **transfer curve** replaces the Print stage. It is exactly the identity at default settings. **Positive** (a Slide-only setting) takes the same curve. `process.path.render_path` makes the choice (`TRANSFER`, `POSITIVE`, or `PRINT` for every negative), and the engine routes the base and exposure stages to `features/transparency` on it; the negative's processors hold no slide branch. A Flat master of a slide takes the same base, then the flat log map.

**Normalization** (`TransparencyBaseProcessor`) does no metering to shape the window; on a Positive frame it meters for Auto Density/Auto Grade.

*   **Neutral decode.** This path always decodes without the as-shot white balance, whatever Linear RAW says (`effective_linear_raw`), because the camera matrix folds the multipliers back in. Decode, CPU and GPU matrix and every cache token ask that one helper, or the raw green-to-red ratio tints the render. The same helper makes the loader read a TIFF as literal linear data whatever its profile. **Positive** is exempt (`TiffLoader.load`'s `positive_source` parameter): its embedded profile decides the decode (untagged falls back to sRGB).
*   **Highlight reconstruction** (`highlight_reconstruction` on `ProcessConfig`, `effective_highlight_reconstruction`) sets libraw's `highlight_mode`: `0` Clip (default), `2` Blend, or Reconstruct `3`-`9` (the sidebar's Reconstruct is level `5`). It applies to either Transparency render and is forced to `0` off Transparency, under Narrowband and on a merged bracket, whose clip-detection weighting (`features/hdr/logic.py`) needs pixels near the sensor ceiling. `WorkspaceConfig.__post_init__` holds that invariant once `hdr` names the bracket; `HdrWorker.run`'s solve zeroes it per frame, since it runs before the merge exists. On this path, active reconstruction bakes the real white balance into the decode and skips the matrix fold (`highlight_reconstruction_bakes_wb`, `should_fold_camera_wb`), because libraw finds clipping from its own per-channel gains. A non-Clip mode normalizes libraw's gains against the widest multiplier, which darkens the decode; `highlight_reconstruction_bright_gain` passes that ratio back as libraw's `bright` scale wherever a real white balance (baked or native) meets a non-Clip mode.
*   **Camera matrix.** RAW is decoded `output_color=raw`. The decoder's XYZ→camera matrix (libraw `rgb_xyz_matrix`, on `PipelineContext.cam_xyz`) becomes the *working→camera* matrix, is **row-normalized there**, then inverted (dcraw's `cam_xyz_coeff` order); normalizing after inversion gives a magenta cast that grows with saturation. It is applied in **linear**, before the log. A source without a matrix (scanner TIFF, JPEG) passes through. The WB fold asks `should_fold_camera_wb`. An explicit Input ICC swaps `cam_xyz` for `wb_only_cam_xyz` (`AppController._effective_cam_xyz`), so only the white-balance fold remains: `_XYZ_TO_WORKING` through the same forward/normalize/invert steps gives exactly identity, which the soft-proof and export ICC bypass also assume.
*   **Fixed bounds.** The stretch is a constant window: floor at the decoder's white level ($D = 0$), ceiling `TRANSFER_DENSITY_RANGE` $= 3.0$ decades below, the same for every frame. White and Black Point (`per_channel_point_offsets`, shared with the measured path) shift the window additively. The dye-crosstalk unmix is skipped.

**Print** (`TransferProcessor` → `apply_transfer_curve`) inverts the window and deviates only by what the user moved. With $n$ the normalized signal and $R$ the density range:

$$D = R \cdot n - \Delta_{exp} + R \cdot \text{cmy} \qquad D \leftarrow p + (D - p) \cdot c$$
$$D \mathrel{+}= \text{cmy}_{sh} \cdot w_{sh} + \text{cmy}_{hl} \cdot (1 - w_{sh}) \qquad w_{sh} = \sigma\big(k_{wb}(D - z_{wb})\big)$$
$$D \leftarrow D - t \cdot \text{softplus}(D - k_t,\ w_t) + s \cdot \text{softplus}(k_s - D,\ w_s) \qquad I_{out} = \text{ODT}\big(2^{b} \cdot 10^{-D}\big)$$

At neutral settings $\Delta_{exp} = 0$, $c = 1$, $t = s = 0$, so $D = R \cdot n$ and $10^{-Rn}$ inverts the normalization exactly in float32. The toe compresses *above* its knee and the shoulder *below* its own; `softplus` is monotonic for $t, s \in [-1, 1]$.

**Display rendering.** $\text{ODT}$ is applied last: a fixed baseline exposure $b$ = `transfer_baseline_ev` = 0.7 EV (darktable's raw default), then Narkowicz's closed-form fit to the ACES RRT plus sRGB ODT. The decode anchors to the **sensor white level** (`no_auto_bright`, `adjust_maximum_thr=0`), so a frame exposed below clipping arrives dark without it. Both terms are fixed.

**Positive** (`ProcessConfig.positive_source`) skips both. `apply_transfer_curve` returns the scene stage's output clipped to display range, and the Print sliders still act. The GPU packs `baseline_gain` as `1.0` and carries the skip in the `zone_taper.y` lane.

The controls are the Print sliders, each neutral at its default:
*   **Print Density** → $\Delta_{exp}$ (stops, `transfer_density_stops` $= 2.0$ per unit, higher is darker).
*   **ISO-R Grade** → $c$ (`transfer_grade_ref` $/$ grade, so the shipped $100$ is unity).
*   **Toe** / **Shoulder** → $t, s$, with knees $k_t = 1.6$ and $k_s = 0.35$ density. Their **Width** sliders → $w_t, w_s$, scaled about `transfer_width_ref` $= 2.5$.
*   White balance → per-channel density offsets through `filtration_offsets`. Its **Shadows / Highlights** split → the Regional CMY term (`wb_split_geometry`).
*   **Shadows / Highlights Density** → the Zone Density term.

The last two apply after contrast and before the knees. R/G/B trims go through `per_channel_toe_shoulder` and `per_channel_widths`.

**Auto Density/Auto Grade** (`transfer_auto_terms`) meter as on a negative, restated on this curve. `auto_meter_for_mode` starts them off on Slide, rewriting only the setting still at the mode being left (as `cast_removal_for_mode` does); a live merge holds both off (`WorkspaceConfig.__post_init__`). Auto Density places the anchor at `transfer_contrast_pivot`, the mid-gray density Grade rotates about (`transfer_assumed_anchor` is that reference's fraction of the fixed window). Auto Grade uses `effective_grade_range`/`grade_to_slope` unchanged. Shadow Reach, Highlight Hold and Shadow Hold act as on a negative, the holds through the Zone Density kernel. `shadow_reach_density`/`highlight_hold_density` map onto this curve by tonal position.

Zone Density's geometry comes from the print path's (`zone_geometry()`) by **tonal position**, because the scales differ (print $d_{min}$ 0.06 to $d_{max}$ 2.3, here 0 to `TRANSFER_DENSITY_RANGE`): the shadow center 1.50 re-centers to 1.93, with sharpness scaled likewise. `wb_split_geometry()` derives the Regional CMY center $z_{wb}$ and sharpness $k_{wb}$ the same way, from `anchor_target_density` and the print's fixed sharpness of 3.0.

The shadow lift is **tapered to nothing at the bottom of the window** (`_black_taper`, smoothstep over `ZONE_BLACK_TAPER` = 1.0 density), because no paper black stops it lifting the black point. Split Grade is not mirrored: the transfer curve has no per-zone slope.

> `transfer_grade_ref` and `transfer_width_ref` **mirror** `DEFAULT_WORKSPACE_CONFIG`. If they drift apart, the default render stops being the identity. `test_transparency_transfer.py` asserts they agree.

The paper model cannot be the identity (`d_max` floors the blacks, `anchor_target_density` places mid-gray, the midtone snap and paper-white reference stay live at neutral), so its controls and the normalization tuning do not apply here. Downstream stages run as usual.

Dye Separation and Separation Damping stay. Each channel's $k$ (`dye_separation` + `dye_separation_trim_red/green/blue`) applies to density directly, $D'_{ch} = \bar D + k_{eff,ch}(D_{ch} - \bar D)$. $\bar D$ is the mean of the channels, each softly capped (softplus) at the higher of 2.5 D and the pixel's lightest channel plus 1.5 D, so a channel far denser than the rest near the dense end of the window (mostly noise) does not spread onto the other two. The cap is set on the capture density, where the window clamp always sits at 3 D, so Print Density, Grade, WB and the knees cannot move a clamped channel back under it. The cap only rises with its inputs, so $\bar D$ never falls as a channel darkens. Separation Damping applies §3's $k_{eff}(c)$ law to each channel's $k$ from the shared chroma $c$, measured on the capped channels.

**Capture settings.** `linear_raw` and `sensor_matrix` are sticky global settings, each *inert* where it does not apply.

Linear RAW decodes with `user_wb=[1,1,1,1]`. `camera_to_working_matrix` folds the as-shot multipliers back in, normalized to green so exposure does not move, which makes the render independent of the decode with no shader change (the folded matrix rides the existing `cam*` rows). Linear RAW is inert on this path only; with Positive Source on, it decides the decode again.

`unmix_block_reason` refuses the sensor matrix for E-6 on every path: it describes a narrowband light, which a slide is not scanned under.

The GPU mirror is a separate shader, not a branch in `exposure.wgsl`. Like that shader it emits **display-encoded** values (`oetf_encode` at the end of the pass); the CPU stays linear to the end of the engine.

### Linear Output
**Code**: `negpy.services.export.linear_output`

With render intent **Linear** the darkroom pipeline is bypassed. The source decodes to its native linear buffer, lossless geometry (EXIF orientation plus user rotation and flip) is applied, and the output is a 16-bit file: TIFF (default, at the Export panel's TIFF compression) or standalone lossless JPEG XL. An Effort slider (1 to 9, default 7) trades JPEG XL encoder speed for compression. No normalization, exposure or color management runs, and a correction runs only when its toggle is on. The IR sidecar, when present and not consumed by ICE, uses the same format; it is single-channel grayscale, and on the JXL gray path `transfer=LINEAR` is its only tag.

**Only the TIFF path is untagged.** A JPEG XL codestream always carries a `ColorEncoding`, even with `photometric`, `primaries` and `transfer` unset (true of `imagecodecs.jpegxl_encode()` and `cjxl`). `_write_jxl()` pins `transfer=LINEAR` but leaves `primaries` at the sRGB default, so a spec-compliant JXL viewer applies a wrong sRGB→display transform to camera or scanner-native primaries. `imagecodecs` cannot pass `PRIMARIES.CUSTOM` chromaticities or an ICC profile (`jpegxl_encode()` has no `iccprofile=`, though libjxl's `JxlEncoderSetICCProfile` has one). NegPy re-imports the file correctly, because `JxlLoader` reads no color metadata (`imagecodecs.jpegxl_decode()` surfaces none).

*   **Pakon RAW**: uint16 scanner data is scaled by an expansion factor. F135 (14-bit sensor) defaults to 4× (`PAKON_EXPANSION`); F335 (16-bit, detected by file size) defaults to 1×, off. The 2k Square and Panoram specs are assumed 14-bit (not verified). 4× puts a typical F135 negative peak around 50 to 55% of range. The Expansion combo overrides it, and the TIFF's ImageDescription tag records the factor.
*   **LinearRaw DNG**: 4-channel VueScan (RGB+IR) and 3-channel SilverFast HDRi files are read through tifffile, bypassing rawpy. IR, when present, is written as a separate grayscale file with an `_ir` suffix. Expansion defaults to off; 2× and 4× are available. Only a main-image page (`NewSubfileType` 0) counts as LinearRaw: Adobe DNG Converter's reduced LinearRaw previews sit beside a CFA, so its files are camera RAW.

    A 3-channel LinearRaw DNG that libraw fails to unpack falls back to the same tifffile decode, replaying the `LinearizationTable`, `BlackLevel`/`WhiteLevel` and `DefaultCropOrigin`/`DefaultCropSize` tags. This covers DNG 1.7 JPEG-XL ([rawpy#207](https://github.com/letmaik/rawpy/issues/207)). No camera-to-XYZ matrix is computed; only the as-shot white balance (`AsShotNeutral`) applies, under the **Linear RAW** toggle. The fallback is failure-driven. Linear Output's 3-channel path uses the same decode (`_peek_linear_dng_rgb`). A default crop that trims the RGB plane while a same-file HDRi IR page stays full size raises an error.
*   **Camera RAW**: demosaiced by rawpy (AHD on a CFA sensor, or the Demosaic panel's choice) with `user_wb=[1,1,1,1]`, `output_color=raw`, `gamma=(1,1)`, `no_auto_bright=True`. ImageDescription records the resolved algorithm (`resolve_demosaic` in `loaders/helpers.py`). Green-normalized as-shot WB multipliers go into XMP as `RAW-WB: R G B` inside `dc:description`, and the source filename into `crs:RawFileName`. No expansion.
*   **Coolscan NEF** (`negpy.infrastructure.loaders.nef_loader`): TIFF-structured, with the full-res RGB image in a SubIFD chain (tag 0x014A). The loader picks the largest RGB SubIFD by pixel count and rejects files with CFA/Bayer SubIFDs (camera NEFs). The loader assumes nothing about linearity or color space. No IR; extra channels are dropped. No expansion.
*   **Flextight FFF** (`negpy.infrastructure.loaders.fff_loader`): two variants. Uncompressed 16-bit RGB (FlexColor export) is a big-endian TIFF with the image in a top-level IFD, selected by pixel count because SubfileType is unreliable. SGI LogLuv (raw `.3fr`/`.fff`, compression tags 34676/34677) decodes through `negpy.infrastructure.loaders.logluv`, LogLuv32/24 → CIE XYZ → linear sRGB, ported from [flexcolor-tool](https://github.com/rohanpandula/flexcolor-tool) (MIT).

    LogLuv values exceed 1.0, so the decoder applies per-channel percentile normalization (0.2th/99.8th) after XYZ → linear RGB, giving gain-normalized linear transmittance. FlexColor metadata comes from tags 50457 (Apple plist: film stock, film type, gamma, DPI, scan date) and 46279 (firmware version, scanner serial). No IR; extra channels dropped. No expansion.
*   **Noritsu RAW** (`negpy.infrastructure.loaders.noritsu_loader`): headerless BGR 16-bit little-endian dumps. Dimensions come from file size against a table of known Noritsu scan sizes (exact match, known width with novel height, novel-width fallback). BGR is swapped to RGB. 12-bit data in 16-bit range; default expansion 16× (`NORITSU_EXPANSION`).
*   **TIFF** (standalone `_decode_tiff`, independent of `TiffLoader` and its sRGB linearization): a 4th channel is IR only when ExtraSamples is 0 (UNSPECIFIED) or missing; 1 and 2 (alpha) are dropped. Sidecar IR (`_ir.tif`, with optional `_ir_valid` mask) and IR in secondary TIFF pages (SilverFast iSRD) are detected. **Input gamma** declares the source encoding (linear, 1.8, 2.2 or sRGB) for linearization before export. Expansion is available, off by default.
*   **Trichrome triplets**: all three narrowband exposures decode and merge through `merge_rgb_triplet()` into one TIFF. The red exposure is primary (and supplies WB and device metadata); green and blue paths come from the frame's `RgbScanConfig`. No sensor correction.

    **Merge to TIFF Negative** (`services/export/frame_merge.py`) replaces a triplet or a stitch with one untagged 16-bit linear negative, written from the render decode (`ImageProcessor._load_source_f32`), so it loads back to the buffer the pipeline inverts. `decode_params` decides what is baked: a triplet decodes with flat-field off, so the gain map stays live in the edit; a stitch keeps it on, because its registration holds only on per-part corrected buffers. `merged_edit` clears what was baked (`rgbscan` and the sensor matrix, plus `stitch` and `flatfield` for a stitch), `carry_edit` locks those cards against the roll, and `forget_composite` dissolves a stitch's record. A single-shot stitch's preview unmixes the assembled canvas while its export unmixes per part; the merged file holds the per-part version. Discovery never groups a TIFF whose description names a merge (`is_merged_source`). Refused: a bracket (float32 detail below a 16-bit step, render exposure baked in), a transfer or positive frame (`needs_camera_matrix`), and a source whose format label hides the assembly, such as a LinearRaw DNG (`describes_a_merge`).
*   **Stitch composites**: each part is decoded and corrected (flatfield plus sensor correction for single-shot parts, flatfield only for triplet parts), then assembled through `stitch_composite()` as in §2.

**IR Dust Removal** (when an IR channel is available, off by default): IR-based dust and scratch correction on the linear buffer before writing.

**Optional corrections** (camera RAW only), all off by default: *White Balance* multiplies by the green-normalized as-shot WB gains; *Flat Field* applies the configured Flat Field gain; *Sensor Correction* applies the crosstalk unmixing matrix. Stitch composites always get Flat Field and sensor correction per part, to avoid seams. The corrections run after EXIF orientation and before the user rotation and flips, because the Flat Field gain map is laid out on the EXIF-oriented decode.

*Lens Correction* (any source, off by default) applies the Optics card's correction. The embedded warp runs as in `prepare_lens_source` (single camera RAW, after Flat Field, before the unmix); the manual `k1` runs on RGB and IR after the user rotation and flips, with `GeometryProcessor`'s `apply_radial_distortion`. A half-frame asset skips it, since its file holds the whole scan. The description records "lens" when either ran.

`wb_bake_block_reason()` makes *White Balance* inert for a Trichrome triplet or a Single-Shot Narrowband capture ("trichrome"/"narrowband", "" otherwise), because as-shot gains correct a broadband scene. Single-Shot Narrowband is detected from the Narrowband toggle or a calibrated sensor matrix.

**Output metadata.** The file holds raw pixels plus Make, Model and DateTime from the source; source ICC profiles, EXIF color space tags and XMP color metadata are never copied. For **TIFF**, the description records source format, expansion, white balance and applied corrections (including whether ICE ran) and ends with "no color management"; as-shot WB also goes into XMP. For Flextight FFF, Make includes film stock and type from the plist, and Model includes the scanner serial.

**JPEG XL** carries the same record: `imagecodecs.jpegxl_encode()` has no metadata parameter, so `_write_jxl()` writes the description, Make, Model, DateTime and XMP into the container's Exif and `xml ` boxes after the encode. The IR sidecar is untagged in both formats.

### Peek Negative
**Code**: `AppController.toggle_negative_peek` / `_paint_negative_peek`

Not a pipeline stage. The decoded source (`AppState.preview_raw`, before flat-field, sensor unmix and every defect repair) runs through `GeometryProcessor` and `CropProcessor`, then `working_oetf_encode` for display. Nothing else runs, and `content_rect` is cleared.

The peek applies `camera_to_working_matrix` before the encode. Unlike `should_fold_camera_wb`, it folds in the as-shot multipliers whenever the decode skipped them (`effective_linear_raw`), narrowband included. `lightbox_level` then applies one scalar gain that puts a high percentile of the frame at display white, measured before the crop and applied after; a frame with no signal returns None and paints unlifted. The frame is *not* marked `splash`; it takes the normal working-to-display conversion with `proof` set False.

### Peek Embedded Preview
**Code**: `AppController.toggle_embedded_peek` / `_paint_embedded_peek`

The camera's own JPEG (`PreviewManager.try_splash_preview`) through `GeometryProcessor` and `CropProcessor`, nothing else. Its context uses the preview's own shape, not `original_res`. It is marked `splash`, so the display transform takes it as sRGB. It is held in `AppState.preview_embedded` until the frame changes, with the active asset's half-frame slice applied. No metric is derived from it.

---

## 4. Local Contrast (CLAHE)
**Code**: `negpy.features.lab.logic.apply_clahe` (CPU) / `negpy/features/lab/shaders/clahe_{hist,cdf,apply}.wgsl` (GPU)

CLAHE on CIELAB $L^{\ast}$, computed from linear working RGB (Adobe RGB, D65). Chroma ($a^{\ast}/b^{\ast}$) is untouched. CPU and GPU mirror each other bin for bin; the parity test pins them to about 1e-6.

*   **Fixed $8\times8$ tile grid** over the full frame at every render scale, so the preview predicts the export. 256 bins over $L^{\ast} \in [0, 100]$.
*   **Clip limit**: $\max(1, \lfloor \text{strength} \cdot 2.5 \cdot N_{tile} / 256 \rfloor)$ counts. The excess is spread evenly across all bins, remainder to the lowest.
*   **Per-pixel remap**: a smoothstep-weighted bilinear blend of the four neighboring tile CDFs (tile centers at $(\text{pos}/\text{dims}) \cdot 8 - 0.5$, edge-clamped), then
    $$L_{final} = (1 - \alpha) \cdot L + \alpha \cdot \text{CDF}(L) \cdot 100$$
    with $\alpha$ = `clahe_strength`.
*   A tiled full-res export hands the CDF of its downsampled meter render to every tile (`clahe_cdf_override`), so the tiles share one mapping.

The control is `lab.clahe_strength`. Defect repair (§5) runs earlier, on the linear source.

---

## 5. Retouching
**Code**: `negpy.features.retouch`

Three sources find defects: the IR channel, statistical detection and manual strokes. Each gives the same continuous **defect score** ($1$ = clean film, floor $0.02$ = certain defect), repaired by one shared fill (step 3 below), plus a routed inpaint for defects too wide for the fill.

This is not a pipeline stage. Every repair is baked into the **linear source before geometry and normalization**, like flat-field, so the meters read cleaned film and CPU and GPU agree without either implementing it.

*   **Light leak** (`leak_remove`, `features/leak/`; Print and Transfer paths, single frames only): the first bake, after flat-field and unmix, so every repair copies leak-free film. It reduces a leak only under the shapes in `leak_shapes`, placed by hand or by Auto. Every estimate runs on a 1024 px copy of the frame, so preview and export read the same leak.

    1.  **Model**: a leak is one light, so it adds exposure to the three emulsion layers in one ratio $k$, its color, times one strength map: $u_c = u_{scene,c} + A(s)\,F(t)\,k_c$. Each layer's density over the base $D$ (channel density through the crosstalk unmix when one is set) maps to excess exposure $u = 10^{D/\gamma} - 1$, $\gamma = 0.6$, the toe form $D = D_{min} + \gamma \log_{10}(1 + E)$. With **Film Dmax** $D_s$ set, the curve is $D = D_{min} + D_s \tanh(\gamma \log_{10}(1 + E) / D_s)$; its inverse stops at $0.995 D_s$, and a layer past that keeps its density. $D_s$ is never read from the frame, whose densest film is as often a bright subject as the film's ceiling, but never sits under that film's density (99.9th percentile per layer).
    2.  **Evidence**: the crop maps back through the geometry to a picture box. Each side with film outside it gives a rebate profile, the 95th-percentile transmittance of film pixels in 8 px bins of a strip up to 8% of the long side deep, so sprocket holes, bed and edge print drop out; $T_b$ is `film_base` refined to the thinnest bin. Rebate holds no scene, so its glow is the leak's own. In the picture, a leak reads in the **shadow spike**: the mode of the lowest quartile of $\log(1 + u)$ in 0.04 bins, where it holds 6% of the pixels. A leak only adds, in every layer, so a spike under the clean film's in any layer by more than 0.1 is a darker scene, not a reading.
    3.  **Color**: from lit rebate the shape crosses; else each layer's median ratio to the strongest over the cells that read strongest; else neutral. **Back** gives $k = (1, 0.2, 0.05)$, light through the base reaching the red-sensitive layer first; **Front** gives $(0.5, 0.8, 1)$, light on the emulsion side reaching the blue-sensitive layer first, as gate leaks on a color negative read. **Color Trim** $t$ scales the field in the layer $k$ leads in by $1 + t$. On B&W film every shape reads its color off the film, with no trim.
    4.  **Reduction**: the leak $L$ comes off the local mean $\bar u$ (a 1 px blur at the work size) through a softplus floor of width $0.1L$, and each pixel's $u$ takes the gain $g = \mathrm{floor}(\bar u - L)/\bar u$, so grain is scaled, not cut to bare base. A layer thinner than the base keeps its density, and a pixel nothing came off keeps its value. Shapes are read in order, each on what the ones before it left, and their fields add, so two over one leak remove it once.
    5.  **Band** (two ends of a line and a width each side, in long sides): $A(s)$ along the line times $F(t) = \cos^2(\tfrac{\pi}{2} t^{q})$ across, $t$ 0 on the line and 1 at either edge, $q = 4^{1 - 2f}$ at the shape's Falloff $f$, tapered past the line's ends over the mean width. In 24 px cells along the line, each side reads the spike within half the width less that of the clean film past its edge, divided by $F$ there; where both sides step darker, as a dark subject on the line with sky past the edges, the cell reads the cap. The readings project onto $k$ by least squares; a cell with no spike fits a straight scene ramp per layer plus $a F k$ with Tukey weights. A smooth curve (penalty $10^{-0.1}$ on second differences) through the readings, with Tukey's biweights, sets $A$; Strength scales it.
    6.  **Edge** (the ends of a stretch of picture edge and a reach $R_e$): $A(s)\,G(t / R(s))\,k$ at depth $t$, 1 outside the edge. Light through a gap spreads in the base and reaches further where it is stronger, so each 24 px cell has its own strength and reach, read from 12 depth bins of spikes less the clean film's past $R_e$. The fade is $G(x) = (e^{-3x} - e^{-3})/(1 - e^{-3})$, or, for the soft shadow of a gap at the gate, a shoulder $e^{-(3x)^2/2}$ eased to nothing at 1, taken where it fits the pooled readings twice as well.
        *   **Crossing**: a leak's light is continuous across the frame line. It crosses where the shadows right inside carry the rebate's light at the line at half to twice its level, in at least half the read cells, standing 1.75 times or more over the clean film's spike and at most 10 times the strip's thin end. A lit cell carries the glow where the picture's 5th percentile within 3 px inside is at 65% or more of the rebate's median within 3 px outside, in every layer the rebate shows; where fewer than half the lit cells do, the glow is fog or edge print, and an edge on Auto color reads no leak in the picture there.
        *   **Rebate profile**: with lit rebate, the leak follows its level $\ell(s)$ along the edge, the rebate's light within 3 px of the line interpolated over the lit cells (the strip's profile where fewer than two read it, or where that light is over 10 times the strip's thin end): $A = c\,\ell$ and $R = r\,(\ell/\ell_{max})^p$, $p \in \{0, 0.5, 1, 1.5, 2\}$, fitted over every reading under their 25th percentile (pinball loss), as scene only adds; $c = 1$ where the leak crosses, else $c \le 1$. With no lit rebate, each cell fits its own strength and reach, and a cell with too few readings, as over smooth sky, takes them from its neighbors through the band's robust curve.
        *   **Steep tip**: rebate that rises at least 2 times over 0.6% of the long side away from the edge, lit at the line and carried by the picture's first row, is light piped along the base from the film's cut end. It is read per layer and per 1 px column: the rebate within 3 px of the edge is the strength, the fade $e^{-3x}$ eased to nothing at $x = 2$, its reach the shorter of the rebate's own and a fit under the picture's 25th percentile, and each layer's scale is capped per cell by the 10th percentile of the local mean over that strength.
        *   **Frame line inside a wide crop**: a crop set past the frame line holds rebate, and the edge reads from the line. Going in, film is rebate while its detail (the mean of the layers' $\log(1 + u)$ against its 3 px blur) stays under 1.5 times that of the rebate 4 to 8 px outside, starting within 1.67 times its level, up to 6% of the long side in. Where light-struck leader meets the gap, the step between them reads as texture too: smooth film within 20 px and 1.33 times dimmer is still rebate, and the search runs 6% on from there, to 12% at most. The frame line is the sharpest detail within 5 px of where it rises for two steps with the 10 px past it at twice the rebate's detail; under 5 px is no rebate. That rebate holds no scene, so along the whole side its width in from the line and out to the next frame's texture, read per 24 px cell and fitted as a straight line (cells off it by more than 3 px or twice the median offset set aside; the outer width is 0 where fewer than three cells read it), is cleared by a field of 1.5 times its local mean times $\min(\text{Strength}, 1)$. Where no crossing is proven, the clearing also needs the rebate lit (a median of 0.5) and the picture past it at 2.5 times the strip's lower-quartile detail, as a smooth stretch of picture reads as rebate too.
    7.  **Spot** (a center and the ends of two axes): $A\,F(\rho)\,k$, $\rho$ the distance from the center over the oval. In 8 sectors, the spike within half the oval is read against the ring past its edge as a band's sides are; the sectors' median is $A$, or the cap where no sector reads and one stepped. Shadows under a leak pile up at the leak itself, so where the core's 1st percentile of $u_c / F$ reads within 1.5 times that, its layers give $k$.
    8.  **Paint** (brush strokes): weight $W$, 1 inside, fading to 0 at the edge over the Falloff's share of its deepest point's depth, smoothstep. In 16 px cells, the spike inside is read against a 24 px ring of film around the region, carried in by normalized convolution over 3 cells; each layer's map is the lower envelope of its cells (an opening over 5 cells, a blur over 2), as a bright subject only lifts a cell. Its color is the 1st percentile of $u / W$ in the cells over half the strongest, where that is within 1.5 times the envelope, else the envelope's own layers.
    9.  **Caps**: along a band or edge, the field is capped by the 1st percentile of $u / (k F)$ near the line in the leak's strongest layer, smoothed over a cell in log and never above a cell's own reading; under a proven crossing, pixels under 0.25 in that layer are left out, as bare film there is rebate or a sprocket hole's rim. A spot caps by the same percentile in its core, a painted region per cell. Where a falloff reaches past its leak, a local cap keeps a dark subject from going to bare base: per 48-cell grid, the 1st percentile of each layer's $\bar u$, smoothed over 0.6 cells in log, joined by soft min. A steep tip and the rebate clearing are not held by it.
    10. **Recover Layers**: a layer inside the leak's area (over 10% of its peak), driven by its color (at least half its strongest layer's exposure) and within 5% of its ceiling (its shoulder with Film Dmax, else its 99.9th-percentile density; opened by 3 × 3) has lost its picture. Where another layer holds it, its $\log(1 + u)$ is rebuilt as $a\,g + b$ of the mean $g$ of the others, $a$ and $b$ a guided filter's local fit (ε 0.01) over film where the leak adds under 5%, in the smallest box reaching 3, 6, 12 or 24% of the long side each way where that film weighs 5%. Film saturated in every layer is marked for **Show Burned**.
    11. **Grain and Contrast**: where leak came off (a smoothstep of the luminance density removed over 0.05 D, blurred over 0.2% of the long side; never bed or sprocket hole), **Grain** blends toward He et al.'s guided filter of luminance density by itself (radius 0.06% of the long side, at least 2 px; $\varepsilon = (2\sigma)^2$, $\sigma$ the grain's robust spread) and **Contrast** adds up to 1.5 times the detail against a blur over 2% of the long side. The luminance shift scales all three channels.
    12. **Auto**: **bands** come from each 16 px cell's spike over the frame's 40th percentile, smoothed by 2 cells, and its Radon transform as line means every 2°. Of the 16 strongest lines, one is a leak when the bare rebate it crosses is lit over 7 bins by at least 2.0 and 6 of the side's spread, and the rebate 4 to 8 px outside the frame line by 2.0 too, since edge print never reaches the line; a line crossing no rebate must rank in the top 6, stand 6 robust deviations over its angle's lines and lift by 50. Lines within 35° and one width of a found band are the same leak; at most 3 are found. **Edges** read each picture side 0.3 long sides deep: cells where the leak crosses at 0.5 are strong, gaps of three cells join, and runs closer than the side's strong length merge into one edge where a run of three cells holds it, reaching its cells' 90th-percentile reach. A side with no such run whose crop holds lit rebate passing the no-crossing test above gets an edge 0.03 deep, which clears it. Auto places no spot or paint: with no rebate to vouch for it, a lamp or a lit window reads the same.
    13. **Slides** (Transfer path): a slide thins with exposure from its black, so a leak lowers its density. It is read as a negative, $T' = (T_f / T)^{0.6 / \gamma_s}$ per channel, $\gamma_s = 1.8$, $T_f = 10^{-5}$, so its black reads as base and clear film as the densest film. Everything above runs on $T'$, Film Dmax scaled by $0.6 / \gamma_s$, and the pixels removal moved convert back.
    14. **From Roll** (`leak/roll.py`) carries a frame's shapes to others by the picture box: each point keeps its place in the box, and each width and brush size its share of the box's long side. Nothing measured carries: each frame's bake reads the leak on its own film.

*   **Infrared (IR) dust removal** (scans with an IR channel: Coolscan, SilverFast iSRD, VueScan DNG):
    Dust blocks infrared while the dyes pass it, so the IR plane is a defect map. Concepts are ported from digital-fauxice (see `NOTICE.md`).

    The IR plane is downsampled for detection and **follows the buffer** being repaired: at most 1.5× under it, capped at 3600 px. Detection far below repair resolution inflates the mask and the fill supports, so a defect on a tonal edge prints a dark blotch. Footprint constants below are pixels at a 1600 px reference and scale with the plane; the score's 3×3 erode does not. Only *large* regions under the dead floor count as non-film, since an opaque hair blocks IR as fully as the holder.

    1.  **Normalized ratio**: $r = IR / \text{blur}(\text{dilate}(IR))$, about $1.0$ on clean film, lower under defects. IR more than 4σ over the clean-film median is scanner glare, such as a lit frame edge, and stays out of the base, because dust only dims IR. A crosstalk fit first divides out the image's IR ghost. Per frame, the clean-film floor is then rescaled onto the identity point and the dip scale onto a reference σ, stretching only, so a scanner at or above the reference renders bit-identically.
    2.  **Division tier**: semi-transparent dust attenuates, so the image is recovered as $RGB / r^{\gamma}$, with per-channel γ fitted per frame. The gain never lifts a pixel past its local clean base (defect-excluded mean $- \sigma$); a grain-biased local max prints dark rings.
    3.  **Score-weighted fill**: the ratio maps to $s \in [0.02, 1]$ (the IR Threshold slider moves the ramp). Nothing is thresholded. Cores and hairs are rebuilt as a multiscale score-weighted average over nested supports, $\text{fill} = \left(\sum I \cdot s \cdot w\right) / \left(\sum s \cdot w\right)$. The finest support with clean data wins. Three rungs sit at the detection plane's footprint and one coarse rung at the reference footprint, which alone reaches across a wide defect.

        **Original-floor rule**: dust is dark in negative transmittance, so a repair may only lighten. The rule lets this path measure each support's clean fraction on the raw score and correct an all-defect support downstream. It compares the *low-frequency* deficit, since per pixel it keeps grain peaks and clips troughs.
    4.  **Grain transplant**: both repair paths paste back detail high-passed from the nearest pixel the score calls clean, mirrored *through* the donor so a row across a hair does not streak into bands. Deep in a wide defect the paste flattens and the multiscale blend carries the interior.
    5.  **Routed inpaint**: defects with chebyshev radius ≥ 5 at detection scale go to structure-following inpaint, alpha-feathered, with the same grain transplant. A 2% frame budget bounds it; the fill always runs.

    B&W silver and Kodachrome block IR like dust. Such frames are auto-detected (the IR plane mirrors the image) and skipped.

*   **ICE at scan time** (Scan tab, nkscan backend only):
    The nkscan driver runs its own openICE port during the scan. The repair is baked into the file, not an edit.

*   **IR removal, the OpenICE method** (`ir_method = "openice"`, `negpy/features/retouch/openice.py`):
    A port of openICE (see `NOTICE.md`), a reverse-engineering of Nikon Scan's Digital ICE. It replaces steps 1 to 4 above, shares no code with them, keeps the routed inpaint, and runs at the same point.

    All work is in log density, $D(v) = \frac{M}{16\ln 2}\ln(vM + 1)$ with $M = 65535$.

    1.  **Calibration**, once per frame from a roughly 281 px box downsample. The dye→IR crosstalk $c$ is a least-squares slope of IR deviation on red deviation over the four 4×4 quadrants of every all-clear-film 8×8 tile, gated to $|\delta^{IR}/\delta^{R}| \le 0.2$ and weighted $(\delta^R)^2(\sum IR)^2$. Two IR²-weighted reference levels give $IR_\text{ref} = (IR_\text{raw} - c\,R_\text{ref})/(1-c)$.
    2.  **IR gate**: $g = (d_{IR} - c\,d_R)/(1-c) - \theta$, at $IR_\text{ref}$ on clear film, dropping by the light a defect stole.
    3.  **Clean-confidence weight**: $w = \text{clamp}\!\left(1 - (IR_\text{ref} + b - g)/\Delta,\ 0.02,\ 1\right)$ on the horizontal 3-tap minimum of the gate. $b$ and $\Delta$ come per frame from the gate's median and MAD σ over sampled whole rows (after crosstalk subtraction and the min, at working resolution), because ICE's fixed values do not transfer between scanners. The IR Threshold slider biases $\Delta$ between 2σ and 11σ.
    4.  **Two normalized-convolution pyramids** over four scales: 9×9 octagon (69 cells), 5×5 octagon (21), 3×3 binomial tent (16), pixel. $C_\ell = \sum k_i w_i$; gate pyramid $P_\ell$ and color pyramid $L_\ell$ are $\left(\sum k_i w_i x_i\right)/C_\ell$.
    5.  **Reconstruction**: $\text{acc} = L_0 + \gamma_{ch}(IR_\text{ref} - P_0)$, then per scale $\text{detail}^{(\ell)} = (L_\ell - L_{\ell-1}) \cdot 1.25$ through a dead zone $[\beta_\text{lo}, \beta_\text{hi}]$ (min and max of $P_\ell - P_{\ell-1}$ over a 5-point cross). Detail within the IR's own contrast is dust and is dropped. The rest passes, less the threshold, scaled by $\min(2C_1, 1)$, $C_2$, $w^2$.
    6.  **Give up rather than invent**: a pixel is left alone when $w \ge 1$, when any channel reconstructs non-positive, or when one of four 9-sample probes on the 9×9 perimeter is entirely below the dust floor (ICE's 6.5% of clear film, carried as a transmittance fraction).

        Output is $\max(L_3, \text{acc} + \text{dither})$ ("only fill, never darken"), written at full strength wherever $w < 1$, as ICE does. Unlifted pixels are written back verbatim, since $D^{-1}(D(v))$ is not bit-exact in float32.
    7.  **Grain**: ICE's zero-mean density dither, $\text{dither} = \text{env}(x)\,(u - \tfrac{1}{2})\,\alpha_{ch}x$, $u$ uniform, $\alpha = (0.015, 0.015, 0.025)$, with a parabolic envelope over $[D(\lfloor 0.01M \rfloor), D(\lfloor 0.99M \rfloor)]$ peaking at 1 mid-band, and none unless $x + \text{dither}$ stays in the band. The draw is a hash of the absolute pixel coordinate, not ICE's frame-global LCG, so a banded pass matches a whole-frame one.
    8.  **Routing**: unreconstructable defects go to the structure-following inpaint. Components over 0.2% of the frame are dropped first, since some scanners report the rebate as one huge defect that would blow the 2% budget.

    Byte-exactness is out of scope (80-bit x87 intermediates, frame-global grain order). Density through calibration, gate, all four confidences and all four gate-pyramid levels match the C reference to float32 noise, and the give-up maps agree.

*   **Automatic dust removal** (`dust_remove`, with Threshold and Size): a statistical detector on the visible scan, feeding the same fill.

    1.  **Detection plane**: the source min-pooled to the IR detection resolution (`ir_detect_target`) by the same erode-then-average, because an area average dilutes a two-pixel speck below the grain. The preview keeps its own min-pooled plane from load. A segmented RGBI preview filters its IR channel before each segment is reduced.
    2.  **Detection proxy**: percentile-normalized source **density** ($-\log_{10}$ linear luminance, 0.5 to 99.5% window), grade-independent in every process mode.
    3.  **Excess and σ**: background = max of a median over $3\times$ Size and a morphological opening by Size, so specks cannot lift their own background. The excess is averaged $3\times3$ and divided by its local MAD σ over $4\times$ Size, both on 8-bit quantized planes. Windows scale with the plane, so Size is a film footprint.
    4.  **Gate**: Spot and Hair Threshold each map to a seed bar in σ (3 loosest, 9 default, geometric above the default toward 48; 1.0 is off). Seeds grow through connected pixels to a grow bar within $2\times$ Size. A compact component must clear the spot bar, raised per pixel by the wide-window σ over a knee of 0.02; a hair-thin one the hair bar, raised more steeply by the median σ along it over a knee of 0.075, so a seed grown along a tonal edge fails. That σ is capped at twice the σ with the hairs filled, so a deep hair on flat film does not raise its own bar. Stat maps are cached, so a Threshold change re-runs only the gate.
    5.  **Specks → score**: at the floor on the defect, ramping to clean across a small pad, then the shared fill. Where IR Removal already repaired a pixel, the optical mark is released.
    6.  **Hairs → routed inpaint**: strongly elongated defects (thinness $= \text{area}/\text{thickness}^2$, bending-invariant unlike PCA aspect) go to Navier-Stokes inpaint, encoded against the local clean range and alpha-feathered over the dilated PSF skirt.
    7.  **Exclusions**: right-painted strokes (source-normalized points, width at the heal reference scale) release the optical detections they cover: score back to clean, hair mask cleared. A mark crossing the rim keeps its repair outside the brush. The rim feathers like a manual heal, and the band is the whole path, smoothed and round-capped. This runs on the detection plane after the gate and before the fill, and leaves the IR and manual routes alone. An all-clean score skips the bake. Strokes fold into the source token.

*   **Manual heals (Heal and Scratch tools)**: a painted stroke is a **search area, not a stamp**: damaged pixels inside it are measured, and clean grain comes back byte-identical.

    The measure is a **two-sided** $|z|$ (dust is bright in density and a **scratch dark**): density high-passed against a local mean (window $3\times$ the brush radius), averaged $3\times3$, over its MAD σ in the stroke's neighborhood. **Hysteresis**: $|z| \ge 8$ seeds, and connected pixels to $|z| \ge 2$ join. A stroke that never reaches 8 seeds against its own maximum; below $|z| = 5$ it is a no-op. The kept region is padded like the detector's.

    The fill runs with the **original-floor rule off**, since a scratch must be free to darken. Instead it measures each support's clean fraction *above* the score floor, with supports at the film footprint, not buffer pixels. Too-wide cores route to the inpaint. Sparse defects (strokes and detected specks) are repaired each in a padded crop, with the support ladder still from the whole frame, so results match a full-buffer repair.

*   **Transport scratches (the line tool)**: a nearly straight scratch a few px wide along the film, visible only when integrated along the line.

    A click gives a start point. The frame is band-passed *across* the scratch and normalized by a **local** noise scale. The line's **slope** is fitted, pulled toward the click. **Extent and width are measured**: the repair covers the stretch where the ridge holds, and the band grows outward by the same hysteresis as the brush, so a stored line stays valid at any resolution. The mask takes the same skirt pad and fill, original-floor rule off. **Line Sensitivity** sets the ridge bar for both and re-measures placed lines. There is no auto-detection, because a full-length ridge is as likely a horizon.

*   **Clone strokes (the Clone tool)**: the last source bake, so a clone copies repaired film. Each stroke stores raw-frame points, a destination-to-source offset, strength, feather and tone match; strokes apply in order. The patch is the shifted source over a capsule mask whose edge ramps over `feather` × the radius, times strength; off-frame source pixels carry no weight. **Match Tone** scales the source by $\bar d / \bar s$, the Gaussian means ($\sigma$ = brush radius) of destination and source outside the brush, clamped to ±6 stops. A new stroke on the last bake applies only itself.

*   **Resolution independence**: coordinates and sizes are relative to the full-resolution RAW. A half frame reports the cropped and split full-resolution dimensions even when its preview uses a half-size RAW decode. Strokes are in raw-frame coordinates and the repair runs before geometry, so rotations, flips and distortion correction need no mapping.

---

## 6. Lab Scanner Mode
**Code**: `negpy.features.lab`

Steps, in order:

1.  **Chroma Denoise**: a band split of $a^{\ast}b^{\ast}$ (Fuji, US 5,739,922). The low band is a Gaussian blur of $\sigma$ = Chroma Denoise × scale factor (`chroma_denoise_sigma`), with `gaussian_kernel_1d` taps on both engines (`chroma_lowpass_h/v.wgsl`). Chroma above it is grain or scan noise, and only an edge keeps it: $ab' = \text{low} + k\,(ab - \text{low})$, $k = \text{smoothstep}(1.0, 2.5, \sigma\,|\nabla \text{low}|)$ (`CHROMA_EDGE_LO`/`HI`, central differences on the edge-replicated low band). L stays untouched, so grain stays as luminance grain. Below $\sigma = 0.25$ px the stage is skipped.

2.  **Global Saturation**: a lightness-preserving chroma scale ($a^{\ast}/b^{\ast}$) in CIELAB, independent of the print curve.

    Below 1.0 it is a flat scale. Above 1.0 it is **gamut-aware** (`gamut_aware_chroma_scale` in `kernel/image/logic.py`, mirrored in `lab.wgsl`), because a per-channel RGB clamp would shift hue. `_in_gamut_lab` checks whether the boosted (L,a,b) decodes to linear working RGB in $[0,1]$. In-gamut pixels get the flat scale; the rest get a softplus-style knee toward their own headroom, the max in-gamut scale found by bisection to under 0.1% error in 10 iterations.

3.  **Skin Protection** (`skin_chroma_rein` in `kernel/image/logic.py`, mirrored in `lab.wgsl`, `lab.skin_protection`, default 0.5): a soft chroma ceiling inside a skin mask, after saturation. It also runs at Chroma 1.0.

    The mask (`_skin_weight`) multiplies a Gaussian on hue at $52°\pm20°$, a chroma window at full weight to $C^{\ast}=35$ and zero by $60$, and a lightness rolloff below $L^{\ast}=15$ and above $95$. Pixels under $C^{\ast}=2$ get zero.

    With $w$ the mask weight and $s$ the slider:

    $$C_{\text{ceil}} = \frac{22}{s}, \qquad C_0 = 0.6\,C_{\text{ceil}}, \qquad C_{\text{knee}} = C_0 + (C_{\text{ceil}}-C_0)\left(1 - e^{-(C-C_0)/(C_{\text{ceil}}-C_0)}\right)$$

    for $C > C_0$, identity below, $C_{\text{out}} = C + w\,(C_{\text{knee}} - C)$. $a^{\ast}$ and $b^{\ast}$ scale by $C_{\text{out}}/C$, so hue and $L^{\ast}$ are kept. The ceiling fades out continuously as $s \to 0$. Chroma is only reduced, so `saturation = 0.0` still reaches gray and the gamut knee's fit holds.

4.  **Sharpening**: **Method** picks Unsharp Mask or Deconvolution. Both share Amount, Radius and Masking and the $\text{radius}$ Gaussian taps from `gaussian_kernel_1d` (`sharpen_k` storage buffer), convolved identically by `cv2.sepFilter2D` and the separable WGSL passes, so CPU and GPU match bit for bit.

    **Unsharp Mask** on $L$ with halo suppression (`lab_sharpen_h/v.wgsl`):

    $$L_{diff} = L - \text{blur}(L, \sigma), \qquad \sigma = \text{radius}$$
    $$\text{gain} = \text{amount} \cdot 2.5 \cdot \text{smoothstep}(0.25, 0.33, |L_{diff}|) \cdot m \cdot g(L)$$
    $$L_{final} = \text{clamp}\big(L + L_{diff}\cdot\text{gain},\; L_{min}-2,\; L_{max}+1\big)$$
    *   **Radius** (px): blur $\sigma$ in pixels of the full-size render. A resized export scales the radius down with the image.
    *   **Masking** ($m$): $\text{smoothstep}(0.5t, t, |\nabla L|)$ on the boxed gradient, $t = 10\cdot\text{masking}$; off at 0.
    *   **Shadow gain** ($g(L)$, fixed): $\tfrac{1}{3} + \tfrac{2}{3}\,\text{smoothstep}(0, 35, L)$, since the thinnest negative has the most grain (Gallagher & Gindele, US 7,228,004). Both methods apply it; Deconvolution reads $L^{\ast}$ of the observed $Y$.
    *   The noise gate over $[0.25, 0.33]$ is sized for $|L_{diff}|$ at a 1 px radius. The clamp to the local $3\times3$ range is tighter above (+1) than below (−2), because $L^{\ast}$ USM exaggerates light halos.

    **Deconvolution**: Richardson-Lucy on linear luminance $Y$ with a Gaussian PSF (`rl_*.wgsl`).

    $$\hat{o}_{n+1} = \hat{o}_n \cdot \left(K \otimes \frac{o}{\max(K \otimes \hat{o}_n,\ \epsilon)}\right), \qquad \hat{o}_0 = o = Y$$

    Iterations are $\text{clamp}(\text{round}(10\cdot\text{radius}), 5, 20)$, so preview and export match. No early stop or damping: the edge mask alone governs grain, as in RawTherapee. Applied as a chroma-preserving RGB ratio:

    $$\mathrm{RGB}_{out} = \mathrm{clamp}\left(\mathrm{RGB} \cdot \max\left(1 + \left(\frac{\hat{o}_N}{\max(o,\epsilon)} - 1\right) \cdot \mathrm{amount} \cdot m,\ 0\right),\ 0,\ 1\right)$$

5.  **Glow**: blurred highlights added back in linear light.

    $$I_{out} = I + B_{glow} \cdot s_{glow}$$
    $$B_{glow} = \text{GaussianBlur}(I \cdot m_{hl})$$

    *   $m_{hl}$: **display-domain** highlight mask, linear ramp from code value 0.5 to 1.0.
    *   Equal on all three channels; the sum is clamped at the stage output.

6.  **Halation**: red scatter from the film base. A larger Gaussian than Glow, additive in linear light, with the mask on **linear reflectance** ($t = 0.65$) so it follows scene exposure, not Grade and Density.

    $$I_{out} = I + B_{hal} \cdot s_{hal}$$
    $$B_{hal} = \text{GaussianBlur}(I_R \cdot m_{lin} \cdot C_{hal})$$

    *   $I_R$: red channel, the scatter source.
    *   $m_{lin}$: linear ramp from reflectance 0.65 to 1.0.
    *   $C_{hal}$: tint weights $(1.0,\ 0.3,\ 0.05)$.

---

## 7. Alternative Processes
**Code**: `negpy.features.lith`, `negpy.features.cyanotype`, `negpy.features.sabattier`; config `negpy.features.altprocess`

One optional process (B&W Negative mode only), between Lab and Toning so the toners act on it. Lith, cyanotype and Sabattier are mutually exclusive (one `alt_process` enum), default none; then both engines skip the stage.

### 7.1 Lith

Two phases (Rudman's "New Rules", Moersch Lessons 1-6): a long low-gamma highlight branch fixed by exposure, then infectious development that drives the shadows to Dmax almost vertically, with no separation past it. On the print density $D_0$, with the paper's $D_{max}$:

$$D_0' = D_0 + 0.301 E, \qquad D_h = f_{max}\left(1 - e^{-f_{rate} (D_0 + 0.301 E v) / f_{max}}\right)$$
$$D = D_h + (D_{max} - D_h) \cdot \sigma\!\left(\frac{D_0' - K}{w}\right)$$

*   **Exposure** $E$ (stops) shifts the image up the exposure axis and brings the knee forward. The highlight branch takes only a fraction $v$ (`foot_veil`), so paper white does not fog by a full quarter-stop of density at +2.
*   **Snatch Point** sets the knee density $K = D_{max}(\text{knee}_{\text{lo}} - \text{knee}_{\text{span}}\cdot\text{snatch})$. A later snatch widens the black band.
*   **Abruptness** sets the knee width $w$; the sharpest end reaches $w \approx 0.03$.
*   **Color** comes from the paper: its $(a^{\ast}, b^{\ast})$ path on $u = D/D_{max}$ (Kong & Shore, *J. Imaging Sci. Technol.* 51(3), 2007), with anchors at $u = 0.10/0.35/0.65/1.00$: peach, ochre, **olive**, neutral. Keep the olive knot: warmtone lith goes green between warm highlights and cold blacks. $L^{\ast}$ comes from the density, so the RGB→Lab transform is skipped. The path is `PaperProfile.lith_path`.

Not modeled: semiquinone and bromide diffusion halos, pepper fog, snowballs and other faults.

### 7.2 Cyanotype

No development: the print depends on the light through the negative and on the sensitizer's density range, which is the contrast control (Ware). Midtones compress within that range, so the mid gamma is below one.

On $D_0$, with exposure $E$ in stops and exposure scale $S$ in log D:

$$t = \mathrm{clamp}\!\left(\frac{D_0 + 0.301 E}{S},\ 0,\ 1\right), \qquad v = 2t - 1$$
$$u_0 = (1-m)\,t + \tfrac{m}{2}\left(1 + v\,|v|\right)$$
$$u_b = u_0\left(1 - B\left(1 - \beta u_0\right)\right), \qquad u = u_b + T\left(r\,u_0 - u_b\right)$$
$$D = D_{max}\,(1 + T g)\,u$$

*   **Exposure** $E$: more UV drives more of the scale into blue.
*   **Exposure Scale** $S$: the negative density range printed, so the contrast. A short scale clips both ends of a normal negative.
*   Midtone compression is the reverse-S term, $m = 0.45$, written $v|v|$ so both engines evaluate it identically and blended with a straight line so the center slope is $1-m$ (a zero slope would posterize).
*   **Bleach** $B$ strips pigment highlights-first, leaving the deepest shadow at $\beta = 0.15$ of its density at full strength. **Tannin** $T$ re-develops the bleached iron as iron tannate, slightly past the blue ($r = 1.05$), with more covering power ($g = 0.15$).
*   **Color**: an $(a^{\ast}, b^{\ast})$ path on $u$, with anchors at $u = 0.00/0.15/0.55/1.00$: rag white, **green highlight stain**, mid blue, Prussian blue; $L^{\ast}$ from density. Tannin mixes toward an iron-tannate direction scaled by $u$, so a partial bleach splits blue-brown. **Sensitizer** picks $D_{max}$ and path: Classic (Herschel) 0.95 and greener, New (Ware) 1.40 and deeper.
*   No silver, so the six chemical toners are skipped. Split toning still applies.

Not modeled: **solarization**, bronzing, paper texture.

### 7.3 Sabattier

A print re-exposed to white light part-way through development. Dense tones hold; light tones reverse, the lighter the more, so the scale folds at the re-exposure density. Bromide from the developing dense areas restrains re-development of the adjoining light ones: the Mackie line, a light line along every edge between dark and light. On the print density $D$, with $R = \rho D_{max}$, strength $s$, fold width $w = 0.08$ and edge width $w_e = 0.10$ (`SABATTIER_CONSTANTS`):

$$g = \sigma\!\left(\frac{D - R}{w_e}\right), \qquad D' = D + s\,\big(1 - \ell\,\text{blur}_{\sigma_m}(g)\big)\, w \ln\!\big(1 + e^{(R - D)/w}\big)$$

*   **Strength** $s$ (0 to 2, default 1.3): below 1 the light tones flatten toward the fold, at 1 they sit on it, and above 1 they reverse. Paper white comes back at $sR$, clamped to $D_{max}$.
*   **Re-exposure** $\rho$ (0.1 to 0.9 of $D_{max}$, default 0.45): the density the fold sits at.
*   **Agitation** $a$ (0 to 1, default 0.7): $\sigma_m = (1 - a)$ % of the frame's short side, the Mackie lines' width; at 1 there are none ($\ell = 0$). The blur is the stage's one spatial term, a separable Gaussian in render pixels (`line_sigma_px`) whose taps come from `gaussian_kernel_1d` on both engines; a tiled export grows its halo by the kernel's half-width.
*   The print is neutral silver, so the toners act on it as on a plain print.

Not modeled: the warm second-development tone of chlorobromide papers, pseudo-solarization of the negative.

---

## 8. Toning
**Code**: `negpy.features.toning`

*   **Chemical Toning** (B&W Negative mode only): six baths (**Selenium**, **Sepia**, **Gold**, **Iron Blue**, **Copper**, **Vanadium Green**) modeled as a **silver ledger** in density space (`TONING_CONSTANTS`). The mean density $D_0$ is the silver reservoir; each toner converts a fraction, locked away from later baths (Rudman, Ilford).
    *   **Susceptibility**: $c_i$ is a pure function of $D_0$; sequence only decides who claims silver first.
        *   *Silver-proportional, shadows first*: Selenium $c = S \cdot (D_0/D_{ref})^{p}$. Iron Blue and Copper likewise, with $D_{ref} = 0.9$.
        *   *Bleach-limited, highlights first*: Sepia, Gold, Vanadium $c = S \cdot (1 - D_0/D_{ref})^{p}$.

        Strength $> 1$ is a longer bath; conversion caps at 1.
    *   **The ledger**: untoned fraction $a$ starts at 1. In bath order (selenium → sepia → gold → blue → copper → vanadium) each claims $f_i = a \cdot c_i$, $a \mathrel{-}= f_i$. **Gold** also plates the sulfide (sepia) fraction.
    *   **Covering power**: $D_{ch} = D_{0,ch} \cdot \left(a + \sum_i f_i \cdot g_{i,ch}\right)$, $I_{out} = 10^{-D_{ch}}$. Each channel uses its own input density, so existing color (a lith print) survives; on a gray print $D_{0,ch} = D_0$.

        Gain triplets $g_i$: selenium all $\ge 1$; sepia lower covering power; gold slight intensification; Prussian blue net $> 1$ with G at exactly $1.00$ (so sepia-plus-blue splits green); copper net $< 1$; vanadium R/B absorbed with slight density loss.

*   **On a lith print** (`LITH_TONING_CONSTANTS`, when the Lith stage is on): only two baths change:
    *   **Selenium**: $D_{ref} = 1.2$, exponent $1.0$; a green-led gain turns the green-black shadow magenta (Moersch), with a mean well above 1 for Dmax lift.
    *   **Gold**: flat $c = S$, gain blue-violet.
    *   Sepia, iron blue, copper and vanadium are unchanged.

*   **Split Toning** (all modes): an additive tint in LAB ($a^{\ast}b^{\ast}$). With $L$ the CIELAB lightness ($0$ to $100$):
    $$m_{shadow} = 1 - \text{smoothstep}(0,\ 50,\ L), \qquad m_{highlight} = \text{smoothstep}(50,\ 100,\ L)$$
    For each region, with hue $\theta$, strength $S$ and mask $m$:
    $$a^{\ast} \mathrel{+}= \cos\theta \cdot 20 \cdot S \cdot m, \qquad b^{\ast} \mathrel{+}= \sin\theta \cdot 20 \cdot S \cdot m$$

---

## 9. Finish
**Code**: `negpy.features.finish`

Post-crop print finishing in scene-linear, before the output transform. Order: edge burn → filed carrier; layout extras run at compositing time. The uncropped tool previews (crop, analysis region, tilt/swing) skip the filed carrier and the layout on both engines.

*   **Edge Burn (Vignette)**: $I_{out} = I \cdot 2^{-s \cdot m}$, with $s$ the burn in stops (negative = hold back) and $m$ a cosine falloff mask. **Roundness** morphs from radial to rectangular; **Size** sets the falloff midpoint.

*   **Filed Carrier**: the clear rebate prints between two boundaries, framed by an unexposed margin ($0.7 \cdot w$) in the mat color (`PrintService.effective_paper_linear`, scene-linear since this stage precedes the output OETF and the mat does not).

    `rebate_tone` pushes a neutral density ramp off the film base (the normalization's thin bound plus `REBATE_BASE_MARGIN` $= 0.15$ log units) through the frame's exposure kernel, saturation and toning into a 64-entry table over the exposure fraction $t$ that reaches the paper, sampled at $t = u^3$. The table is divided by its $t = 0$ entry and multiplies the mat color, so bare paper meets the mat exactly. With Paper Black on the rebate stops at the paper's D-max. Alternative processes, slides and flat masters use plain light ($1 - t$). Both engines read the one table (GPU storage buffer), so the shader has no print model.

    The picture side is the camera gate: it only wobbles (`carrier_profiles()` rows 0-3, no slider), prints soft and has a small fixed corner radius. The paper side is the filed aperture: **Roughness** swings it off rows 4-7, straight file strokes with raised-cosine nicks weighted toward the corners. **Corners** rounds only the aperture. The aperture is evaluated in its own frame, offset from the picture by `CARRIER_OFFSET_X/Y` of $w$, so the rebate is wider at the top and left. The aperture also gates the picture (weight $A_{in} A_{out}$). Profiles and offset are fixed-seed, so one carrier prints the whole roll.

    A fine **2-D fBm** (`carrier_noise`) displaces both boundaries' distance fields. It is hand-rolled hash value noise (u32 wrap-around only), so WGSL reproduces it bit for bit. Cell size scales with $w$.

    **Flare**: exposure added to $t$, peaking on each filed edge over $0.25 \cdot w$ and gated by the other three edges, so it stops at the aperture's corners. It reads through the same table.

*   **Layout extras** (`services/export/print.py` plus the `layout.wgsl` mirror): **bottom-weighted mat** and **match paper white** (mat color from paper white run through the toning stack).

*   **Export sharpening** (`apply_resize_sharpening`): a Print or Pixels export runs the Lab unsharp mask again on the scaled content, inside the mat, at a fixed $\sigma$ of `RESIZE_SHARPEN_RADIUS` exported pixels. It runs on the CPU after either engine's layout, so it needs no shader. An *Original* export and a flat master skip it.

*   **Preview paper size**: an interactive render sizes the paper from the preview long edge (`preview_render_size`), not the export DPI, and resamples the content to fit, never above its own resolution, because the canvas quotes zoom against the pipeline's buffer. The display shader magnifies instead.
