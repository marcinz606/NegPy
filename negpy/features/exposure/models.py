from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict


class RenderIntent(StrEnum):
    """
    How the Print stage renders the positive.

    PRINT — the full photographic-paper look (the default NegPy conversion).
    FLAT  — a low-contrast, neutral "digital intermediate" master intended for
            further editing in Lightroom/Darktable/Photoshop. The mask-neutralized
            inversion is kept, but the creative print decisions (auto density/grade,
            cast removal, toe/shoulder) and the downstream creative
            stages (lab, local, toning, finish) are bypassed so maximal tonal and
            colour information is preserved with gentle highlight/shadow roll-off.
    """

    PRINT = "print"
    FLAT = "flat"


@dataclass(frozen=True)
class ExposureConfig:
    """
    Print parameters (Density, Grade, Color).
    """

    density: float = 1.0
    grade: float = 115.0
    # Per-layer contrast trims in ISO-R points (crossover correction).
    grade_trim_red: float = 0.0
    grade_trim_green: float = 0.0
    grade_trim_blue: float = 0.0
    wb_cyan: float = 0.0
    wb_magenta: float = 0.0
    wb_yellow: float = 0.0
    shadow_cyan: float = 0.0
    shadow_magenta: float = 0.0
    shadow_yellow: float = 0.0
    highlight_cyan: float = 0.0
    highlight_magenta: float = 0.0
    highlight_yellow: float = 0.0
    # Neutral zone density offsets (ΔD, achromatic): + = denser = darker print.
    # Slider ranges are asymmetric — an equal ΔD reads far smaller near d_max
    # than near d_min (density is log10).
    shadow_density: float = 0.0
    highlight_density: float = 0.0
    # Split grade: zone contrast in ISO-R points (negative = harder), global
    # value + per-layer trims like Grade.
    shadow_grade: float = 0.0
    highlight_grade: float = 0.0
    shadow_grade_trim_red: float = 0.0
    shadow_grade_trim_green: float = 0.0
    shadow_grade_trim_blue: float = 0.0
    highlight_grade_trim_red: float = 0.0
    highlight_grade_trim_green: float = 0.0
    highlight_grade_trim_blue: float = 0.0
    toe: float = 0.0
    toe_width: float = 2.5
    shoulder: float = 0.0
    shoulder_width: float = 2.5
    # Per-layer knee trims on top of the global toe/shoulder (endpoint crossover).
    toe_trim_red: float = 0.0
    toe_trim_green: float = 0.0
    toe_trim_blue: float = 0.0
    shoulder_trim_red: float = 0.0
    shoulder_trim_green: float = 0.0
    shoulder_trim_blue: float = 0.0
    # Per-layer knee width trims (roll-off extent, sharpness crossover).
    toe_width_trim_red: float = 0.0
    toe_width_trim_green: float = 0.0
    toe_width_trim_blue: float = 0.0
    shoulder_width_trim_red: float = 0.0
    shoulder_width_trim_green: float = 0.0
    shoulder_width_trim_blue: float = 0.0
    paper_dmin: bool = False
    # Black point compensation: map paper Dmax to display black.
    true_black: bool = True
    # Additive trim on the paper's variable midtone gamma (tanh S-curve).
    midtone_gamma: float = 0.0
    # Per-layer Snap trims on top of the global midtone gamma (midtone crossover).
    midtone_gamma_trim_red: float = 0.0
    midtone_gamma_trim_green: float = 0.0
    midtone_gamma_trim_blue: float = 0.0
    cast_removal_strength: float = 0.5
    auto_exposure: bool = True
    auto_normalize_contrast: bool = True
    render_intent: str = RenderIntent.PRINT
    paper_profile: str = "neutral"

    def __post_init__(self) -> None:
        """
        Legacy migration: grade used to be a 0-5 paper-grade number
        (ladder R = 150 - 20*G). Real ISO R values start at 50, so any
        stored value <= 5 is unambiguously legacy — convert it with the old
        ladder so previously saved edits keep their rendered look.
        """
        if self.grade <= 5.0:
            object.__setattr__(self, "grade", 150.0 - 20.0 * self.grade)
        # Legacy: cast_removal was a bool toggle; MIGRATIONS renames the key, coerce its value.
        if isinstance(self.cast_removal_strength, bool):
            object.__setattr__(self, "cast_removal_strength", 1.0 if self.cast_removal_strength else 0.0)


EXPOSURE_CONSTANTS: Dict[str, Any] = {
    # Max absolute density offset applied by CMY white-balance sliders (slider ±1 → ±this density).
    # ↑ widens colour-balance range per slider unit; ↓ narrows it.
    "cmy_max_density": 0.2,
    # Scales the density slider's effect on the exposure pivot.
    # ↑ density slider moves the midtone more aggressively; ↓ gentler response.
    "density_multiplier": 0.2,
    # Target density where the reference tone (assumed_anchor) should print on paper.
    # ↑ reference tone prints darker; ↓ reference tone prints brighter.
    "anchor_target_density": 0.74,
    # Zone Density (ΔD) weights: mid-sparing sigmoids centred in the three-quarter/
    # quarter tones (offsets from anchor_target_density), so midtones get neither
    # offset. Mirrored as literals in exposure.wgsl — change both together.
    "zone_density_sharpness": 4.0,
    "zone_density_shadow_offset": 0.75,
    "zone_density_highlight_offset": -0.40,
    # Default normalized midtone reference in [0,1] log space (used when auto_exposure=False).
    # ↑ curve pivots brighter (assumes denser negative); ↓ pivots darker.
    "assumed_anchor": 0.46,
    # Minimum ISO R paper exposure range (hardest/highest-contrast grade allowed).
    # ↑ raises the maximum achievable slope; ↓ allows even harder grades.
    "iso_r_min": 50.0,
    # Maximum ISO R paper exposure range (softest/lowest-contrast grade allowed).
    # ↑ lowers the minimum achievable slope; ↓ forces a higher contrast floor.
    "iso_r_max": 180.0,
    # Hard floor on the per-channel straight-line slope k.
    # ↑ prevents very flat curves; ↓ allows lower contrast.
    "slope_min": 2.0,
    # Hard ceiling on the per-channel straight-line slope k.
    # ↑ allows steeper (higher-contrast) curves; ↓ caps maximum contrast.
    "slope_max": 10.0,
    # Physical paper black density (maximum density, D_max).
    # ↑ deeper blacks; ↓ lighter shadow floor.
    "d_max": 2.3,
    # Physical paper white density (minimum density, D_min / paper base).
    # ↑ denser paper white (slightly compressed highlights); ↓ purer paper white.
    "d_min": 0.06,
    # Global multiplier applied to both toe and shoulder slider values before the curve.
    # ↑ amplifies slider sensitivity (more roll-off per unit); ↓ dampens it.
    "toe_shoulder_strength": 0.85,
    # ── Asymmetric H&D print curve (toe-linear-shoulder) ─────────────────────
    # Straight-line midtone of slope k flanked by independently-tunable toe
    # (shadow roll-off toward paper black d_max) and shoulder (highlight roll-off
    # toward paper white d_min), each a smooth softplus bound — the film/print
    # convention (toe = shadows, shoulder = highlights). Sharpness comes from the
    # *_sharpness_base / width, the slider sets roll-off *height*.
    # Softplus sharpness coefficient for the shadow (toe) knee: a_sh = this * width_ref / toe_width.
    # ↑ snappier shadow transition; ↓ softer, more gradual roll-off to paper black.
    "toe_sharpness_base": 4.0,
    # Softplus sharpness coefficient for the highlight (shoulder) knee: a_hl = this * width_ref / shoulder_width.
    # ↑ snappier highlight transition; ↓ softer roll-off to paper white.
    "shoulder_sharpness_base": 3.0,
    # Reference width used to normalise both sharpness coefficients (units match slider range).
    # ↑ both knees sharpen for a given width slider value; ↓ both soften.
    "toeshoulder_width_ref": 2.5,
    # D the toe/shoulder slider lowers the black ceil / lifts the highlight floor
    # per unit (pre-scaled by toe_shoulder_strength). +toe = lifted blacks;
    # +shoulder = compressed (greyer) highlights.
    # Density lift of the paper-black ceiling per positive toe unit: d_max_eff = d_max − toe·this.
    # ↑ toe slider lifts blacks more aggressively; ↓ gentler shadow lift.
    # Larger than shoulder_height: density is log10, so a ΔD near d_max is
    # perceptually far smaller than the same ΔD near d_min — this evens out the
    # toe vs shoulder slider strength in L*.
    "toe_height": 0.90,
    # Density lift of the paper-white floor per positive shoulder unit: d_min_eff = d_min + shoulder·this.
    # ↑ shoulder slider compresses highlights more per unit; ↓ gentler compression.
    "shoulder_height": 0.35,
    # Grade -> straight-line slope k: k = grade_contrast_scale * density_range / er
    # (er = ISO R / 100). Calibrated so R115 reproduces the legacy mid-curve slope.
    # Calibration factor in k = this · density_range / (ISO_R/100).
    # ↑ all grades produce higher slopes (more contrast); ↓ flatter curves system-wide.
    "grade_contrast_scale": 2.9,
    # Side length of the block-median pre-filter grid for robust exposure analysis.
    # ↑ finer grid (less dust/specular rejection); ↓ coarser (stronger outlier rejection).
    "analysis_grid": 1024,
    # Base percentile clip added to the luma-range histogram analysis (robust floor/ceil detection).
    # ↑ clips more histogram tails (tighter black/white points); ↓ uses fuller histogram range.
    "base_luma_clip": 0.01,
    # Colour Clip neutral/default percentile: robust per-tail clip for per-channel
    # balance (orange-mask cast removal), independent of luma range. The slider spans
    # log-interpolated percentiles around this neutral.
    # Default neutral percentile for per-channel colour clip / cast-removal analysis.
    # ↑ more outlier-resistant balance detection; ↓ more relaxed (includes more extreme tones).
    "base_color_clip": 1.0,
    # Percentile used to sample per-channel shadow references for cast detection.
    # ↑ samples even darker shadow tones (closer to paper black); ↓ lighter reference tones.
    "shadow_neutral_percentile": 98.0,
    # Scan-exposure warning: linear level treated as sensor-white clipping (film base
    # and scene shadows live near sensor white in a negative scan, so clipped pixels
    # collapse distinct densities to D=0).
    "scan_clip_level": 0.99,
    # Per-channel clipped fraction above which the Analysis panel warns.
    "scan_clip_warn": 0.01,
    # Cast Removal: max normalized shadow cast (green - channel) corrected, bounding the tilt.
    # Hard clamp on automatic per-channel slope tilt during cast removal.
    # ↑ allows stronger shadow neutralization; ↓ limits correction (less risk of overcorrection).
    "cast_removal_max_offset": 0.1,
    # Cast Removal neutral axis: per-channel refs at a highlight/midtone/shadow luma band, each
    # over the band's lowest-chroma pixels (relative quantile). R/B fit green's axis — a quadratic
    # through all three, else a line through mid+shadow. Bands are normalized luma.
    "neutral_axis_highlight_band": (0.10, 0.30),
    "neutral_axis_mid_band": (0.40, 0.60),
    "neutral_axis_shadow_band": (0.72, 0.92),
    # Lowest-chroma fraction of each band kept as the near-neutral set.
    "neutral_axis_chroma_quantile": 0.30,
    # Above this median chroma the set isn't trustworthy -> fall back to the shadow-only tie.
    "neutral_axis_chroma_cap": 0.35,
    "neutral_axis_min_pixels": 64,
    # Clamp on each channel's deviation from green at any anchor (generous: refs are clean).
    "midtone_cast_max_offset": 0.2,
    # Curvature clamp (fraction of slope, <0.5): keeps the per-channel core monotonic on [0,1].
    "neutral_axis_curv_max_ratio": 0.45,
    # Percentile of scene luminance sampled as the raw metered anchor.
    # ↑ samples darker histogram tones as key; ↓ samples brighter tones.
    "anchor_meter_percentile": 50.0,
    # Safety band around assumed_anchor that clamps the auto-metered result.
    # ↑ allows wider exposure swing between frames; ↓ tighter, more conservative auto-exposure.
    "anchor_meter_band": 0.12,
    # Auto Density: fraction the anchor moves from the assumed key toward the measured median.
    # Fraction of the distance from assumed_anchor toward the metered anchor that is applied.
    # ↑ auto-exposure responds more strongly to measured key; ↓ stays closer to assumed anchor.
    "anchor_meter_strength": 0.2,
    # Grade-coupled baseline toe/shoulder: hard grades (high slope) get more roll-off by default.
    # Adds slope-proportional toe to hard grades: toe_eff += this · slope_norm.
    # ↑ hard grades get more automatic shadow roll-off; ↓ decouples toe from grade.
    # 0.15 · (0.35 / 0.90): holds the baseline ΔD (this · toe_height) at its
    # calibrated value, independent of the perceptual toe_height.
    "toe_grade_strength": 0.15 * 0.35 / 0.90,
    # Adds slope-proportional shoulder to hard grades: shoulder_eff += this · slope_norm.
    # ↑ hard grades compress highlights more automatically; ↓ decouples shoulder from grade.
    "shoulder_grade_strength": 0.12,
    # Auto Grade nominal-frame contrast = auto_grade_target * auto_grade_nominal_ratio.
    # Target contrast multiplier for Auto Grade: effective_range = this · blend(nominal, measured_ratio).
    # ↑ aims for higher printed contrast across all frames; ↓ targets lower contrast.
    "auto_grade_target": 0.6,
    # Auto Grade adaptation strength (partial slope normalization): 0 = fixed, 1 = full.
    # How strongly Auto Grade adapts slope to scene range (0 = ignore scene, 1 = fully normalize).
    # ↑ grade changes more aggressively with scene contrast variation; ↓ closer to a fixed grade.
    "auto_grade_strength": 0.6,
    # Canonical floor_ceil/textural ratio of a normal tone distribution (~2.0); default-range fallback.
    # Reference floor_ceil/textural ratio for a "normal" negative (used as Auto Grade blend anchor).
    # ↑ system treats denser negatives as normal (grades down harder frames); ↓ expects flatter negatives.
    "auto_grade_nominal_ratio": 2.0,
    # Percentile margin for measuring the "textural" scene range (rejects specular highlights and dust).
    # ↑ includes more histogram (wider textural range); ↓ tighter (more robust to extreme outliers).
    "textural_range_clip": 10.0,
    # ── Flat / digital-intermediate master (RenderIntent.FLAT) ──────────────
    # A true log-video master: the normalized log signal is emitted directly as the
    # code value (positive-oriented 1 - val), with NO 10^-D decode and NO sRGB OETF,
    # so the result is flat/milky and fully invertible for downstream editing.
    # code = clip(flat_log_lift + flat_log_gain*(1 - val), 0, 1). Fixed (no per-frame
    # metering) so a roll of equally-exposed scans renders identically.
    # Log-master contrast (range of code values used); <1 keeps it flat.
    # ↑ more contrast (less editing headroom); ↓ flatter, milkier, more latitude.
    "flat_log_gain": 0.65,
    # Code value the scene shadow (val=1) lands on; the black/shadow lift.
    # ↑ greyer shadows (more lift); ↓ deeper shadows in the master.
    "flat_log_lift": 0.10,
    # ── Variable-gamma paper S-curve ─────────────────────────────────────────
    # Extra local gamma added at the midtone centre (around the reference tone) via
    # v += gamma·width·tanh((v − v_star)/width), easing to zero toward toe/shoulder —
    # a real paper characteristic curve's continuously varying gamma. Anchor-preserving.
    # Extra midtone gamma at the curve centre (0 disables the S-shape).
    # ↑ snappier midtones (more contrast around the reference tone); ↓ closer to a straight line.
    "paper_midtone_gamma": 0.15,
    # Density half-width over which the midtone gamma boost eases to the tails.
    # ↑ wider, more gradual S; ↓ tighter, more localized midtone boost.
    "paper_gamma_width": 0.6,
}
