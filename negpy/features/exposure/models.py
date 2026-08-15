from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, Tuple


class RenderIntent(StrEnum):
    """
    How the Print stage renders the positive.

    PRINT: the full photographic-paper look (the default conversion).
    FLAT:  a low-contrast master for further editing elsewhere. Keeps the
           mask-neutralized inversion, bypasses the creative print decisions
           (auto density/grade, cast removal, toe/shoulder) and the lab, local,
           toning and finish stages.
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
    # Ranges are asymmetric: density is log10, so an equal ΔD reads smaller near d_max.
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
    # On shows the paper's natural Dmax as a lifted black. Consumed inverted as
    # bpc (= not paper_black), so the default keeps black point compensation on.
    paper_black: bool = False
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
    # Density-domain saturation, composed into the dye_mix kernel slot (see
    # papers.py: resolve_saturation_matrix). 1.0 = identity; trims are per-layer.
    dye_separation: float = 1.0
    dye_separation_trim_red: float = 0.0
    dye_separation_trim_green: float = 0.0
    dye_separation_trim_blue: float = 0.0
    # Tapers dye_separation by each pixel's own chroma (see
    # logic.separation_damping_gain). Inert at dye_separation 1.0: it only
    # redistributes that slider's push.
    separation_damping: float = 0.0

    def __post_init__(self) -> None:
        """
        Legacy: grade was a 0-5 paper grade (R = 150 - 20*G). ISO R starts at
        50, so a stored value <= 5 is legacy. Convert it with the old ladder.
        """
        if self.grade <= 5.0:
            object.__setattr__(self, "grade", 150.0 - 20.0 * self.grade)
        # Legacy: cast_removal was a bool toggle; MIGRATIONS renames the key, coerce its value.
        if isinstance(self.cast_removal_strength, bool):
            object.__setattr__(self, "cast_removal_strength", 1.0 if self.cast_removal_strength else 0.0)


EXPOSURE_CONSTANTS: Dict[str, Any] = {
    # Max absolute density offset per CMY white-balance slider unit.
    "cmy_max_density": 0.2,
    # Scales the density slider's effect on the exposure pivot.
    "density_multiplier": 0.2,
    # Density the reference tone (assumed_anchor) prints at.
    "anchor_target_density": 0.75,
    # Zone Density (ΔD) weights: mid-sparing sigmoids centred in the three-quarter/
    # quarter tones (offsets from anchor_target_density), so midtones get neither
    # offset. Mirrored as literals in exposure.wgsl: change both together.
    "zone_density_sharpness": 4.0,
    "zone_density_shadow_offset": 0.75,
    "zone_density_highlight_offset": -0.40,
    # Default normalized midtone reference in [0,1] log space (auto_exposure off).
    "assumed_anchor": 0.46,
    # ISO R bounds: hardest and softest grade allowed.
    "iso_r_min": 50.0,
    "iso_r_max": 180.0,
    # Bounds on the per-channel straight-line slope k.
    "slope_min": 2.0,
    "slope_max": 10.0,
    # Physical paper black (D_max) and paper white (D_min) densities.
    "d_max": 2.3,
    "d_min": 0.06,
    # Global multiplier on the toe and shoulder slider values.
    "toe_shoulder_strength": 0.85,
    # Asymmetric H&D print curve: a straight midtone of slope k between a toe
    # (shadow roll-off to d_max) and a shoulder (highlight roll-off to d_min),
    # each a smooth softplus bound. The sliders set roll-off *height*; the
    # *_sharpness_base / width set sharpness.
    # a_sh = toe_sharpness_base * width_ref / toe_width (shoulder is the same).
    "toe_sharpness_base": 4.0,
    "shoulder_sharpness_base": 3.0,
    # Reference width that normalises both sharpness coefficients.
    "toeshoulder_width_ref": 2.5,
    # Density moved per slider unit: d_max_eff = d_max - toe*this,
    # d_min_eff = d_min + shoulder*this. toe_height is larger because density is
    # log10, so a ΔD near d_max reads smaller in L* than the same ΔD near d_min.
    "toe_height": 0.90,
    "shoulder_height": 0.35,
    # k = grade_contrast_scale * density_range / (ISO_R/100). Calibrated so R115
    # reproduces the legacy mid-curve slope.
    "grade_contrast_scale": 2.9,
    # Side length of the block-median pre-filter grid for exposure analysis.
    "analysis_grid": 1024,
    # Base percentile clip on the luma-range histogram analysis.
    "base_luma_clip": 0.01,
    # Neutral per-tail clip for per-channel balance (orange-mask cast removal),
    # independent of luma range. The slider spans percentiles around this.
    "base_color_clip": 1.0,
    # Percentile sampled as the per-channel shadow reference for cast detection.
    "shadow_neutral_percentile": 98.0,
    # Scan-exposure warning: linear level treated as sensor-white clipping. Film
    # base and scene shadows sit near sensor white, so clipped pixels collapse
    # distinct densities to D=0.
    "scan_clip_level": 0.99,
    # Per-channel clipped fraction above which the Analysis panel warns.
    "scan_clip_warn": 0.01,
    # Max normalized shadow cast (green - channel) that Cast Removal corrects.
    "cast_removal_max_offset": 0.1,
    # Cast Removal neutral axis: per-channel refs at a highlight/midtone/shadow luma
    # band, each over the band's lowest-chroma pixels. R/B fit green's axis with a
    # quadratic through all three, else a line through mid+shadow. Bands are
    # normalized luma.
    "neutral_axis_highlight_band": (0.10, 0.30),
    "neutral_axis_mid_band": (0.40, 0.60),
    "neutral_axis_shadow_band": (0.72, 0.92),
    # Lowest-chroma fraction of each band kept as the near-neutral set.
    "neutral_axis_chroma_quantile": 0.30,
    # Above this median corrected chroma (pass 2) the set is not trustworthy:
    # fall back to the shadow-only tie.
    "neutral_axis_chroma_cap": 0.29,
    # Pass-1 chroma ceiling: admits strong correctable casts, rejects saturated content.
    "neutral_axis_first_pass_cap": 0.55,
    "neutral_axis_min_pixels": 64,
    # Confidence sample-size half-point: the size term is n / (n + this).
    "neutral_axis_confidence_n0": 256,
    # Mid/shadow deviation-difference dead zone (a plausible crossover passes free)
    # and the roll-off width of the confidence agreement term beyond it.
    "neutral_axis_agreement_deadzone": 0.10,
    "neutral_axis_agreement_scale": 0.20,
    # Width (percentile points) of the luma-extreme band the same-pixel color
    # floor refs read; Color Clip sets the band's depth.
    "color_bounds_band_width": 4.0,
    # Clamp on each channel's deviation from green at any anchor.
    "midtone_cast_max_offset": 0.2,
    # Curvature clamp (fraction of slope, <0.5): keeps the per-channel core
    # monotonic on [0,1].
    "neutral_axis_curv_max_ratio": 0.45,
    # Percentile of scene luminance sampled as the raw metered anchor.
    "anchor_meter_percentile": 50.0,
    # Safety band around assumed_anchor that clamps the auto-metered result.
    "anchor_meter_band": 0.12,
    # Auto Density: fraction of the distance from assumed_anchor toward the
    # metered anchor that is applied.
    "anchor_meter_strength": 0.2,
    # Grade-coupled baseline roll-off: toe_eff += this * slope_norm, so hard
    # grades get more toe. The 0.35/0.90 factor holds the baseline ΔD
    # (this * toe_height) constant against the perceptual toe_height.
    "toe_grade_strength": 0.15 * 0.35 / 0.90,
    "shoulder_grade_strength": 0.12,
    # Auto Grade: effective_range = auto_grade_target * blend(nominal, measured_ratio).
    "auto_grade_target": 0.6,
    # Auto Grade adaptation strength: 0 = fixed grade, 1 = full slope normalization.
    "auto_grade_strength": 0.5,
    # floor_ceil/textural ratio of a normal negative; the Auto Grade blend anchor.
    "auto_grade_nominal_ratio": 2.0,
    # Percentile margin for the "textural" scene range (rejects speculars and dust).
    "textural_range_clip": 10.0,
    # Flat / digital-intermediate master (RenderIntent.FLAT). A log-video master:
    # the normalized log signal becomes the code value directly, with no 10^-D
    # decode and no sRGB OETF, so it stays flat and fully invertible.
    # code = clip(flat_log_lift + flat_log_gain*(1 - val), 0, 1). Fixed, with no
    # per-frame metering, so a roll of equal scans renders identically.
    # Log-master contrast; <1 keeps it flat.
    "flat_log_gain": 0.65,
    # Code value the scene shadow (val=1) lands on.
    "flat_log_lift": 0.10,
    # Variable-gamma paper S-curve. Extra local gamma at the midtone centre via
    # v += gamma*width*tanh((v - v_star)/width), easing to zero toward the toe and
    # shoulder, like a real paper curve. Anchor-preserving.
    # Extra midtone gamma at the centre (0 disables the S-shape).
    "paper_midtone_gamma": 0.15,
    # Density half-width over which that boost eases to the tails.
    "paper_gamma_width": 0.6,
    # Chroma (RMS dye-density spread above paper base) that Separation Damping
    # leaves untouched: below it a pixel takes the full Dye Separation push, above
    # it the inverse. Both ends are failure modes: too high and every pixel takes
    # the same push (the frame-wide matrix again), too low and the crossing leaves
    # the picture. Inlined as a WGSL literal: change both.
    "separation_damping_ref_spread": 0.35,
}

# Auto Density / Auto Grade targets the user can retune (Set Targets dialog).
# App-global, not per-image: one calibration per install. key -> (min, max).
TUNABLE_TARGETS: Dict[str, Tuple[float, float]] = {
    "anchor_target_density": (0.4, 1.1),
    "anchor_meter_strength": (0.0, 1.0),
    "anchor_meter_band": (0.0, 0.4),
    "auto_grade_target": (0.3, 0.8),
    "auto_grade_strength": (0.0, 1.0),
}
DEFAULT_TARGETS: Dict[str, float] = {k: float(EXPOSURE_CONSTANTS[k]) for k in TUNABLE_TARGETS}

# Render caches key on config hashes, which don't see a global-constant edit.
# Both engines fold this into the exposure stage's key so it re-runs.
TARGETS_REVISION = 0


def apply_targets(values: Dict[str, float]) -> None:
    """Overlay user target overrides onto EXPOSURE_CONSTANTS; invalidates render caches."""
    global TARGETS_REVISION
    EXPOSURE_CONSTANTS.update({k: float(v) for k, v in values.items() if k in TUNABLE_TARGETS})
    TARGETS_REVISION += 1
