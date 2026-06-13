from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ExposureConfig:
    """
    Print parameters (Density, Grade, Color).
    """

    density: float = 1.0
    grade: float = 115.0
    linear_raw: bool = False
    wb_cyan: float = 0.0
    wb_magenta: float = 0.0
    wb_yellow: float = 0.0
    shadow_cyan: float = 0.0
    shadow_magenta: float = 0.0
    shadow_yellow: float = 0.0
    highlight_cyan: float = 0.0
    highlight_magenta: float = 0.0
    highlight_yellow: float = 0.0
    toe: float = 0.0
    toe_width: float = 2.5
    shoulder: float = 0.0
    shoulder_width: float = 2.5
    paper_dmin: bool = True
    flare: bool = False
    auto_shadow_neutral: bool = True
    auto_exposure: bool = True
    auto_normalize_contrast: bool = True

    def __post_init__(self) -> None:
        """
        Legacy migration: grade used to be a 0-5 paper-grade number
        (ladder R = 150 - 20*G). Real ISO R values start at 50, so any
        stored value <= 5 is unambiguously legacy — convert it with the old
        ladder so previously saved edits keep their rendered look.
        """
        if self.grade <= 5.0:
            object.__setattr__(self, "grade", 150.0 - 20.0 * self.grade)


EXPOSURE_CONSTANTS: Dict[str, Any] = {
    "cmy_max_density": 0.2,
    "density_multiplier": 0.2,
    "anchor_target_density": 0.74,
    "assumed_anchor": 0.46,
    "iso_r_min": 50.0,
    "iso_r_max": 180.0,
    "slope_min": 2.0,
    "slope_max": 11.0,
    "d_max": 2.3,
    "d_min": 0.06,
    "curve_asymptote": 2.7,
    "dmax_shoulder": 5.0,
    "paper_toe_nu": 3.0,
    "toe_onset_density": 1.2,
    "toe_shoulder_strength": 0.85,
    "analysis_grid": 1024,
    "base_drange_clip": 0.01,
    "shadow_neutral_percentile": 97.5,
    "shadow_neutral_max_offset": 0.125,
    "anchor_meter_percentile": 50.0,
    "anchor_meter_band": 0.12,
    # Auto Density partial metering: the anchor moves only this fraction from the
    # assumed key toward the measured median. Full re-centering forces every frame
    # to mid-key (dark negs print too bright, bright negs too dark), so it is kept
    # gentle to preserve the scene's intended key.
    "anchor_meter_strength": 0.25,
    # Auto Grade base contrast: the effective grade range for a nominal frame is
    # auto_grade_target * auto_grade_nominal_ratio. Larger = punchier overall.
    "auto_grade_target": 0.6,
    # Adaptation strength of Auto Grade (partial normalization, like an
    # auto-printer's slope control). 0 = fixed contrast (ignore the scene),
    # 1 = fully hold printed textural contrast constant — which overcorrects
    # (flat scenes go harsh, contrasty scenes go muddy). ~0.4 leans contrast
    # toward the scene without printing to a hard extreme.
    "auto_grade_strength": 0.4,
    # Nominal floor_ceil / textural ratio of a well-exposed frame: pure
    # percentile geometry of a roughly-normal tone distribution (P10-P90 =
    # 2.56 sigma, P0.5-P99.5 = 5.15 sigma -> ~2.0). Used to derive the default
    # grade range (auto_grade_target * this) when no per-frame textural range is
    # measured, so the fallback obeys the same printed-contrast rule instead of a
    # standalone magic number.
    "auto_grade_nominal_ratio": 2.0,
    # Preferred midtone print gamma for an average viewing surround
    # (Bartleson-Breneman: ~1.05-1.15, not 1:1). The design intent that
    # auto_grade_target is calibrated against; documentation only.
    "target_system_gamma": 1.10,
    "textural_range_clip": 10.0,
    "auto_density_target_offset": 0.0,
    # Veiling-glare / print-flare floor: a uniform light added to print
    # reflectance, out = (r + f) / (1 + f) with r normalized to paper white.
    # Lifts the deepest blacks and softens the toe (film look) while leaving
    # paper white fixed. The amount applied when the Flare toggle is on
    # (ExposureConfig.flare); classic order ~0.005-0.02.
    "flare_fraction": 0.005,
}
