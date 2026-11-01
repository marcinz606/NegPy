"""Fade Restoration: the restoration matrix a Slide renders with, and the per-image estimate
of the dye survivals."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Sequence

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.system.i18n import mark, tr
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    effective_crosstalk_matrix,
    measure_neutral_axis_from_log,
)
from negpy.features.process.capture_color import camera_to_working_matrix, is_source_profile
from negpy.features.process.models import ProcessConfig
from negpy.features.process.path import RenderPath, render_path

#: Below this density spread a channel's neutral axis is too flat to be signal.
SPREAD_FLOOR = 0.02
#: Spreads within this fraction of the largest agree: no evidence of differential fade.
AGREEMENT_TOLERANCE = 0.05
#: Bound on an implied ratio, a reciprocal pair; the factor stays well conditioned.
RATIO_BOUNDS = (0.2, 5.0)
#: Bound on `fade_ratio_r`, a fraction of the original density, so it cannot exceed 1.0.
RED_SURVIVAL_BOUNDS = (0.05, 1.0)
#: Percentile of each channel's extremes for the frame's own bounds, read first.
COLOR_EXTREME_PCT = 0.00001
#: Below this confidence the first reading holds colored content, not grays.
MIN_CONFIDENCE = 0.2
#: The fallback bounds' percentile: a scan whose extreme cells are noise or scanner zeros sets no bounds.
ROBUST_EXTREME_PCT = 0.1
#: The fallback's own gate, stricter because its sector-free pass reads more colored content.
ROBUST_MIN_CONFIDENCE = 0.4
#: A cell at or below this log transmittance holds a decoded zero, not a density.
CLIPPED_LOG = -5.0
#: Percentile of the analysis grid that Remove Stain reads as each channel's white.
WHITE_PCT = 99.9
#: To suggest Remove Stain, the brightest channel's white is within this density of full scale.
STAIN_FULL_SCALE_TOLERANCE = 0.1
#: To suggest Remove Stain, blue is the darkest channel at the white, by at least this density.
STAIN_BLUE_MIN = 0.2
#: Remove Stain puts the white at least this log density below full scale, so clear areas do not clip.
FADE_WHITE_HEADROOM = 0.1


def resolve_fade_matrix(ratio_r: float, ratio_g: float, ratio_b: float) -> np.ndarray:
    """Restoration operator for a faded dye set: the inverse of diag(ar, ar*ag, ar*ab).

    Not row-normalized: fade changes each layer's neutral density, and that is the signal.
    `ratio_g` and `ratio_b` are survival ratios to red; `ratio_r` is red's absolute survival.
    """
    ar, ag, ab = float(ratio_r), float(ratio_g), float(ratio_b)
    return np.diag([1.0 / ar, 1.0 / (ar * ag), 1.0 / (ar * ab)])


def fade_matrix(process: ProcessConfig) -> Optional[np.ndarray]:
    """The restoration matrix, or None at unit survival or on a negative. A Positive slide takes
    it too, since a scanner file opens Positive."""
    ratios = (process.fade_ratio_r, process.fade_ratio_g, process.fade_ratio_b)
    if render_path(process) is RenderPath.PRINT or all(r == 1.0 for r in ratios):
        return None
    return resolve_fade_matrix(*ratios)


def fade_white_space(cam_xyz: Optional[Sequence[Sequence[float]]], camera_wb: Optional[Sequence[float]]) -> str:
    """The space a camera RAW's log white is measured in: its camera matrix with the as-shot white
    balance, which every RAW decode applies at the decode or in the matrix. "" for a source with no
    camera matrix, whose white is the file's own full scale."""
    if cam_xyz is None or is_source_profile(cam_xyz):
        return ""
    matrix = camera_to_working_matrix(cam_xyz, camera_wb)
    if matrix is None:
        return ""
    return hashlib.sha1(np.round(np.asarray(matrix, dtype=np.float64), 4).tobytes()).hexdigest()[:12]


def unmix_with_fade(process: ProcessConfig, process_mode: Optional[str], white_space: str = "") -> Optional[np.ndarray]:
    """The log-density matrix a render applies: the crosstalk unmix, then the fade restoration of
    the densities it recovers (`fade @ unmix`). None when neither applies. *white_space* is the
    render's `fade_white_space`. A stored white makes it a (3, 4) affine `M (L - w) + t`, the last
    column the log offset `t - M w`: Remove Stain maps the white to a neutral t; on a camera RAW,
    whose white sits below full scale, t = w keeps the white and scales density above it."""
    unmix = effective_crosstalk_matrix(process, process_mode)
    fade = fade_matrix(process)
    matrix = unmix if fade is None else (fade if unmix is None else fade @ unmix)
    white = _fade_white(process, white_space)
    if white is None:
        return matrix
    if process.fade_white_anchor:
        target = np.full(3, _white_target(white))
    elif white_space and matrix is not None:
        target = white
    else:
        return matrix
    linear = np.eye(3) if matrix is None else matrix
    return np.hstack([linear, (target - linear @ white)[:, None]])


def _fade_white(process: ProcessConfig, white_space: str = "") -> Optional[np.ndarray]:
    """The stored per-channel white (log transmittance) when it applies to this render, else None.
    A white measured in another space than the render's reads the wrong channels, so it does not apply."""
    if render_path(process) is RenderPath.PRINT or process.fade_white_space != white_space:
        return None
    white = np.array([process.fade_white_r, process.fade_white_g, process.fade_white_b], dtype=np.float64)
    return white if bool(np.any(white != 0.0)) else None


def _white_target(white: np.ndarray) -> float:
    """The cleanest channel's white, capped FADE_WHITE_HEADROOM below full scale, so a dim scan keeps its exposure."""
    return min(float(white.max()), -FADE_WHITE_HEADROOM)


def fade_white_target(process: ProcessConfig, white_space: str = "") -> Optional[float]:
    """The neutral log level Remove Stain maps the stored white to in this render, else None."""
    white = _fade_white(process, white_space) if process.fade_white_anchor else None
    return None if white is None else _white_target(white)


def fade_white_stale(process: ProcessConfig, white_space: str) -> bool:
    """Whether a camera RAW's fade needs its white measured again: Fade Restoration or Remove Stain
    acts, and the stored white was measured in another space (or never) for this render."""
    if not white_space or render_path(process) is RenderPath.PRINT or process.fade_white_space == white_space:
        return False
    return process.fade_white_anchor or fade_matrix(process) is not None


#: The return shape of `measure_neutral_axis_from_log`: midtone, shadow, highlight or None, confidence.
NeutralAxisRefs = tuple[tuple[float, float, float], tuple[float, float, float], object, float]


def measure_neutral_axis_ratios(grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reason) from the transparency meter's grid, metered against this frame's own
    bounds: the fixed transfer window finds no grays in a faded slide. `refs` is None, with a
    reason, when no trustworthy neutral axis exists."""
    return neutral_axis_from_grid(grid)


def neutral_axis_from_grid(grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
    """`measure_neutral_axis_ratios` on a prefiltered log grid."""
    refs = measure_neutral_axis_from_log(grid, _measured_slide_bounds(grid, COLOR_EXTREME_PCT), None, 0.0)
    if refs is not None and refs[3] >= MIN_CONFIDENCE:
        return refs, ""
    # A faded slide's grays carry the fade's cast, which can put them in the foliage or sky
    # sector; the unfiltered set is the last try.
    bounds = _measured_slide_bounds(grid, ROBUST_EXTREME_PCT)
    for exclude in (True, False):
        refs = measure_neutral_axis_from_log(grid, bounds, None, 0.0, exclude_subjects=exclude)
        if refs is not None and refs[3] >= ROBUST_MIN_CONFIDENCE:
            return refs, ""
    return None, tr("No reliable grays found on this frame.")


def _measured_slide_bounds(grid: ImageBuffer, extreme_pct: float = COLOR_EXTREME_PCT) -> LogNegativeBounds:
    """This frame's own bounds in the slide orientation (floor above ceiling): the luma span
    from the base clip, each channel's offset from its extreme percentiles. The negative
    analyzer does not fit, since its dense end is chroma-gated for an orange mask."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    cells = grid.reshape(-1, 3)
    unclipped = cells[(cells > CLIPPED_LOG).all(axis=1)]
    if len(unclipped):
        cells = unclipped

    def _pct(p: float) -> list:
        return [float(np.percentile(cells[:, ch], p)) for ch in range(3)]

    base = float(EXPOSURE_CONSTANTS["base_luma_clip"])
    luma_f, luma_c = _pct(100.0 - base), _pct(base)
    color_f, color_c = _pct(100.0 - extreme_pct), _pct(extreme_pct)
    mean_f, mean_c = sum(luma_f) / 3.0, sum(luma_c) / 3.0
    mid_f, mid_c = sorted(color_f)[1], sorted(color_c)[1]
    floors = tuple(mean_f + (color_f[ch] - mid_f) for ch in range(3))
    ceils = tuple(mean_c + (color_c[ch] - mid_c) for ch in range(3))
    return LogNegativeBounds(floors, ceils)


def measure_white(grid: ImageBuffer) -> Optional[tuple[float, float, float]]:
    """Each channel's white: the log transmittance of its brightest cells in the analysis grid, or
    None with no cell left. A cell at full scale in every channel is light past the film (a mount gap,
    a leak or a blown specular), and a decoded zero is no reading, so neither counts."""
    cells = grid.reshape(-1, 3)
    cells = cells[(cells > CLIPPED_LOG).all(axis=1) & (cells < 0.0).any(axis=1)]
    if not len(cells):
        return None
    r, g, b = (float(v) for v in np.percentile(cells, WHITE_PCT, axis=0))
    return r, g, b


def white_anchored_grid(grid: ImageBuffer, white: tuple[float, float, float]) -> np.ndarray:
    """The grid as density above each channel's white, the space Remove Stain renders in.
    A cell brighter than the white stops at full scale, as a decode of the divided scan would."""
    return np.minimum(grid - np.asarray(white, dtype=np.float32), 0.0).astype(np.float32)


def stain_suggested(white: tuple[float, float, float]) -> bool:
    """Whether the white reads as a yellow stain on clear film: the brightest channel near full
    scale, and blue clearly the darkest. A dim scan with no white object never qualifies."""
    brightest = max(white)
    return brightest >= -STAIN_FULL_SCALE_TOLERANCE and white[2] <= min(white[0], white[1]) and brightest - white[2] >= STAIN_BLUE_MIN


class EstimateMethod(StrEnum):
    """MIDTONE reads the midtone grays' own densities. SPREAD reads each channel's midtone-to-shadow
    spread, which cancels a per-channel density offset but follows a flat, noisy shadow."""

    MIDTONE = "midtone"
    SPREAD = "spread"


@dataclass(frozen=True)
class FadeEstimate:
    """Survivals from one Estimate. red is None when the method leaves Red Survival as it is;
    reason is "" for a real estimate (see fade_ratios_from_neutral_axis). white is the frame's
    measured white, None when the frame is not a slide or has no cell to read it from."""

    red: Optional[float]
    green: float
    blue: float
    reason: str
    white: Optional[tuple[float, float, float]] = None


def fade_ratios_from_neutral_axis(refs: Optional[NeutralAxisRefs], method: str = EstimateMethod.SPREAD) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from neutral references made by `measure_neutral_axis_ratios`.

    Ratios are relative to red, the fade matrix's reference channel. `reason` is "" for a real
    estimate, else the fail-closed condition that fired, with 1.0 returned for both."""
    if refs is None:
        return 1.0, 1.0, tr("No grays to measure.")
    mid, shadow = refs[0], refs[1]
    if method == EstimateMethod.MIDTONE:
        # The refs are log transmittance, so density is the negative.
        spreads = tuple(-float(mid[ch]) for ch in range(3))
    else:
        spreads = tuple(float(mid[ch]) - float(shadow[ch]) for ch in range(3))
    r, g, b = spreads
    abs_spreads = (abs(r), abs(g), abs(b))
    if min(abs_spreads) < SPREAD_FLOOR:
        return 1.0, 1.0, tr("The grays in one channel are too flat to measure.")
    if max(abs_spreads) - min(abs_spreads) <= AGREEMENT_TOLERANCE * max(abs_spreads):
        return 1.0, 1.0, tr("The three layers faded equally, so there is nothing to estimate.")
    ratio_g = g / r
    ratio_b = b / r
    lo, hi = RATIO_BOUNDS
    if not (lo <= ratio_g <= hi) or not (lo <= ratio_b <= hi):
        clamped_g = min(max(ratio_g, lo), hi)
        clamped_b = min(max(ratio_b, lo), hi)
        sentences = [
            tr(template, raw=f"{raw:.1f}", lo=f"{lo:g}", hi=f"{hi:g}", clamped=f"{c:.1f}")
            for template, raw, c in (
                (mark("Green read {raw}, outside {lo} to {hi}, so it is set to {clamped}."), ratio_g, clamped_g),
                (mark("Blue read {raw}, outside {lo} to {hi}, so it is set to {clamped}."), ratio_b, clamped_b),
            )
            if raw != c
        ]
        if ratio_g > hi or ratio_b > hi:
            sentences.append(tr("Red faded more than this range covers. Lower Red Survival by eye."))
        reason = " ".join(sentences)
        return clamped_g, clamped_b, reason
    return ratio_g, ratio_b, ""


def estimate_fade(grid: ImageBuffer, white_anchor: bool = False, camera_raw: bool = False) -> dict[EstimateMethod, FadeEstimate]:
    """Both methods' estimates from one measurement of the grays. `grid` is the transparency
    meter's grid with the crosstalk unmix, no fade and no Remove Stain (`TransparencyBaseProcessor`).
    MIDTONE also sets Red Survival so the best-kept layer lands at 1.0; a ratio above 1 says red
    faded most. With `white_anchor`, or on a `camera_raw` whose render scales density above its
    white (`unmix_with_fade`), the grays are read as density above the frame's white."""
    white = measure_white(grid)
    if white_anchor and white is None:
        return {m: FadeEstimate(None, 1.0, 1.0, tr("No clear area to measure the stain on this frame.")) for m in EstimateMethod}
    anchored = white is not None and (white_anchor or camera_raw)
    refs, reason = neutral_axis_from_grid(white_anchored_grid(grid, white) if anchored else grid)
    out = {}
    for method in EstimateMethod:
        if refs is None:
            out[method] = FadeEstimate(None, 1.0, 1.0, reason, white)
            continue
        g, b, why = fade_ratios_from_neutral_axis(refs, method)
        red = None
        if method == EstimateMethod.MIDTONE and not (why and g == 1.0 and b == 1.0):
            lo, hi = RED_SURVIVAL_BOUNDS
            red = min(max(1.0 / max(g, b, 1.0), lo), hi)
        out[method] = FadeEstimate(red, g, b, why, white)
    return out
