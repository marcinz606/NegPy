"""Fade Restoration: the restoration matrix a Slide renders with, and the per-image estimate
of the dye survivals."""

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple, Optional, Sequence

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.system.i18n import mark, tr
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    effective_crosstalk_matrix,
    measure_neutral_axis_from_log,
)
from negpy.features.process.capture_color import camera_matrix_parts, is_source_profile
from negpy.features.process.logic import effective_highlight_reconstruction
from negpy.features.process.models import ProcessConfig
from negpy.features.process.path import RenderPath, render_path

#: Below this midtone density a channel's grays are too flat to be signal.
SPREAD_FLOOR = 0.02
#: Densities within this fraction of the largest agree: no evidence of differential fade.
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
STAIN_FULL_SCALE_TOLERANCE = 0.2
#: To suggest Remove Stain, blue's white is at least this density below both red's and green's.
STAIN_BLUE_MIN = 0.08
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


def fade_white_space(
    cam_xyz: Optional[Sequence[Sequence[float]]],
    camera_wb: Optional[Sequence[float]],
    crosstalk: Optional[np.ndarray] = None,
    highlight_mode: int = 0,
) -> str:
    """The space a camera RAW's log white is measured in: its white-balanced camera channels
    (`density_channels`) after the *crosstalk* unmix, named by the as-shot white balance, the unmix
    and the decode's *highlight_mode*, which rebuilds the clipped pixels the white is read from. The
    primaries after them do not name it. "" for a source with no camera matrix, whose white is the
    file's own full scale."""
    if cam_xyz is None or is_source_profile(cam_xyz):
        return ""
    gains, primaries = camera_matrix_parts(cam_xyz, camera_wb)
    if primaries is None:
        return ""
    named = b"" if gains is None else np.round(np.diag(gains).astype(np.float64), 4).tobytes()
    if crosstalk is not None:
        named += b"unmix" + np.round(np.asarray(crosstalk, dtype=np.float64), 4).tobytes()
    named += b"highlight" + int(highlight_mode).to_bytes(1, "little")
    return hashlib.sha1(_WHITE_SPACE_CHANNELS + named).hexdigest()[:12]


def render_white_space(
    process: ProcessConfig, process_mode: Optional[str], cam_xyz: Optional[Sequence[Sequence[float]]], camera_wb: Optional[Sequence[float]]
) -> str:
    """`fade_white_space` of a render with *process*: its crosstalk unmix and highlight mode included,
    "" on a negative, which takes no fade."""
    if render_path(process) is RenderPath.PRINT:
        return ""
    crosstalk = effective_crosstalk_matrix(process, process_mode)
    return fade_white_space(cam_xyz, camera_wb, crosstalk, effective_highlight_reconstruction(process))


# Names the channels in `fade_white_space`, so a tag hashed any other way reads as stale.
_WHITE_SPACE_CHANNELS = b"camera-gains"


class FadeDensity(NamedTuple):
    """`fade_density`'s result."""

    matrix: Optional[np.ndarray]
    anchor: Optional[tuple[float, float, float]]


def fade_density(process: ProcessConfig, process_mode: Optional[str], white_space: str = "") -> FadeDensity:
    """The log-density matrix a render applies, `F @ U` (the fade after the crosstalk unmix), and
    the anchor it scales about. A stored white w, read after U, makes the matrix the affine
    `F (U L - w) + t` about `a = U^-1 w`. The anchor is a camera RAW's a, else None
    (docs/PIPELINE.md, Fade restoration). `fade_density_shift` adds density to every channel after it."""
    density = _fade_density(process, process_mode, white_space)
    shift = float(process.fade_density_shift)
    if shift == 0.0 or render_path(process) is RenderPath.PRINT:
        return density
    m = np.eye(3) if density.matrix is None else np.asarray(density.matrix, dtype=np.float64)
    affine = m.copy() if m.shape[1] == 4 else np.hstack([m, np.zeros((3, 1))])
    affine[:, 3] -= shift
    return FadeDensity(affine, density.anchor)


def _fade_density(process: ProcessConfig, process_mode: Optional[str], white_space: str) -> FadeDensity:
    unmix = effective_crosstalk_matrix(process, process_mode)
    fade = fade_matrix(process)
    matrix = unmix if fade is None else (fade if unmix is None else fade @ unmix)
    terms = _white_terms(process, white_space, fade is not None)
    if terms is None:
        return FadeDensity(matrix, None)
    white, target = terms
    a = white if unmix is None else _unmixed_point(unmix, white)
    linear = np.eye(3) if matrix is None else matrix
    affine = np.hstack([linear, (target - linear @ a)[:, None]])
    return FadeDensity(affine, (float(a[0]), float(a[1]), float(a[2])) if white_space else None)


def _unmixed_point(unmix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """The log point that *unmix* maps to *point*; *point* itself when *unmix* has no inverse."""
    try:
        return np.linalg.solve(np.asarray(unmix, dtype=np.float64), point)
    except np.linalg.LinAlgError:
        return point


def _white_terms(process: ProcessConfig, white_space: str, fade_acts: bool) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(white, target) of the stored white's affine, both after the crosstalk unmix, or None when no
    white acts. *fade_acts*: Fade Restoration applies. Crosstalk alone keeps a neutral neutral, so a
    camera RAW needs no anchor for it."""
    white = _fade_white(process, white_space)
    if white is None:
        return None
    if process.fade_white_anchor:
        return white, np.full(3, _white_target(white))
    if white_space and fade_acts:
        level = np.asarray(density_anchor(white, False), dtype=np.float64)
        return level, level
    return None


def density_anchor(white: Sequence[float], remove_stain: bool) -> tuple[float, float, float]:
    """The log point a camera RAW's fade scales density about: each channel's white with Remove Stain,
    else the white's cleanest channel on all three. A neutral point keeps a gray neutral when the white
    is a colored highlight, which is all a slide shows when it holds no clear film."""
    if remove_stain:
        return (float(white[0]), float(white[1]), float(white[2]))
    level = float(max(white))
    return (level, level, level)


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
    acts, and the stored white was measured in another space (or never)."""
    if not white_space or render_path(process) is RenderPath.PRINT or process.fade_white_space == white_space:
        return False
    return process.fade_white_anchor or fade_matrix(process) is not None


#: A white a render measured for itself: (r, g, b, fade_white_space).
MeasuredWhite = tuple[float, float, float, str]


def resolve_fade_white(process: ProcessConfig, white_space: str, grid: Callable[[], ImageBuffer]) -> Optional[MeasuredWhite]:
    """The white a render measures on the Estimate *grid* when the stored one is stale, else None.
    Each engine measures it before the density stage, so a frame renders alike everywhere."""
    if not fade_white_stale(process, white_space):
        return None
    r, g, b = measure_white(grid()) or (0.0, 0.0, 0.0)
    return (r, g, b, white_space)


def with_fade_white(process: ProcessConfig, white: Optional[MeasuredWhite]) -> ProcessConfig:
    """*process* carrying a white `resolve_fade_white` measured; unchanged with none."""
    if white is None:
        return process
    return replace(process, fade_white_r=white[0], fade_white_g=white[1], fade_white_b=white[2], fade_white_space=white[3])


#: The return shape of `measure_neutral_axis_from_log`: midtone, shadow, highlight or None, confidence.
NeutralAxisRefs = tuple[tuple[float, float, float], tuple[float, float, float], object, float]


def tone_band_refs(grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
    """The tone bands' grays as neutral-axis refs: the share-weighted neutral centre of the shadow
    bands (1 and 2), the midtone bands (3 and 4) and the highlight bands (5 and 6) of `neutral_bands`."""
    bands = neutral_bands(grid)

    def group(*index: int) -> Optional[tuple[float, float, float]]:
        if bands is None:
            return None
        sel = np.isin(bands.index, index)
        if not sel.any():
            return None
        c = (bands.share[sel, None] * bands.centre[sel]).sum(axis=0) / bands.share[sel].sum()
        return (float(c[0]), float(c[1]), float(c[2]))

    mid, shadow, highlight = group(3, 4), group(1, 2), group(5, 6)
    if mid is None or shadow is None:
        return None, tr("No reliable grays found on this frame.")
    return (mid, shadow, highlight, 1.0), ""


def neutral_axis_from_grid(grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reason) from the transparency meter's prefiltered log grid, metered against this
    frame's own bounds: the fixed transfer window finds no grays in a faded slide. `refs` is None,
    with a reason, when no trustworthy neutral axis exists."""
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

    base = float(EXPOSURE_CONSTANTS["base_luma_clip"])
    qs = [100.0 - base, base, 100.0 - extreme_pct, extreme_pct]
    luma_f, luma_c, color_f, color_c = ([float(v) for v in p] for p in negpy_core.slide_extremes(_cells(grid), CLIPPED_LOG, qs))
    mean_f, mean_c = sum(luma_f) / 3.0, sum(luma_c) / 3.0
    mid_f, mid_c = sorted(color_f)[1], sorted(color_c)[1]
    floors = tuple(mean_f + (color_f[ch] - mid_f) for ch in range(3))
    ceils = tuple(mean_c + (color_c[ch] - mid_c) for ch in range(3))
    return LogNegativeBounds(floors, ceils)


def measure_white(grid: ImageBuffer) -> Optional[tuple[float, float, float]]:
    """Each channel's white: the log transmittance of its brightest cells in the analysis grid, or
    None with no cell left. A cell at full scale in every channel is light past the film (a mount gap,
    a leak or a blown specular), and a decoded zero is no reading, so neither counts. A camera RAW's
    white-balance gains lift a channel past full scale where another channel clipped, so a channel
    stops at full scale, as the decode with Highlight Recovery does."""
    white = negpy_core.fade_white(_cells(grid), CLIPPED_LOG, WHITE_PCT)
    if white is None:
        return None
    r, g, b = (float(v) for v in white)
    return r, g, b


def _cells(grid: ImageBuffer) -> np.ndarray:
    """*grid* as the contiguous float32 (..., 3) array the fade kernels read."""
    return np.ascontiguousarray(grid, dtype=np.float32)


def white_anchored_grid(grid: ImageBuffer, white: tuple[float, float, float]) -> np.ndarray:
    """The grid as density above each channel's white, the space Remove Stain renders in.
    A cell brighter than the white stops at full scale, as a decode of the divided scan would."""
    return np.minimum(grid - np.asarray(white, dtype=np.float32), 0.0).astype(np.float32)


def stain_suggested(white: tuple[float, float, float]) -> bool:
    """Whether the white reads as a yellow stain on clear film: the brightest channel near full
    scale, and blue below both red and green. A red cast leaves blue near green, and a dim scan
    with no white object never qualifies."""
    return max(white) >= -STAIN_FULL_SCALE_TOLERANCE and min(white[0], white[1]) - white[2] >= STAIN_BLUE_MIN


@dataclass(frozen=True)
class FadeEstimate:
    """Survivals from one Estimate. red is None when no estimate was made;
    reason is "" for a real estimate (see fade_ratios_from_neutral_axis). white is the frame's
    measured white, None when the frame is not a slide or has no cell to read it from."""

    red: Optional[float]
    green: float
    blue: float
    reason: str
    white: Optional[tuple[float, float, float]] = None
    #: The crossover table measured after these survivals (`measure_crossover`), None with none.
    crossover: Optional[tuple[float, ...]] = None


def fade_ratios_from_neutral_axis(refs: Optional[NeutralAxisRefs]) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from the midtone grays' own densities in neutral references made by
    `neutral_axis_from_grid`. Ratios are relative to red, the fade matrix's reference channel.
    `reason` is "" for a real estimate, else the fail-closed condition that fired, with 1.0 for both."""
    if refs is None:
        return 1.0, 1.0, tr("No grays to measure.")
    # The refs are log transmittance, so density is the negative.
    spreads = tuple(-float(refs[0][ch]) for ch in range(3))
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


def _blend_refs(a: NeutralAxisRefs, b: NeutralAxisRefs, mix: float) -> NeutralAxisRefs:
    def lerp(x, y):
        return tuple(float((1.0 - mix) * u + mix * v) for u, v in zip(x, y))

    highlight = lerp(a[2], b[2]) if a[2] is not None and b[2] is not None else (a[2] if mix < 0.5 else b[2])
    return lerp(a[0], b[0]), lerp(a[1], b[1]), highlight, float((1.0 - mix) * a[3] + mix * b[3])


def gray_refs(grid: ImageBuffer, mix: float = 0.0) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reason): Estimate's grays, each ref moved in log by *mix* from the gray finder (0) to
    the tone bands (1)."""
    mix = min(max(float(mix), 0.0), 1.0)
    if mix <= 0.0:
        return neutral_axis_from_grid(grid)
    if mix >= 1.0:
        return tone_band_refs(grid)
    return blend_gray_readings(gray_readings(grid), mix)


#: Both readings `gray_refs` blends: (the gray finder's, the tone bands'), each (refs, reason).
GrayReadings = tuple[tuple[Optional[NeutralAxisRefs], str], tuple[Optional[NeutralAxisRefs], str]]


def gray_readings(grid: ImageBuffer) -> GrayReadings:
    """Both readers' grays on *grid*. They do not depend on Mix, so a Gray Mix drag reads them once."""
    return neutral_axis_from_grid(grid), tone_band_refs(grid)


def blend_gray_readings(readings: GrayReadings, mix: float) -> tuple[Optional[NeutralAxisRefs], str]:
    """`gray_refs` from *readings*: each ref moved in log by *mix*. Where the first reader finds no
    grays, the second's act at every mix (`toward_no_change` scales their estimate)."""
    (ours, why), (theirs, why_nc) = readings
    mix = min(max(float(mix), 0.0), 1.0)
    if ours is None and theirs is not None:
        return theirs, why_nc
    if mix <= 0.0:
        return ours, why
    if mix >= 1.0:
        return theirs, why_nc
    if theirs is None:
        return ours, why
    return _blend_refs(ours, theirs, mix), ""


def tone_bands_only(readings: GrayReadings) -> bool:
    """Whether only the tone bands find grays in *readings*."""
    return readings[0][0] is None and readings[1][0] is not None


def estimate_red(green: float, blue: float) -> float:
    """Estimate's Red Survival for *green* and *blue*: the best-kept layer at 1."""
    lo, hi = RED_SURVIVAL_BOUNDS
    return min(max(1.0 / max(green, blue, 1.0), lo), hi)


def toward_no_change(estimate: FadeEstimate, mix: float) -> FadeEstimate:
    """*estimate* with its green and blue raised to *mix*, so 0 keeps every survival at 1 and 1 is
    the estimate. Red follows as Estimate sets it; the crossover table holds only at 1."""
    if estimate.red is None:
        return estimate
    mix = min(max(float(mix), 0.0), 1.0)
    if mix >= 1.0:
        return estimate
    g, b = estimate.green**mix, estimate.blue**mix
    return replace(estimate, red=estimate_red(g, b), green=g, blue=b, crossover=None)


def estimate_fade_with(
    grid: ImageBuffer,
    reader: Callable[[ImageBuffer], tuple[Optional[NeutralAxisRefs], str]],
    white_anchor: bool = False,
    camera_raw: bool = False,
    white_space: str = "",
    crossover: bool = True,
) -> FadeEstimate:
    """The estimate from the grays *reader* finds on `grid`, the transparency meter's grid with the
    crosstalk unmix, no fade and no Remove Stain (`TransparencyBaseProcessor`). Red Survival puts
    the best-kept layer at 1.0; a ratio above 1 says red faded most. With `white_anchor`, or on a
    `camera_raw` whose render scales density above its white, the grays are read above that point
    (`density_anchor`). It carries the crossover table measured after its survivals, zero at its
    midtone; *crossover* False leaves the table out, the slowest step, for a live slider."""
    white = measure_white(grid)
    if white_anchor and white is None:
        return FadeEstimate(None, 1.0, 1.0, tr("No clear area to measure the stain on this frame."))
    camera_raw = camera_raw or bool(white_space)
    refs, reason = reader(_gray_grid(grid, white, white_anchor, camera_raw))
    if refs is None:
        return FadeEstimate(None, 1.0, 1.0, reason, white)
    g, b, why = fade_ratios_from_neutral_axis(refs)
    red = None
    if not (why and g == 1.0 and b == 1.0):
        red = estimate_red(g, b)
    table = None
    if red is not None and crossover:
        table = _table_after(grid, refs, white, white_anchor, camera_raw, white_space, (red, g, b))
    return FadeEstimate(red, g, b, why, white, table)


def _table_after(
    grid: ImageBuffer,
    refs: NeutralAxisRefs,
    white: Optional[tuple[float, float, float]],
    white_anchor: bool,
    camera_raw: bool,
    white_space: str,
    survivals: tuple[float, float, float],
    shift: float = 0.0,
) -> Optional[tuple[float, ...]]:
    """The crossover table measured on the Estimate *grid* after *survivals* and *shift*, zero at the
    midtone of *refs* (read above the anchor when one acts) through the same fade."""
    density = survival_density(*survivals, white, white_anchor, camera_raw, white_space, shift)
    mid = np.asarray(refs[0], dtype=np.float32).reshape(1, 1, 3)
    if white is not None and (white_anchor or camera_raw):
        mid = mid + np.asarray(density_anchor(white, white_anchor), dtype=np.float32)
    zero = float(_mean_log(through_density(mid, density).reshape(1, 3).astype(np.float64))[0])
    return measure_crossover(through_density(grid, density), zero)


def _gray_grid(grid: ImageBuffer, white: Optional[tuple[float, float, float]], white_anchor: bool, camera_raw: bool) -> ImageBuffer:
    """*grid* as the grays are read: above the render's anchor when one acts."""
    if white is not None and (white_anchor or camera_raw):
        return white_anchored_grid(grid, density_anchor(white, white_anchor))
    return grid


def survival_crossover(
    grid: ImageBuffer,
    reader: Callable[[ImageBuffer], tuple[Optional[NeutralAxisRefs], str]],
    survivals: tuple[float, float, float],
    white_anchor: bool = False,
    camera_raw: bool = False,
    white_space: str = "",
    shift: float = 0.0,
) -> Optional[tuple[float, ...]]:
    """The table Estimate would store with *survivals* and *shift* set by hand: the same grid, gray
    reader and zero, so a survival change keeps the Gray Mix the table was read with."""
    white = measure_white(grid)
    if white_anchor and white is None:
        return None
    camera_raw = camera_raw or bool(white_space)
    refs, _ = reader(_gray_grid(grid, white, white_anchor, camera_raw))
    return None if refs is None else _table_after(grid, refs, white, white_anchor, camera_raw, white_space, survivals, shift)


# Crossover table: the per-density lean of the neutrals that the survivals leave. Ported from
# NegativeConverter's expired-film rescue (MIT, Copyright (c) 2025 lex; NOTICE.md) and moved from
# the display-encoded positive to log density after the fade matrix.

#: Band edges over the frame's own density range, dense end first.
CROSSOVER_BAND_EDGES = (0.0, 0.06, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
#: Percentiles of the weighted mean density that set the frame's own range.
CROSSOVER_RANGE_PCT = (0.5, 99.5)
#: Channel weights of the mean density. Equal weights keep each pixel's mean dye density, the
#: neutral the transfer curve renders; the dye channels carry no luma.
CROSSOVER_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
#: Channel spread (D) at half neutral weight: the source's 0.12 display units at a mid gray.
CROSSOVER_NEUTRAL_SPREAD = 0.25
#: Mean-shift radius (D, Chebyshev): the source's 0.06 display units at a mid gray.
CROSSOVER_CLUSTER_RADIUS = 0.12
#: Largest channel offset (D), about one stop: a larger residual says the survivals are wrong.
CROSSOVER_LIMIT = 0.3
#: A band needs this share of the total neutral weight and this much weight to count.
CROSSOVER_MIN_SHARE = 0.004
CROSSOVER_MIN_WEIGHT = 8.0
#: Below this density span the frame has no range to band.
CROSSOVER_MIN_SPAN = 0.05
CROSSOVER_MEAN_SHIFT_STEPS = 3
#: Bins of the per-pixel offset table between the outer band anchors.
CROSSOVER_BINS = 64
#: Fewest readings `neutral_bands` measures on.
CROSSOVER_MIN_CELLS = 64
#: Values per band in a stored table: mean log, share, then the lean of each channel.
CROSSOVER_BAND_SIZE = 5


#: Below this spread (D) of any band's lean away from the reference the frame has no crossover to fix:
#: the source's 0.02 display units for naming a shadow or highlight cast, at a mid gray.
CROSSOVER_FLOOR = 0.04


#: `negpy_core.neutral_bands`' numbers in its order, the last the readings a grid needs.
_BAND_CONSTS = (
    list(CROSSOVER_BAND_EDGES[1:-1]),
    list(CROSSOVER_WEIGHTS),
    CROSSOVER_RANGE_PCT,
    CROSSOVER_MIN_SPAN,
    CROSSOVER_NEUTRAL_SPREAD,
    CROSSOVER_CLUSTER_RADIUS,
    CROSSOVER_MEAN_SHIFT_STEPS,
    CROSSOVER_MIN_WEIGHT,
    CLIPPED_LOG,
    CROSSOVER_MIN_CELLS,
)


def _mean_log(cells: np.ndarray) -> np.ndarray:
    w = np.asarray(CROSSOVER_WEIGHTS, dtype=np.float64)
    return cells @ w


class NeutralBands(NamedTuple):
    """Each band's near-neutral centre (log, per channel), its mean log and its share of the neutral
    weight, for the bands with enough weight, dense end first."""

    centre: np.ndarray
    level: np.ndarray
    share: np.ndarray
    index: np.ndarray


def neutral_bands(grid: ImageBuffer) -> Optional[NeutralBands]:
    """The neutral centre of each density band of a log *grid*: soft neutral weight, then mean-shift
    onto each band's largest near-neutral cluster. None with too few cells."""
    found = negpy_core.neutral_bands(_cells(grid), _BAND_CONSTS)
    if found is None:
        return None
    centre, weight, total = found
    share = weight / total
    keep = ~np.isnan(centre[:, 0]) & (share >= CROSSOVER_MIN_SHARE) & (weight >= CROSSOVER_MIN_WEIGHT)
    if not keep.any():
        return None
    kept = centre[keep]
    return NeutralBands(kept, _mean_log(kept), share[keep], np.nonzero(keep)[0])


def measure_crossover(grid: ImageBuffer, zero: Optional[float] = None) -> Optional[tuple[float, ...]]:
    """The lean of the neutrals band by band on a log *grid* after the fade matrix, as a flat tuple:
    the mean log the table is zero at, then (mean log, share, lean r, g, b) per band. A lean is the
    band's neutral centre minus its mean log, so it sums to zero under CROSSOVER_WEIGHTS. *zero* is
    the Estimate's midtone; without it the midtone of the grid's own grays, else the share-weighted
    mean log."""
    bands = neutral_bands(grid)
    if bands is None:
        return None
    if zero is None:
        refs, _ = neutral_axis_from_grid(grid)
        zero = float(_mean_log(np.asarray(refs[0], dtype=np.float64)[None, :])[0]) if refs is not None else None
    if zero is None:
        zero = float((bands.share * bands.level).sum() / bands.share.sum())
    out: list[float] = [round(zero, 5)]
    for centre, level, share in zip(bands.centre, bands.level, bands.share):
        out.extend(round(float(v), 5) for v in (level, share, *(centre - level)))
    return tuple(out)


def crossover_bands(table: Sequence[float]) -> np.ndarray:
    """A stored table's bands as rows of (mean log, share, lean r, g, b), dense end first."""
    return np.asarray(table[1:], dtype=np.float64).reshape(-1, CROSSOVER_BAND_SIZE)


def crossover_zero(table: Sequence[float]) -> float:
    """The mean log a stored table is zero at: the Estimate's midtone."""
    return float(table[0])


def crossover_reference(table: Sequence[float]) -> np.ndarray:
    """The lean at the table's zero, where the survivals were fit: the cast they own."""
    rows = crossover_bands(table)
    x = np.array([crossover_zero(table)])
    return np.array([_shape_cubic(rows[:, 0], rows[:, 2 + c], x)[0] for c in range(3)])


def crossover_amount(table: Optional[Sequence[float]]) -> float:
    """The largest channel spread of a band's lean away from the reference, in D."""
    if not table:
        return 0.0
    residual = crossover_bands(table)[:, 2:] - crossover_reference(table)
    return float((residual.max(axis=1) - residual.min(axis=1)).max())


#: The Gray Mix values Estimate tries (`pick_neutrals`).
NEUTRALS_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def neutral_spread(table: Sequence[float]) -> float:
    """The share-weighted channel spread of a table's neutral bands, a cast included: how far from
    gray the survivals leave the frame's grays."""
    rows = crossover_bands(table)
    spread = rows[:, 2:].max(axis=1) - rows[:, 2:].min(axis=1)
    return float((rows[:, 1] * spread).sum() / rows[:, 1].sum())


def pick_neutrals(
    grid: ImageBuffer, readings: GrayReadings, white_anchor: bool = False, camera_raw: bool = False, white_space: str = ""
) -> Optional[tuple[float, FadeEstimate]]:
    """(mix, estimate) for the NEUTRALS_CANDIDATES value whose estimate leaves the least
    `neutral_spread` in its crossover table, the first on a tie; None when none estimates. The
    candidates run on threads, as negpy_core releases the GIL for the table's work."""
    from concurrent.futures import ThreadPoolExecutor

    def estimate(mix: float) -> FadeEstimate:
        return estimate_fade_with(grid, lambda _: blend_gray_readings(readings, mix), white_anchor, camera_raw, white_space)

    with ThreadPoolExecutor(len(NEUTRALS_CANDIDATES)) as pool:
        estimates = list(pool.map(estimate, NEUTRALS_CANDIDATES))
    best: Optional[tuple[float, float, FadeEstimate]] = None
    for mix, est in zip(NEUTRALS_CANDIDATES, estimates):
        if est.red is None or est.crossover is None:
            continue
        score = neutral_spread(est.crossover)
        if best is None or score < best[0] - 1e-6:
            best = (score, mix, est)
    return None if best is None else (best[1], best[2])


def crossover_casts(table: Optional[Sequence[float]]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(shadow, highlight): the share-weighted lean beyond the reference of the dense bands (edges
    0.06 to 0.3) and the light bands (0.6 to 0.9), as `describeExpiredAnalysis` groups them."""
    if not table:
        return None, None
    rows = crossover_bands(table)
    residual = rows[:, 2:] - crossover_reference(table)
    # The stored rows hold only kept bands, so the groups go by mean log within the stored range.
    t = (rows[:, 0] - rows[0, 0]) / max(rows[-1, 0] - rows[0, 0], 1e-12)

    def group(lo: float, hi: float) -> Optional[np.ndarray]:
        sel = (t >= lo) & (t <= hi)
        if not sel.any():
            return None
        return (rows[sel, 1:2] * residual[sel]).sum(axis=0) / rows[sel, 1].sum()

    return group(0.0, 0.35), group(0.6, 1.0)


#: Hue names in `castName`'s order around the hue circle.
_HUE_NAMES = (mark("red"), mark("yellow"), mark("green"), mark("cyan"), mark("blue"), mark("magenta"))


def cast_name(lean: Optional[np.ndarray]) -> Optional[str]:
    """The hue a lean in log transmittance tilts toward, translated, as NegativeConverter's `castName`
    names an RGB offset; None under CROSSOVER_FLOOR. More light in a channel is more of its color."""
    if lean is None:
        return None
    r, g, b = (float(v) for v in lean)
    hi, lo = max(r, g, b), min(r, g, b)
    chroma = hi - lo
    if chroma < CROSSOVER_FLOOR:
        return None
    if hi == r:
        hue = ((g - b) / chroma) % 6
    elif hi == g:
        hue = (b - r) / chroma + 2
    else:
        hue = (r - g) / chroma + 4
    return tr(_HUE_NAMES[int(math.floor(hue + 0.5)) % 6])


def crossover_description(table: Sequence[float]) -> str:
    """One line on what a table found: the crossover amount, then the hue the shadows and the
    highlights lean toward beyond the midtone."""
    shadow, highlight = crossover_casts(table)
    size = crossover_amount(table)
    amount = tr("Crossover {amount} D.", amount=f"{size:.2f}")
    if size < CROSSOVER_FLOOR:
        return amount + " " + tr("Too small to correct.")
    parts = []
    shadow_name, highlight_name = cast_name(shadow), cast_name(highlight)
    if shadow_name:
        parts.append(tr("Shadows lean {hue}.", hue=shadow_name))
    if highlight_name:
        parts.append(tr("Highlights lean {hue}.", hue=highlight_name))
    return " ".join([amount, *parts])


def _shape_cubic(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic through (xs, ys), flat beyond the ends."""
    n = len(xs)
    if n == 1:
        return np.full_like(x, ys[0], dtype=np.float64)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    d = np.zeros(n)
    d[0], d[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0.0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    xc = np.clip(x, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, n - 2)
    t = (xc - xs[i]) / h[i]
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * d[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * d[i + 1]


class CrossoverOffsets(NamedTuple):
    """The per-pixel offset table: CROSSOVER_BINS rows of log offsets between mean logs lo and hi."""

    bins: np.ndarray
    lo: float
    hi: float


def crossover_offsets(table: Optional[Sequence[float]], crossover: float) -> Optional[CrossoverOffsets]:
    """The offsets a render adds in log, -(lean - ref) * crossover, where ref is the
    lean at the table's zero, or None when nothing moves or the crossover is under CROSSOVER_FLOOR.
    Each row sums to zero under CROSSOVER_WEIGHTS and is scaled down as a whole to CROSSOVER_LIMIT, so
    the correction keeps the mean density and its own hue."""
    if not table or crossover == 0.0 or crossover_amount(table) < CROSSOVER_FLOOR:
        return None
    rows = crossover_bands(table)
    ref = crossover_reference(table)
    xs = rows[:, 0]
    lo, hi = float(xs[0]), float(xs[-1])
    # Bands are dense end first, so the anchors already rise in mean log.
    x = np.linspace(lo, hi, CROSSOVER_BINS) if hi > lo else np.full(CROSSOVER_BINS, lo)
    lean = np.stack([_shape_cubic(xs, rows[:, 2 + c], x) for c in range(3)], axis=1)
    offset = -(lean - ref) * crossover
    offset -= _mean_log(offset)[:, None]
    peak = np.abs(offset).max(axis=1, keepdims=True)
    offset *= np.minimum(1.0, CROSSOVER_LIMIT / np.maximum(peak, 1e-12))
    if float(np.abs(offset).max()) <= 1e-4:
        return None
    return CrossoverOffsets(offset.astype(np.float32), lo, hi)


def apply_crossover(img_log: ImageBuffer, offsets: Optional[CrossoverOffsets]) -> ImageBuffer:
    """*img_log* plus the offset row its own mean log selects, linear between bins. Float32 in the
    shader's order (normalization.wgsl), so the engines agree."""
    if offsets is None:
        return img_log
    img = np.ascontiguousarray(img_log, dtype=np.float32)
    bins = np.ascontiguousarray(offsets.bins, dtype=np.float32)
    # The span in float32, as the shader subtracts the uniform's lo and hi.
    lo = np.float32(offsets.lo)
    span = np.float32(offsets.hi) - lo
    return negpy_core.apply_crossover(img, bins, float(lo), float(span))


def crossover_uniform(offsets: Optional[CrossoverOffsets]) -> bytes:
    """The shader's table: (lo, hi, on, 0) then CROSSOVER_BINS rows of vec4, zeros when off."""
    if offsets is None:
        return bytes(16 * (CROSSOVER_BINS + 1))
    rows = np.zeros((CROSSOVER_BINS, 4), dtype=np.float32)
    rows[:, :3] = offsets.bins
    return np.array([offsets.lo, offsets.hi, 1.0, 0.0], dtype=np.float32).tobytes() + rows.tobytes()


def crossover_space(density: FadeDensity, white_space: str, process: ProcessConfig) -> str:
    """Names the grid a crossover table is measured on: the render's fade matrix, its anchor, the white
    space and *process*'s meter area. A table tagged with another name was read on other densities."""
    named = b"" if density.matrix is None else np.round(np.asarray(density.matrix, dtype=np.float64), 5).tobytes()
    if density.anchor is not None:
        named += b"anchor" + np.round(np.asarray(density.anchor, dtype=np.float64), 5).tobytes()
    area = (process.analysis_buffer, process.analysis_rect, process.meter_rect)
    named += b"area" + repr(area).encode()
    return hashlib.sha1(b"crossover" + named + white_space.encode()).hexdigest()[:12]


#: A table a render measured for itself: (table, fade_crossover_space).
MeasuredCrossover = tuple[Optional[tuple[float, ...]], str]


def crossover_strength(process: ProcessConfig) -> float:
    """The table's strength in a render: Crossover on a slide whose survivals act (an Estimate that
    succeeded, or survivals set by hand), else 0."""
    if fade_matrix(process) is None:
        return 0.0
    return float(process.fade_crossover)


def resolve_crossover(
    process: ProcessConfig, space: str, grid: Callable[[], ImageBuffer], measure: bool = True
) -> tuple[Optional[tuple[float, ...]], Optional[MeasuredCrossover]]:
    """(table, measured): the table a render applies, and the one it measured on the after-fade *grid*
    when the frame has none, else None. A stored table applies under any tag, because a roll gives every
    frame the table of the frame it was pushed from. Nothing is read at zero strength (`crossover_strength`).
    Without *measure* (a tile of an export) a missing table does not apply."""
    if crossover_strength(process) == 0.0:
        return None, None
    if process.fade_crossover_table is not None:
        return process.fade_crossover_table, None
    if not measure:
        return None, None
    table = measure_crossover(grid())
    return table, (table, space)


def with_fade_crossover(process: ProcessConfig, measured: Optional[MeasuredCrossover]) -> ProcessConfig:
    """*process* carrying a table `resolve_crossover` measured; unchanged with none."""
    if measured is None:
        return process
    return replace(process, fade_crossover_table=measured[0], fade_crossover_space=measured[1])


#: The white space a camera RAW's survival density takes when Estimate has none to name: any
#: non-empty space makes `fade_density` anchor the matrix at the white, as a RAW render does.
ESTIMATE_WHITE_SPACE = "estimate"


def survival_density(
    red: float,
    green: float,
    blue: float,
    white: Optional[tuple[float, float, float]],
    white_anchor: bool,
    camera_raw: bool,
    white_space: str = "",
    shift: float = 0.0,
) -> FadeDensity:
    """The fade a render with these survivals and *shift* applies after the crosstalk unmix, on the Estimate grid."""
    from negpy.features.process.models import ProcessMode

    space = white_space or (ESTIMATE_WHITE_SPACE if camera_raw else "")
    w = white or (0.0, 0.0, 0.0)
    process = replace(
        ProcessConfig(),
        process_mode=ProcessMode.E6,
        fade_ratio_r=red,
        fade_ratio_g=green,
        fade_ratio_b=blue,
        fade_white_anchor=white_anchor,
        fade_white_r=w[0],
        fade_white_g=w[1],
        fade_white_b=w[2],
        fade_white_space=space,
        fade_density_shift=shift,
    )
    return fade_density(process, ProcessMode.E6, space)


def through_density(grid: ImageBuffer, density: FadeDensity) -> np.ndarray:
    """*grid* through *density*'s matrix and anchor, as `unmix_log_image` applies them."""
    from negpy.features.exposure.normalization import unmix_log_image

    return np.asarray(unmix_log_image(np.ascontiguousarray(grid, dtype=np.float32), density.matrix, density.anchor))
