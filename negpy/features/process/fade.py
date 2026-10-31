"""Per-image estimate of the green and blue dye survival for Fade Restoration.

Neutral content constrains only the ratios between channels, so red's absolute survival is
never estimated.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.system.i18n import tr
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    measure_neutral_axis_from_log,
    prefilter_log_grid,
)
from negpy.features.process.capture_color import apply_camera_matrix
from negpy.features.process.models import ProcessConfig, ProcessMode

#: Below this density spread a channel's neutral axis is too flat to be signal.
SPREAD_FLOOR = 0.02
#: Spreads within this fraction of the largest agree: no evidence of differential fade.
AGREEMENT_TOLERANCE = 0.05
#: Bound on an implied ratio, a reciprocal pair; the factor stays well conditioned.
RATIO_BOUNDS = (0.2, 5.0)
#: Bound on `fade_ratio_r`, a fraction of the original density, so it cannot exceed 1.0.
RED_SURVIVAL_BOUNDS = (0.05, 1.0)

#: The return shape of `measure_neutral_axis_from_log`: midtone, shadow, highlight or None, confidence.
NeutralAxisRefs = tuple[tuple[float, float, float], tuple[float, float, float], object, float]


def measure_neutral_axis_ratios(image: ImageBuffer, roi: Optional[tuple], analysis_buffer: float) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reason) from the raw capture, metered against this frame's own bounds: the fixed
    transfer window finds no grays in a faded slide. `refs` is None, with a reason, when no
    trustworthy neutral axis exists."""
    grid = prefilter_log_grid(image, roi, analysis_buffer)
    bounds = _measured_slide_bounds(grid)
    refs = measure_neutral_axis_from_log(grid, bounds, None, 0.0)
    if refs is None:
        return None, tr("No reliable grays found on this frame.")
    return refs, ""


def _measured_slide_bounds(grid: ImageBuffer) -> LogNegativeBounds:
    """This frame's own bounds in the slide orientation (floor above ceiling): the luma span
    from the base clip, each channel's offset from its extreme percentiles. The negative
    analyzer does not fit, since its dense end is chroma-gated for an orange mask."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    def _pct(p: float) -> list:
        return [float(np.percentile(grid[:, :, ch], p)) for ch in range(3)]

    base = float(EXPOSURE_CONSTANTS["base_luma_clip"])
    luma_f, luma_c = _pct(100.0 - base), _pct(base)
    color_f, color_c = _pct(100.0 - 0.00001), _pct(0.00001)
    mean_f, mean_c = sum(luma_f) / 3.0, sum(luma_c) / 3.0
    mid_f, mid_c = sorted(color_f)[1], sorted(color_c)[1]
    floors = tuple(mean_f + (color_f[ch] - mid_f) for ch in range(3))
    ceils = tuple(mean_c + (color_c[ch] - mid_c) for ch in range(3))
    return LogNegativeBounds(floors, ceils)


class EstimateMethod(StrEnum):
    """MIDTONE reads the midtone grays' own densities. SPREAD reads each channel's midtone-to-shadow
    spread, which cancels a per-channel density offset but follows a flat, noisy shadow."""

    MIDTONE = "midtone"
    SPREAD = "spread"


@dataclass(frozen=True)
class FadeEstimate:
    """Survivals from one Estimate. red is None when the method leaves Red Survival as it is;
    reason is "" for a real estimate (see fade_ratios_from_neutral_axis)."""

    red: Optional[float]
    green: float
    blue: float
    reason: str


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
        out = [(n, raw, c) for n, raw, c in ((tr("green"), ratio_g, clamped_g), (tr("blue"), ratio_b, clamped_b)) if raw != c]
        reason = "; ".join(
            tr(
                "Implied {channel} {raw} is outside {lo}–{hi}, set to {clamped}",
                channel=n,
                raw=f"{raw:.1f}",
                lo=f"{lo:g}",
                hi=f"{hi:g}",
                clamped=f"{c:.1f}",
            )
            for n, raw, c in out
        )
        if any(raw > hi for _n, raw, _c in out):
            reason += ". " + tr("Red has faded further than this range covers: lower Red Survival by eye")
        return clamped_g, clamped_b, reason
    return ratio_g, ratio_b, ""


def estimate_fade_ratios(
    image: ImageBuffer, process_mode: Optional[str], roi: Optional[tuple], analysis_buffer: float
) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from the raw capture, failing closed to 1.0 with a reason.
    Runs on request, never during a render."""
    if process_mode is not None and str(process_mode) != str(ProcessMode.E6):
        return 1.0, 1.0, tr("This frame is not a slide.")
    refs, reason = measure_neutral_axis_ratios(image, roi, analysis_buffer)
    if refs is None:
        return 1.0, 1.0, reason
    return fade_ratios_from_neutral_axis(refs)


def estimate_fade(
    image: ImageBuffer,
    process_mode: Optional[str],
    roi: Optional[tuple],
    analysis_buffer: float,
    cam_matrix: Optional[np.ndarray] = None,
) -> dict[EstimateMethod, FadeEstimate]:
    """Both methods' estimates from one measurement of the grays. MIDTONE also sets Red Survival so
    the best-kept layer lands at 1.0; a ratio above 1 says red faded most. `cam_matrix` takes a
    camera buffer into the working space the render applies the fade in."""
    if process_mode is not None and str(process_mode) != str(ProcessMode.E6):
        reason = tr("This frame is not a slide.")
        return {m: FadeEstimate(None, 1.0, 1.0, reason) for m in EstimateMethod}
    if cam_matrix is not None:
        image = apply_camera_matrix(negpy_core.nan_to_num(np.ascontiguousarray(image, dtype=np.float32), 1e-6, 1.0, 1e-6), cam_matrix)
    refs, reason = measure_neutral_axis_ratios(image, roi, analysis_buffer)
    out = {}
    for method in EstimateMethod:
        if refs is None:
            out[method] = FadeEstimate(None, 1.0, 1.0, reason)
            continue
        g, b, why = fade_ratios_from_neutral_axis(refs, method)
        red = None
        if method == EstimateMethod.MIDTONE and not (why and g == 1.0 and b == 1.0):
            lo, hi = RED_SURVIVAL_BOUNDS
            red = min(max(1.0 / max(g, b, 1.0), lo), hi)
        out[method] = FadeEstimate(red, g, b, why)
    return out


def fade_estimate_available(process: ProcessConfig) -> bool:
    """Whether the Estimate action can run: Transparency only."""
    return process.process_mode == ProcessMode.E6
