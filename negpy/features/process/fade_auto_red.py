"""Restore Range: restores a faded slide's density span through Red Survival, and keeps its median where it printed."""

from typing import Any, Callable, Optional, Sequence

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.system.i18n import tr

#: Percentiles of the mean density Restore Range reads, light end first: the 0.5 and 99.5 % points, the median
#: and the quantiles between.
RANGE_PCT = (0.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.5)
#: Below this density span a frame has no range to read.
RANGE_MIN_SPAN = 0.05
#: Restore Range: the lowest Red Survival it sets. A slide that never held a full range (fog, a dull day) reads
#: as a heavy fade, so the stretch stops here.
AUTO_RED_MIN = 0.4
#: Restore Range: the density span it maps the frame's 0.5 to 99.5 % span onto, below a full slide's span because
#: a stretch by Red Survival also steepens the transfer curve's shoulder and toe.
AUTO_RED_SPAN = 1.6
#: Restore Range: the lowest density its shift leaves the frame's 0.5 % point at. The points are grid-cell means,
#: so the brightest pixels sit above them.
AUTO_RED_LIGHT = 0.2
#: Estimate suggests Restore Range only where its Red would be this share of Estimate's or less: the frame's
#: span is well short of AUTO_RED_SPAN, so a frame near the target stays quiet.
AUTO_RED_HINT_RATIO = 0.85


def range_points(density: ImageBuffer) -> Optional[tuple[float, ...]]:
    """The per-cell mean of a capture-density grid (the transfer curve's input, D = range * normalized
    log) at RANGE_PCT, light end first; None with too few cells or no range."""
    found = negpy_core.mean_percentiles(np.ascontiguousarray(density, dtype=np.float32), list(RANGE_PCT), 64)
    if found is None:
        return None
    points = tuple(float(v) for v in found)
    if points[-1] - points[0] < RANGE_MIN_SPAN:
        return None
    return points


def fade_range_points(log_grid: ImageBuffer, process: Any) -> Optional[tuple[float, ...]]:
    """`range_points` of the meter grid after the fade and Crossover, through the slide's window with
    its White and Black Point, as the transfer curve reads it."""
    from negpy.features.exposure.normalization import LogNegativeBounds, normalize_log_image
    from negpy.features.transparency.logic import TRANSFER_DENSITY_RANGE, transfer_bounds, transfer_point_offsets

    floors, ceils = transfer_bounds()
    wp3, bp3 = transfer_point_offsets(process)
    bounds = LogNegativeBounds(tuple(f + w for f, w in zip(floors, wp3)), tuple(c + b for c, b in zip(ceils, bp3)))
    return range_points(np.asarray(normalize_log_image(log_grid, bounds)) * TRANSFER_DENSITY_RANGE)


def survival_points(
    grid: ImageBuffer,
    process: Any,
    survivals: tuple[float, float, float],
    white_anchor: bool,
    camera_raw: bool,
    white_space: str,
    shift: float = 0.0,
) -> Optional[tuple[float, ...]]:
    """`fade_range_points` of the Estimate *grid* after *survivals* and *shift*, as a render with them
    reads it. The crossover table keeps each cell's mean density, so it does not move the points."""
    from negpy.features.process.fade import measure_white, survival_density, through_density

    density = survival_density(*survivals, measure_white(grid), white_anchor, camera_raw or bool(white_space), white_space, shift)
    return fade_range_points(through_density(grid, density), process)


def auto_red(
    points_at: Callable[[float], Optional[Sequence[float]]], red: float, shift: float = 0.0
) -> tuple[Optional[dict[str, float]], str]:
    """(changes, reason): the Red Survival that maps the frame's 0.5 to 99.5 % density span onto
    AUTO_RED_SPAN, held to AUTO_RED_MIN and 1.0, and the density shift that keeps the median where it
    printed, cut so the 0.5 % point prints no lighter than AUTO_RED_LIGHT or than before. *points_at* gives
    the points at a red with *shift*. Every density scales by red / new red, so one reading finds it."""
    from negpy.features.process.fade import RED_SURVIVAL_BOUNDS

    points = points_at(red)
    if points is None or float(points[-1] - points[0]) < RANGE_MIN_SPAN:
        return None, tr("No range to read on this frame.")
    span = float(points[-1] - points[0])
    new = round(min(max(float(red) * span / AUTO_RED_SPAN, AUTO_RED_MIN, RED_SURVIVAL_BOUNDS[0]), RED_SURVIVAL_BOUNDS[1]), 2)
    after = points_at(new)
    gain = 0.0
    if after is not None:
        gain = min(float(np.median(after)) - float(np.median(points)), float(after[0]) - min(AUTO_RED_LIGHT, float(points[0])))
    return {"fade_ratio_r": new, "fade_density_shift": round(float(shift) - gain, 4)}, ""
