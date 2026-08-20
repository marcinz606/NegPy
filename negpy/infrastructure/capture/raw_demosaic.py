"""Linear sensor-RGB demosaic for calibration metering.

Follows NegPy's canonical RAW decode (`ImageProcessor._decode_sensor_rgb`): sensor-native
`output_color=raw`, no white balance, linear gamma, 16-bit — so calibration meters the film base
the same way the RGB-Scan merge later reads the channels. rawpy is imported lazily so the module
stays import-safe.

It deviates in one parameter, deliberately: `adjust_maximum_thr=0.0` (see `linear_demosaic`). The
canonical decode still runs LibRaw's default, where each frame is scaled by its own brightest
pixel — harmless for a single rendered image, fatal for a meter comparing frames. Whether the
canonical path wants the same fix is a separate question (it changes rendered output, so it needs
its own verification); it is NOT covered here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _linearity_limit(raw, positions: Optional[Sequence[int]] = None) -> int:
    """The camera's calibrated linear-range ceiling (raw ADC scale), not the format's generic max.

    `white_level` is the format's theoretical max code value; `camera_white_level_per_channel` is
    the manufacturer's own calibration of where the sensor stops responding linearly, often lower
    (a Nikon D800: 15311 vs a generic 16383). Trusting the generic number lets already non-linear
    or clipped photosites read as clean. Falls back to `white_level` where a body carries no
    calibrated table. `positions` restricts the minimum to specific CFA channel indices; None
    takes it across all of them.
    """
    generic = int(raw.white_level or 0)
    per_channel = raw.camera_white_level_per_channel
    if not per_channel:
        return generic
    values = [per_channel[j] for j in positions] if positions is not None else per_channel
    calibrated = int(min(values))
    # A calibrated limit can't legitimately exceed the format's own ADC ceiling; a body
    # reporting one anyway is bad metadata, not a wider range.
    return min(calibrated, generic) if generic > 0 else calibrated


def _user_sat(raw) -> Optional[int]:
    """`_linearity_limit`, converted to the post-black-subtraction scale `postprocess()` wants.

    `user_sat` is compared after LibRaw subtracts the black level from every photosite, so the
    raw-scale limit must be corrected by the same amount or every body with a non-zero black
    level is silently under-scaled (invisible on a body whose black level happens to be 0).
    """
    limit = _linearity_limit(raw)
    if limit <= 0:
        return None
    black = min(raw.black_level_per_channel or [0])
    return max(1, limit - black)


def linear_demosaic(path: str, half_size: bool = False) -> np.ndarray:
    """Decode one RAW to a sensor-native, linear, 16-bit HxWx3 array (R=0, G=1, B=2).

    `half_size=True` bins each 2×2 Bayer quad straight into one RGB pixel (no interpolation)
    for a ~4× faster decode — used by calibration, which only meters a uniform base patch, so
    full resolution is wasted (and the raw-Bayer clip check reads full-res separately). Bayer
    only: X-Trans automatically falls back to a full-size decode because 2×2 binning aliases
    its 6×6 CFA.
    """
    import rawpy

    from negpy.infrastructure.loaders.helpers import get_best_demosaic_algorithm, is_xtrans

    with rawpy.imread(path) as raw:
        algo = get_best_demosaic_algorithm(raw)
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            # Scale against the camera's white level ONLY, never the frame's own brightest pixel.
            # LibRaw's default (adjust_maximum_thr=0.75) switches the scaling reference to the image
            # maximum once that exceeds 75 % of the white level, so each frame is normalised by its
            # own content. That makes the decode non-linear in exposure: rig data showed the metered
            # base pinned across a range of LED levels, because the scaling grew exactly as fast as
            # the light, which reads as an LED or shutter defect and is neither. 0.0 disables the
            # substitution and makes the demosaiced scale a fixed multiple of the raw counts, which is
            # what a meter measuring absolute light requires and what CLIP_CEILING assumes.
            adjust_maximum_thr=0.0,
            # Pin the scale reference to the camera's calibrated linearity limit (see
            # `_user_sat`), not LibRaw's generic ADC ceiling — otherwise a body whose real
            # limit sits below the format max reports up to 65535 for photosites that are
            # already non-linear or clipped, and CLIP_CEILING never sees it.
            user_sat=_user_sat(raw),
            use_camera_wb=False,
            user_wb=[1, 1, 1, 1],
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            demosaic_algorithm=algo,
            half_size=half_size and not is_xtrans(raw),
            user_flip=0,
        )
    return np.asarray(rgb)


#: Window below the ROI's own maximum searched for a saturation plateau, and the run of bins
#: immediately above a candidate peak compared against it for density. A real pileup is denser by
#: an order of magnitude or more; a clean sensor's falling tail never clears this ratio (verified
#: down to a 4-count read-noise sigma). Both are counts, so they hold regardless of what the raw
#: format's own ceiling is — this path needs no white level at all.
_PLATEAU_WINDOW = 256
_PLATEAU_NEIGHBOR_BINS = 8
_PLATEAU_DENSITY_RATIO = 6.0
_MIN_PLATEAU_SITES = 4  # a plateau is a pile, never a handful of photosites on a small ROI


def _plateau_clip_fraction(values: np.ndarray, dark_frame_guard: int) -> float:
    """Fraction of photosites pinned on a saturation plateau, found from the data's shape alone.

    A sensor can saturate below the white level its metadata publishes — or on a body with no
    metadata at all. Saturation still has a shape: photosites pile up against the highest level
    the sensor reaches, while noise above that pile falls off smoothly. The pile is not always the
    ROI's own maximum — a handful of noise-elevated sites (dust, a hot pixel, ordinary shot noise)
    routinely sit a few dozen counts above the true plateau, so anchoring on `values.max()` itself
    can land in that sparse tail and miss the pile beneath it. Scanning the top `_PLATEAU_WINDOW`
    counts for the densest bin, rather than assuming the maximum sample instead, finds it either
    way. `dark_frame_guard` (typically the raw white level, loosely, or 0 to skip the guard) keeps
    a frame nowhere near saturation from having its noise floor mistaken for a plateau.
    """
    top = int(values.max())
    if dark_frame_guard > 0 and top * 2 < dark_frame_guard:
        return 0.0
    lo = max(0, top - _PLATEAU_WINDOW)
    counts = np.bincount(values[values >= lo] - lo, minlength=_PLATEAU_WINDOW + 1)
    peak_bin = int(np.argmax(counts))
    peak = int(counts[peak_bin])
    if peak < _MIN_PLATEAU_SITES:
        return 0.0
    neighbors = counts[peak_bin + 1 : peak_bin + 1 + _PLATEAU_NEIGHBOR_BINS]
    neighbor_density = float(neighbors.mean()) if neighbors.size else 0.0
    if peak < _PLATEAU_DENSITY_RATIO * max(neighbor_density, 1.0):
        return 0.0
    anchor = lo + peak_bin
    return float(np.count_nonzero(values >= anchor)) / values.size


def raw_channel_clip_fraction(path: str, channel_index: int, roi, saturation_margin: int = 16) -> tuple[float, float]:
    """Fraction of *raw Bayer* photosites for one channel past two independent points, inside the
    ROI: (linearity_fraction, plateau_fraction).

    A demosaiced channel can read clean while its source photosites are already past either point
    — interpolation averages a bad site with clean neighbours and hides it. Metering the raw sites
    (before demosaic/color) catches that, which matters for ETTR where the base is deliberately
    exposed near the ceiling. The two are semantically different and must not be merged into one
    number here: `linearity_fraction` (`_linearity_limit`, when the body publishes a calibrated
    ceiling) can overstate real clipping — a body's calibrated ceiling is often a conservative
    linearity limit, not where it actually saturates, so a channel legitimately using the top of
    its range reads as partly "past" it. `plateau_fraction` (`_plateau_clip_fraction`) finds the
    true saturation pile from the data's shape, needs no metadata, and is what should gate a hard
    abort; the caller is expected to budget the two separately. `roi` is any object with a
    `.pixels(w, h)` method (duck-typed to avoid an infra→services import). channel_index: R=0,
    G=1, B=2. Returns (0.0, 0.0) if the channel can't be resolved."""
    import rawpy

    with rawpy.imread(path) as raw:
        img = raw.raw_image_visible
        colors = raw.raw_colors_visible
        letter = "RGB"[channel_index]
        desc = raw.color_desc.decode("ascii", errors="ignore")  # e.g. "RGBG": 0=R,1=G,2=B,3=G
        wanted = [j for j, c in enumerate(desc) if c.upper() == letter]
        if not wanted:
            return 0.0, 0.0
        h, w = img.shape[:2]
        x0, y0, x1, y1 = roi.pixels(w, h)
        sub_img = img[y0:y1, x0:x1]
        mask = np.isin(colors[y0:y1, x0:x1], wanted)
        if not mask.any():
            return 0.0, 0.0
        values = sub_img[mask]

        linearity_fraction = 0.0
        # raw_image_visible is absolute ADC counts (black not subtracted), matching the limit's scale.
        limit = _linearity_limit(raw, wanted)
        if limit > 0:
            threshold = max(0, limit - saturation_margin)
            linearity_fraction = float(np.mean(values >= threshold))

        plateau_fraction = _plateau_clip_fraction(values, int(raw.white_level or 0))
        return linearity_fraction, plateau_fraction
