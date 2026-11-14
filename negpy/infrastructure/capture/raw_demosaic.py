"""Linear sensor-RGB demosaic for calibration metering.

Follows NegPy's canonical RAW decode (`ImageProcessor._decode_sensor_rgb`): sensor-native
`output_color=raw`, no white balance, linear gamma, 16-bit — so calibration meters the film base
the same way the RGB-Scan merge later reads the channels. rawpy is imported lazily so the module
stays import-safe.

It shares both scale-reference parameters with the canonical decode
(`image_processor.py:_decode_sensor_rgb`): `adjust_maximum_thr=0.0`, which disables LibRaw's
per-frame maximum substitution and scales each frame by a fixed reference instead of its own
brightest pixel — required for a meter comparing frames — and `user_sat=_user_sat(raw)` (see
`_user_sat`), which pins that reference to the camera's calibrated linearity limit rather than
LibRaw's generic ADC ceiling.
"""

from __future__ import annotations

from typing import Optional, Sequence

import negpy_core
import numpy as np


def _has_corrupt_colordata(raw) -> bool:
    generic = int(raw.white_level or 0)
    per_channel = raw.camera_white_level_per_channel
    if not per_channel:
        return False
    positive = [int(v) for v in per_channel if v and v > 0]
    if not positive:
        return False
    calibrated = min(positive)
    black = max(getattr(raw, "black_level_per_channel", None) or [0])
    return calibrated <= black or (generic > 0 and calibrated < generic // 2)


def _optical_black(raw) -> int:
    full = getattr(raw, "raw_image", None)
    sizes = getattr(raw, "sizes", None)
    if full is None or sizes is None:
        return 0
    tm = int(getattr(sizes, "top_margin", 0) or 0)
    lm = int(getattr(sizes, "left_margin", 0) or 0)
    if lm >= 24:
        strip = full[tm:, 8 : lm - 8]
    elif tm >= 24:
        strip = full[8 : tm - 8, lm:]
    else:
        return 0
    return int(np.median(strip)) if strip.size else 0


def _repair_corrupt_black(raw) -> int:
    """Compensate `raw_image_visible` when LibRaw misparsed `black_level_per_channel` from an
    unrecognised MakerNote layout, using the masked optical-black margin."""
    ob = _optical_black(raw)
    cblack = list(getattr(raw, "black_level_per_channel", None) or [0, 0, 0, 0])
    if ob <= max(cblack):
        return 0
    full = getattr(raw, "raw_image", None)
    vis = getattr(raw, "raw_image_visible", None)
    cols = getattr(raw, "raw_colors_visible", None)
    pat = cols[:2, :2] if cols is not None and cols.shape >= (2, 2) else getattr(raw, "raw_pattern", None)
    if full is not None and vis is not None and pat is not None and pat.shape == (2, 2) and full[0, 0] > 0:
        for r in range(2):
            for c in range(2):
                ch = int(pat[r, c])
                cb = int(cblack[ch])
                d = ob - cb
                if d > 0:
                    plane = vis[r::2, c::2]
                    np.maximum(plane, d + cb, out=plane)
                    plane -= d
        full[0, 0] = 0
    return ob


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
    if not per_channel or _has_corrupt_colordata(raw):
        return generic
    values = [per_channel[j] for j in positions] if positions is not None else per_channel
    positive = [int(v) for v in values if v and v > 0]
    if not positive:
        return generic
    calibrated = min(positive)
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
    if _has_corrupt_colordata(raw):
        black = _repair_corrupt_black(raw)
    else:
        black = min(raw.black_level_per_channel or [0])
    return max(1, limit - black)


def as_shot_wb(path: str) -> Optional[list]:
    """The RAW's as-shot white balance as [R, G, B] multipliers, None when it carries none."""
    import rawpy

    from negpy.infrastructure.loaders.helpers import camera_wb_multipliers

    with rawpy.imread(path) as raw:
        return camera_wb_multipliers(raw)


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


#: Plateau search window below the ROI's maximum, in counts, so this path needs no white level. A real pileup
#: is `_PLATEAU_DENSITY_RATIO` times denser than each of the next codes above it; a clean sensor's mode is not.
_PLATEAU_WINDOW = 256
_PLATEAU_NEIGHBOR_BINS = 8
_PLATEAU_DENSITY_RATIO = 6.0
_MIN_PLATEAU_SITES = 4  # a plateau is a pile, never a handful of photosites on a small ROI


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
    its range reads as partly "past" it. `plateau_fraction` (`negpy_core.raw_clip_fractions`) finds the
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
        # raw_image_visible is absolute ADC counts (black not subtracted), matching the limit's scale.
        limit = _linearity_limit(raw, wanted)
        found = negpy_core.raw_clip_fractions(
            np.ascontiguousarray(img[y0:y1, x0:x1], dtype=np.uint16),
            np.ascontiguousarray(colors[y0:y1, x0:x1], dtype=np.uint8),
            wanted,
            max(0, limit - saturation_margin) if limit > 0 else None,
            int(raw.white_level or 0),
            _PLATEAU_WINDOW,
            _MIN_PLATEAU_SITES,
            _PLATEAU_NEIGHBOR_BINS,
            _PLATEAU_DENSITY_RATIO,
        )
        return (0.0, 0.0) if found is None else found
