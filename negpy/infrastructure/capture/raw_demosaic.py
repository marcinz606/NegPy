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

import numpy as np


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
            # Scale against the camera's white level ONLY — never the frame's own brightest pixel.
            # LibRaw's default (adjust_maximum_thr=0.75) silently switches the scaling reference to
            # the image maximum once that exceeds 75 % of the white level, so each frame is
            # normalised by its own content. That makes the decode non-linear in exposure: rig data
            # showed the metered base pinned at ~34400 across LED levels 128-160 (the scaling grew
            # exactly as fast as the light), which reads as an LED/shutter defect and is neither.
            # 0.0 disables the substitution, making the demosaiced scale a fixed multiple of the raw
            # counts — which is what a meter measuring absolute light requires, and what
            # CLIP_CEILING assumes.
            adjust_maximum_thr=0.0,
            use_camera_wb=False,
            user_wb=[1, 1, 1, 1],
            output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            demosaic_algorithm=algo,
            half_size=half_size and not is_xtrans(raw),
            user_flip=0,
        )
    return np.asarray(rgb)


def raw_channel_clip_fraction(path: str, channel_index: int, roi, saturation_margin: int = 16) -> float:
    """Fraction of *raw Bayer* photosites for one channel that are clipped, inside the ROI.

    A demosaiced channel can read below saturation while its source photosites are already at
    the sensor ceiling — interpolation averages a clipped site with clean neighbours and hides
    it. Metering the raw sites (before demosaic/colour) catches that, which matters for ETTR
    where the base is deliberately exposed near the ceiling. `roi` is any object with a
    `.pixels(w, h)` method (duck-typed to avoid an infra→services import). channel_index: R=0,
    G=1, B=2. Returns 0.0 if the channel/white level can't be resolved."""
    import rawpy

    with rawpy.imread(path) as raw:
        img = raw.raw_image_visible
        colors = raw.raw_colors_visible
        # No white level → no raw refinement (the demosaiced clip guard still runs). Never guess
        # img.max() instead: that is an image-dependent reference (the adjust_maximum_thr failure
        # class), and on a uniform base the guess sits inside the noise — the quieter the sensor,
        # the more photosites land within `saturation_margin` of their own maximum, reading as
        # heavy clipping on a frame that clips nowhere (measured: ~0.9 % at σ=8 DN, ~40 % at σ=4).
        white = int(raw.white_level or 0)
        if white <= 0:
            return 0.0
        letter = "RGB"[channel_index]
        desc = raw.color_desc.decode("ascii", errors="ignore")  # e.g. "RGBG": 0=R,1=G,2=B,3=G
        wanted = [j for j, c in enumerate(desc) if c.upper() == letter]
        if not wanted:
            return 0.0
        h, w = img.shape[:2]
        x0, y0, x1, y1 = roi.pixels(w, h)
        sub_img = img[y0:y1, x0:x1]
        mask = np.isin(colors[y0:y1, x0:x1], wanted)
        if not mask.any():
            return 0.0
        threshold = max(0, white - saturation_margin)
        return float(np.mean(sub_img[mask] >= threshold))
