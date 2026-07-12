"""Spot densitometer: hover position -> normalized-log coords -> darkroom read-out.
Densities are relative to this scan's normalization, not absolute."""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R
from negpy.features.exposure.analysis import zone_of_encoded
from negpy.kernel.image.logic import working_oetf_decode

_PRINT_DENSITY_MAX = 4.0

_ROMAN = ("0", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
_THIRDS = ("", "⅓", "⅔")


def map_display_to_norm(
    nx: float,
    ny: float,
    disp_w: int,
    disp_h: int,
    content_rect: Optional[Tuple[int, int, int, int]],
    active_roi: Optional[Tuple[int, int, int, int]],
    crop_full: bool,
    norm_w: int,
    norm_h: int,
) -> Optional[Tuple[int, int]]:
    """
    Normalized display coords (borders included) -> pixel coords in the
    normalized-log frame (post-geometry, uncropped); None outside the content.
    `content_rect` = (off_x, off_y, w, h) inside the borders; `active_roi` =
    crop (y1, y2, x1, x2), skipped when `crop_full` shows the uncropped frame.
    """
    fx, fy = nx * disp_w, ny * disp_h
    if content_rect is not None:
        off_x, off_y, cw, ch = content_rect
        if cw <= 0 or ch <= 0 or not (off_x <= fx < off_x + cw and off_y <= fy < off_y + ch):
            return None
        u, v = (fx - off_x) / cw, (fy - off_y) / ch
    else:
        u, v = nx, ny

    if crop_full or active_roi is None:
        x, y = u * norm_w, v * norm_h
    else:
        y1, y2, x1, x2 = active_roi
        x, y = x1 + u * (x2 - x1), y1 + v * (y2 - y1)
    return (
        int(min(max(x, 0), norm_w - 1)),
        int(min(max(y, 0), norm_h - 1)),
    )


@dataclass(frozen=True)
class DensitometerReading:
    """One probed pixel, in darkroom units (relative to this scan's bounds)."""

    dd_rgb: Tuple[float, float, float]  # ΔD above base, per channel
    val_luma: float  # Rec.709 luma of the val triplet (curve-dot x)
    print_density: float  # reflection density of the displayed tone
    zone: float  # print zone 0..10 (analysis.zone_of_encoded ruler, V = 18% gray)


def compute_reading(
    val_rgb: Tuple[float, float, float],
    bounds: Any,
    display_rgb: Tuple[float, float, float],
) -> DensitometerReading:
    """`bounds` is the LogNegativeBounds the image was normalized with."""
    dd = tuple(float(val_rgb[ch]) * (float(bounds.ceils[ch]) - float(bounds.floors[ch])) for ch in range(3))
    val_luma = LUMA_R * float(val_rgb[0]) + LUMA_G * float(val_rgb[1]) + LUMA_B * float(val_rgb[2])
    lum = LUMA_R * float(display_rgb[0]) + LUMA_G * float(display_rgb[1]) + LUMA_B * float(display_rgb[2])
    t = float(working_oetf_decode(np.asarray([lum], dtype=np.float32))[0])
    t = max(t, 10.0**-_PRINT_DENSITY_MAX)
    print_density = min(_PRINT_DENSITY_MAX, -math.log10(t))
    zone = float(zone_of_encoded(lum))
    return DensitometerReading((dd[0], dd[1], dd[2]), val_luma, print_density, zone)


def zone_roman(zone: float) -> str:
    """Zone in roman numerals with ⅓-stop fractions (e.g. 4.33 -> 'IV⅓')."""
    thirds = int(round(min(max(zone, 0.0), 10.0) * 3.0))
    base, frac = divmod(thirds, 3)
    if base >= 10:
        return _ROMAN[10]
    return _ROMAN[base] + _THIRDS[frac]


def format_reading(reading: DensitometerReading) -> str:
    dd = reading.dd_rgb
    return f"ΔD {dd[0]:.2f}·{dd[1]:.2f}·{dd[2]:.2f} · D {reading.print_density:.2f} · {zone_roman(reading.zone)}"
