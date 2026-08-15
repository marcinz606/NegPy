from typing import Any, Dict, Sequence, Tuple

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.image.logic import _LAB_EPS, _LAB_KAPPA, lab_to_rgb_working
from negpy.kernel.image.validation import ensure_image

LITH_CONSTANTS: Dict[str, Any] = {
    # Highlight branch (first development phase).
    # Ceiling of the exposure-only branch. Lith highlights are "controlled by exposure, not
    # development" (Rudman), so they saturate well below Dmax and only the knee carries a
    # pixel to black.
    "foot_max": 0.70,
    # dD/dD0 at zero, the foot gamma. The published range is 0.2-0.5: a lot of exposure
    # squeezed into very little density is what reads as creamy.
    "foot_rate": 0.60,
    # Fraction of the over-exposure the highlight branch sees. Lith highlights are meant to
    # carry a veil, not print as bare paper, and over-exposing two to four stops is what puts
    # tone in them. Too low and the whole highlight range collapses to paper white and reads
    # blown. Exposure 0 still gives clean white, so the slider keeps that end.
    "foot_veil": 0.60,
    # Infectious knee (second phase).
    # Snatch maps to the knee density: knee = d_max*(knee_lo - knee_span*snatch). snatch 0
    # puts the knee past Dmax so nothing fires, and 1 pulls it down to a wide, undifferentiated
    # "lith band" (Moersch).
    "knee_lo": 1.25,
    "knee_span": 0.70,
    # Abruptness maps to the knee width: w = abrupt_lo - abrupt_span*abruptness. Moersch's
    # hydroquinone-rich Solution A end of the A:B ratio gives a very narrow knee, his "almost
    # abrupt blackening", where the next shadow zone goes black with no separation at all.
    "abrupt_lo": 0.30,
    "abrupt_span": 0.27,
    # Hue path.
    # Density fractions u = D/d_max at which a paper's four (a*, b*) anchors sit: peach,
    # ochre, olive, neutral.
    "path_u": (0.10, 0.35, 0.65, 1.00),
}


def _hue_path(u: np.ndarray, path: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """(a*, b*) along the paper's density hue path, clamped outside the anchors."""
    xp = np.asarray(LITH_CONSTANTS["path_u"], dtype=np.float32)
    a = np.interp(u, xp, np.asarray([p[0] for p in path], dtype=np.float32)).astype(np.float32)
    b = np.interp(u, xp, np.asarray([p[1] for p in path], dtype=np.float32)).astype(np.float32)
    return a, b


def apply_lith(
    img: ImageBuffer,
    path: Sequence[Sequence[float]],
    d_max: float,
    enabled: bool = False,
    exposure: float = 0.0,
    snatch: float = 0.55,
    abruptness: float = 0.6,
) -> ImageBuffer:
    """
    Lith (infectious) development of a linear-reflectance print.

    Two branches on print density D0: a low-gamma highlight branch fixed by
    exposure alone, and a near-vertical infectious knee whose position is set by
    the snatch point. Between them the print has no midtones to speak of — past
    the knee everything sits on the flat Dmax "lith band".

    Color is the paper's own (a*, b*) path, indexed on the *output* density
    fraction: peach, ochre, olive, neutral. There is no strength control; the
    paper picked in the Exposure panel is the control. Small-particle silver is
    strongly chromatic and turns neutral only once the grains grow and pack
    (Kong & Shore 2007), and grain size tracks development stage, which tracks
    density.
    """
    if not enabled:
        return img

    c = LITH_CONSTANTS
    arr = np.clip(img.astype(np.float32), 1e-6, 1.0)
    luma = arr[:, :, 0] * np.float32(0.2126) + arr[:, :, 1] * np.float32(0.7152) + arr[:, :, 2] * np.float32(0.0722)

    base = -np.log10(np.clip(luma, 1e-6, 1.0))
    over = np.float32(0.301 * exposure)
    d0 = base + over

    foot_max = np.float32(c["foot_max"])
    d_foot = base + over * np.float32(c["foot_veil"])
    d_h = foot_max * (np.float32(1.0) - np.exp(-np.float32(c["foot_rate"]) * d_foot / foot_max))

    knee = np.float32(d_max * (c["knee_lo"] - c["knee_span"] * snatch))
    width = np.float32(max(c["abrupt_lo"] - c["abrupt_span"] * abruptness, 0.01))

    g = 1.0 / (1.0 + np.exp(-np.clip((d0 - knee) / width, -30.0, 30.0)))
    dens = (d_h + (np.float32(d_max) - d_h) * g).astype(np.float32)

    grey = np.power(np.float32(10.0), -dens)
    a_star, b_star = _hue_path(np.clip(dens / np.float32(d_max), 0.0, 1.0), path)
    f = np.where(grey > np.float32(_LAB_EPS), np.cbrt(grey), np.float32(_LAB_KAPPA) * grey + np.float32(16.0 / 116.0))
    lab = np.dstack([np.float32(116.0) * f - np.float32(16.0), a_star, b_star])

    return ensure_image(np.clip(lab_to_rgb_working(lab), 0.0, 1.0))
