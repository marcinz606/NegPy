from typing import Any, Dict, Sequence, Tuple

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.altprocess.models import Sensitizer
from negpy.kernel.image.logic import _LAB_EPS, _LAB_KAPPA, lab_to_rgb_working
from negpy.kernel.image.validation import ensure_image

CYANOTYPE_CONSTANTS: Dict[str, Any] = {
    # Transfer.
    # Weight of the reverse-S mixed against a straight line. Cyanotype compresses the
    # midtones, which is the whole reason a cyanotype digital-negative curve exists, so the
    # mid gamma must sit below 1. The straight-line half is what stops the centre from going
    # perfectly flat and posterizing a gradient.
    "mid_compress": 0.45,
    # Toning.
    # Washing soda strips Prussian blue highlights-first. At full strength the deepest shadow
    # keeps this fraction of its density and everything above it clears to paper.
    "bleach_floor": 0.15,
    # Tannin re-develops the bleached iron as iron tannate, slightly past where the blue was,
    # and the tannate covers more than the pigment it replaced.
    "tannin_restore": 1.05,
    "tannin_dmax_gain": 0.15,
    # (a*, b*) of iron tannate at full density, scaled by the density fraction so the
    # highlights stay a pale stain rather than jumping to full brown.
    "brown_dir": (9.0, 20.0),
    # Hue path.
    # Density fractions u = D/d_max of the four (a*, b*) anchors: rag paper, the green
    # highlight stain (Prussian blue mixed with the residual yellow sensitiser, which Ware
    # describes on both formulas), mid blue, Prussian blue.
    "path_u": (0.00, 0.15, 0.55, 1.00),
}

# Per-sensitiser Dmax and hue path. Ware's densitometry puts classic (Herschel) Dmax well
# below a good modern print on the red channel, Prussian blue's absorption peak, and the
# new (Ware) process holds far more pigment through the wash, so it goes deeper and
# cleaner.
SENSITIZERS: Dict[str, Dict[str, Any]] = {
    Sensitizer.CLASSIC: {
        "d_max": 0.95,
        "path": ((0.5, 3.0), (-4.0, 6.0), (-7.0, -17.0), (-6.0, -25.0)),
    },
    Sensitizer.NEW: {
        "d_max": 1.40,
        "path": ((0.3, 2.0), (-2.0, 2.0), (-9.0, -24.0), (-8.0, -34.0)),
    },
}


def sensitizer_constants(sensitizer: str) -> Dict[str, Any]:
    return SENSITIZERS.get(Sensitizer(sensitizer), SENSITIZERS[Sensitizer.CLASSIC])


def _hue_path(u: np.ndarray, path: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """(a*, b*) along the sensitiser's density hue path, clamped outside the anchors."""
    xp = np.asarray(CYANOTYPE_CONSTANTS["path_u"], dtype=np.float32)
    a = np.interp(u, xp, np.asarray([p[0] for p in path], dtype=np.float32)).astype(np.float32)
    b = np.interp(u, xp, np.asarray([p[1] for p in path], dtype=np.float32)).astype(np.float32)
    return a, b


def apply_cyanotype(
    img: ImageBuffer,
    sensitizer: str = Sensitizer.CLASSIC,
    enabled: bool = False,
    exposure: float = 0.0,
    scale: float = 1.4,
    bleach: float = 0.0,
    tannin: float = 0.0,
) -> ImageBuffer:
    """
    Cyanotype printing of a linear-reflectance print.

    The iron processes have no development stage to snatch: the print is fixed by
    how much UV got through the negative and by how long a density range the
    sensitiser can hold. That range is the contrast control — Ware measures ~1.0
    to 1.2 for the traditional formula against ~2.4 for the new one, and his
    Simple Cyanotype ships as three variants at 1.8 / 2.3 / 2.7. Within it the
    midtones compress, so the mid gamma runs below one.

    Color is Prussian blue, which absorbs around 700nm and so carries most of its
    density in red: the print never goes black, it goes blue, and the highlights
    print green where the residual yellow sensitiser mixes into the blue. Bleach
    then tannin is the standard toning pair — washing soda strips the pigment
    highlights-first, tannic acid re-develops the iron as a brown tannate.

    Not modelled: solarisation. It is real, but it is a wet-print artefact — the
    reversed shadows regain their density on drying as Prussian white oxidises
    back to Prussian blue, so the finished print does not carry it.
    """
    if not enabled:
        return img

    c = CYANOTYPE_CONSTANTS
    sc = sensitizer_constants(sensitizer)
    d_max = np.float32(sc["d_max"])

    arr = np.clip(img.astype(np.float32), 1e-6, 1.0)
    luma = arr[:, :, 0] * np.float32(0.2126) + arr[:, :, 1] * np.float32(0.7152) + arr[:, :, 2] * np.float32(0.0722)

    base = -np.log10(np.clip(luma, 1e-6, 1.0))
    t = np.clip((base + np.float32(0.301 * exposure)) / np.float32(max(scale, 0.1)), 0.0, 1.0).astype(np.float32)

    m = np.float32(c["mid_compress"])
    v = np.float32(2.0) * t - np.float32(1.0)
    u0 = (np.float32(1.0) - m) * t + m * np.float32(0.5) * (np.float32(1.0) + v * np.abs(v))

    b_amt = np.float32(bleach)
    u_b = u0 * (np.float32(1.0) - b_amt * (np.float32(1.0) - np.float32(c["bleach_floor"]) * u0))

    t_amt = np.float32(tannin)
    u = u_b + t_amt * (u0 * np.float32(c["tannin_restore"]) - u_b)
    dens = (d_max * (np.float32(1.0) + t_amt * np.float32(c["tannin_dmax_gain"])) * u).astype(np.float32)

    frac = np.clip(u, 0.0, 1.0).astype(np.float32)
    a_blue, b_blue = _hue_path(frac, sc["path"])
    brown = c["brown_dir"]
    a_star = a_blue + t_amt * (np.float32(brown[0]) * frac - a_blue)
    b_star = b_blue + t_amt * (np.float32(brown[1]) * frac - b_blue)

    grey = np.power(np.float32(10.0), -dens)
    f = np.where(grey > np.float32(_LAB_EPS), np.cbrt(grey), np.float32(_LAB_KAPPA) * grey + np.float32(16.0 / 116.0))
    lab = np.dstack([np.float32(116.0) * f - np.float32(16.0), a_star, b_star])

    return ensure_image(np.clip(lab_to_rgb_working(lab), 0.0, 1.0))
