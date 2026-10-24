from typing import Any, Dict

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.altprocess.models import Sensitizer
from negpy.kernel.image.logic import WORKING_LAB

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
    # highlight stain (Prussian blue mixed with the residual yellow sensitizer, which Ware
    # describes on both formulas), mid blue, Prussian blue.
    "path_u": (0.00, 0.15, 0.55, 1.00),
}

# Per-sensitizer Dmax and hue path. Ware's densitometry puts classic (Herschel) Dmax well
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
    sensitizer can hold. That range is the contrast control — Ware measures ~1.0
    to 1.2 for the traditional formula against ~2.4 for the new one, and his
    Simple Cyanotype ships as three variants at 1.8 / 2.3 / 2.7. Within it the
    midtones compress, so the mid gamma runs below one.

    Color is Prussian blue, which absorbs around 700nm and so carries most of its
    density in red: the print never goes black, it goes blue, and the highlights
    print green where the residual yellow sensitizer mixes into the blue. Bleach
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
    f32 = np.float32
    scalars = [
        f32(sc["d_max"]),
        f32(0.301 * exposure),
        f32(max(scale, 0.1)),
        f32(c["mid_compress"]),
        f32(bleach),
        f32(c["bleach_floor"]),
        f32(tannin),
        f32(c["tannin_restore"]),
        f32(c["tannin_dmax_gain"]),
    ]
    brown = (f32(c["brown_dir"][0]), f32(c["brown_dir"][1]))
    path_u = [f32(v) for v in c["path_u"]]
    anchors = [(f32(a), f32(b)) for a, b in sc["path"]]
    return negpy_core.cyanotype(np.ascontiguousarray(img, dtype=np.float32), scalars, brown, path_u, anchors, WORKING_LAB)
