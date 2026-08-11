"""
Capture colorimetry: the camera's own colour matrix.

RAW is decoded `output_color=raw` (sensor-native RGB), so a buffer arriving in the
pipeline is in the camera's primaries, not the working space. The print path never
needed the distinction — it derives colour from measured film density and a paper
model — but a transparency transfer does: without this matrix a slide renders in
sensor primaries and no raw converter agrees with it.

The construction follows dcraw/libraw `cam_xyz_coeff`: build the *working->cam*
matrix, row-normalize it so each camera channel answers 1 to a neutral, and only
then invert. The white balance owns the grey point, not this matrix.

The order matters, and reversing it is not a subtle error. Normalizing the rows of
the already-inverted matrix also sends neutral to neutral -- so greys, and any
near-neutral frame, look right either way -- but it is a different transform for
everything else, and the error grows with saturation. Measured against libraw's own
cam->sRGB on a saturated sunset it doubled both R/G and B/G, which renders as a
magenta cast that reads like wildly excessive saturation.
"""

from typing import Optional, Sequence

import numpy as np

# Adobe RGB (1998) D65 — the working space (see infrastructure.display.color_spaces).
_XYZ_TO_WORKING = np.array(
    [
        [2.0413690, -0.5649464, -0.3446944],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0134474, -0.1183897, 1.0154096],
    ],
    dtype=np.float64,
)
_WORKING_TO_XYZ = np.linalg.inv(_XYZ_TO_WORKING)


def camera_to_working_matrix(
    cam_xyz: Optional[Sequence[Sequence[float]]],
    camera_wb: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    """
    Working-space-from-camera 3x3 for a camera's XYZ->cam matrix (libraw's
    `rgb_xyz_matrix`, whose 4th row is the unused CMYG slot).

    `camera_wb` folds the as-shot multipliers in, for buffers decoded WITHOUT white
    balance (Linear RAW). The row normalization below makes this matrix assume a
    neutral camera signal, so an unbalanced one renders with a heavy cast — a typical
    2:1 raw green-to-red ratio is a green frame, not a subtle shift. Folding the
    multipliers in reconstructs the signal the matrix expects, which makes the render
    independent of how the buffer was decoded. Pass None when white balance was applied
    at decode.

    None when no matrix is available (a scanner TIFF, a JPEG) or when it is
    singular/degenerate — the caller then treats the buffer as already in the
    working space, which is the correct reading for a profiled source.
    """
    if cam_xyz is None:
        return None
    m = np.asarray(cam_xyz, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 3 or m.shape[0] < 3:
        return None
    m = m[:3, :]
    if not np.all(np.isfinite(m)) or abs(float(np.linalg.det(m))) < 1e-9:
        return None

    # Normalize the forward (working->cam) rows, then invert — dcraw's order. Doing it
    # the other way round distorts every non-neutral colour; see the module docstring.
    forward = m @ _WORKING_TO_XYZ
    sums = forward.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(sums)) or np.any(np.abs(sums) < 1e-9):
        return None
    out = np.linalg.pinv(forward / sums)
    if not np.all(np.isfinite(out)):
        return None

    if camera_wb is not None:
        wb = np.asarray(camera_wb, dtype=np.float64).reshape(-1)[:3]
        if wb.shape[0] == 3 and np.all(np.isfinite(wb)) and float(wb.min()) > 0.0 and float(wb[1]) > 0.0:
            # Normalized to green, matching libraw's own scaling reference, so only the
            # channel ratios are applied and overall exposure is untouched.
            out = out @ np.diag(wb / wb[1])

    return np.ascontiguousarray(out, dtype=np.float32)


def apply_camera_matrix(img: np.ndarray, matrix: Optional[np.ndarray]) -> np.ndarray:
    """
    Camera primaries -> working space. A None matrix passes the buffer through.

    Negatives are kept: they are real out-of-gamut colours, and clipping here
    would bake a hue shift into the transfer before any tone curve sees it.
    """
    if matrix is None:
        return img
    flat = img.reshape(-1, 3).astype(np.float32, copy=False)
    return (flat @ np.asarray(matrix, dtype=np.float32).T).reshape(img.shape)
