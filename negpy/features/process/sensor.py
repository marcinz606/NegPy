"""Sensor (CFA) crosstalk calibration for single-shot narrowband scans.

The camera's color-filter passbands overlap the narrowband source's bands, so a
pure red/green/blue exposure leaks into the other channels — a fixed property of
the sensor+light pair, independent of film. Calibrated once from three bare-light
exposures and corrected with a 3x3 unmix in the LINEAR domain, before the
log/inversion where the film-dye crosstalk (Density Mixer) lives.
"""

from typing import Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import get_analysis_crop
from negpy.features.process.models import ProcessConfig, ProcessMode

_EPS = 1e-4


def measure_capture(img: ImageBuffer, center: float = 0.5) -> tuple[float, float, float]:
    """Mean linear RGB over the central ``center`` fraction (dodges vignetting)."""
    region = get_analysis_crop(np.asarray(img[:, :, :3], dtype=np.float64), (1.0 - center) / 2.0)
    m = region.reshape(-1, 3).mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def build_sensor_matrix(
    rgb_red: tuple[float, float, float],
    rgb_green: tuple[float, float, float],
    rgb_blue: tuple[float, float, float],
) -> tuple[float, ...]:
    """Correction matrix (9 floats, row-major) from three bare-light capture means.

    Columns of S are each band's sensor response; each column is divided by its
    own-channel value (unit diagonal — per-capture exposure cancels, white balance
    stays with downstream normalization), then inverted to un-mix.
    """
    s = np.column_stack([rgb_red, rgb_green, rgb_blue]).astype(np.float64)
    diag = np.diag(s).copy()
    if np.any(diag <= _EPS * s.max(axis=0)):
        raise ValueError("a capture has no signal in its own channel — check the R/G/B assignment")
    s_norm = s / diag
    try:
        correction = np.linalg.inv(s_norm)
    except np.linalg.LinAlgError:
        raise ValueError("captures are not independent — three distinct single-band exposures are required") from None
    return tuple(float(x) for x in correction.reshape(-1))


def apply_sensor_correction(img: ImageBuffer, matrix: Optional[tuple]) -> ImageBuffer:
    """Un-mix a linear (H, W, 3) capture with the baked 3x3; identity when None."""
    if matrix is None:
        return img
    m = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    out = np.einsum("ck,hwk->hwc", m, img[:, :, :3].astype(np.float32, copy=False))
    return np.clip(out, 0.0, None)


def unmix_block_reason(process: ProcessConfig) -> str:
    """Why the unmix cannot apply — "transparency", "linear_raw", or "" when it can.

    A **transparency** is refused outright. The matrix is only meaningful for a capture
    made under narrowband light, and narrowband is not supported for slides: no profile
    can be built for a broadband capture, so a matrix on a slide is either inapplicable
    or inherited from a sticky rig setting that no longer describes the render. It stays
    baked in the config either way, so switching the frame back to a negative restores it.

    **Linear RAW** off is the other block. Matrices are always calibrated from neutral-WB
    decodes (the dialog forces use_camera_wb=False); a RAW buffer then carries the camera's
    as-shot per-channel gains, and a diagonal gain does not commute with the non-diagonal
    unmix — the leftover term is sign-flipped, so the correction overcorrects rather than
    degrading gracefully.

    Single source of truth: the sidebar greys the panel on this and names the reason, the
    pipeline skips on effective_sensor_matrix below, so the two cannot disagree.
    """
    if process.process_mode == ProcessMode.E6:
        return "transparency"
    if not process.linear_raw:
        return "linear_raw"
    return ""


def sensor_unmix_available(process: ProcessConfig) -> bool:
    """Whether the unmix can apply at all; see unmix_block_reason for why not."""
    return not unmix_block_reason(process)


def effective_sensor_matrix(process: ProcessConfig) -> Optional[tuple]:
    """The baked matrix, or None when the basis makes it invalid to apply."""
    if not sensor_unmix_available(process):
        return None
    return process.sensor_matrix


def sensor_token(process: ProcessConfig) -> str:
    """Identity of the baked sensor matrix, folded into the render source hash."""
    matrix = effective_sensor_matrix(process)
    if matrix is None:
        return ""
    return "|sn:" + ",".join(f"{v:.6g}" for v in matrix)
