"""Sensor (CFA) crosstalk calibration for single-shot narrowband scans.

The camera's color-filter passbands overlap the narrowband source's bands, so a
pure red/green/blue exposure leaks into the other channels — a fixed property of
the sensor+light pair, independent of film. Calibrated once from three bare-light
exposures and corrected with a 3x3 unmix on the LINEAR capture, before the
log/inversion where the film-dye crosstalk (Density Mixer) lives.

Where the film passes almost none of a band's light, the subtraction leaves a value
smaller than its own grain and calibration error, and the log after it turns that
into maximum density. `SensorUnmix` picks how that case is handled.
"""

from typing import Optional

import cv2
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import get_analysis_crop
from negpy.features.process.logic import narrowband_allowed
from negpy.features.process.models import ProcessConfig, SensorUnmix

_EPS = 1e-4

# Soft floor: no channel is corrected below this fraction of its raw value.
SOFT_FLOOR = 0.15

# Adaptive: full correction while the weakest channel keeps ADAPTIVE_FULL of its signal,
# none below ADAPTIVE_NONE. No channel is ever corrected below ADAPTIVE_NONE of its raw value.
ADAPTIVE_NONE = 0.05
ADAPTIVE_FULL = 0.25
# The strength map is computed at this long edge, so grain does not steer it and preview
# and export get the same map.
_GUIDE_EDGE = 2048

# Density: the film base is the thinnest negative in the frame interior.
_BASE_PERCENTILE = 99.5
_BASE_BUFFER = 0.1
_LOG_FLOOR = 1e-6


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


def apply_sensor_correction(img: ImageBuffer, matrix: Optional[tuple], mode: str = SensorUnmix.LINEAR) -> ImageBuffer:
    """Un-mix a linear (H, W, 3) capture with the baked 3x3; identity when None."""
    if matrix is None:
        return img
    m = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    x = img[:, :, :3].astype(np.float32, copy=False)
    mode = SensorUnmix(mode)
    if mode == SensorUnmix.DENSITY:
        out = _density_unmix(x, m)
        if out is not None:
            return out
    out = np.einsum("ck,hwk->hwc", m, x)
    if mode == SensorUnmix.SOFT_FLOOR:
        out = _soft_floor(x, out, m)
    elif mode == SensorUnmix.ADAPTIVE:
        out = _fade_unreliable(x, out, m)
    return np.clip(out, 0.0, None)


def _smoothstep(lo: float, hi: float, v: np.ndarray) -> np.ndarray:
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _soft_floor(x: np.ndarray, full: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Smooth maximum of the unmix and SOFT_FLOOR of each channel's raw value, per channel.
    Never below either; the knee is as wide as the floor."""
    floor = x * (SOFT_FLOOR * np.diag(m).astype(np.float32))
    gap = full - floor
    gap *= gap
    gap += floor * floor
    np.sqrt(gap, out=gap)
    gap += full
    gap += floor
    gap *= 0.5
    return gap


def _fade_unreliable(x: np.ndarray, full: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Blend each pixel from the full unmix toward its uncorrected value as its weakest
    channel's surviving signal falls. One strength per pixel, so the hue does not turn."""
    diag = np.diag(m).astype(np.float32)
    h, w = x.shape[:2]
    scale = min(1.0, _GUIDE_EDGE / max(h, w))
    guide = x if scale >= 1.0 else cv2.resize(x, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    surviving = np.einsum("ck,hwk->hwc", m, guide) / np.maximum(guide * diag, _EPS * _EPS)
    strength = _smoothstep(ADAPTIVE_NONE, ADAPTIVE_FULL, surviving.min(axis=2)).astype(np.float32)
    if scale < 1.0:
        strength = cv2.resize(strength, (w, h), interpolation=cv2.INTER_LINEAR)

    raw = x * diag
    full -= raw  # now the (negative) amount the unmix removes
    for c in range(3):
        removed = -full[:, :, c]
        cap = np.where(removed > 0, (1.0 - ADAPTIVE_NONE) * raw[:, :, c] / np.maximum(removed, _EPS * _EPS), 1.0)
        np.minimum(strength, cap, out=strength)
    full *= strength[:, :, None]
    full += raw
    return full


def density_unmix_matrix(matrix, base) -> np.ndarray:
    """The sensor matrix linearised in log space around ``base``. Rows sum to 1."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    b = np.asarray(base, dtype=np.float64)
    return (m * b[None, :]) / (m @ b)[:, None]


def film_base(x: np.ndarray) -> np.ndarray:
    """Per-channel film base estimate: the thinnest negative in the frame interior."""
    region = get_analysis_crop(np.asarray(x[::8, ::8, :3], dtype=np.float32), _BASE_BUFFER)
    return np.percentile(region.reshape(-1, 3), _BASE_PERCENTILE, axis=0).astype(np.float64)


def _density_unmix(x: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """The unmix applied to log transmittance, exact at the film base; never reaches zero.
    None when the base gives no usable linearisation point."""
    base = film_base(x)
    at_base = m.astype(np.float64) @ base
    if np.any(base <= _LOG_FLOOR) or np.any(at_base <= _LOG_FLOOR):
        return None
    c = density_unmix_matrix(m, base).astype(np.float32)
    logs = np.log(np.maximum(x, _LOG_FLOOR))
    logs -= np.log(base).astype(np.float32)
    out = np.einsum("ck,hwk->hwc", c, logs)
    out += np.log(at_base).astype(np.float32)
    return np.exp(out, out=out)


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
    if not narrowband_allowed(process):
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
    mode = SensorUnmix(process.sensor_unmix)
    return "|sn:" + ",".join(f"{v:.6g}" for v in matrix) + ("" if mode == SensorUnmix.LINEAR else f"|{mode}")
