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

# Two-Scale: exact Linear while its gain on a channel's own value stays within TWO_SCALE_START
# of its gain at the film base; the capped detail takes over fully at TWO_SCALE_FULL.
TWO_SCALE_START = 1.15
TWO_SCALE_FULL = 1.5
# Color-layer blur as a fraction of the long edge, so preview and export split at the same detail.
_TWO_SCALE_SIGMA = 1.0 / 1200.0

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
    if mode in (SensorUnmix.DENSITY, SensorUnmix.TWO_SCALE):
        out = _density_unmix(x, m) if mode == SensorUnmix.DENSITY else _two_scale_unmix(x, m)
        if out is not None:
            return out
    out = np.einsum("ck,hwk->hwc", m, x)
    if mode == SensorUnmix.SOFT_FLOOR:
        out = _soft_floor(x, out, m)
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


def _two_scale_unmix(x: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """Linear where the unmix amplifies a channel no more than it does at the film base.
    Elsewhere the color comes from a blurred, soft-floored unmix and the detail is added back
    with the base's log gain, so grain is never amplified past it. Never reaches zero.
    None when the base gives no usable linearisation point."""
    base = film_base(x)
    if np.any(base <= _LOG_FLOOR) or np.any(m.astype(np.float64) @ base <= _LOG_FLOOR):
        return None
    c = density_unmix_matrix(m, base).astype(np.float32)
    own = np.diag(m).astype(np.float32) / np.diag(c)
    h, w = x.shape[:2]
    sigma = max(1.0, max(h, w) * _TWO_SCALE_SIGMA)
    # The color layer and the weight are smooth by construction, so they are built small.
    step = max(1, int(sigma // 2))
    small = x if step == 1 else cv2.resize(x, (-(-w // step), -(-h // step)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / step)
    color = np.maximum(_soft_floor(small, cv2.transform(small, m), m), _LOG_FLOOR)
    weight = _smoothstep(TWO_SCALE_START, TWO_SCALE_FULL, (small * own / color).max(axis=2))
    if step > 1:
        small, color, weight = (cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) for a in (small, color, weight))
    out = np.clip(cv2.transform(x, m), 0.0, None)
    hit = weight > 0
    if hit.any():
        detail = np.log(np.maximum(x[hit], _LOG_FLOOR)) - np.log(np.maximum(small[hit], _LOG_FLOOR))
        capped = color[hit] * np.exp(detail @ c.T)
        linear = out[hit]
        out[hit] = linear + weight[hit][:, None] * (capped - linear)
    return out


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
