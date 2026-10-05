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
from numba import prange

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import get_analysis_crop
from negpy.features.process.logic import narrowband_allowed
from negpy.features.process.models import ProcessConfig, SensorUnmix
from negpy.kernel.system.parallel import parallel_njit

_EPS = 1e-4

# Two-Scale's color layer: no channel is corrected below this fraction of its raw value.
COLOR_FLOOR = 0.15

# Two-Scale: exact Linear while its gain on a channel's own value stays within TWO_SCALE_START
# of its gain at the film base; the capped detail takes over fully at TWO_SCALE_FULL.
TWO_SCALE_START = 1.15
TWO_SCALE_FULL = 1.5
# Color-layer blur as a fraction of the long edge, so preview and export split at the same detail.
_TWO_SCALE_SIGMA = 1.0 / 1200.0

# Density: the film base is the thinnest negative in the frame interior.
_BASE_BUFFER = 0.1
# The base is the median color of this share of an averaged grid, the cells thinnest in all
# three channels together. _BASE_PERCENTILE is each channel's thin end.
_BASE_SHARE = 0.005
_BASE_GRID_EDGE = 512
_BASE_PERCENTILE = 99.5
# A channel whose thin end sits within this fraction of its maximum is clipped.
_CLIP_PLATEAU = 0.998
_LOG_FLOOR = 1e-6
# No channel is read denser than this transmission relative to the base, 3 D, past any film:
# a dead or black-clipped pixel would otherwise reach the other channels through the log.
_DENSITY_RANGE = 1e-3
# Two-Scale: a channel below this fraction of its blurred neighborhood is a defect, not film,
# and carries no detail.
_DEFECT_RATIO = 1.0 / 64.0


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


def apply_sensor_correction(img: ImageBuffer, matrix: Optional[tuple], mode: str) -> ImageBuffer:
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
    out = np.empty_like(x)
    _unmix_kernel(x, m, out)
    return out


@parallel_njit(cache=True, fastmath=True)
def _unmix_kernel(x: np.ndarray, m: np.ndarray, out: np.ndarray) -> None:
    """The unmix, clipped at zero."""
    h, w = x.shape[0], x.shape[1]
    for i in prange(h):
        for j in range(w):
            x0, x1, x2 = x[i, j, 0], x[i, j, 1], x[i, j, 2]
            for c in range(3):
                out[i, j, c] = max(m[c, 0] * x0 + m[c, 1] * x1 + m[c, 2] * x2, 0.0)


@parallel_njit(cache=True, fastmath=True)
def _log_unmix_kernel(x: np.ndarray, c: np.ndarray, offset: np.ndarray, floor: np.ndarray, out: np.ndarray) -> None:
    """exp(c @ ln x + offset) per pixel, with each channel held at its ``floor`` or above."""
    h, w = x.shape[0], x.shape[1]
    for i in prange(h):
        for j in range(w):
            l0 = np.log(max(x[i, j, 0], floor[0]))
            l1 = np.log(max(x[i, j, 1], floor[1]))
            l2 = np.log(max(x[i, j, 2], floor[2]))
            for k in range(3):
                out[i, j, k] = np.exp(c[k, 0] * l0 + c[k, 1] * l1 + c[k, 2] * l2 + offset[k])


@parallel_njit(cache=True, fastmath=True)
def _two_scale_layer_kernel(
    blurred: np.ndarray,
    m: np.ndarray,
    own: np.ndarray,
    floor: float,
    log_floor: float,
    start: float,
    full: float,
    color: np.ndarray,
    weight: np.ndarray,
) -> None:
    """Soft-floored unmix of the blurred capture, and the blend weight: a smoothstep on the
    largest ratio of a channel's unmix gain to its gain at the film base (``own``). The gain
    is the unfloored unmix's, so a channel the floor holds up still reads as amplified."""
    h, w = blurred.shape[0], blurred.shape[1]
    for i in prange(h):
        for j in range(w):
            x0, x1, x2 = blurred[i, j, 0], blurred[i, j, 1], blurred[i, j, 2]
            ratio = 0.0
            for c in range(3):
                u = m[c, 0] * x0 + m[c, 1] * x1 + m[c, 2] * x2
                ratio = max(ratio, blurred[i, j, c] * own[c] / max(u, log_floor))
                f = floor * m[c, c] * blurred[i, j, c]
                d = u - f
                color[i, j, c] = max(0.5 * (u + f + np.sqrt(d * d + f * f)), log_floor)
            t = min(max((ratio - start) / (full - start), 0.0), 1.0)
            weight[i, j] = t * t * (3.0 - 2.0 * t)


@parallel_njit(cache=True, fastmath=True)
def _two_scale_kernel(
    x: np.ndarray,
    blurred: np.ndarray,
    color: np.ndarray,
    weight: np.ndarray,
    m: np.ndarray,
    c: np.ndarray,
    floor: float,
    defect: float,
    out: np.ndarray,
) -> None:
    """Linear, blended by ``weight`` toward ``color`` carrying the detail x / blurred through c.
    A channel under ``defect`` of its neighborhood carries no detail."""
    h, w = x.shape[0], x.shape[1]
    for i in prange(h):
        for j in range(w):
            x0, x1, x2 = x[i, j, 0], x[i, j, 1], x[i, j, 2]
            wt = weight[i, j]
            if wt > 0.0:
                b0, b1, b2 = max(blurred[i, j, 0], floor), max(blurred[i, j, 1], floor), max(blurred[i, j, 2], floor)
                d0 = np.log(x0 / b0) if x0 >= defect * b0 else 0.0
                d1 = np.log(x1 / b1) if x1 >= defect * b1 else 0.0
                d2 = np.log(x2 / b2) if x2 >= defect * b2 else 0.0
            for k in range(3):
                lin = max(m[k, 0] * x0 + m[k, 1] * x1 + m[k, 2] * x2, 0.0)
                if wt > 0.0:
                    capped = color[i, j, k] * np.exp(c[k, 0] * d0 + c[k, 1] * d1 + c[k, 2] * d2)
                    lin += wt * (capped - lin)
                out[i, j, k] = lin


def density_unmix_matrix(matrix, base) -> np.ndarray:
    """The sensor matrix linearised in log space around ``base``. Rows sum to 1."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    b = np.asarray(base, dtype=np.float64)
    return (m * b[None, :]) / (m @ b)[:, None]


def film_base(x: np.ndarray) -> Optional[np.ndarray]:
    """Film base color: the median of the frame-interior cells thinnest in all three channels
    together. One shared cell set, so a colored object cannot pass for the base in a single
    channel. Only the color is meaningful. None when the thin end is clipped in the capture."""
    sample = np.ascontiguousarray(get_analysis_crop(x[::4, ::4, :3], _BASE_BUFFER), dtype=np.float32)
    flat = sample.reshape(-1, 3)
    if flat.shape[0] == 0 or np.any(np.percentile(flat, _BASE_PERCENTILE, axis=0) >= _CLIP_PLATEAU * flat.max(axis=0)):
        return None
    h, w = sample.shape[:2]
    scale = _BASE_GRID_EDGE / max(h, w)
    if scale < 1.0:
        sample = cv2.resize(sample, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    logs = np.log(np.maximum(sample.reshape(-1, 3), _LOG_FLOOR)).astype(np.float64)
    # A cell is as thin as its least-thin channel, each channel against its own thin end.
    thin = (logs - np.percentile(logs, _BASE_PERCENTILE, axis=0)).min(axis=1)
    return np.exp(np.median(logs[thin >= np.quantile(thin, 1.0 - _BASE_SHARE)], axis=0))


def _density_unmix(x: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """The unmix applied to log transmittance, exact at the film base; never reaches zero.
    None when the base gives no usable linearisation point."""
    base = film_base(x)
    if base is None:
        return None
    at_base = m.astype(np.float64) @ base
    if np.any(base <= _LOG_FLOOR) or np.any(at_base <= _LOG_FLOOR):
        return None
    c = density_unmix_matrix(m, base)
    offset = (np.log(at_base) - c @ np.log(base)).astype(np.float32)
    out = np.empty_like(x)
    _log_unmix_kernel(x, c.astype(np.float32), offset, (base * _DENSITY_RANGE).astype(np.float32), out)
    return out


def _two_scale_unmix(x: np.ndarray, m: np.ndarray) -> Optional[np.ndarray]:
    """Linear where the unmix amplifies a channel no more than it does at the film base.
    Elsewhere the color comes from a blurred, soft-floored unmix and the detail is added back
    with the base's log gain, so grain is never amplified past it. Never reaches zero.
    None when the base gives no usable linearisation point."""
    base = film_base(x)
    if base is None or np.any(base <= _LOG_FLOOR) or np.any(m.astype(np.float64) @ base <= _LOG_FLOOR):
        return None
    c = density_unmix_matrix(m, base).astype(np.float32)
    own = (np.diag(m) / np.diag(c)).astype(np.float32)
    h, w = x.shape[:2]
    sigma = max(1.0, max(h, w) * _TWO_SCALE_SIGMA)
    # The color layer and the weight are smooth by construction, so they are built small.
    step = max(1, int(sigma // 2))
    small = x if step == 1 else cv2.resize(x, (-(-w // step), -(-h // step)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / step)
    color = np.empty_like(small)
    weight = np.empty(small.shape[:2], np.float32)
    _two_scale_layer_kernel(small, m, own, COLOR_FLOOR, _LOG_FLOOR, TWO_SCALE_START, TWO_SCALE_FULL, color, weight)
    if step > 1:
        small, color, weight = (cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) for a in (small, color, weight))
    out = np.empty_like(x)
    _two_scale_kernel(x, small, color, weight, m, c, _LOG_FLOOR, _DEFECT_RATIO, out)
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
