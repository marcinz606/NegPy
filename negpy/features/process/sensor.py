"""Sensor (CFA) crosstalk of a sensor+light pair for single-exposure narrowband scans: calibrated from three
bare-light exposures, corrected by a 3x3 unmix on the linear capture, before the log where film-dye crosstalk lives.
Where the film passes almost none of a band, the subtraction falls below grain and the log turns it into maximum
density; `SensorUnmix` picks how that case is handled.
"""

from typing import Optional

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import get_analysis_crop
from negpy.features.process.models import ProcessConfig, ProcessMode, SensorUnmix

_EPS = 1e-4
# Three captures whose normalized responses are this ill-conditioned are one light three times.
_MAX_CONDITION = 1e3

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
# Two-Scale: a channel this far below the other two against its blurred neighborhood is a dead or
# black-clipped photosite; a real edge moves all three channels together.
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
    not_independent = "captures are not independent — three distinct single-band exposures are required"
    # Near-singular: the inverse would amplify capture noise into the correction.
    if np.linalg.cond(s_norm) > _MAX_CONDITION:
        raise ValueError(not_independent)
    try:
        correction = np.linalg.inv(s_norm)
    except np.linalg.LinAlgError:
        raise ValueError(not_independent) from None
    return tuple(float(x) for x in correction.reshape(-1))


def apply_sensor_correction(img: ImageBuffer, matrix: Optional[tuple], mode: str) -> ImageBuffer:
    """Un-mix a linear (H, W, 3) capture with the baked 3x3; identity when None."""
    if matrix is None:
        return img
    m = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    x = np.ascontiguousarray(img[:, :, :3], dtype=np.float32)
    mode = SensorUnmix(mode)
    if mode in (SensorUnmix.DENSITY, SensorUnmix.TWO_SCALE):
        out = _density_unmix(x, m) if mode == SensorUnmix.DENSITY else _two_scale_unmix(x, m)
        if out is not None:
            return out
    out = np.empty_like(x)
    _unmix_kernel(x, m, out)
    return out


def _unmix_kernel(x: np.ndarray, m: np.ndarray, out: np.ndarray) -> None:
    """The unmix, clipped at zero."""
    negpy_core.sensor_unmix(x, m, out)


def _log_unmix_kernel(x: np.ndarray, c: np.ndarray, offset: np.ndarray, floor: np.ndarray, out: np.ndarray) -> None:
    """exp(c @ ln x + offset) per pixel, with each channel held at its ``floor`` or above."""
    negpy_core.sensor_log_unmix(x, c, offset, floor, out)


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
    negpy_core.sensor_two_scale_layer(blurred, m, own, floor, log_floor, start, full, color, weight)


def _two_scale_kernel(
    x: np.ndarray,
    blurred: np.ndarray,
    color: np.ndarray,
    weight: np.ndarray,
    m: np.ndarray,
    c: np.ndarray,
    floor: float,
    log_defect: float,
    out: np.ndarray,
) -> None:
    """Linear, blended by ``weight`` toward ``color`` carrying the detail x / blurred through c.
    A channel whose detail sits ``log_defect`` below both others takes their mean instead."""
    negpy_core.sensor_two_scale(x, blurred, color, weight, m, c, floor, log_defect, out)


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
    # A cell is as thin as its least-thin channel, each channel against its own thin end.
    base = negpy_core.film_base(sample, (_BASE_PERCENTILE, _CLIP_PLATEAU, float(_BASE_GRID_EDGE), _LOG_FLOOR, _BASE_SHARE))
    return None if base is None else np.array(base)


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
    """Linear where the unmix amplifies a channel no more than at the film base. Elsewhere a blurred, soft-floored
    unmix gives the color and the detail returns at the base's log gain, so grain is never amplified past it.
    Never reaches zero; None when the base gives no usable linearisation point."""
    base = film_base(x)
    if base is None or np.any(base <= _LOG_FLOOR) or np.any(m.astype(np.float64) @ base <= _LOG_FLOOR):
        return None
    c = density_unmix_matrix(m, base).astype(np.float32)
    own = (np.diag(m) / np.diag(c)).astype(np.float32)
    h, w = x.shape[:2]
    sigma = max(1.0, max(h, w) * _TWO_SCALE_SIGMA)
    # The color layer and the weight are smooth by construction, so they are built small.
    step = max(1, int(sigma // 2))
    small = x if step == 1 else negpy_core.resize_area(np.ascontiguousarray(x, dtype=np.float32), (-(-w // step), -(-h // step)))
    small = negpy_core.gaussian_blur(np.ascontiguousarray(small, dtype=np.float32), (0, 0), sigma / step)
    color = np.empty_like(small)
    weight = np.empty(small.shape[:2], np.float32)
    _two_scale_layer_kernel(small, m, own, COLOR_FLOOR, _LOG_FLOOR, TWO_SCALE_START, TWO_SCALE_FULL, color, weight)
    if step > 1:
        small, color, weight = (negpy_core.resize_linear(np.ascontiguousarray(a, dtype=np.float32), (w, h)) for a in (small, color, weight))
    out = np.empty_like(x)
    _two_scale_kernel(x, small, color, weight, m, c, _LOG_FLOOR, float(np.log(_DEFECT_RATIO)), out)
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
    mode = SensorUnmix(process.sensor_unmix)
    return "|sn:" + ",".join(f"{v:.6g}" for v in matrix) + ("" if mode == SensorUnmix.LINEAR else f"|{mode}")
