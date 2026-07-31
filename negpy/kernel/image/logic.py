import hashlib
import os
from typing import Any, Optional
import numpy as np
from numba import njit, prange  # type: ignore
from negpy.kernel.system.parallel import parallel_njit
from negpy.domain.types import LUMA_R, LUMA_G, LUMA_B
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


@njit(cache=True, fastmath=True)
def _get_luminance_jit(img: np.ndarray) -> np.ndarray:
    """
    Rec. 709 luminance.
    """
    h, w, _ = img.shape
    res = np.empty((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            res[y, x] = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
    return res


@njit(cache=True, fastmath=True)
def _to_uint16_jit(img: np.ndarray) -> np.ndarray:
    """
    Scale to uint16 (clips & handles NaNs).
    """
    res = np.empty_like(img, dtype=np.uint16)
    img_flat = img.reshape(-1)
    res_flat = res.reshape(-1)

    for i in range(len(img_flat)):
        val = img_flat[i]
        if np.isnan(val):
            v = 0.0
        else:
            v = val * 65535.0

        if v < 0.0:
            v = 0.0
        elif v > 65535.0:
            v = 65535.0

        res_flat[i] = np.uint16(v)
    return res


@njit(cache=True, fastmath=True)
def _to_uint8_jit(img: np.ndarray) -> np.ndarray:
    """
    Scale to uint8 (clips & handles NaNs).
    """
    res = np.empty_like(img, dtype=np.uint8)
    img_flat = img.reshape(-1)
    res_flat = res.reshape(-1)

    for i in range(len(img_flat)):
        val = img_flat[i]
        if np.isnan(val):
            v = 0.0
        else:
            v = val * 255.0

        if v < 0.0:
            v = 0.0
        elif v > 255.0:
            v = 255.0

        res_flat[i] = np.uint8(v)
    return res


@njit(cache=True, fastmath=True)
def uint8_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT conversion from uint8 to float32 [0.0, 1.0].
    """
    h, w, c = img.shape
    res = np.empty((h, w, c), dtype=np.float32)
    inv_255 = 1.0 / 255.0
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                res[y, x, ch] = np.float32(img[y, x, ch]) * inv_255
    return res


@njit(cache=True, fastmath=True)
def uint16_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT conversion from uint16 to float32 [0.0, 1.0].
    """
    h, w, c = img.shape
    res = np.empty((h, w, c), dtype=np.float32)
    inv_65535 = 1.0 / 65535.0
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                res[y, x, ch] = np.float32(img[y, x, ch]) * inv_65535
    return res


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB gamma-encoded float32 image to linear light (IEC 61966-2-1)."""
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


# Working-space output transform: Adobe RGB (1998) TRC — a pure 563/256 power with no
# linear segment. Applied at the pipeline boundary; composes with the Adobe RGB ICC.
# Mirrored in WGSL oetf_encode/oetf_decode.
_WORKING_GAMMA = 563.0 / 256.0  # 2.19921875


@parallel_njit(cache=True, fastmath=True)
def _oetf_encode_flat(flat: np.ndarray, inv_gamma: float) -> np.ndarray:
    """Row-parallel working-space encode over a flattened buffer (shape-agnostic)."""
    n = flat.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        x = flat[i]
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        out[i] = x**inv_gamma
    return out


@parallel_njit(cache=True, fastmath=True)
def _oetf_decode_flat(flat: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse of _oetf_encode_flat."""
    n = flat.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        e = flat[i]
        if e < 0.0:
            e = 0.0
        out[i] = e**gamma
    return out


def working_oetf_encode(img: np.ndarray) -> np.ndarray:
    """Scene-linear -> display-encoded code values [0,1] (Adobe RGB TRC)."""
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    return _oetf_encode_flat(flat, np.float32(1.0 / _WORKING_GAMMA)).reshape(img.shape)


def working_oetf_decode(img: np.ndarray) -> np.ndarray:
    """Inverse of working_oetf_encode."""
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    return _oetf_decode_flat(flat, np.float32(_WORKING_GAMMA)).reshape(img.shape)


# CIELAB in the working space (Adobe RGB 1998, D65): Adobe RGB primaries.
# Mirrors the WGSL rgb_to_lab; OpenCV's float Lab scale (L 0-100).
_WORKING_TO_XYZ = np.array(
    [
        [0.5767309, 0.1855540, 0.1881852],
        [0.2973769, 0.6273491, 0.0752741],
        [0.0270343, 0.0706872, 0.9911085],
    ],
    dtype=np.float32,
)
_XYZ_TO_WORKING = np.array(
    [
        [2.0413690, -0.5649464, -0.3446944],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0134474, -0.1183897, 1.0154096],
    ],
    dtype=np.float32,
)
_WORKING_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
_LAB_EPS = 0.008856
_LAB_KAPPA = 7.787


@parallel_njit(cache=True, fastmath=True)
def _rgb_to_lab_kernel(px: np.ndarray, m: np.ndarray, white: np.ndarray, eps: float, kappa: float) -> np.ndarray:
    """Row-parallel linear working RGB -> CIELAB (D65) over an (N, 3) pixel list."""
    n = px.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    c = np.float32(16.0 / 116.0)
    for i in prange(n):
        r = px[i, 0]
        g = px[i, 1]
        b = px[i, 2]
        if r < 0.0:
            r = 0.0
        if g < 0.0:
            g = 0.0
        if b < 0.0:
            b = 0.0
        xr = (m[0, 0] * r + m[0, 1] * g + m[0, 2] * b) / white[0]
        yr = (m[1, 0] * r + m[1, 1] * g + m[1, 2] * b) / white[1]
        zr = (m[2, 0] * r + m[2, 1] * g + m[2, 2] * b) / white[2]
        fx = xr ** (1.0 / 3.0) if xr > eps else kappa * xr + c
        fy = yr ** (1.0 / 3.0) if yr > eps else kappa * yr + c
        fz = zr ** (1.0 / 3.0) if zr > eps else kappa * zr + c
        out[i, 0] = 116.0 * fy - 16.0
        out[i, 1] = 500.0 * (fx - fy)
        out[i, 2] = 200.0 * (fy - fz)
    return out


@parallel_njit(cache=True, fastmath=True)
def _lab_to_rgb_kernel(lab: np.ndarray, m: np.ndarray, white: np.ndarray, eps: float, kappa: float) -> np.ndarray:
    """Row-parallel inverse: CIELAB (D65) -> linear working RGB over an (N, 3) pixel list."""
    n = lab.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    c = np.float32(16.0 / 116.0)
    for i in prange(n):
        fy = (lab[i, 0] + 16.0) / 116.0
        fx = lab[i, 1] / 500.0 + fy
        fz = fy - lab[i, 2] / 200.0
        fx3 = fx * fx * fx
        fy3 = fy * fy * fy
        fz3 = fz * fz * fz
        xr = (fx3 if fx3 > eps else (fx - c) / kappa) * white[0]
        yr = (fy3 if fy3 > eps else (fy - c) / kappa) * white[1]
        zr = (fz3 if fz3 > eps else (fz - c) / kappa) * white[2]
        r = m[0, 0] * xr + m[0, 1] * yr + m[0, 2] * zr
        g = m[1, 0] * xr + m[1, 1] * yr + m[1, 2] * zr
        b = m[2, 0] * xr + m[2, 1] * yr + m[2, 2] * zr
        out[i, 0] = r if r > 0.0 else 0.0
        out[i, 1] = g if g > 0.0 else 0.0
        out[i, 2] = b if b > 0.0 else 0.0
    return out


@njit(inline="always")
def _in_gamut_lab(l_val: float, a: float, b: float, m: np.ndarray, white: np.ndarray, eps: float, kappa: float, c: float) -> bool:
    """Whether (L, a, b) decodes to linear working RGB within [0,1] on all channels."""
    fy = (l_val + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    fx3 = fx * fx * fx
    fy3 = fy * fy * fy
    fz3 = fz * fz * fz
    xr = (fx3 if fx3 > eps else (fx - c) / kappa) * white[0]
    yr = (fy3 if fy3 > eps else (fy - c) / kappa) * white[1]
    zr = (fz3 if fz3 > eps else (fz - c) / kappa) * white[2]
    r = m[0, 0] * xr + m[0, 1] * yr + m[0, 2] * zr
    g = m[1, 0] * xr + m[1, 1] * yr + m[1, 2] * zr
    bl = m[2, 0] * xr + m[2, 1] * yr + m[2, 2] * zr
    tol = np.float32(1e-4)
    one = np.float32(1.0)
    return r >= -tol and r <= one + tol and g >= -tol and g <= one + tol and bl >= -tol and bl <= one + tol


# Skin-tone mask for skin_chroma_rein. Axis-aligned reduction of the CIELAB
# skin locus: hue is the axis that stays put across the whole tonal range
# (literature places skin at 40-65deg; a rendered portrait frame here measured
# median ~52deg, p10-p90 ~41-73deg), chroma is bounded (skin ~12-40, sunburnt
# ~60, a pure saturated red ~104), lightness is free apart from the two ends
# where the hue angle turns to noise. The chroma window is what keeps saturated
# reds out of the band -- hue alone cannot tell a face from a red car, since
# pure red sits at ~35-40deg in this working space.
_SKIN_HUE_CENTER_DEG = np.float32(52.0)
_SKIN_HUE_WIDTH_DEG = np.float32(25.0)
_SKIN_CHROMA_FULL = np.float32(55.0)
_SKIN_CHROMA_ZERO = np.float32(85.0)
_SKIN_L_LO = np.float32(15.0)
_SKIN_L_HI = np.float32(95.0)

# Chroma ceiling the rein pulls toward: _SKIN_CEIL_AT_FULL / strength, so the
# ceiling runs off past any reachable chroma as strength approaches 0 and the
# control fades out continuously. A lerp to some finite "off" ceiling instead
# would step discontinuously the moment the slider left 0. The knee starts at a
# fraction of the ceiling so ordinary skin below it passes through untouched.
_SKIN_CEIL_AT_FULL = np.float32(22.0)
_SKIN_KNEE_START_FRAC = np.float32(0.6)


@njit(inline="always")
def _smoothstep_scalar(e0: float, e1: float, x: float) -> float:
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return float(t * t * (np.float32(3.0) - np.float32(2.0) * t))


@njit(inline="always")
def _skin_weight(l_val: float, a: float, b: float) -> float:
    """0 (not skin) to 1 (dead centre of the skin locus). Near-neutral pixels
    have an undefined hue angle, so gate on chroma before trusting atan2."""
    chroma = np.sqrt(a * a + b * b)
    if chroma < np.float32(2.0):
        return 0.0
    hue_deg = np.degrees(np.arctan2(b, a))
    dist = hue_deg - _SKIN_HUE_CENTER_DEG
    # Wrap to [-180, 180] so e.g. 179 vs -179 reads as 2deg apart, not 358.
    dist = dist - np.float32(360.0) * np.round(dist / np.float32(360.0))
    x = dist / _SKIN_HUE_WIDTH_DEG
    w_hue = np.exp(np.float32(-0.5) * x * x)
    w_chroma = np.float32(1.0) - _smoothstep_scalar(_SKIN_CHROMA_FULL, _SKIN_CHROMA_ZERO, chroma)
    w_light = _smoothstep_scalar(np.float32(0.0), _SKIN_L_LO, l_val) * (
        np.float32(1.0) - _smoothstep_scalar(_SKIN_L_HI, np.float32(100.0), l_val)
    )
    return float(w_hue * w_chroma * w_light)


@parallel_njit(cache=True, fastmath=True)
def _skin_chroma_rein_kernel(lab: np.ndarray, strength: float) -> np.ndarray:
    """Soft chroma ceiling inside the skin mask -- see skin_chroma_rein."""
    n = lab.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    one = np.float32(1.0)
    ceiling = _SKIN_CEIL_AT_FULL / strength
    start = _SKIN_KNEE_START_FRAC * ceiling
    span = ceiling - start
    for i in prange(n):
        l_val = lab[i, 0]
        a = lab[i, 1]
        b = lab[i, 2]
        chroma = np.sqrt(a * a + b * b)
        scale = one
        if chroma > start:
            w = _skin_weight(l_val, a, b)
            if w > 0.0:
                knee = start + span * (one - np.exp(-(chroma - start) / span))
                scale = (chroma + w * (knee - chroma)) / chroma
        out[i, 0] = l_val
        out[i, 1] = a * scale
        out[i, 2] = b * scale
    return out


def skin_chroma_rein(lab: np.ndarray, strength: float) -> np.ndarray:
    """
    Pull skin-hued pixels down toward a chroma ceiling, leaving hue and L*
    untouched (a* and b* scale together). Independent of the Chroma scale, so
    it also reins in skin that arrived over-chromatic from the print curve.

    One-directional: chroma is only ever reduced, never added. That is what
    keeps saturation=0.0 reaching true grey without a special case, and what
    makes it safe to run after the gamut knee -- a smaller chroma cannot
    overshoot a gamut the full-strength push already fitted inside.

    `strength` drives the ceiling itself rather than a blend amount, so the
    control has one meaning: how tightly skin chroma is reined. The knee is the
    same softplus shape as gamut_aware_chroma_scale's and the print curve's
    toe/shoulder bounds. Accepts any array whose last axis is (L, a, b).
    """
    arr = np.ascontiguousarray(lab, dtype=np.float32)
    if strength <= 0.0:
        return arr
    out = _skin_chroma_rein_kernel(arr.reshape(-1, 3), np.float32(strength))
    return out.reshape(arr.shape)


@parallel_njit(cache=True, fastmath=True)
def _gamut_aware_chroma_scale_kernel(
    lab: np.ndarray, saturation: float, m: np.ndarray, white: np.ndarray, eps: float, kappa: float, iters: int
) -> np.ndarray:
    """
    Per-pixel gamut-aware chroma scale: pixels comfortably inside the display
    gamut get the full flat `saturation` scale, unchanged from a plain a*/b*
    multiply. Pixels whose full-strength push would clip get a smooth
    softplus-style knee toward their own actual in-gamut headroom instead of
    an abrupt per-channel RGB clamp -- same knee shape as the print curve's
    toe/shoulder bounds. Bisection converges to <0.1% error in 10 iterations
    (measured against a 24-iteration reference); going lower starts costing
    real precision, going higher buys nothing visible.

    An independent per-channel hard clamp shifts hue even though the a*/b*
    scale itself preserves hue exactly (uniform scaling of both components
    leaves atan2(b,a) unchanged) -- clamping only the channel(s) that overshot
    changes the R:G:B ratio the eye actually sees. This sidesteps that by
    never overshooting in the first place.
    """
    n = lab.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    c = np.float32(16.0 / 116.0)
    one = np.float32(1.0)
    for i in prange(n):
        l_val = lab[i, 0]
        a = lab[i, 1]
        b = lab[i, 2]
        if saturation <= one:
            # Desaturation never overshoots the gamut -- moving toward the
            # achromatic axis can't push a channel further out of range.
            eff = saturation
        else:
            if _in_gamut_lab(l_val, a * saturation, b * saturation, m, white, eps, kappa, c):
                # Full push already lands in gamut -- use it directly. Without this,
                # bisecting only within [1, saturation] always converges lo toward
                # saturation itself (an artifact of that being the search's own
                # upper bound, not a real constraint), and the knee formula below
                # then misreads "the boundary is right at the edge of what I asked
                # for" and throttles pixels that were never going to clip at all.
                eff = saturation
            else:
                lo = one
                hi = saturation
                still_ok = _in_gamut_lab(l_val, a, b, m, white, eps, kappa, c)
                for _ in range(iters):
                    mid = (lo + hi) / np.float32(2.0)
                    if still_ok and _in_gamut_lab(l_val, a * mid, b * mid, m, white, eps, kappa, c):
                        lo = mid
                    else:
                        hi = mid
                s_max = lo
                if s_max < one + np.float32(1e-4):
                    s_max = one + np.float32(1e-4)
                knee = s_max - one
                eff = one + knee * (one - np.exp(-(saturation - one) / knee))
        out[i, 0] = l_val
        out[i, 1] = a * eff
        out[i, 2] = b * eff
    return out


def gamut_aware_chroma_scale(lab: np.ndarray, saturation: float, iters: int = 10) -> np.ndarray:
    """Public entry point for _gamut_aware_chroma_scale_kernel -- see its docstring.
    Accepts any array whose last axis is (L, a, b); returns the same shape."""
    arr = np.ascontiguousarray(lab, dtype=np.float32)
    out = _gamut_aware_chroma_scale_kernel(
        arr.reshape(-1, 3), np.float32(saturation), _XYZ_TO_WORKING, _WORKING_WHITE, np.float32(_LAB_EPS), np.float32(_LAB_KAPPA), iters
    )
    return out.reshape(arr.shape)


def rgb_to_lab_working(img: np.ndarray) -> np.ndarray:
    """Linear working RGB -> CIELAB (D65). No transfer decode — the buffer is linear.

    Accepts any array whose last axis is the 3 RGB channels ((H, W, 3), (N, 3), ...)."""
    arr = np.ascontiguousarray(img, dtype=np.float32)
    out = _rgb_to_lab_kernel(arr.reshape(-1, 3), _WORKING_TO_XYZ, _WORKING_WHITE, np.float32(_LAB_EPS), np.float32(_LAB_KAPPA))
    return out.reshape(arr.shape)


def lab_to_rgb_working(lab: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab_working: CIELAB (D65) -> linear working RGB (no encode)."""
    arr = np.ascontiguousarray(lab, dtype=np.float32)
    out = _lab_to_rgb_kernel(arr.reshape(-1, 3), _XYZ_TO_WORKING, _WORKING_WHITE, np.float32(_LAB_EPS), np.float32(_LAB_KAPPA))
    return out.reshape(arr.shape)


@njit(cache=True, fastmath=True)
def _float_to_uint8_luma_jit(img: np.ndarray) -> np.ndarray:
    """
    Luminance -> uint8.
    """
    scale = 255.0
    dtype = np.uint8

    if img.ndim == 2:
        h, w = img.shape
        res = np.empty((h, w), dtype=dtype)
        for y in range(h):
            for x in range(w):
                v = img[y, x] * scale + 0.5
                if v < 0:
                    v = 0
                elif v > scale:
                    v = scale
                res[y, x] = dtype(v)
        return res
    else:
        h, w, c = img.shape
        res = np.empty((h, w), dtype=dtype)
        for y in range(h):
            for x in range(w):
                lum = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
                v = lum * scale + 0.5
                if v < 0:
                    v = 0
                elif v > scale:
                    v = scale
                res[y, x] = dtype(v)
        return res


@njit(cache=True, fastmath=True)
def _float_to_uint16_luma_jit(img: np.ndarray) -> np.ndarray:
    """
    Luminance -> uint16.
    """
    scale = 65535.0
    dtype = np.uint16

    if img.ndim == 2:
        h, w = img.shape
        res = np.empty((h, w), dtype=dtype)
        for y in range(h):
            for x in range(w):
                v = img[y, x] * scale + 0.5
                if v < 0:
                    v = 0
                elif v > scale:
                    v = scale
                res[y, x] = dtype(v)
        return res
    else:
        h, w, c = img.shape
        res = np.empty((h, w), dtype=dtype)
        for y in range(h):
            for x in range(w):
                lum = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
                v = lum * scale + 0.5
                if v < 0:
                    v = 0
                elif v > scale:
                    v = scale
                res[y, x] = dtype(v)
        return res


def float_to_uint_luma(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    Fuses luminance calculation and bit-depth conversion.
    Dispatches to specialized JIT kernels based on bit_depth.
    """
    if bit_depth == 16:
        res_16: np.ndarray = _float_to_uint16_luma_jit(img)
        return res_16
    res_8: np.ndarray = _float_to_uint8_luma_jit(img)
    return res_8


def float_to_uint16(img: np.ndarray) -> np.ndarray:
    """Converts float32 [0,1] buffer to uint16."""
    res: np.ndarray = _to_uint16_jit(np.ascontiguousarray(img, dtype=np.float32))
    return res


def float_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts float32 [0,1] buffer to uint8."""
    res: np.ndarray = _to_uint8_jit(np.ascontiguousarray(img, dtype=np.float32))
    return res


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Broadens single-channel or 2D arrays to 3-channel RGB.
    """
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img] * 3, axis=-1)
    return img


def apply_exif_orientation(arr: np.ndarray, orientation: Optional[int]) -> np.ndarray:
    """
    Bake an EXIF orientation value (1-8) into pixels so the array displays upright.
    Works on HxW (IR) and HxWxC (RGB) arrays. Returns the input unchanged for 1/None.
    """
    if not orientation or orientation == 1:
        return arr
    if orientation == 2:
        return np.ascontiguousarray(np.fliplr(arr))
    if orientation == 3:
        return np.ascontiguousarray(np.rot90(arr, 2))
    if orientation == 4:
        return np.ascontiguousarray(np.flipud(arr))
    if orientation == 5:
        return np.ascontiguousarray(np.swapaxes(arr, 0, 1))
    if orientation == 6:  # rotate 90° CW
        return np.ascontiguousarray(np.rot90(arr, 3))
    if orientation == 7:
        return np.ascontiguousarray(np.rot90(np.swapaxes(arr, 0, 1), 2))
    if orientation == 8:  # rotate 90° CCW
        return np.ascontiguousarray(np.rot90(arr, 1))
    return arr


def get_luminance(img: np.ndarray) -> np.ndarray:
    """
    Calculates relative luminance. Supports (H, W, 3) and (N, 3) arrays.
    """
    if img.ndim == 3:
        return ensure_image(_get_luminance_jit(np.ascontiguousarray(img.astype(np.float32))))

    return LUMA_R * img[..., 0] + LUMA_G * img[..., 1] + LUMA_B * img[..., 2]


def calculate_file_hash(file_path: str) -> str:
    """
    Fingerprint using file size + head/tail samples.
    """
    try:
        file_size = os.path.getsize(file_path)
        hasher = hashlib.sha256()
        hasher.update(str(file_size).encode())

        with open(file_path, "rb") as f:
            hasher.update(f.read(1024 * 1024))
            if file_size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, os.SEEK_END)
                hasher.update(f.read(1024 * 1024))

        return hasher.hexdigest()
    except Exception as e:
        import uuid

        logger.error(f"Hash error for {file_path}: {e}")
        return f"err_{uuid.uuid4()}"


def prepare_thumbnail(img: Any, size: int) -> Any:
    """
    Resizes and pads an image to a square of given size.
    Returns a PIL.Image.
    """
    from PIL import Image

    # Copy to avoid mutating original
    img_copy = img.copy()
    img_copy.thumbnail((size, size), Image.Resampling.LANCZOS)

    # Create dark square background
    square_img = Image.new("RGB", (size, size), (14, 17, 23))
    # Center the thumbnail
    offset_x = (size - img_copy.width) // 2
    offset_y = (size - img_copy.height) // 2
    square_img.paste(img_copy, (offset_x, offset_y))

    return square_img
