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


# Working-space output transform: ProPhoto RGB (ROMM) TRC — gamma 1.8 with a linear
# toe (slope 16) below 1/512. Applied at the pipeline boundary; composes with the
# ProPhoto ICC. Mirrored in WGSL oetf_encode/oetf_decode.
_WORKING_GAMMA = 1.8
_ROMM_LIN_BREAK = 1.0 / 512.0  # linear-domain toe break (encode)
_ROMM_ENC_BREAK = 16.0 / 512.0  # encoded-domain toe break (decode) = 1/32


@parallel_njit(cache=True, fastmath=True)
def _oetf_encode_flat(flat: np.ndarray, inv_gamma: float, lin_break: float) -> np.ndarray:
    """Row-parallel ProPhoto ROMM encode over a flattened buffer (shape-agnostic)."""
    n = flat.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        x = flat[i]
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        out[i] = x * 16.0 if x < lin_break else x**inv_gamma
    return out


@parallel_njit(cache=True, fastmath=True)
def _oetf_decode_flat(flat: np.ndarray, gamma: float, enc_break: float) -> np.ndarray:
    """Inverse of _oetf_encode_flat."""
    n = flat.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        e = flat[i]
        if e < 0.0:
            e = 0.0
        out[i] = e / 16.0 if e < enc_break else e**gamma
    return out


def working_oetf_encode(img: np.ndarray) -> np.ndarray:
    """Scene-linear -> display-encoded code values [0,1] (ProPhoto ROMM TRC)."""
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    return _oetf_encode_flat(flat, np.float32(1.0 / _WORKING_GAMMA), np.float32(_ROMM_LIN_BREAK)).reshape(img.shape)


def working_oetf_decode(img: np.ndarray) -> np.ndarray:
    """Inverse of working_oetf_encode."""
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    return _oetf_decode_flat(flat, np.float32(_WORKING_GAMMA), np.float32(_ROMM_ENC_BREAK)).reshape(img.shape)


# CIELAB in the working space (ProPhoto RGB / ROMM, D50): ProPhoto primaries.
# Mirrors the WGSL rgb_to_lab; OpenCV's float Lab scale (L 0-100).
_PROPHOTO_TO_XYZ = np.array(
    [
        [0.7976749, 0.1351917, 0.0313534],
        [0.2880402, 0.7118741, 0.0000857],
        [0.0000000, 0.0000000, 0.8252100],
    ],
    dtype=np.float32,
)
_XYZ_TO_PROPHOTO = np.array(
    [
        [1.3459433, -0.2556075, -0.0511118],
        [-0.5445989, 1.5081673, 0.0205351],
        [0.0000000, 0.0000000, 1.2118128],
    ],
    dtype=np.float32,
)
_D50_WHITE = np.array([0.96422, 1.00000, 0.82521], dtype=np.float32)
_LAB_EPS = 0.008856
_LAB_KAPPA = 7.787


@parallel_njit(cache=True, fastmath=True)
def _rgb_to_lab_kernel(px: np.ndarray, m: np.ndarray, white: np.ndarray, eps: float, kappa: float) -> np.ndarray:
    """Row-parallel linear ProPhoto RGB -> CIELAB (D50) over an (N, 3) pixel list."""
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
    """Row-parallel inverse: CIELAB (D50) -> linear ProPhoto RGB over an (N, 3) pixel list."""
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


def rgb_to_lab_working(img: np.ndarray) -> np.ndarray:
    """Linear ProPhoto RGB -> CIELAB (D50). No transfer decode — the buffer is linear.

    Accepts any array whose last axis is the 3 RGB channels ((H, W, 3), (N, 3), ...)."""
    arr = np.ascontiguousarray(img, dtype=np.float32)
    out = _rgb_to_lab_kernel(arr.reshape(-1, 3), _PROPHOTO_TO_XYZ, _D50_WHITE, np.float32(_LAB_EPS), np.float32(_LAB_KAPPA))
    return out.reshape(arr.shape)


def lab_to_rgb_working(lab: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab_working: CIELAB (D50) -> linear ProPhoto RGB (no encode)."""
    arr = np.ascontiguousarray(lab, dtype=np.float32)
    out = _lab_to_rgb_kernel(arr.reshape(-1, 3), _XYZ_TO_PROPHOTO, _D50_WHITE, np.float32(_LAB_EPS), np.float32(_LAB_KAPPA))
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
