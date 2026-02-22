import hashlib
import os
from typing import Any
import numpy as np
from numba import njit, prange  # type: ignore
from negpy.domain.types import LUMA_R, LUMA_G, LUMA_B
from negpy.kernel.image.validation import ensure_image
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


@njit(parallel=True, cache=True, fastmath=True)
def _get_luminance_jit(img: np.ndarray) -> np.ndarray:
    """
    Rec. 709 luminance.
    """
    h, w, _ = img.shape
    res = np.empty((h, w), dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            res[y, x] = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
    return res


@njit(parallel=True, cache=True, fastmath=True)
def _to_uint16_jit(img: np.ndarray) -> np.ndarray:
    """
    Scale to uint16 (clips & handles NaNs).
    """
    res = np.empty_like(img, dtype=np.uint16)
    img_flat = img.reshape(-1)
    res_flat = res.reshape(-1)

    for i in prange(len(img_flat)):
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


@njit(parallel=True, cache=True, fastmath=True)
def _to_uint8_jit(img: np.ndarray) -> np.ndarray:
    """
    Scale to uint8 (clips & handles NaNs).
    """
    res = np.empty_like(img, dtype=np.uint8)
    img_flat = img.reshape(-1)
    res_flat = res.reshape(-1)

    for i in prange(len(img_flat)):
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


@njit(parallel=True, cache=True, fastmath=True)
def uint8_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT conversion from uint8 to float32 [0.0, 1.0].
    """
    h, w, c = img.shape
    res = np.empty((h, w, c), dtype=np.float32)
    inv_255 = 1.0 / 255.0
    for y in prange(h):
        for x in range(w):
            for ch in range(3):
                res[y, x, ch] = np.float32(img[y, x, ch]) * inv_255
    return res


@njit(parallel=True, cache=True, fastmath=True)
def uint16_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT conversion from uint16 to float32 [0.0, 1.0].
    """
    h, w, c = img.shape
    res = np.empty((h, w, c), dtype=np.float32)
    inv_65535 = 1.0 / 65535.0
    for y in prange(h):
        for x in range(w):
            for ch in range(3):
                res[y, x, ch] = np.float32(img[y, x, ch]) * inv_65535
    return res


@njit(parallel=False, cache=True, fastmath=True)
def uint8_to_float32_seq(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT conversion from uint8 to float32 [0.0, 1.0]. (Sequential)
    """
    h, w, c = img.shape
    res = np.empty((h, w, c), dtype=np.float32)
    inv_255 = 1.0 / 255.0
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                res[y, x, ch] = np.float32(img[y, x, ch]) * inv_255
    return res


@njit(parallel=False, cache=True, fastmath=True)
def uint16_to_float32_seq(img: np.ndarray) -> np.ndarray:
    """
    Fast JIT conversion from uint16 to float32 [0.0, 1.0]. (Sequential)
    """
    h, w, c = img.shape
    res = np.empty((h, w, c), dtype=np.float32)
    inv_65535 = 1.0 / 65535.0
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                res[y, x, ch] = np.float32(img[y, x, ch]) * inv_65535
    return res


@njit(parallel=True, cache=True, fastmath=True)
def _float_to_uint8_luma_jit(img: np.ndarray) -> np.ndarray:
    """
    Luminance -> uint8.
    """
    scale = 255.0
    dtype = np.uint8

    if img.ndim == 2:
        h, w = img.shape
        res = np.empty((h, w), dtype=dtype)
        for y in prange(h):
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
        for y in prange(h):
            for x in range(w):
                lum = LUMA_R * img[y, x, 0] + LUMA_G * img[y, x, 1] + LUMA_B * img[y, x, 2]
                v = lum * scale + 0.5
                if v < 0:
                    v = 0
                elif v > scale:
                    v = scale
                res[y, x] = dtype(v)
        return res


@njit(parallel=True, cache=True, fastmath=True)
def _float_to_uint16_luma_jit(img: np.ndarray) -> np.ndarray:
    """
    Luminance -> uint16.
    """
    scale = 65535.0
    dtype = np.uint16

    if img.ndim == 2:
        h, w = img.shape
        res = np.empty((h, w), dtype=dtype)
        for y in prange(h):
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
        for y in prange(h):
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
    res: np.ndarray = _to_uint16_jit(np.ascontiguousarray(img.astype(np.float32)))
    return res


def float_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts float32 [0,1] buffer to uint8."""
    res: np.ndarray = _to_uint8_jit(np.ascontiguousarray(img.astype(np.float32)))
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
