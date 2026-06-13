import hashlib
import os
from typing import Any, Optional
import numpy as np
from numba import njit  # type: ignore
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


# CIELAB in the working space (Adobe RGB 1998, D65): sRGB transfer (matches the encoding) +
# Adobe RGB primaries. Mirrors the WGSL rgb_to_lab; OpenCV's float Lab scale (L 0-100).
_ADOBE_RGB_TO_XYZ = np.array(
    [
        [0.5767309, 0.1855540, 0.1881852],
        [0.2973769, 0.6273491, 0.0752741],
        [0.0270343, 0.0706872, 0.9911085],
    ],
    dtype=np.float32,
)
_XYZ_TO_ADOBE_RGB = np.array(
    [
        [2.0413690, -0.5649464, -0.3446944],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0134474, -0.1183897, 1.0154096],
    ],
    dtype=np.float32,
)
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
_LAB_EPS = 0.008856
_LAB_KAPPA = 7.787


def rgb_to_lab_working(img: np.ndarray) -> np.ndarray:
    """
    sRGB-encoded Adobe-RGB-primaried image -> CIELAB (D65). Working-space-correct
    replacement for cv2.cvtColor(..., COLOR_RGB2LAB), which assumes sRGB primaries.
    """
    rgb = np.clip(img.astype(np.float32), 0.0, None)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92).astype(np.float32)
    xyz = lin @ _ADOBE_RGB_TO_XYZ.T
    xyz = xyz / _D65_WHITE
    f = np.where(xyz > _LAB_EPS, np.cbrt(xyz), _LAB_KAPPA * xyz + 16.0 / 116.0).astype(np.float32)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


def lab_to_rgb_working(lab: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_lab_working: CIELAB (D65) -> sRGB-encoded Adobe RGB."""
    lab = lab.astype(np.float32)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    f3 = f**3
    xyz = np.where(f3 > _LAB_EPS, f3, (f - 16.0 / 116.0) / _LAB_KAPPA).astype(np.float32)
    xyz = xyz * _D65_WHITE
    lin = xyz @ _XYZ_TO_ADOBE_RGB.T
    lin = np.clip(lin, 0.0, None)
    rgb = np.where(lin > 0.0031308, 1.055 * lin ** (1.0 / 2.4) - 0.055, 12.92 * lin)
    return rgb.astype(np.float32)


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
