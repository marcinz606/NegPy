import hashlib
import os
from typing import Any, Optional
import negpy_core
import numpy as np
from negpy.domain.types import LUMA_R, LUMA_G, LUMA_B
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _to_uint16_jit(img: np.ndarray) -> np.ndarray:
    """
    Scale a float32 buffer to uint16: rounds to nearest, clips, NaN to 0.

    The +0.5 rounds. A bare cast truncates toward zero, which biases every sample down by
    half a level and doubles the quantisation error: the error distribution runs [-1, 0]
    instead of [-0.5, +0.5].
    """
    return negpy_core.to_uint16(np.ascontiguousarray(img, dtype=np.float32))


def _to_uint8_jit(img: np.ndarray) -> np.ndarray:
    """
    Scale a float32 buffer to uint8: rounds to nearest, clips, NaN to 0. See _to_uint16_jit.
    """
    return negpy_core.to_uint8(np.ascontiguousarray(img, dtype=np.float32))


def uint8_to_float32(img: np.ndarray) -> np.ndarray:
    """
    uint8 to float32 [0.0, 1.0].
    """
    return negpy_core.uint8_to_float32(np.ascontiguousarray(img, dtype=np.uint8))


def rgb_to_rgba_into(src: np.ndarray, dst: np.ndarray) -> None:
    """Copies the three color planes of float32 ``src`` into the RGBA ``dst``; alpha is left
    as is. numpy cannot vectorize a 3-of-4 strided copy."""
    negpy_core.rgb_to_rgba_into(np.asarray(src, dtype=np.float32), dst)


def rgba_to_rgb_into(src: np.ndarray, dst: np.ndarray, oy: int, ox: int) -> None:
    """Writes the RGB of float32 ``src[oy:oy+h, ox:ox+w]`` into the (h, w, 3) ``dst``."""
    negpy_core.rgba_to_rgb_into(np.asarray(src, dtype=np.float32), dst, int(oy), int(ox))


def uint16_to_float32(img: np.ndarray) -> np.ndarray:
    """
    uint16 to float32 [0.0, 1.0].
    """
    return negpy_core.uint16_to_float32(np.ascontiguousarray(img, dtype=np.uint16))


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB gamma-encoded float32 image to linear light (IEC 61966-2-1)."""
    return negpy_core.srgb_to_linear(np.ascontiguousarray(img, dtype=np.float32))


# Working-space output transform: the Adobe RGB (1998) TRC, a pure 563/256 power with no
# linear segment. Applied at the pipeline boundary, and it composes with the Adobe RGB ICC.
# Mirrored in WGSL oetf_encode/oetf_decode.
_WORKING_GAMMA = 563.0 / 256.0  # 2.19921875


def _oetf_encode_flat(flat: np.ndarray, inv_gamma: float) -> np.ndarray:
    """Working-space encode of a flat buffer: clamped to 0..1, raised to `inv_gamma`."""
    return negpy_core.oetf_encode(np.ascontiguousarray(flat, dtype=np.float32), inv_gamma)


def _oetf_decode_flat(flat: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse of _oetf_encode_flat."""
    return negpy_core.oetf_decode(np.ascontiguousarray(flat, dtype=np.float32), gamma)


WORKING_INV_GAMMA = np.float32(1.0 / _WORKING_GAMMA)


def working_oetf_encode(img: np.ndarray) -> np.ndarray:
    """Scene-linear -> display-encoded code values [0,1] (Adobe RGB TRC)."""
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    return _oetf_encode_flat(flat, WORKING_INV_GAMMA).reshape(img.shape)


def working_oetf_decode(img: np.ndarray) -> np.ndarray:
    """Inverse of working_oetf_encode."""
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    return _oetf_decode_flat(flat, np.float32(_WORKING_GAMMA)).reshape(img.shape)


# CIELAB in the working space (Adobe RGB 1998, D65), on Adobe RGB primaries. Mirrors the
# WGSL rgb_to_lab, on OpenCV's float Lab scale (L 0-100).
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


def _matmul_3x3_kernel(px: np.ndarray, m: np.ndarray) -> np.ndarray:
    """3x3 matrix multiply over an (N, 3) pixel array."""
    return negpy_core.matmul_3x3(np.ascontiguousarray(px, dtype=np.float32), m)


def apply_primaries_transform(img: np.ndarray, src_to_xyz: np.ndarray) -> np.ndarray:
    """Apply a primaries-only colour transform (no TRC re-decode from the input profile).

    By the time export calls this, the buffer is already encoded with the
    *working-space* OETF (not the input profile's own TRC — decoding it with
    the wrong one is the double-TRC bug this bypass exists to avoid). The
    XYZ_to_working @ src_to_XYZ matrix is a linear-light operator, so it must
    run on decoded values: decode, matrix-multiply, re-encode.
    """
    m_total = np.ascontiguousarray((_XYZ_TO_WORKING.astype(np.float64) @ src_to_xyz).astype(np.float32))
    linear = working_oetf_decode(img)
    h, w = linear.shape[:2]
    flat = linear.reshape(-1, 3).astype(np.float32, copy=False)
    out = _matmul_3x3_kernel(flat, m_total)
    out = np.clip(out.reshape(h, w, 3), 0.0, 1.0)
    return working_oetf_encode(out)


def _rgb_to_lab_kernel(px: np.ndarray, m: np.ndarray, white: np.ndarray, eps: float, kappa: float) -> np.ndarray:
    """Linear working RGB -> CIELAB (D65) over an (N, 3) pixel list."""
    return negpy_core.rgb_to_lab(np.ascontiguousarray(px, dtype=np.float32), m, white, eps, kappa)


def _lab_to_rgb_kernel(lab: np.ndarray, m: np.ndarray, white: np.ndarray, eps: float, kappa: float) -> np.ndarray:
    """Inverse: CIELAB (D65) -> linear working RGB over an (N, 3) pixel list."""
    return negpy_core.lab_to_rgb(np.ascontiguousarray(lab, dtype=np.float32), m, white, eps, kappa)


def _skin_weight(l_val: float, a: float, b: float) -> float:
    """0 (not skin) to 1 (dead centre of the skin locus); 0 below a chroma of 2, where the hue is noise."""
    return negpy_core.skin_weight(l_val, a, b)


def _skin_chroma_rein_kernel(lab: np.ndarray, strength: float) -> np.ndarray:
    """Soft chroma ceiling inside the skin mask over an (N, 3) Lab list -- see skin_chroma_rein."""
    return negpy_core.skin_chroma_rein(np.ascontiguousarray(lab, dtype=np.float32), strength)


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
    return negpy_core.gamut_aware_chroma_scale(np.ascontiguousarray(lab, dtype=np.float32), saturation, m, white, eps, kappa, iters)


def gamut_aware_chroma_scale(lab: np.ndarray, saturation: float, iters: int = 10) -> np.ndarray:
    """Public entry point for _gamut_aware_chroma_scale_kernel -- see its docstring.
    Accepts any array whose last axis is (L, a, b); returns the same shape."""
    arr = np.ascontiguousarray(lab, dtype=np.float32)
    out = _gamut_aware_chroma_scale_kernel(
        arr.reshape(-1, 3), np.float32(saturation), _XYZ_TO_WORKING, _WORKING_WHITE, np.float32(_LAB_EPS), np.float32(_LAB_KAPPA), iters
    )
    return out.reshape(arr.shape)


# The working Lab constants, in the order negpy_core's Lab-stage kernels take them.
WORKING_LAB = (_WORKING_TO_XYZ, _XYZ_TO_WORKING, _WORKING_WHITE, np.float32(_LAB_EPS), np.float32(_LAB_KAPPA))


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


def float_to_uint_luma(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    Fuses luminance calculation and bit-depth conversion: an HxWx3 float32 buffer (or an
    HxW one, taken as luminance) to HxW uint8, or uint16 for ``bit_depth`` 16.
    """
    return negpy_core.luma_to_uint(np.ascontiguousarray(img, dtype=np.float32), 16 if bit_depth == 16 else 8)


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
        return negpy_core.luminance(np.ascontiguousarray(img[..., :3], dtype=np.float32))

    return LUMA_R * img[..., 0] + LUMA_G * img[..., 1] + LUMA_B * img[..., 2]


_HEAD_TAIL = 1024 * 1024
_INTERIOR_CHUNKS = 16
_INTERIOR_CHUNK = 256 * 1024


def file_hashes(file_path: str) -> tuple[str, str]:
    """(current, legacy) fingerprints in one pass.

    Current = size + 1 MiB head + 1 MiB tail + 16 evenly-spaced 256 KiB interior chunks.
    Head/tail alone collide across same-size scans of one frame, whose container header
    and trailer are byte-identical. Legacy is the pre-interior digest, computed here only
    so persisted edits can be rehomed onto the current one; nothing else may use it.

    Files <= 2 MiB have no interior, so both digests are equal — those identities are
    unchanged by the interior sampling and need no migration.
    """
    try:
        file_size = os.path.getsize(file_path)
        hasher = hashlib.sha256()
        legacy = hashlib.sha256()
        size_bytes = str(file_size).encode()
        hasher.update(size_bytes)
        legacy.update(size_bytes)

        with open(file_path, "rb") as f:
            head = f.read(_HEAD_TAIL)
            hasher.update(head)
            legacy.update(head)
            if file_size > 2 * _HEAD_TAIL:
                f.seek(-_HEAD_TAIL, os.SEEK_END)
                tail = f.read(_HEAD_TAIL)
                hasher.update(tail)
                legacy.update(tail)
                # Chunks overlap once the interior is under 4 MiB, which only makes coverage
                # contiguous. The read count stays bounded either way.
                step = (file_size - 2 * _HEAD_TAIL) // _INTERIOR_CHUNKS
                for i in range(_INTERIOR_CHUNKS):
                    f.seek(_HEAD_TAIL + i * step)
                    hasher.update(f.read(_INTERIOR_CHUNK))

        return hasher.hexdigest(), legacy.hexdigest()
    except Exception as e:
        import uuid

        logger.error(f"Hash error for {file_path}: {e}")
        return f"err_{uuid.uuid4()}", ""


def calculate_file_hash(file_path: str) -> str:
    """Content fingerprint used as the edit-store identity. See :func:`file_hashes`."""
    return file_hashes(file_path)[0]


# The padding around the picture in a square thumbnail from an older cache.
THUMBNAIL_PAD = (14, 17, 23)


def prepare_thumbnail(img: Any, size: int) -> Any:
    """The PIL image scaled so its long edge is *size*."""
    from PIL import Image, ImageOps

    return ImageOps.contain(img, (size, size), Image.Resampling.LANCZOS)


def percentile_from_sorted(sorted_grid: np.ndarray, q: float) -> np.ndarray:
    """
    The three channels' `np.percentile(grid[:, :, ch], q)`, read off `sorted_channel_grid`.

    Bit-exact needs numpy's arithmetic, not its definition: virtual index `(n-1)*(q/100)`
    in float64, the far half interpolated back from the upper sample, and a weight that
    keeps `q`'s own type -- numpy interpolates in float32 for a Python `q` and float64 for
    an `np.float64` one, and both kinds of caller exist here.
    """
    n = sorted_grid.shape[0]
    virtual = (n - 1) * np.true_divide(q, 100)
    lo = min(max(int(np.floor(virtual)), 0), n - 1)
    hi = min(lo + 1, n - 1)
    t = virtual - lo
    if type(q) in (int, float):
        t = float(t)
    a, b = sorted_grid[lo], sorted_grid[hi]
    diff = b - a
    return b - diff * (1 - t) if t >= 0.5 else a + diff * t
