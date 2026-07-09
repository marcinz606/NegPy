"""
3D LUT-based ICC transform for 16-bit RGB buffers.

PIL's ImageCms cannot transform 16-bit RGB Images (Pillow has no 16-bit
RGB mode in fromarray), so we sample the profile transform at 8-bit grid
points and apply the resulting LUT to uint16 data via trilinear
interpolation. The interpolation keeps the 16-bit dynamic range smooth;
per-sample fidelity is 8-bit (bounded by LUT size and profile non-linearity).
"""

from typing import Any
import numpy as np
from numba import prange  # type: ignore
from PIL import Image, ImageCms

from negpy.kernel.system.parallel import parallel_njit

DEFAULT_LUT_SIZE = 33


def build_3d_lut(
    p_src: Any,
    p_dst: Any,
    rendering_intent: ImageCms.Intent,
    flags: ImageCms.Flags,
    size: int = DEFAULT_LUT_SIZE,
) -> np.ndarray:
    """Return an (N,N,N,3) float32 LUT in [0,1], indexed by [r, g, b]."""
    axis = np.linspace(0, 255, size).round().astype(np.uint8)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.ascontiguousarray(np.stack((r, g, b), axis=-1))  # (N,N,N,3) uint8

    flat = np.ascontiguousarray(grid.reshape(size, size * size, 3))
    src_img = Image.fromarray(flat, mode="RGB")
    transform = ImageCms.buildTransform(
        p_src,
        p_dst,
        "RGB",
        "RGB",
        renderingIntent=rendering_intent,
        flags=flags,
    )
    dst_img = ImageCms.applyTransform(src_img, transform)
    lut = np.asarray(dst_img, dtype=np.float32).reshape(size, size, size, 3) / 255.0
    return np.ascontiguousarray(lut)


@parallel_njit(cache=True, fastmath=True)
def _apply_lut_u16_jit(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    h = img.shape[0]
    w = img.shape[1]
    n = lut.shape[0]
    scale = np.float32(n - 1) / np.float32(65535.0)
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            rf = np.float32(img[y, x, 0]) * scale
            gf = np.float32(img[y, x, 1]) * scale
            bf = np.float32(img[y, x, 2]) * scale
            r0 = int(rf)
            g0 = int(gf)
            b0 = int(bf)
            if r0 >= n - 1:
                r0 = n - 2
            if g0 >= n - 1:
                g0 = n - 2
            if b0 >= n - 1:
                b0 = n - 2
            if r0 < 0:
                r0 = 0
            if g0 < 0:
                g0 = 0
            if b0 < 0:
                b0 = 0
            r1 = r0 + 1
            g1 = g0 + 1
            b1 = b0 + 1
            dr = rf - r0
            dg = gf - g0
            db = bf - b0
            for c in range(3):
                c000 = lut[r0, g0, b0, c]
                c001 = lut[r0, g0, b1, c]
                c010 = lut[r0, g1, b0, c]
                c011 = lut[r0, g1, b1, c]
                c100 = lut[r1, g0, b0, c]
                c101 = lut[r1, g0, b1, c]
                c110 = lut[r1, g1, b0, c]
                c111 = lut[r1, g1, b1, c]
                c00 = c000 * (1.0 - db) + c001 * db
                c01 = c010 * (1.0 - db) + c011 * db
                c10 = c100 * (1.0 - db) + c101 * db
                c11 = c110 * (1.0 - db) + c111 * db
                c0 = c00 * (1.0 - dg) + c01 * dg
                c1 = c10 * (1.0 - dg) + c11 * dg
                v = c0 * (1.0 - dr) + c1 * dr
                vi = int(v * 65535.0 + 0.5)
                if vi < 0:
                    vi = 0
                elif vi > 65535:
                    vi = 65535
                out[y, x, c] = vi
    return out


def apply_lut_u16(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate `lut` onto a uint16 RGB image."""
    img_c = np.ascontiguousarray(img)
    lut_c = np.ascontiguousarray(lut, dtype=np.float32)
    return _apply_lut_u16_jit(img_c, lut_c)


@parallel_njit(cache=True, fastmath=True)
def _apply_lut_f32_jit(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    h = img.shape[0]
    w = img.shape[1]
    n = lut.shape[0]
    scale = np.float32(n - 1)
    out = np.empty_like(img)
    for y in prange(h):
        for x in range(w):
            rf = min(max(img[y, x, 0], np.float32(0.0)), np.float32(1.0)) * scale
            gf = min(max(img[y, x, 1], np.float32(0.0)), np.float32(1.0)) * scale
            bf = min(max(img[y, x, 2], np.float32(0.0)), np.float32(1.0)) * scale
            r0 = int(rf)
            g0 = int(gf)
            b0 = int(bf)
            if r0 >= n - 1:
                r0 = n - 2
            if g0 >= n - 1:
                g0 = n - 2
            if b0 >= n - 1:
                b0 = n - 2
            r1 = r0 + 1
            g1 = g0 + 1
            b1 = b0 + 1
            dr = rf - r0
            dg = gf - g0
            db = bf - b0
            for c in range(3):
                c000 = lut[r0, g0, b0, c]
                c001 = lut[r0, g0, b1, c]
                c010 = lut[r0, g1, b0, c]
                c011 = lut[r0, g1, b1, c]
                c100 = lut[r1, g0, b0, c]
                c101 = lut[r1, g0, b1, c]
                c110 = lut[r1, g1, b0, c]
                c111 = lut[r1, g1, b1, c]
                c00 = c000 * (1.0 - db) + c001 * db
                c01 = c010 * (1.0 - db) + c011 * db
                c10 = c100 * (1.0 - db) + c101 * db
                c11 = c110 * (1.0 - db) + c111 * db
                c0 = c00 * (1.0 - dg) + c01 * dg
                c1 = c10 * (1.0 - dg) + c11 * dg
                out[y, x, c] = c0 * (1.0 - dr) + c1 * dr
    return out


def apply_lut_f32(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate `lut` onto a float32 RGB image in [0, 1]."""
    # no-op (no copy) when input is already contiguous f32 — hot display path
    img_c = np.ascontiguousarray(img, dtype=np.float32)
    lut_c = np.ascontiguousarray(lut, dtype=np.float32)
    return _apply_lut_f32_jit(img_c, lut_c)


def apply_icc_u16_rgb(
    img_u16: np.ndarray,
    p_src: Any,
    p_dst: Any,
    rendering_intent: ImageCms.Intent,
    flags: ImageCms.Flags,
    size: int = DEFAULT_LUT_SIZE,
) -> np.ndarray:
    """Apply an ICC RGB→RGB transform to a (H,W,3) uint16 image."""
    lut = build_3d_lut(p_src, p_dst, rendering_intent, flags, size)
    return apply_lut_u16(img_u16, lut)


def apply_icc_u16_greyscale(
    img_u16: np.ndarray,
    p_src: Any,
    p_dst: Any,
    rendering_intent: ImageCms.Intent,
    flags: ImageCms.Flags,
    size: int = DEFAULT_LUT_SIZE,
) -> np.ndarray:
    """Apply an ICC transform to a (H,W) uint16 greyscale image.

    Extracts the neutral axis (R=G=B diagonal) of the 3D LUT to build a 1D
    grey→grey curve, then applies it via a full 65536-entry lookup table.
    """
    lut = build_3d_lut(p_src, p_dst, rendering_intent, flags, size)
    idx = np.arange(size)
    diag = lut[idx, idx, idx, 0]  # (size,) float32 in [0, 1]
    xs = np.linspace(0, 65535, size, dtype=np.float64)
    full_lut = np.interp(np.arange(65536, dtype=np.float64), xs, diag.astype(np.float64))
    full_lut = np.clip(full_lut * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    return full_lut[img_u16]
