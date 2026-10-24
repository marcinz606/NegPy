"""
3D LUT-based ICC transform for 16-bit RGB buffers.

PIL's ImageCms cannot transform 16-bit RGB Images (Pillow has no 16-bit
RGB mode in fromarray), so we sample the profile transform at 8-bit grid
points and apply the resulting LUT to uint16 data via trilinear
interpolation. The interpolation keeps the 16-bit dynamic range smooth;
per-sample fidelity is 8-bit (bounded by LUT size and profile non-linearity).
"""

from typing import Any, Optional

import negpy_core
import numpy as np
from PIL import Image, ImageCms

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


def apply_lut_u16(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate `lut` onto a uint16 RGB image."""
    img_c = np.ascontiguousarray(img)
    lut_c = np.ascontiguousarray(lut, dtype=np.float32)
    return negpy_core.apply_lut_u16(img_c, lut_c)


def apply_lut_f32(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate `lut` onto a float32 RGB image in [0, 1]."""
    # No copy when the input is already contiguous f32, on the hot display path
    img_c = np.ascontiguousarray(img, dtype=np.float32)
    lut_c = np.ascontiguousarray(lut, dtype=np.float32)
    return negpy_core.apply_lut_f32(img_c, lut_c)


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


_ENC_LUT_SIZE = 65536


def apply_matrix_trc_u8(img_u8: np.ndarray, src_icc: bytes, dst_icc: bytes) -> Optional[np.ndarray]:
    """Relative-colorimetric transform between two matrix/TRC profiles on a (H,W,3)
    uint8 image: TRC decode, inv(dst_pcs) @ src_pcs with gamut clip, TRC encode.
    None when either profile is not matrix/TRC; the caller falls back to lcms."""
    from negpy.infrastructure.display.icc_profile import (
        extract_pcs_matrix,
        extract_trc_decode_samples,
        is_matrix_trc_profile,
    )

    if not (is_matrix_trc_profile(src_icc) and is_matrix_trc_profile(dst_icc)):
        return None
    m_src = extract_pcs_matrix(src_icc)
    m_dst = extract_pcs_matrix(dst_icc)
    if m_src is None or m_dst is None:
        return None
    dec = extract_trc_decode_samples(src_icc, np.linspace(0.0, 1.0, 256))
    dst_dec_dense = extract_trc_decode_samples(dst_icc, np.linspace(0.0, 1.0, _ENC_LUT_SIZE))
    if dec is None or dst_dec_dense is None:
        return None
    try:
        m = np.linalg.inv(m_dst) @ m_src
    except np.linalg.LinAlgError:
        return None
    # The accumulate guards interp against a flat toe in a table-based curve.
    x_enc = np.linspace(0.0, 1.0, _ENC_LUT_SIZE)
    lin_grid = x_enc
    enc_rows = []
    for ch in range(3):
        d = np.maximum.accumulate(np.clip(dst_dec_dense[ch], 0.0, 1.0))
        enc_rows.append(np.interp(lin_grid, d, x_enc))
    enc_lut = np.ascontiguousarray(np.stack(enc_rows).astype(np.float32))
    dec_lut = np.ascontiguousarray(np.clip(dec, 0.0, 1.0).astype(np.float32))
    m32 = np.ascontiguousarray(m.astype(np.float32))
    return negpy_core.matrix_trc_u8(np.ascontiguousarray(img_u8), dec_lut, m32, enc_lut)
