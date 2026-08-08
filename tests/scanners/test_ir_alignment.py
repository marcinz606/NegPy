"""Regression coverage for the two-pass IR registration fix (genesys 'Transparency
Adapter Infrared' and Plustek-style separate IR scans land off by a few pixels).

Whole-pixel only: interpolating the IR plane biases the per-frame noise sigma
that _ir_normalize_ratio (retouch/logic.py, #659/#715) calibrates against."""

import cv2
import numpy as np

from negpy.infrastructure.scanners.sane_backend import _align_ir_to_rgb


def _texture(h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((h, w), dtype=np.float32)
    return cv2.GaussianBlur(base, (0, 0), 1.5)


def _shift(img, dx, dy):
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def test_align_ir_to_rgb_recovers_whole_pixel_offset_exactly():
    # An integer carriage offset (the realistic case) must round-trip bit-exact —
    # this is the whole point of not interpolating.
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = np.roll(np.roll(base, 2, axis=1), -3, axis=0)  # shifted +2 px right, 3 px up

    aligned = _align_ir_to_rgb(rgb, ir)

    sl = (slice(8, -8), slice(8, -8))
    np.testing.assert_allclose(aligned[sl], base[sl], atol=1e-6)


def test_align_ir_to_rgb_never_interpolates():
    # Every output pixel must be a value copied verbatim from the input IR plane
    # (rounded to the nearest whole-pixel shift), never a blend of neighbours.
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = _shift(base, 1.5, -2.0)  # sub-pixel carriage offset

    aligned = _align_ir_to_rgb(rgb, ir)

    assert np.isin(aligned, ir).all()


def test_align_ir_to_rgb_corrects_carriage_offset():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = _shift(base, 1.5, -2.0)  # second-pass carriage landed a bit high and left

    aligned = _align_ir_to_rgb(rgb, ir)

    sl = (slice(8, -8), slice(8, -8))
    err_aligned = np.abs(aligned[sl] - base[sl]).mean()
    err_raw = np.abs(ir[sl] - base[sl]).mean()
    assert err_aligned < err_raw * 0.5


def test_align_ir_to_rgb_preserves_dtype_and_shape():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    ir = (_shift(base, 1.0, 1.0) * 65535).astype(np.uint16)

    aligned = _align_ir_to_rgb((rgb * 65535).astype(np.uint16), ir)

    assert aligned.dtype == ir.dtype
    assert aligned.shape == ir.shape


def test_align_ir_to_rgb_shape_mismatch_is_noop():
    rgb = np.zeros((10, 10, 3), np.uint16)
    ir = np.zeros((12, 10), np.uint16)
    assert _align_ir_to_rgb(rgb, ir) is ir


def test_align_ir_to_rgb_zero_shift_is_noop_identity():
    base = _texture()
    rgb = np.stack([base, base, base], axis=-1)
    assert _align_ir_to_rgb(rgb, base) is base
